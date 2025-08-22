import re
from typing import Dict, Any, Tuple, Optional, List
from functools import lru_cache
import asyncio

from rapidfuzz.distance import Levenshtein
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore_score
from bert_score.scorer import BERTScorer

_SMOOTH = SmoothingFunction().method1

# Compiled regex patterns (already optimized in original)
BELIEF_OPEN  = re.compile(r"<belief>", re.I)
BELIEF_CLOSE = re.compile(r"</belief>|<\\belief>", re.I)
RESP_OPEN    = re.compile(r"<response>", re.I)
RESP_CLOSE   = re.compile(r"</response>|<\\response>", re.I)
MEM_OPEN     = re.compile(r"<memory>", re.I)
MEM_CLOSE    = re.compile(r"</memory>|<\\memory>", re.I)


# Optimized parsing with early returns and reduced regex operations
def parse_text(text: str) -> Tuple[str, str, str]:
    """Optimized version with early returns and reduced string operations."""
    if not text:
        return "", "", ""
    
    belief, response, memory = "", text, ""
    
    # Parse belief section
    b_open = BELIEF_OPEN.search(text)
    b_close = BELIEF_CLOSE.search(text, b_open.end() if b_open else 0)
    b_span = None
    
    if b_open and b_close:
        belief = text[b_open.end():b_close.start()].strip()
        b_span = (b_open.start(), b_close.end()) 
    elif (not b_open) and b_close:
        belief = text[:b_close.start()].strip()
        b_span = (0, b_close.end())

    # Parse memory section
    m_open = MEM_OPEN.search(text)
    if m_open:
        m_close = MEM_CLOSE.search(text, m_open.end())
        memory = text[m_open.end():m_close.start()].strip() if m_close else text[m_open.end():].strip()

    # Parse response section
    r_open = RESP_OPEN.search(text)
    if r_open:
        r_close = RESP_CLOSE.search(text, r_open.end())
        if r_close:
            response = text[r_open.end():r_close.start()].strip()
        else:
            r_end = m_open.start() if (m_open and m_open.start() >= r_open.end()) else len(text)
            response = text[r_open.end():r_end].strip()
    else:
        # Build leftover text more efficiently
        leftover_parts = []
        last_end = 0
        
        if b_span is not None:
            leftover_parts.append(text[last_end:b_span[0]])
            last_end = b_span[1]
        
        if m_open:
            leftover_parts.append(text[last_end:m_open.start()])
            if m_close:
                leftover_parts.append(text[m_close.end():])
        else:
            leftover_parts.append(text[last_end:])
            
        response = ''.join(leftover_parts).strip()
    
    return belief, response, memory


# Enhanced caching with LRU
@lru_cache(maxsize=8)
def _get_bert_scorer(model_type: str, device: str) -> BERTScorer:
    return BERTScorer(
        model_type=model_type,    
        rescale_with_baseline=True,
        lang='en',
        device=device
    )

# Thread-safe rouge scorer cache
_ROUGE_SCORERS: Dict[str, rouge_scorer.RougeScorer] = {}
_rouge_lock = asyncio.Lock() if asyncio else None

async def _get_rouge_scorer(rouge_type: str) -> rouge_scorer.RougeScorer:
    """Async thread-safe rouge scorer getter."""
    if rouge_type in _ROUGE_SCORERS:
        return _ROUGE_SCORERS[rouge_type]
    
    if _rouge_lock:
        async with _rouge_lock:
            if rouge_type not in _ROUGE_SCORERS:
                _ROUGE_SCORERS[rouge_type] = rouge_scorer.RougeScorer([rouge_type], use_stemmer=False)
    else:
        _ROUGE_SCORERS[rouge_type] = rouge_scorer.RougeScorer([rouge_type], use_stemmer=False)
    
    return _ROUGE_SCORERS[rouge_type]


# Optimized BLEU with pre-computed weights
_BLEU_WEIGHTS_CACHE = {}

def bleu(ref: str, output: str) -> float:
    """Optimized BLEU with cached weights and early validation."""
    if not ref or not output:
        return 0.0
        
    ref_t, out_t = ref.split(), output.split()
    if not ref_t or not out_t:
        return 0.0
        
    n = max(1, min(4, len(ref_t), len(out_t)))
    
    # Cache weights computation
    if n not in _BLEU_WEIGHTS_CACHE:
        if n < 4:
            _BLEU_WEIGHTS_CACHE[n] = tuple((1.0/n if i < n else 0.0) for i in range(4))
        else:
            _BLEU_WEIGHTS_CACHE[n] = (0.25, 0.25, 0.25, 0.25)
    
    weights = _BLEU_WEIGHTS_CACHE[n]
    return float(sentence_bleu([ref_t], out_t, weights=weights, smoothing_function=_SMOOTH))


async def rouge(ref: str, output: str, rouge_type: str) -> float:
    """Async rouge computation with early validation."""
    if not ref or not output:
        return 0.0
    scorer = await _get_rouge_scorer(rouge_type)
    return float(scorer.score(ref, output)[rouge_type].fmeasure)


def edit_sim(ref: str, output: str) -> float:
    """Edit similarity with early validation."""
    if not ref and not output:
        return 1.0
    if not ref or not output:
        return 0.0
    return Levenshtein.normalized_similarity(ref, output) 


def bertscore(ref: str, output: str, model: Optional[str] = None, device: str = "cpu") -> float:
    """Optimized BERTScore with validation and caching."""
    if not ref or not output:
        return 0.0
    
    model = model or "microsoft/deberta-xlarge-mnli"  # Default model
    scorer = _get_bert_scorer(model, device)
    P, R, F1 = scorer.score([output], [ref]) 
    return float(F1.mean())


def aggregate(weighted_scores: List[Tuple[float, float]]) -> float:
    """Optimized aggregation with single pass."""
    if not weighted_scores:
        return 0.0
    
    total_score = total_weight = 0.0
    for score, weight in weighted_scores:
        total_score += weight * score
        total_weight += weight
    
    return total_score / total_weight if total_weight > 0 else 0.0


# Pre-compile metric type sets for faster lookup
ROUGE_TYPES = {"rougeL", "rouge1", "rouge2"}
EDIT_TYPES = {"edit", "edit_sim", "levenshtein"}


async def run_metrics(ref: str, output: str, metrics_cfg: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    """Optimized metrics computation with async support and batching."""
    if not metrics_cfg:
        return 0.0, {}
    
    breakdown = {}
    weighted = []
    
    # Group metrics by type for potential batching
    rouge_metrics = []
    other_metrics = []
    
    for m in metrics_cfg:
        metric_type = m["type"]
        if metric_type in ROUGE_TYPES:
            rouge_metrics.append(m)
        else:
            other_metrics.append(m)
    
    # Process ROUGE metrics
    for m in rouge_metrics:
        metric_type = m["type"]
        w = float(m.get("weight", 1.0))
        value = await rouge(ref, output, rouge_type=metric_type)
        breakdown[metric_type] = value
        value = max(0.0, min(1.0, value))
        weighted.append((value, w))
    
    # Process other metrics
    for m in other_metrics:
        metric_type = m["type"]
        w = float(m.get("weight", 1.0))
        
        if metric_type == "bleu":
            value = bleu(ref, output)
            breakdown["bleu"] = value
        elif metric_type in EDIT_TYPES:
            value = edit_sim(ref, output)
            breakdown["edit_sim"] = value
        elif metric_type == "bertscore":       
            value = bertscore(ref, output, m.get("model"), m.get("device", "cpu"))
            breakdown["bertscore"] = value
        else:                        
            value = 0.0
            breakdown[metric_type] = value
            print(f"WARNING: Metric {metric_type} not supported for rewards.")
        
        value = max(0.0, min(1.0, float(value)))
        weighted.append((value, w))
    
    return aggregate(weighted), breakdown


async def compute_reward(data_source, generation, ground_truth, response_metrics, belief_metrics={}, extra_info=None):
    pred_belief, pred_resp, pred_memory = parse_text(generation)
    ref_belief, ref_resp, ref_memory = parse_text(ground_truth)

    print("\n========== [Generation] ==========")
    print(generation)
    print("=======================================\n")

    if not ref_belief:
        print("[Warning] Ground truth belief is empty.")
    if not ref_resp:
        print("[Warning] Ground truth response is empty.")

    # Input validation logs
    if pred_belief:
        # Belief reward
        if belief_metrics:
            belief_score, _ = await run_metrics( 
                ref_belief, pred_belief, belief_metrics
            )
        else:
            belief_score = 0.0
    else:
        print("[Warning] Generation missing belief.")
        belief_score = -1.0
            
        
    if pred_resp:
        # Response reward
        resp_score, _ = await run_metrics( 
            ref_resp, pred_resp, response_metrics
        )
    else:
        print("[Warning] Generation missing response.")
        resp_score = -1.0

    reward_dict = {
        "belief": belief_score, 
        "response": resp_score
    }
    return reward_dict