import re
from typing import Dict, Any, Tuple, Optional, List, TypedDict
from functools import lru_cache
import asyncio, random
import threading
from threading import Lock

from rapidfuzz.distance import Levenshtein
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore_score
from bert_score.scorer import BERTScorer
import os, json
from uuid import uuid4
from openai import AsyncOpenAI
try:
    # new-style exceptions (names may vary by SDK version)
    from openai import APIConnectionError, APITimeoutError, RateLimitError
except Exception:
    APIConnectionError = APITimeoutError = RateLimitError = Exception


import weakref
import threading

_GLOBAL_JUDGE_SEM = threading.Semaphore(int(os.getenv("JUDGE_CONCURRENCY", "16")))

_CLIENTS_BY_LOOP = weakref.WeakKeyDictionary()
_SEMS_BY_LOOP = weakref.WeakKeyDictionary()

_SMOOTH = SmoothingFunction().method1

# Compiled regex patterns (already optimized in original)
BELIEF_OPEN  = re.compile(r"<belief>", re.I)
BELIEF_CLOSE = re.compile(r"</belief>|<\\belief>", re.I)
RESP_OPEN    = re.compile(r"<response>", re.I)
RESP_CLOSE   = re.compile(r"</response>|<\\response>", re.I)
MEM_OPEN     = re.compile(r"<memory>", re.I)
MEM_CLOSE    = re.compile(r"</memory>|<\\memory>", re.I)

# Thread-safe scorer caches with locks
_ROUGE_SCORERS: Dict[str, rouge_scorer.RougeScorer] = {}
_ROUGE_LOCK = Lock()

_BERT_SCORERS: Dict[str, BERTScorer] = {}
_BERT_LOCK = Lock()

# Global lock for BERTScore initialization to prevent conflicts
_BERT_INIT_LOCK = Lock()


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


def _get_rouge_scorer_thread_safe(rouge_type: str) -> rouge_scorer.RougeScorer:
    """Thread-safe rouge scorer getter with proper locking."""
    with _ROUGE_LOCK:
        if rouge_type not in _ROUGE_SCORERS:
            print(f"Creating new ROUGE scorer for type: {rouge_type}")
            _ROUGE_SCORERS[rouge_type] = rouge_scorer.RougeScorer([rouge_type], use_stemmer=False)
        return _ROUGE_SCORERS[rouge_type]


def _get_bert_scorer_thread_safe(model_type: str, device: str = "cpu") -> BERTScorer:
    """Thread-safe BERTScore scorer getter with proper locking."""
    cache_key = f"{model_type}_{device}"
    
    with _BERT_LOCK:
        if cache_key not in _BERT_SCORERS:
            print(f"Creating new BERTScore scorer for model: {model_type}, device: {device}")
            
            # Use a separate initialization lock to prevent concurrent model loading
            with _BERT_INIT_LOCK:
                # Double-check pattern in case another thread created it while we were waiting
                if cache_key not in _BERT_SCORERS:
                    try:
                        _BERT_SCORERS[cache_key] = BERTScorer(
                            model_type=model_type,
                            rescale_with_baseline=True,
                            lang='en',
                            device=device
                        )
                        print(f"Successfully created BERTScore scorer for {model_type}")
                    except Exception as e:
                        print(f"Error creating BERTScore scorer: {e}")
                        raise
        
        return _BERT_SCORERS[cache_key]


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


def rouge2(ref: str, output: str, rouge_type: str) -> float:
    """Synchronous rouge computation with early validation and thread safety."""
    if not ref or not output:
        return 0.0
    scorer = _get_rouge_scorer_thread_safe(rouge_type)
    return float(scorer.score(ref, output)[rouge_type].fmeasure)


def edit_sim(ref: str, output: str) -> float:
    """Edit similarity with early validation."""
    if not ref and not output:
        return 1.0
    if not ref or not output:
        return 0.0
    return Levenshtein.normalized_similarity(ref, output) 


def bertscore(ref: str, output: str, model: str, device: str = "cpu") -> float:
    """Thread-safe BERTScore with proper locking and validation."""
    if not ref or not output:
        return 0.0

    try:
        scorer = _get_bert_scorer_thread_safe(model, device)
        
        # # Use the scorer in a thread-safe manner
        # with _BERT_INIT_LOCK:  # Use the same lock for scoring to prevent conflicts
        P, R, F1 = scorer.score([output], [ref])
        return float(F1.mean())
        
    except Exception as e:
        print(f"Error in BERTScore computation: {e}")
        return 0.0


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

def rouge(ref, out, rouge_type):
    """Thread-safe ROUGE computation."""
    if not ref or not out: 
        return 0.0
    scorer = _get_rouge_scorer_thread_safe(rouge_type)
    scores = scorer.score(ref, out)
    return float(scores[rouge_type].fmeasure)


@lru_cache(maxsize=8)
def _load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _get_async_client_for_loop() -> AsyncOpenAI:
    loop = asyncio.get_running_loop()
    client = _CLIENTS_BY_LOOP.get(loop)
    if client is None:
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        _CLIENTS_BY_LOOP[loop] = client
    return client

def _get_semaphore_for_loop() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _SEMS_BY_LOOP.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(int(os.getenv("JUDGE_CONCURRENCY_PER_LOOP", "16")))
        _SEMS_BY_LOOP[loop] = sem
    return sem


async def _call_judge_once(model, instructions, user_input, max_output_tokens):
    client = _get_async_client_for_loop()
    loop_sem = _get_semaphore_for_loop()

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _GLOBAL_JUDGE_SEM.acquire)
    try:
        async with loop_sem:
            resp = await client.responses.create(
                model=model,
                instructions=instructions,
                input=user_input,
            )
    finally:
        _GLOBAL_JUDGE_SEM.release()
    raw = (resp.output_text or "").strip()
    data = json.loads(raw) 
    score = float(data.get("score", 0.0))

    #print(f"========={user_input, score}===========")
    # clamp
    if score < 0.0: score = 0.0
    if score > 1.0: score = 1.0
    return score


async def _llm_similarity_score(ref_resp, pred_resp, post=None, model="gpt-5-nano", prompt_path=None, max_output_tokens=20,vmax_retries=4, base_backoff=0.5):
    if not ref_resp or not pred_resp:
        return 0.0

    instructions = _load_text_file(prompt_path)
    user_input = (
        "POST:\n"
        f"{post}\n\n"
        "REFERENCE:\n"
        f"{ref_resp}\n\n"
        "RESPONSE:\n"
        f"{pred_resp}\n\n"
    )

    print('[USER INPUT] ', user_input)

    backoff = base_backoff
    for attempt in range(max_retries + 1):
        try:
            return await _call_judge_once(
                model=model,
                instructions=instructions,
                user_input=user_input,
                max_output_tokens=max_output_tokens,
            )
        except (APIConnectionError, APITimeoutError) as e:
            if attempt == max_retries:
                print(f"[LLM-JUDGE ERROR] network/timeout after {attempt} retries: {e}")
                return 0.0
            await asyncio.sleep(backoff + random.random() * 0.2)
            backoff = min(backoff * 2, 8.0)
        except RateLimitError as e:
            if attempt == max_retries:
                print(f"[LLM-JUDGE ERROR] rate limit after {attempt} retries: {e}")
                return 0.0
            await asyncio.sleep(backoff + random.random() * 0.5)
            backoff = min(backoff * 2, 10.0)
        except Exception as e:
            print(f"[LLM-JUDGE ERROR] create() failed: {type(e).__name__}: {e}")
            return 0.0


async def run_metrics(ref: str, output: str, metrics_cfg: List[Dict[str, Any]], post=None) -> Tuple[float, Dict[str, float]]:
    """Synchronous metrics computation with thread-safe operations."""
    if not metrics_cfg:
        return 0.0, {}
    
    breakdown = {}
    weighted = []
    
    # Process all metrics synchronously with thread safety
    for m in metrics_cfg:
        metric_type = m["type"]
        w = float(m.get("weight", 1.0))
        
        try:
            if metric_type in ROUGE_TYPES:
                value = rouge(ref, output, rouge_type=metric_type)
                breakdown[metric_type] = value
            elif metric_type == "bleu":
                value = bleu(ref, output)
                breakdown["bleu"] = value
            elif metric_type in EDIT_TYPES:
                value = edit_sim(ref, output)
                breakdown["edit_sim"] = value
            elif metric_type == "bertscore":       
                model = m.get("model", "microsoft/deberta-xlarge-mnli")
                device = m.get("device", "cpu")
                value = bertscore(ref, output, model, device)
                breakdown["bertscore"] = value
            elif metric_type == "gpt":
                judge_model = m.get("model", "gpt-5-nano")
                prompt_path = m.get("prompt_path") 
                coro = _llm_similarity_score(
                    ref_resp=ref,
                    post=post,
                    pred_resp=output,
                    model=judge_model,
                    prompt_path=prompt_path
                )
                value = await coro
                #print("lllm value: ", value)

            else:                        
                value = 0.0
                breakdown[metric_type] = value
                print(f"WARNING: Metric {metric_type} not supported for rewards.")
            
            value = max(0.0, min(1.0, float(value)))
            weighted.append((value, w))
            
        except Exception as e:
            print(f"Error computing {metric_type}: {e}")
            value = 0.0
            breakdown[metric_type] = value
            weighted.append((value, w))
    
    return aggregate(weighted), breakdown



async def compute_reward(data_source, generation, ground_truth, response_metrics, belief_metrics={}, extra_info=None):
    # return {
    #     "belief": 2 - len(generation) / 1000.0, 
    #     "response": 2 - len(generation) / 1000.0
    # } # Dummy reward for testing
    # #   
    #original_post = extra_info[""]
    pred_belief, pred_resp, pred_memory = parse_text(generation)
    ref_belief, ref_resp, ref_memory = parse_text(ground_truth)

    post = extra_info.get("post")

    print(f"\n========== [post] ==========\n{post}=======================================\n")
    print(f"\n========== [GT] ==========\n{ground_truth}=======================================\n")

    '''if not ref_belief:
        print("[Warning] Ground truth belief is empty.")'''
    if not ref_resp:
        print("[Warning] Ground truth response is empty.")
    #print("[BELIFHDI METRUC] ", belief_metrics)
    # Input validation logs
    if pred_belief:
        # Belief reward
        if belief_metrics:
            print("[Warning] belief metric is true.")
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
            ref_resp, pred_resp, response_metrics, post=post
        )
    else:
        print("[Warning] Generation missing response.")
        resp_score = -1.0

    reward_dict = {
        "belief": belief_score, 
        "response": resp_score
    }
    return reward_dict

