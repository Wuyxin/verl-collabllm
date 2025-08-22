import re
from typing import Dict, Any, Tuple, Optional

from rapidfuzz.distance import Levenshtein
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore_score
from functools import cache
from bert_score.scorer import BERTScorer
_SMOOTH = SmoothingFunction().method1

'''
    Implements custom score/reward for VERL PPO. 
    Supported metrics: 
            - BERTScore
            - Sentence BLEU - add smoothing??
            - ROUGE-1/2/L
            - edit_sim
    This script clamps the each metric to [0,1] and returns the weighted average
    of the normalized scores. The final output is in [0,1]
    We cache the BERT model and cache Rouge scorers for faster computation

'''

BELIEF_OPEN  = re.compile(r"<belief>", re.I)
BELIEF_CLOSE = re.compile(r"</belief>|<\\belief>", re.I)
RESP_OPEN    = re.compile(r"<response>", re.I)
RESP_CLOSE   = re.compile(r"</response>|<\\response>", re.I)
MEM_OPEN     = re.compile(r"<memory>", re.I)
MEM_CLOSE    = re.compile(r"</memory>|<\\memory>", re.I)


'''
    Returns parsed belief, response, and memory text from a string with
    <belief>...</belief>, <response>...</response>, and <memory>...</memory> tokens.

    Edge Cases to consider:
      - If there is no opening belief token, takes everything from beginning to closing belief token as belief.
      - Accepts wrong backslash for closing tokens: <\belief>, <\response>, <\memory>
      - If there is no opening or closing belief token, belief = ""
      - If there is no <response>, assumes the rest of the text (outside possible belief/memory) is the response
      - If there is a <response> but no </response>, it runs until <memory> if present, else to the end
      - If there is a <memory> but no </memory>, it runs until the end
'''
def parse_text(text):
    belief, response, memory = "", text, ""

    b_open = BELIEF_OPEN.search(text)
    b_close = BELIEF_CLOSE.search(text, b_open.end()) if b_open else BELIEF_CLOSE.search(text)
    b_span = None
    if b_open and b_close:
        belief = text[b_open.end():b_close.start()].strip()
        b_span = (b_open.start(), b_close.end()) 
    elif (not b_open) and b_close:
        belief = text[:b_close.start()].strip()
        b_span = (0, b_close.end())

    m_open = MEM_OPEN.search(text)
    m_close = MEM_CLOSE.search(text, m_open.end()) if m_open else None
    if m_open:
        memory = text[m_open.end():m_close.start()].strip() if m_close else text[m_open.end():].strip()

    r_open = RESP_OPEN.search(text)
    if r_open:
        r_close = RESP_CLOSE.search(text, r_open.end())
        if r_close:
            response = text[r_open.end():r_close.start()].strip()
        else:
            r_end = m_open.start() if (m_open and m_open.start() >= r_open.end()) else len(text)
            response = text[r_open.end():r_end].strip()
    else:
        leftover = text
        if b_span is not None:
            leftover = leftover[:b_span[0]] + leftover[b_span[1]:]
        else:
            leftover = BELIEF_CLOSE.sub("", BELIEF_OPEN.sub("", leftover))
        if m_open:
            leftover = leftover[:m_open.start()] + (leftover[m_close.end():] if m_close else "")
        response = leftover.strip()
    return belief, response, memory


@cache
def _get_bert_scorer(model_type, device):
    return BERTScorer(
        model_type=model_type,    
        rescale_with_baseline=True,
        lang='en',
        device=device
    )

_ROUGE_SCORERS: Dict[str, rouge_scorer.RougeScorer] = {}

# Strict ROUGE w/ no stemmer
def _get_rouge_scorer(rouge_type: str) -> rouge_scorer.RougeScorer:
    key = rouge_type
    if key not in _ROUGE_SCORERS:
        _ROUGE_SCORERS[key] = rouge_scorer.RougeScorer([key], use_stemmer=False)
    return _ROUGE_SCORERS[key]

# using sentence bleu for short outputs
def bleu(ref, output):
    ref_t, out_t = ref.split(), output.split()
    n = max(1, min(4, len(ref_t), len(out_t)))
    if n < 4:
        weights = tuple((1.0/n if i < n else 0.0) for i in range(4))
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    return float(sentence_bleu([ref_t], out_t, weights=weights, smoothing_function=_SMOOTH))

def rouge(ref, output, rouge_type) -> float:
    scorer = _get_rouge_scorer(rouge_type)
    return float(scorer.score(ref, output)[rouge_type].fmeasure)

def edit_sim(ref, output):
    return Levenshtein.normalized_similarity(ref, output) 

def bertscore(ref: str, output: str, model: Optional[str] = None, device=None) -> float:
    if device is None:
        device="cpu"
    scorer = _get_bert_scorer(model, device)
    P, R, F1 = scorer.score([output], [ref]) 
    return float(F1.mean())

def aggregate(weighted_scores):
    num = sum(w * s for s, w in weighted_scores)
    den = sum(w for _, w in weighted_scores)
    return (num / den) if den > 0 else 0.0

# if no weights are provided, everything is weighted equally
def run_metrics(ref, output, metrics_cfg):
    breakdown = {}
    weighted = []
    for m in metrics_cfg:
        metric_type = m["type"]
        w = float(m.get("weight", 1.0))    
        if metric_type in ("rougeL", "rouge1", "rouge2"): 
            value = rouge(ref, output, rouge_type=metric_type)
            breakdown[metric_type] = value
        elif metric_type == "bleu":
            value = bleu(ref, output)
            breakdown["bleu"] = value
        elif metric_type in ("edit","edit_sim","levenshtein"):
            value = edit_sim(ref, output)
            breakdown["edit_sim"] = value
        elif metric_type == "bertscore":       
            value = bertscore(ref, output, m.get("model"), m.get("device"))
            breakdown["bertscore"] = value
        else:                        
            value= 0.0
            breakdown[metric_type] = value
            print(f"WARNING: Metric {m} not supported for rewards.")
        value = max(0.0, min(1.0, float(value)))
        weighted.append((value, w))
    return aggregate(weighted), breakdown


# default weights are always 1
EXAMPLE_CFG = {
    "reward_belief": True,  # False if we only reward actions
    "weights": {"belief": 0.3, "response": 0.7}, 
    "belief_metrics": [
        {"type": "bertscore", "weight": 1.0, "model": None, "device": "cpu"}
    ],
    "response_metrics": [
        {"type": "rougeL", "weight": 0.5},
        {"type": "bleu", "weight": 0.25},
        {"type": "edit_sim", "weight": 0.25},
    ],
}

# entra_info should be a dict where the key "reward_cfg" is a dict with the example structure above
# records individual metric breakdowns for debugging purposes
def compute_reward(data_source, solution_str, ground_truth, extra_info=None):
    cfg = {**(extra_info["custom_reward_config"] or {})}

    pred_belief, pred_resp, pred_memory = parse_text(solution_str)
    ref_belief, ref_resp, ref_memory = parse_text(ground_truth)
    '''print("[SOlUTION STR]******************\n", solution_str, '\n******************')
    if pred_belief=="" or pred_resp=="":
        print("ERROR SOLUTION STR EMPTY")
    if ref_belief == "":
        print("ERROR GOLD BELIEF EMPTY")
    if ref_resp == "":
        print("ERROR GOLD REPSONSE MEPTY")'''

    reward_belief = bool(cfg.get("reward_belief", True)) 
    belief_score, belief_breakdown = (0.0, {})
    if reward_belief:
        belief_score, belief_breakdown = run_metrics(ref_belief, pred_belief, cfg.get("belief_metrics", []))

    resp_score, resp_breakdown = run_metrics(ref_resp, pred_resp, cfg.get("response_metrics", []))

    weight_b = float(cfg.get("weights", {}).get("belief", 0.0))
    weight_r = float(cfg.get("weights", {}).get("response", 1.0))

    total =(weight_b * belief_score + weight_r * resp_score)/max(weight_b + weight_r, 1e-6)
    total = max(0.0, min(1.0, float(total)))     # normalize btwn 0 and 1

    return total