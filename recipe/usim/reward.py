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

'''
    Returns parsed belief and resonse text from string with <belief> and <response> tokens.
    If we have a belief but no response tags, assume the rest outside is a response.

    Edge Cases to consider:
        - Accepts wrong backslash for tokens, i.e. <\belief>
        - If there is no opening or closing belief token, belief is ""
        - If there is no <response>, assumes the rest of the text (outside possible belief tokens) is the response
        - If there is a <response> but no <\response>, we assume the response runs until the end of the text
'''

def parse_text(text: str) -> Tuple[str, str]:
    belief, response = "", text
    belief_open = BELIEF_OPEN.search(text)
    if belief_open:
        belief_closed = BELIEF_CLOSE.search(text, belief_open.end())
        if belief_closed:
            belief = text[belief_open.end():belief_closed.start()].strip()
    response_open = RESP_OPEN.search(text)
    if response_open:
        response_close = RESP_CLOSE.search(text, response_open.end())
        response = text[response_open.end(): response_close.start()].strip() if response_close else text[response_open.end():].strip()
    elif belief:
        response = BELIEF_CLOSE.sub("", BELIEF_OPEN.sub("", text)).strip()
    return belief, response


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
def compute_reward(data_source, solution_str, ground_truth, extra_info=None, reward_config=None):
    cfg = {**(reward_config or {})}

    pred_belief, pred_resp = parse_text(solution_str)
    ref_belief, ref_resp = parse_text(ground_truth)
    if pred_belief=="" or pred_resp=="":
        print("ERROR SOLUTION STR EMPTY")
        print(solution_str)
        #print("----------end----------- \n")
        #print("----------end----------- \n")
    if ref_belief == "":
        print("ERROR GOLD BELIEF EMPTY")
        #print("++++++++++++++++++++++== starts: \n ", ground_truth)
        #print("----------end----------- \n")
    if ref_resp == "":
        print("ERROR GOLD REPSONSE MEPTY")

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