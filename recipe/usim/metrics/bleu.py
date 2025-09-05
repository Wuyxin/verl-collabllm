# metrics/bleu.py
import asyncio
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from functools import lru_cache

_SMOOTH = SmoothingFunction().method1
_BLEU_WEIGHTS_CACHE = {}

@lru_cache(maxsize=8)
def _get_bleu_weights(n):
    if n < 4:
        return tuple((1.0 / n if i < n else 0.0) for i in range(4))
    else:
        return (0.25, 0.25, 0.25, 0.25)

async def compute_score(data_source, generation, ground_truth, extra_info=None, **kwargs):
    if not generation or not ground_truth:
        return 0.0

    ref_toks = ground_truth.split()
    gen_toks = generation.split()
    if not ref_toks or not gen_toks:
        return 0.0

    n = max(1, min(4, len(ref_toks), len(gen_toks)))
    weights = _get_bleu_weights(n)
    try:
        return float(sentence_bleu([ref_toks], gen_toks, weights=weights, smoothing_function=_SMOOTH))
    except Exception as e:
        print(f"BLEU computation error: {e}")
        return 0.0
