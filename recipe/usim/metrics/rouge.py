import asyncio
from rouge_score import rouge_scorer
from threading import Lock

_ROUGE_SCORERS = {}
_ROUGE_LOCK = Lock()

async def compute_score(data_source, generation, ground_truth, extra_info=None, **kwargs):
    rouge_type = kwargs.get("rouge_type", "rougeL")
    if not generation or not ground_truth:
        return 0.0

    with _ROUGE_LOCK:
        if rouge_type not in _ROUGE_SCORERS:
            _ROUGE_SCORERS[rouge_type] = rouge_scorer.RougeScorer([rouge_type], use_stemmer=False)
        scorer = _ROUGE_SCORERS[rouge_type]

    try:
        score = scorer.score(ground_truth, generation)[rouge_type].fmeasure
        return float(score)
    except Exception as e:
        print(f"ROUGE-{rouge_type} computation error: {e}")
        return 0.0