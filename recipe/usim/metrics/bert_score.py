import asyncio
from bert_score.scorer import BERTScorer
from threading import Lock

_BERT_SCORERS = {}
_BERT_LOCK = Lock()

async def compute_score(data_source, generation, ground_truth, extra_info=None, **kwargs):
    model = kwargs.get("model", "microsoft/deberta-xlarge-mnli")
    device = kwargs.get("device", "cpu")
    cache_key = f"{model}_{device}"

    if not generation or not ground_truth:
        return 0.0

    with _BERT_LOCK:
        if cache_key not in _BERT_SCORERS:
            try:
                _BERT_SCORERS[cache_key] = BERTScorer(
                    model_type=model,
                    lang="en",
                    rescale_with_baseline=True,
                    device=device
                )
            except Exception as e:
                print(f"BERTScore init error: {e}")
                return 0.0

    try:
        scorer = _BERT_SCORERS[cache_key]
        P, R, F1 = scorer.score([generation], [ground_truth])
        return float(F1.mean())
    except Exception as e:
        print(f"BERTScore computation error: {e}")
        return 0.0
