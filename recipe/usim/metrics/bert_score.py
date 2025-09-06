import asyncio
from bert_score.scorer import BERTScorer
from threading import Lock

_BERT_SCORERS = {}
_BERT_LOCK = Lock()
_BERT_INIT_LOCK = Lock()


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


def compute_score(data_source, generation, ground_truth, extra_info=None, **kwargs):
    model = kwargs.get("model", "microsoft/deberta-xlarge-mnli")
    device = kwargs.get("device", "cpu")
    cache_key = f"{model}_{device}"

    if not generation or not ground_truth:
        return 0.0

    try:
        scorer = _get_bert_scorer_thread_safe(model, device)
        
        P, R, F1 = scorer.score([generation], [ground_truth])
        return float(F1.mean())
        
    except Exception as e:
        print(f"Error in BERTScore computation: {e}")
        return 0.0
