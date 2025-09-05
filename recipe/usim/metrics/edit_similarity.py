import asyncio
from rapidfuzz.distance import Levenshtein

async def compute_score(data_source, generation, ground_truth, extra_info=None, **kwargs):
    if not generation and not ground_truth:
        return 1.0
    if not generation or not ground_truth:
        return 0.0
    try:
        return Levenshtein.normalized_similarity(ground_truth, generation)
    except Exception as e:
        print(f"Edit similarity computation error: {e}")
        return 0.0