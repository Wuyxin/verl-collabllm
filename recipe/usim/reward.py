import os
import re
import sys
import asyncio
import importlib.util
import threading
import torch
import litellm
from typing import Dict, Any, Tuple, Optional, List, TypedDict


METRIC_IMPORT_LOCK = threading.Lock()


async def compute_reward(
    data_source,
    generation,
    ground_truth,
    metrics,
    extra_info=None,
    num_retries: int = 6
):

    async def try_compute(metric, ref, pred, **kwargs):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        metric_file_path = os.path.join(current_dir, f"metrics/{metric}.py")
        if not os.path.exists(metric_file_path):
            print(f"[Error] Metric file not found: {metric_file_path}")
            return 0.0

        with METRIC_IMPORT_LOCK:
            if metric in sys.modules:
                module = sys.modules[metric]
            else:
                spec = importlib.util.spec_from_file_location(metric, metric_file_path)
                if spec is None or spec.loader is None:
                    print(f"[Error] Could not load spec for metric '{metric}'.")
                    return 0.0

                module = importlib.util.module_from_spec(spec)
                try:
                    sys.modules[metric] = module
                    spec.loader.exec_module(module)
                except Exception as e:
                    print(f"[Error] Failed to import metric '{metric}': {e}")
                    return 0.0

        if not hasattr(module, "compute_score"):
            print(f"[Error] 'compute_score' not found in {metric_file_path}")
            return 0.0

        compute_score_fn = module.compute_score
        for attempt in range(num_retries):
            try:
                if asyncio.iscoroutinefunction(compute_score_fn):
                    score = await compute_score_fn(
                        data_source, pred, ref, extra_info, **kwargs
                    )
                else:
                    score = compute_score_fn(
                        data_source, pred, ref, extra_info, **kwargs
                    )
                return score
            except Exception as e:
                if attempt == num_retries - 1:
                    print(f"[Error] Final failure computing '{metric}': {e}")
                    return 0.0
                else:
                    print(f"[Retry {attempt+1}] Failed computing '{metric}': {e}")
                    if isinstance(e, litellm.RateLimitError):
                        await asyncio.sleep(1)

    post = extra_info.get("post") if extra_info else None
    reward_dict = {}
    if generation:
        for metric, weight_n_kwargs in metrics.items():
            score = await try_compute(metric, ground_truth, generation, **weight_n_kwargs['kwargs'])
            reward_dict[f"{metric}"] = score
    else:
        for metric in metrics:
            reward_dict[f"{metric}"] = -1.0

    return reward_dict
