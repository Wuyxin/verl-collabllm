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


async def compute_reward(
    data_source,
    generation,
    ground_truth,
    response_metrics,
    belief_metrics={},
    extra_info=None,
    num_retries: int = 6
):
    pred_belief, pred_resp, _ = parse_text(generation)
    ref_belief, ref_resp, _ = parse_text(ground_truth)
    post = extra_info.get("post") if extra_info else None

    async def try_compute(metric_name, metric_type, ref, pred, **kwargs):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        metric_file_path = os.path.join(current_dir, f"metrics/{metric_name}.py")
        if not os.path.exists(metric_file_path):
            print(f"[Error] Metric file not found: {metric_file_path}")
            return 0.0

        module_name = f"{metric_type}_{metric_name}"
        with METRIC_IMPORT_LOCK:
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(module_name, metric_file_path)
                if spec is None or spec.loader is None:
                    print(f"[Error] Could not load spec for {metric_type} metric '{metric_name}'.")
                    return 0.0

                module = importlib.util.module_from_spec(spec)
                try:
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                except Exception as e:
                    print(f"[Error] Failed to import {metric_type} metric '{metric_name}': {e}")
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
                    print(f"[Error] Final failure computing '{metric_name}' ({metric_type}): {e}")
                    return 0.0
                else:
                    print(f"[Retry {attempt+1}] Failed computing '{metric_name}' ({metric_type}): {e}")
                    if isinstance(e, litellm.RateLimitError):
                        await asyncio.sleep(1)

    reward_dict = {}
    # Belief metrics
    if pred_belief:
        for metric, kwargs in belief_metrics.items():
            score = await try_compute(metric, "belief", ref_belief, pred_belief, **kwargs)
            reward_dict[f"belief_{metric}"] = score
    else:
        for metric, kwargs in belief_metrics.items():
            reward_dict[f"belief_{metric}"] = -1.0
        print("[Warning] Generation missing belief.")

    # Response metrics
    if pred_resp:
        for metric, kwargs in response_metrics.items():
            score = await try_compute(metric, "response", ref_resp, pred_resp, **kwargs)
            reward_dict[f"response_{metric}"] = score
    else:
        for metric, kwargs in response_metrics.items():
            reward_dict[f"response_{metric}"] = -1.0
        print("[Warning] Generation missing response.")

    print(f"[Post] \n{post}\n[GT] \n{ref_resp}\n[Gen] \n{pred_resp}\n [Reward] \n{reward_dict}\n" + "-" * 50)

    return reward_dict
