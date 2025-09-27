
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable, Optional, Union

import numpy as np
import copy
import psutil
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
import threading, time
import re
import torch.distributed as dist
import os
import json
from typing import Dict, Any, Tuple, Optional, List, TypedDict



FIELD_PATTERN = re.compile(r"<(?P<key>\w+)>(?P<value>.*?)</\1>", re.DOTALL | re.IGNORECASE)

def parse_fields(text: str) -> Dict[str, str]:
    """
    Parse <field>content</field> blocks into a dictionary.
    Works for any tag name (letters, numbers, underscore).
    """
    matches = FIELD_PATTERN.findall(text or "")
    return {key.lower().strip(" \n"): value.strip(" \n") for key, value in matches}


@register("usim")
class UsimRewardManager(AbstractRewardManager):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        train_metrics: dict,
        val_metrics: dict = None,
        reward_fn_key: str = "data_source",
        compute_score: Optional[Callable] = None,
        tag_log_path=None
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  
        self.compute_score = compute_score or default_compute_score
        self.split = 'train' if self.num_examine == 0 else 'val'

        if self.split == 'train':
            self.field_to_metrics = train_metrics # {'signature': {'signature_reward': 1}}
        else:
            self.field_to_metrics = val_metrics 

        self.field_metric_weights = {
            f"{field}:{metric}": weight_n_kwargs['weight'] \
            for field in self.field_to_metrics \
            for metric, weight_n_kwargs in self.field_to_metrics[field].items()
        }
        self.reward_fn_key = reward_fn_key

        # ADD LOGGER
        # self.tag_log_path=None
        # if tag_log_path:
        #     rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        #     if rank == 0:
        #         os.makedirs(os.path.dirname(tag_log_path) or ".", exist_ok=True)
        #         self.tag_log_path = tag_log_path
        #     # For now, assume only one metric for tags
        #     self.tag_key = next((m for m in self.metrics if "tag" in m.lower()), self.metrics[0])
    
    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # Use asyncio.run to handle the async computation
        return asyncio.run(self._compute_rewards_async(data, return_dict))
    
    def compute_score_sync(self, *args, **kwargs):
        result = asyncio.run(self.compute_score(*args, **kwargs))
        return result

    async def _compute_rewards_async(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        
        data_source = data.non_tensor_batch["data_source"]
        extra_info = data.non_tensor_batch["extra_info"]
        ground_truth = [item["ground_truth"] for item in data.non_tensor_batch["reward_model"]]
        batch_size = len(data_source)

        generations = self.tokenizer.batch_decode(
            data.batch["responses"],
            skip_special_tokens=True,
        )
        generation_fields = [parse_fields(generation) for generation in generations]
        
        keys = [list(set(g.keys()).intersection(set(self.field_to_metrics.keys()))) for g in generation_fields]
        nonempty_rate = sum([1 if len(k) else 0 for k in keys]) / len(keys)
        valid_rate = sum([1 if len(k) == len(self.field_to_metrics) else 0 for k in keys]) / len(keys)

        print(f"generation eg {generations[0]} | generation_fields {generation_fields} | valid_rate {nonempty_rate}")
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(          
                    None,                      
                    self.compute_score_sync,  
                    data_source[i], generation_fields[i].get(field, None), ground_truth[i],
                    self.field_to_metrics[field], extra_info[i],
                )
                for i in range(batch_size) for field in self.field_to_metrics
        ]
        score_dicts = await asyncio.gather(*tasks)
        field_to_score_dict = {
            field: score_dicts[i * batch_size:(i + 1) * batch_size]
            for i, field in enumerate(self.field_to_metrics)
        }

        # Aggregate scores for each metric and field 
        scores_by_fm = {
            f"{field}:{metric}": torch.tensor(
                [score_dict[metric] for score_dict in field_to_score_dict[field]]
            ) for field in self.field_to_metrics for metric in self.field_to_metrics[field]
        }

        # Apply field-metric specific weights
        weighted_scores_by_fm = {
            fm: scores_by_fm[fm] * self.field_metric_weights[fm]
            for fm in self.field_metric_weights
        }

        # Compute mean of weighted scores for each metric
        log_weighted_scores_by_field_metric = {
            f"{self.split}/{fm}": weighted_scores_by_fm[fm].mean(dim=0).item()
            for fm in self.field_metric_weights
        }

        # Combine weighted scores from all field and metrics into a single tensor
        scores = torch.stack(
            [weighted_scores_by_fm[fm] for fm in self.field_metric_weights]
        ).sum(dim=0)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        
        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
            #[Tag edit]
            # if self.tag_log_path:
            #     TAG_RE = re.compile(r"<tag>(.*?)</tag>", flags=re.IGNORECASE | re.DOTALL)
            #     m = TAG_RE.search(responses[i])
            #     tag = m.group(1).strip() if m else ""
            #     idx = (extra_info[i] or {}).get("index", -1)
            #     tag_score = float(scores_by_metrics[self.tag_key][i].item())
            #     local_rows.append({"idx": idx, "tag": tag, "score": tag_score})

        # local_rows = []
        # [Tag edit]
        # if self.tag_log_path:
        #     if dist.is_available() and dist.is_initialized():
        #         world = dist.get_world_size()
        #         rank = dist.get_rank()
        #         if rank == 0:
        #             gathered = [None for _ in range(world)]
        #             dist.gather_object(local_rows, object_gather_list=gathered, dst=0)
        #             lines = []
        #             for rows in gathered:
        #                 for r in rows:
        #                     lines.append(json.dumps(r, ensure_ascii=False))
        #             if lines:
        #                 with open(self.tag_log_path, "a", encoding="utf-8") as f:
        #                     f.write("\n".join(lines) + "\n")
        #         else:
        #             dist.gather_object(local_rows, dst=0)
        #     else:
        #         # single-process
        #         if local_rows:
        #             with open(self.tag_log_path, "a", encoding="utf-8") as f:
        #                 f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in local_rows) + "\n")

        log_weighted_scores_by_field_metric.update(
            {
                f"{self.split}/valid_rate": valid_rate,
                f"{self.split}/nonempty_rate": nonempty_rate,
            }
        )
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": log_weighted_scores_by_field_metric}
        else:
            return reward_tensor, log_weighted_scores_by_field_metric