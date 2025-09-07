
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
#import wandb



@register("usim")
class UsimRewardManager(AbstractRewardManager):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        metric_weights: dict,
        belief_metrics: dict,
        response_metrics: dict, 
        val_response_metrics: dict = None,
        reward_fn_key: str = "data_source",
        compute_score: Optional[Callable] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  
        self.compute_score = compute_score or default_compute_score

        self.split = 'train' if self.num_examine == 0 else 'val'
        print("*"*100, self.split)

        if self.split == 'train':
            self.response_metrics = response_metrics
            self.belief_metrics = belief_metrics
            self.metric_weights = metric_weights
        else:
            self.belief_metrics = {}
            self.response_metrics = val_response_metrics if val_response_metrics else response_metrics
            self.metric_weights = {f'response_{k}': 1.0 for k in self.response_metrics.keys()}

        self.metrics = list(self.metric_weights.keys())
        self.reward_fn_key = reward_fn_key
    
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
        # # Weights and final score
        data_source = data.non_tensor_batch["data_source"]
        extra_info = data.non_tensor_batch["extra_info"]
        ground_truth = [item["ground_truth"] for item in data.non_tensor_batch["reward_model"]]

        responses = self.tokenizer.batch_decode(
            data.batch["responses"],
            skip_special_tokens=True,
        )

        batch_size = len(data_source)

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(          
                    None,                      
                    self.compute_score_sync,  
                    data_source[i], responses[i], ground_truth[i],
                    self.response_metrics, self.belief_metrics, extra_info[i],
                )
                for i in range(batch_size)
        ]
        score_dicts = await asyncio.gather(*tasks)

        # Aggregate scores for each metric across repeated rollouts
        print(self.metrics)
        print('==============================')
        print('score dict', score_dicts)
        scores_by_metrics = {
            metric: torch.tensor(
                [score_dict[metric] for score_dict in score_dicts]
            ) for metric in self.metrics
        }

        # Apply metric-specific weights
        weighted_scores_by_metrics = {
            metric: scores_by_metrics[metric] * self.metric_weights[metric]
            for metric in self.metrics
        }

        # Compute mean of weighted scores for each metric
        log_weighted_scores_by_metrics = {
            k: v
            for metric in self.metrics
            for k, v in {
                f"customized_score/{self.split}/{metric}": weighted_scores_by_metrics[metric].mean(dim=0).item(),
                f"customized_score/{self.split}/{metric}_std": weighted_scores_by_metrics[metric].std(dim=0).item()
            }.items()
        }

        # Combine weighted scores from all metrics into a single tensor
        scores = torch.stack(
            [weighted_scores_by_metrics[metric] for metric in self.metrics]
        ).sum(dim=0)
        print('Avg scores:', log_weighted_scores_by_metrics)
        #wandb.log(log_weighted_scores_by_metrics)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        
        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": log_weighted_scores_by_metrics}
        else:
            return reward_tensor, log_weighted_scores_by_metrics