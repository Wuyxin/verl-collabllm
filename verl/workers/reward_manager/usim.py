
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
        stage1_path=None, #'/lfs/ampere2/0/echoi1/tag_log.jsonl'
        two_stage_training=True,  # if we are training both tag and response 
        global_max=True # whether or not to get best tag globally
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

        self.two_stage_training = two_stage_training
        # ADD LOGGER
        self.tag_log_path=None
        if stage1_path:
            rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
            if rank == 0:
                os.makedirs(os.path.dirname(stage1_path) or ".", exist_ok=True)
                self.tag_log_path = stage1_path
            
        self.tag_key = next((m for m in self.metrics if "tag" in m.lower()), self.metrics[0])

        self.global_max = global_max
        self.global_best = {}  
        self.best_flush_every = 200 
        self._updates_since_flush = 0
        # [UNTESTED] LOAD global best tags from tag_log_path
        if self.global_max and self.tag_log_path and os.path.exists(self.tag_log_path):
            try:
                with open(self.tag_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        key = f"idx:{rec['idx']}" if ("idx" in rec and rec["idx"] is not None and rec["idx"] != -1) else None
                        if key and rec.get("tag"):
                            cur = self.global_best.get(key)
                            if (cur is None) or (float(rec.get("score", 0.0)) > cur["score"]):
                                self.global_best[key] = {"tag": rec["tag"], "score": float(rec.get("score", 0.0))}
            except Exception as e:
                print(f"[warn] failed to seed global_best from {self.tag_log_path}: {e}")
    
    def __call__(self, data: DataProto, return_dict: bool = False, two_stage_training=False) -> torch.Tensor | dict[str, Any]:
        # Use asyncio.run to handle the async computation
        return asyncio.run(self._compute_rewards_async(data, return_dict, two_stage_training=two_stage_training))
    
    def compute_score_sync(self, *args, **kwargs):
        result = asyncio.run(self.compute_score(*args, **kwargs))
        return result

    def _stable_prompt_key(self, i: int, data: DataProto, extra_info_i: dict | None) -> str:
        """Prefer dataset index; else hash the prompt ids for stability."""
        idx = None
        if extra_info_i:
            idx = extra_info_i.get("index")
        if idx is not None and idx != -1:
            return f"idx:{idx}"
        # hash the prompt token ids (deterministic, cross-run stable if dataset unchanged)
        pid = data.batch["prompts"][i].detach().cpu().numpy().astype("int64")
        h = hashlib.sha1(pid.tobytes()).hexdigest()
        return f"sha1:{h}"

    def _maybe_update_global_best(self, key: str, tag_text: str, tag_score: float):
        if not tag_text:
            return
        cur = self.global_best.get(key)
        if (cur is None) or (tag_score > cur["score"]):
            self.global_best[key] = {"tag": tag_text, "score": float(tag_score)}
            self._updates_since_flush += 1

    def _maybe_flush_snapshot(self):
        """Optional: write a small snapshot map periodically next to the log."""
        if not self.tag_log_path:
            return
        if self._updates_since_flush < self.best_flush_every:
            return
        snap = {k: v for k, v in self.global_best.items()}
        p = self.tag_log_path + ".best_snapshot.json"
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False)
            self._updates_since_flush = 0
        except Exception as e:
            print(f"[warn] failed to flush best snapshot: {e}")

    async def _compute_rewards_async(self, data: DataProto, return_dict: bool = False, two_stage_training=True) -> torch.Tensor | dict[str, Any]:
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
        tasks = []
        for i in range(batch_size):
            task_fn = partial(
                self.compute_score_sync,
                data_source[i],
                responses[i],
                ground_truth[i],
                extra_info=extra_info[i],
                response_metrics=self.response_metrics,
                belief_metrics=self.belief_metrics,
            )
            tasks.append(loop.run_in_executor(None, task_fn))
        score_dicts = await asyncio.gather(*tasks)
        field_to_score_dict = {
            field: score_dicts[i * batch_size:(i + 1) * batch_size]
            for i, field in enumerate(self.field_to_metrics)
        }

        # print(self.metrics)
        # print('==============================')
        # print('score dict', score_dicts)
        scores_by_metrics = {
            metric: torch.tensor(
                [score_dict[metric] for score_dict in score_dicts]
            ) for metric in self.metrics
        }

        tag_scores = scores_by_metrics[self.tag_key].detach().cpu() 

        # Apply metric-specific weights
        weighted_scores_by_metrics = {
            metric: scores_by_metrics[metric] * self.metric_weights[metric]
            for metric in self.metrics
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
        #print('Avg scores:', log_weighted_scores_by_metrics)
        #wandb.log(log_weighted_scores_by_metrics)
        

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

        # TWO STAGE TRAINING
        if two_stage_training:
            uids = data.non_tensor_batch.get("uid", None)
            per_row_tag_texts = []
            for i in range(batch_size):
                m = TAG_RE.search(responses[i])
                per_row_tag_texts.append(m.group(1).strip() if m else "")
            
            if self.global_max:
                for i in range(batch_size):
                    key = self._stable_prompt_key(i, data, extra_info[i])
                    self._maybe_update_global_best(key, per_row_tag_texts[i], float(tag_scores[i].item()))
                self._maybe_flush_snapshot()

            best_tag_list= []
            best_tag_reward_list= []
            uids = data.non_tensor_batch.get("uid", None)
            if uids is not None:
                uids_np = np.asarray(uids, dtype=object)
                uniq_uids, first_idx_np = np.unique(uids_np, return_index=True)
                order = uniq_uids[np.argsort(first_idx_np)]

                for uid in order:
                    idxs = np.nonzero(uids_np == uid)[0]
                    if self.global_max:
                        i0 = int(idxs[0])
                        key = self._stable_prompt_key(i0, data, extra_info[i0])
                        rec = self.global_best.get(key) if self.global_max else None
                    
                    if self.global_max and rec is not None:
                        best_tag_list.append(rec["tag"])
                        best_tag_reward_list.append(float(rec["score"]))
                    else:
                        j_local = idxs[np.argmax(tag_scores[idxs].numpy())]
                        best_tag_list.append(per_row_tag_texts[int(j_local)])
                        best_tag_reward_list.append(float(tag_scores[int(j_local)].item()))

        # [Tag edit] only save once if we have multiple processes
        if self.tag_log_path:
            if dist.is_available() and dist.is_initialized():
                world = dist.get_world_size()
                rank = dist.get_rank()
                if rank == 0:
                    gathered = [None for _ in range(world)]
                    dist.gather_object(local_rows, object_gather_list=gathered, dst=0)
                    lines = []
                    for rows in gathered:
                        for r in rows:
                            lines.append(json.dumps(r, ensure_ascii=False))
                    if lines:
                        with open(self.tag_log_path, "a", encoding="utf-8") as f:
                            f.write("\n".join(lines) + "\n")
                else:
                    dist.gather_object(local_rows, dst=0)
            else:
                if local_rows:
                    with open(self.tag_log_path, "a", encoding="utf-8") as f:
                        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in local_rows) + "\n")

        log_weighted_scores_by_field_metric.update(
            {
                f"{self.split}/valid_rate": valid_rate,
                f"{self.split}/nonempty_rate": nonempty_rate,
            }
        )
        if return_dict:
            reward_extra_info = dict(log_weighted_scores_by_metrics)
            if two_stage_training:
                reward_extra_info["best_tag"] = best_tag_list                
                reward_extra_info["best_tag_reward"] = best_tag_reward_list
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor, log_weighted_scores_by_field_metric