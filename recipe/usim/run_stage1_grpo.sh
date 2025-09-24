#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.use_kl_in_reward=false \
    data.train_files=/dfs/project/kgrlm/common/llm_twin/reddit/tagged/train.parquet \
    data.val_files=/dfs/project/kgrlm/common/llm_twin/reddit/tagged/test.parquet \
    data.train_batch_size=1 \
    data.max_prompt_length=4096 \
    data.max_response_length=256 \
    '+data.kwargs={}' \
    '+actor_rollout_ref.kwargs={}' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    '+actor_rollout_ref.model.kwargs={}' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    '+actor_rollout_ref.rollout.stop="<response>"' \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    '+actor_rollout_ref.ref.kwargs={}' \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    reward_model.enable=false \
    custom_reward_function.path=recipe/usim/tag_reward.py \
    custom_reward_function.name=llm_judge_reward_function \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=1 \
    trainer.project_name=aita_debug \
    trainer.experiment_name=debug_one_step