#!/bin/bash
set -x

# Stage 1 GRPO training with tag reward function
# This will truncate responses at <response> tag and only train on tag generation

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.stage_1=true \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.use_kl_in_reward=false \
    data.train_files=path/to/your/train.parquet \
    data.val_files=path/to/your/val.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    reward_model.enable=false \
    custom_reward_function.path=recipe/usim/tag_reward.py \
    custom_reward_function.name=llm_judge_reward_function \
    trainer.n_gpus_per_node=1 \
    trainer.total_epochs=2 \
    trainer.project_name=stage1_grpo_tags \
    trainer.experiment_name=tag_training_llm_judge
