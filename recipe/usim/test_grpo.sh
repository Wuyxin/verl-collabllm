set -x
ENGINE=${1:-vllm}
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

# Run this everytime to make sure the common dir can be accessed by anyone
chmod -R 777 /dfs/project/kgrlm/common/llm_twin

export WANDB_ENTITY=dsp-team
VERL_PATH="./"
DATA_PATH="/dfs/project/kgrlm/common/llm_twin/data/reddit/rl"
OUTPUT_DIR="/dfs/project/kgrlm/common/llm_twin/outputs"

EXP_NAME=qwen2_5_14b_full
VERL_PATH="../verl"
DATA_PATH="/dfs/project/kgrlm/common/llm_twin/data/reddit/rl"
OUTPUT_DIR="/dfs/project/kgrlm/common/llm_twin/outputs/$EXP_NAME"
CACHE_DIR="/dfs/project/kgrlm/common/llm_twin/verl_cache"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export NEW_HF_CACHE=/dfs/project/kgrlm/common/llm_twin/hf-cache/shirwu

export HF_HOME="$NEW_HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$NEW_HF_CACHE/hub"
export TRANSFORMERS_CACHE="$NEW_HF_CACHE/hub"
export HF_DATASETS_CACHE="$NEW_HF_CACHE/datasets"
export XDG_CACHE_HOME="$NEW_HF_CACHE"    
export VLLM_DOWNLOAD_DIR="$NEW_HF_CACHE/hub"
export VERL_CACHE_DIR="$NEW_HF_CACHE/verl-cache"

# CHECKLIST:
# 1) BOTH Jinja templates - w/ or w/o belief formatting?
# 2) Response only vs with belief in data path 

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=False \
    reward_model.reward_manager=usim \
    custom_reward_function.path="$VERL_PATH/recipe/usim/reward.py" \
    custom_reward_function.name="compute_reward" \
    '+reward_model.reward_kwargs.metric_weights={belief: 0.5, response: 0.5}' \
    '+reward_model.reward_kwargs.belief_metrics=[{type: bertscore, weight: 1.0, model: microsoft/deberta-xlarge-mnli, device: cpu}]' \
    '+reward_model.reward_kwargs.response_metrics=[{type: bertscore, weight: 1.0, model: microsoft/deberta-xlarge-mnli, device: cpu}]' \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    +data.cache_dir=$CACHE_DIR \
    data.train_batch_size=8 \
    data.val_batch_size=32 \
    +data.kwargs.chat_template_path="$VERL_PATH/recipe/usim/qwen_multi_role_template_belief.jinja"\
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +actor_rollout_ref.kwargs.custom_chat_template="$VERL_PATH/recipe/usim/qwen_multi_role_template_belief.jinja" \
    actor_rollout_ref.model.path="/dfs/project/kgrlm/common/llm_twin/models/Qwen2.5-14B-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$NUM_GPUS \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$NUM_GPUS \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_reddit' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.val_before_train=False \
    trainer.total_epochs=30 $@

    # actor_rollout_ref.model.lora_rank=16 \
    # actor_rollout_ref.model.lora_alpha=16 \
    # actor_rollout_ref.model.target_modules=all-linear \