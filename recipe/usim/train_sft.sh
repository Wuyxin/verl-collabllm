set -e
set -x  # Print commands

DATA_PATH="/dfs/project/kgrlm/common/llm_twin/data/reddit_debug_sft"
VERL_PATH="/lfs/ampere4/0/echoi1/collabllm/verl-collabllm"
OUTPUT_DIR="/dfs/scratch0/echoi1/delete"

export CUDA_VISIBLE_DEVICES=0,1
export NEW_HF_CACHE=/dfs/scratch0/echoi1/hf-cache

export HF_HOME="$NEW_HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$NEW_HF_CACHE/hub"
export TRANSFORMERS_CACHE="$NEW_HF_CACHE/hub"
export HF_DATASETS_CACHE="$NEW_HF_CACHE/datasets"
export XDG_CACHE_HOME="$NEW_HF_CACHE"    
export VLLM_DOWNLOAD_DIR="$NEW_HF_CACHE/hub"

echo "Starting SFT Training"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"

# CHECKLIST:
# 1) Jinja template - w/ or w/o belief formatting?
# 2) Response only vs with belief in data path 
# 3) Change max_length depending on the data

# Launch training
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$DATA_PATH/train.parquet" \
    data.val_files="$DATA_PATH/test_2p.parquet" \
    +data.kwargs.chat_template_path="$VERL_PATH/recipe/usim/qwen_multi_role_template.jinja"\
    data.multiturn.enable=false \
    data.max_length=6000 \
    data.train_batch_size=2 \
    data.prompt_key=messages \
    data.response_key=generation \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain="Qwen/Qwen2.5-0.5B" \
    model.fsdp_config.model_dtype=bfloat16 \
    model.enable_gradient_checkpointing=true \
    optim.lr=2e-5 \
    optim.warmup_steps_ratio=0.1 \
    optim.lr_scheduler=cosine \
    +trainer.val_before_train=True \
    trainer.total_epochs=3 \
    trainer.project_name=dtwin_sft \
    trainer.experiment_name=medium_dataset_lora \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.save_freq=402 \
    trainer.test_freq=1 \
    trainer.max_ckpt_to_keep=2 \
    trainer.n_gpus_per_node=2 \
    +trainer.tag_eval_enable=true \
    model.lora_rank=16 \
    model.lora_alpha=16 \
    model.target_modules=all-linear 

echo "Training completed"
echo "Model saved to: $OUTPUT_DIR"