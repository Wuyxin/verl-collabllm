MODEL_NAME=qwen2_5_14b_new_judge_bs64_n2/global_step_420
MODEL_DIR=/dfs/project/kgrlm/common/llm_twin/outputs
MODEL_NAME_STR=$(echo "$MODEL_NAME" | tr '/' '_')
echo "$MODEL_NAME_STR"

cd /dfs/project/kgrlm/shirwu/digital-human-lm/verl

python3 -m verl.model_merger merge --backend fsdp \
    --local_dir $MODEL_DIR/$MODEL_NAME/actor \
    --target_dir  /lfs/ampere4/0/$USER/$MODEL_NAME_STR

python recipe/usim/eval_vllm.py \
    --rl_model_merged /lfs/ampere4/0/$USER/$MODEL_NAME_STR \
    --out_dir /dfs/project/kgrlm/common/llm_twin/outputs_compare \
    --prefix $MODEL_NAME_STR