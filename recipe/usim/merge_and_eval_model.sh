MODEL_DIR=/dfs/project/kgrlm/common/llm_twin/outputs
RUN_NAME=qwen2_5_14b_bs64_n2_gptmini_evalbyclaude
STEP=global_step_300

MODEL_NAME=$RUN_NAME/$STEP
MODEL_NAME_STR=$(echo "$MODEL_NAME" | tr '/' '_')
echo "$MODEL_NAME_STR"

AMP_NUM=$(ls -d /lfs/ampere*/ | sed -E 's#.*/ampere([0-9]+)/?#\1#')
echo "The model will be stored in /lfs/ampere$AMP_NUM"

cd /dfs/project/kgrlm/shirwu/digital-human-lm/verl

python3 -m verl.model_merger merge --backend fsdp \
    --local_dir $MODEL_DIR/$MODEL_NAME/actor \
    --target_dir /lfs/ampere$AMP_NUM/0/$USER/$MODEL_NAME_STR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python recipe/usim/eval_vllm.py \
    --gpu_mem_util 0.6 \
    --num_examples 100 \
    --gpus "4,5,6,7" \
    --prefix $STEP \
    --rl_model_merged /lfs/ampere$AMP_NUM/0/$USER/$MODEL_NAME_STR \
    --out_dir /dfs/project/kgrlm/common/llm_twin/outputs_compare/$RUN_NAME \
    --base_model Qwen/Qwen2.5-14B-Instruct \
    --test_parquet /dfs/project/kgrlm/common/llm_twin/data/reddit/persona/test.parquet