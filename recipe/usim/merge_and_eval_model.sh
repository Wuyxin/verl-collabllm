MODEL_NAME=qwen2_5_14b_new_judge_bs64_n2/global_step_420
MODEL_DIR=/dfs/project/kgrlm/common/llm_twin/outputs
MODEL_NAME_STR=$(echo "$MODEL_NAME" | tr '/' '_')
echo "$MODEL_NAME_STR"

AMP_NUM=$(ls -d /lfs/ampere*/ | sed -E 's#.*/ampere([0-9]+)/?#\1#')
echo "The model will be stored in /lfs/ampere$AMP_NUM"

cd /dfs/project/kgrlm/shirwu/digital-human-lm/verl

python3 -m verl.model_merger merge --backend fsdp \
    --local_dir $MODEL_DIR/$MODEL_NAME/actor \
    --target_dir /lfs/ampere$AMP_NUM/0/$USER/$MODEL_NAME_STR

python recipe/usim/eval_vllm.py \
    --num_examples 100 \
    --gpus 0,1,2,3,4,5,6,7 \
    --prefix $MODEL_NAME_STR \
    --rl_model_merged /lfs/ampere$AMP_NUM/0/$USER/$MODEL_NAME_STR \
    --out_dir /dfs/project/kgrlm/common/llm_twin/outputs_compare \
    --base_model Qwen/Qwen2.5-14B-Instruct \
    --test_parquet /dfs/project/kgrlm/common/llm_twin/data/reddit/persona/test.parquet