```
cd /dfs/project/kgrlm/shirwu/digital-human-lm/verl
python3 -m verl.model_merger merge --backend fsdp --local_dir /dfs/project/kgrlm/common/llm_twin/outputs_real_LLM/global_step_100/actor  --target_dir /dfs/project/kgrlm/common/llm_twin/rl_merged/outputs_real_LLM_global_step_100
```

```
cd /dfs/project/kgrlm/shirwu/digital-human-lm/verl
python recipe/usim/eval_vllm.py --rl_model_merged /dfs/project/kgrlm/common/llm_twin/rl_merged/outputs_real_LLM_global_step_100 --prefix reddit_rl_eval_llm_judge
```