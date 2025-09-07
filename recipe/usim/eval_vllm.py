import os
import gc
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import csv


def build_prompts(tokenizer, msgs_batch, max_prompt_tokens):
    prompts = []
    for msgs in msgs_batch:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        if max_prompt_tokens:
            toks = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            if len(toks) > max_prompt_tokens:
                toks = toks[-max_prompt_tokens:] 
                text = tokenizer.decode(toks, skip_special_tokens=True)
        prompts.append(text)
    return prompts


def make_llm(model, gpus, dtype, max_model_len, gpu_mem_util, enable_lora=False):
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        tp = len(gpus.split(","))
    else:
        tp = 1

    llm = LLM(
        model=model,
        dtype=dtype,                     
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_lora=enable_lora
    )
    return llm


def generate_texts(llm, prompts, max_new_tokens, temperature, top_p, request_batch: int = 64, lora_request=None):
    sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)

    outs=[]
    for i in tqdm(range(0, len(prompts), request_batch), desc="Generating"):
        sub = prompts[i:i+request_batch]
        if lora_request is not None:
            results = llm.generate(sub, sp, lora_request=lora_request, use_tqdm=False)
        else:
            results = llm.generate(sub, sp, use_tqdm=False)
        for r in results:
            outs.append((r.outputs[0].text if r.outputs else "").strip())
    return outs


def read_dataset(parquet_path, num_examples):
    ds = load_dataset("parquet", data_files=parquet_path, split="train")
    msgs = []
    gts  = []
    for ex in ds:
        m = ex["prompt"]
        gt = ex["reward_model"]["ground_truth"] if ex.get("reward_model") and ex["reward_model"] is not None else ""
        msgs.append(m)
        gts.append(gt)

    if num_examples is not None:
        msgs = msgs[:num_examples]
        gts  = gts[:num_examples]
    return msgs, gts


def save_table(rows: List[Dict], out_csv: Path, out_md: Path):
    cols = ["prompt", "sft_response", "base_response", "trained_response", "ground_truth"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"Writing CSV to {out_csv}")

    def trunc(s, n=120):
        s = s.replace("\n", " ")
        return s if len(s) <= n else s[:n] + " …"

    with out_md.open("w", encoding="utf-8") as f:
        f.write("| prompt | sft model's response | base model's response | trained model's response | ground truth |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write("| " + " | ".join(
                trunc(r.get("prompt","")),
                ) + " |\n") 

    lines = ["| prompt | sft model's response | base model's response | trained model's response | ground truth |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join([
            trunc(r.get("prompt", "")),
            trunc(r.get("sft_response", "")),
            trunc(r.get("base_response", "")),
            trunc(r.get("trained_response", "")),
            trunc(r.get("ground_truth", "")),
        ]) + " |")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote markdown: {out_md}")

def main():
    ap = argparse.ArgumentParser("Side-by-side outputs with vLLM")
    ap.add_argument("--test_parquet", type=str, default="/dfs/project/kgrlm/common/llm_twin/data/reddit/rl/test.parquet",
                    help="Path to parquet (e.g., /dfs/project/kgrlm/common/llm_twin/data/reddit/rl/test.parquet)")
    ap.add_argument("--num_examples", type=int, default=50,
                    help="Show first N examples (set None to use all)")

    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                    help="HF id or path for base model")
    ap.add_argument("--sft_model", type=str, default=None,
                    help="Optional: SFT model path (merged weights)")
    # RL Policy should be either merged or LoRA
    ap.add_argument("--rl_model_merged", type=str, default=None,
                    help="Merged RL model path (if you already merged LoRA into base)")
    ap.add_argument("--rl_lora_adapter", type=str, default=None,
                    help="Path to LoRA adapter dir for RL policy (PEFT). Use with --base_model.")
    ap.add_argument("--rl_lora_name", type=str, default="rl_adapter",
                    help="Logical name for the LoRA adapter in vLLM")

    ap.add_argument("--gpus", type=str, default="0,1,2,3", help='e.g., "0,1" (sets CUDA_VISIBLE_DEVICES)')
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["auto","float16","bfloat16"])
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_mem_util", type=float, default=0.80)
    ap.add_argument("--max_prompt_tokens", type=int, default=None, help="Optional pre-truncation of prompt tokens")

    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--request_batch", type=int, default=64)
    ap.add_argument("--out_dir", type=str, default="/dfs/project/kgrlm/common/llm_twin/outputs_compare")
    ap.add_argument("--prefix", type=str, default="reddit_rl_eval")

    args = ap.parse_args()

    msgs_list, gold = read_dataset(args.test_parquet, args.num_examples)


    toks= {}
    def get_tok(path):
        if path is None:
            return None
        if path not in toks:
            t = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            if t.pad_token is None: 
                t.pad_token = t.eos_token
            toks[path] = t
        return toks[path]

    base_tok = get_tok(args.base_model)
    sft_tok  = get_tok(args.sft_model) if args.sft_model else None
    rl_merged_tok = get_tok(args.rl_model_merged) if args.rl_model_merged else None
    rl_lora_tok = base_tok if args.rl_lora_adapter and not args.rl_model_merged else None

    prompts_by_model={}
    prompts_by_model["base"] = build_prompts(base_tok, msgs_list, args.max_prompt_tokens)
    if sft_tok:
        prompts_by_model["sft"] = build_prompts(sft_tok, msgs_list, args.max_prompt_tokens)
    if rl_merged_tok:
        prompts_by_model["rl"] = build_prompts(rl_merged_tok, msgs_list, args.max_prompt_tokens)
    elif rl_lora_tok:
        prompts_by_model["rl"] = build_prompts(rl_lora_tok, msgs_list, args.max_prompt_tokens)


    print("[BASe] starting vLLM …")
    base_llm = make_llm(args.base_model, args.gpus, args.dtype, args.max_model_len, args.gpu_mem_util)
    base_out = generate_texts(base_llm,
                              prompts_by_model["base"],
                              args.max_new_tokens,
                              args.temperature,
                              args.top_p,
                              args.request_batch)
    del base_llm; gc.collect()

    sft_out = None
    if "sft" in prompts_by_model:
        print("[SFT] starting vLLM …")
        sft_llm = make_llm(args.sft_model, args.gpus, args.dtype, args.max_model_len, args.gpu_mem_util)
        sft_out = generate_texts(sft_llm,
                                 prompts_by_model["sft"],
                                 args.max_new_tokens,
                                 args.temperature,
                                 args.top_p,
                                 args.request_batch)
        del sft_llm; gc.collect()
    rl_out = None
    if args.rl_model_merged:
        print("[RL] starting vLLM …")
        rl_llm = make_llm(args.rl_model_merged, args.gpus, args.dtype, args.max_model_len, args.gpu_mem_util)
        rl_out = generate_texts(rl_llm,
                                prompts_by_model["rl"],
                                args.max_new_tokens,
                                args.temperature,
                                args.top_p,
                                args.request_batch)
        del rl_llm; gc.collect()
    elif args.rl_lora_adapter:
        lora_llm = make_llm(args.base_model, args.gpus, args.dtype, args.max_model_len,
                            args.gpu_mem_util, enable_lora=True)
        lora_req = LoRARequest(args.rl_lora_name, 1, args.rl_lora_adapter)
        rl_out = generate_texts(lora_llm,
                                prompts_by_model["rl"],
                                args.max_new_tokens,
                                args.temperature,
                                args.top_p,
                                args.request_batch,
                                lora_request=lora_req)
        del lora_llm; gc.collect()

    rows = []
    for i, msgs in enumerate(msgs_list):
        prompt_str = base_tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )

        rows.append({
            "prompt": prompt_str,
            "sft_response": (sft_out[i] if sft_out else ""),
            "base_response": base_out[i],
            "trained_response": (rl_out[i] if rl_out else ""),
            "ground_truth": gold[i] or "",
        })

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.prefix}.table.csv"
    out_md  = out_dir / f"{args.prefix}.table.md"
    save_table(rows, out_csv, out_md)

    (out_dir / f"{args.prefix}.raw.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote JSON output: {(out_dir / f'{args.prefix}.raw.json')}")

    print("done")

if __name__ == "__main__":
    main()
