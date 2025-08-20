import os
from datasets import load_dataset

HF_REPO = "snap-stanford/synthetic_subreddit_advice" 
OUT_DIR = "/lfs/ampere1/0/echoi1/digitial-human-lm/data/reddit_sft" 

raw = load_dataset(HF_REPO)
train_val = raw["train"].train_test_split(test_size=0.10, seed=42, shuffle=True)

def build_row(ex):
    post = ex.get("post") or ""
    resp = ex.get("response") or ""

    system = ""
    md = ex.get("metadata") or {}
    cp = md.get("complete_prompt") or []
    if isinstance(cp, dict):
        cp = [cp]
    if cp:
        item0 = cp[0]
        if isinstance(item0, dict):
            system = item0.get("content", "") or ""
        elif isinstance(item0, str):
            system = item0

    user_prompt = "You are responding to this Reddit post: "+ post
    prompt = (system.rstrip() + "\n\n" + user_prompt) if system else user_prompt

    ch = ex.get("character") or {}
    name = ch.get("name") if isinstance(ch, dict) else None

    return {
        "prompt": prompt,                
        "response": resp,                 
        "extra_info": {"name": name},   
        "name": name,  
    }

train_cols = train_val["train"].column_names
val_cols   = train_val["test"].column_names

train_ds = train_val["train"].map(build_row, remove_columns=train_cols)
val_ds   = train_val["test"].map(build_row, remove_columns=val_cols)

os.makedirs(OUT_DIR, exist_ok=True)

train_ds.to_parquet(os.path.join(OUT_DIR, "train.parquet"))
val_ds.to_parquet(os.path.join(OUT_DIR, "val.parquet"))

print(os.path.join(OUT_DIR, "train.parquet"))
print(os.path.join(OUT_DIR, "val.parquet"))
