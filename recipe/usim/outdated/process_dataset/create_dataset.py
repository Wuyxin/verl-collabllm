import re
import os
import datasets
import json, glob
from datasets import Dataset, DatasetDict

from verl.utils.hdfs_io import copy, makedirs
import argparse
from pathlib import Path
from collections import Counter
from datasets import Features, Value

'''
    Data preprocessing
    ASSUMPTION: response only tag still includes the response tokens.....?
    Original HF dataset only has train
    Check: DO we want to add anything else in extra_info?
    Remove columns that we don't use (metadata etc.)
'''

def write_fraction(ds, out_path, frac=0.02, seed=42):
    n = len(ds)
    k = max(1, int(round(n * frac)))
    sampled = ds.shuffle(seed=seed).select(range(k))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(out_path)
    print(f"Wrote {len(sampled)} rows ({len(sampled)/n:.2%}) to {out_path}")

def extract_response_block(text):
    m = re.search(r'<response>\s*(.*?)\s*</response>', text,flags=re.DOTALL)
    return m.group(1) if m else None

def fix_tags(s):
    if not isinstance(s, str):
        return s
    s = s.replace("<\\belief>", "</belief>").replace("<\\response>", "</response>")
    s = re.sub(r"<\\\s*belief>", "</belief>", s)
    s = re.sub(r"<\\\s*response>", "</response>", s)
    return s

BELIEF_OPEN  = re.compile(r"<\s*belief\s*>", re.I)
BELIEF_CLOSE = re.compile(r"<\s*/\s*belief\s*>", re.I)
RESP_OPEN    = re.compile(r"<\s*response\s*>", re.I)
RESP_CLOSE   = re.compile(r"<\s*/\s*response\s*>", re.I)

def add_missing_closing_tags(s: str) -> str:
    if not isinstance(s, str):
        return s
    text = s

    bo = BELIEF_OPEN.search(text)
    if bo:
        bc = BELIEF_CLOSE.search(text, bo.end())
        ro = RESP_OPEN.search(text, bo.end())
        # missing </belief> and response appears after <belief>
        if bc is None and ro:
            insert_at = ro.start()
            text = text[:insert_at] + "</belief>" + text[insert_at:]
        # (optional) if no <response> at all and no </belief>, close at end
        elif bc is None and not ro:
            text = text + "</belief>"

    ro = RESP_OPEN.search(text)
    if ro:
        rc = RESP_CLOSE.search(text, ro.end())
        if rc is None:
            text = text + "</response>"

    return text

# WARNING: Make sure to change prompt template if doing response_only
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_path', type=str, default=None)
    parser.add_argument('--local_dir', default='/dfs/project/kgrlm/common/llm_twin/data/reddit/rl_real_reddit_filtered')
    parser.add_argument('--hf_repo', default='snap-stanford/filtered_subreddit_users')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_source', default='user-sim/generation')
    #parser.add_argument('--prompt_template', default='recipe/usim/system_prompt/character_template.txt')
    parser.add_argument('--response_only', action='store_true', default=False)
    parser.add_argument('--persona', action='store_true', default=False)  # add the persona
    parser.add_argument('--past_comments', action='store_true', default=False)
    
    args = parser.parse_args()
    # If persona is true: character_template.txt
    # If past_comments is true: character_template_past_comments.txt
    # If neither persona and past_comments is true: character_template_bare.txt
    if args.persona:
        prompt_template_path = "./recipe/usim/system_prompt/character_templates/character_template.txt"
    elif args.past_comments:
        prompt_template_path = "./recipe/usim/system_prompt/character_templates/character_template_past_comments.txt"
    else:
        prompt_template_path = "./recipe/usim/system_prompt/character_templates/character_template_bare.txt"

    # CHANGE ARGS
    data_source = args.data_source 

    if args.jsonl_path:
        paths = sum((glob.glob(p.strip()) for p in args.jsonl_path.split(',')), [])
        def gen(paths):
            for p in paths:
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        ex = json.loads(line)

                        # keep only what you actually read later
                        md = ex.get("metadata") or {}
                        ex["metadata"] = {
                            "comment_author": md.get("comment_author"),
                            "post_author": md.get("post_author"),
                        }
                        yield ex

        raw_train = Dataset.from_generator(
            gen, gen_kwargs={"paths": paths},
            cache_dir=os.path.join(args.local_dir, "hf-cache")  # isolates cache
        )
        raw = DatasetDict({"train": raw_train})
        print("USING JSONL PATH")
    else:
        raw = datasets.load_dataset(args.hf_repo, 'default')

    
    print("Number of rows before filtering ", raw.num_rows)
   

    MIN_ROWS = 50  # adjust if needed

    def normalize_author(md):
        a = (md or {}).get("comment_author")
        if not isinstance(a, str):
            return None
        a = a.strip()
        # exclude "[deleted]" (case-insensitive)
        if a.lower() == "[deleted]":
            return None
        return a

    # 1) Count per-author rows on the full train set, ignoring "[deleted]"
    author_list = [normalize_author(md) for md in raw["train"]["metadata"]]
    author_list = [a for a in author_list if a is not None]
    counts = Counter(author_list)
    print("total author list length ", len(counts))

    # 2) Filter: drop "[deleted]" authors and keep only authors with >= MIN_ROWS
    def keep_batch(batch):
        keep = []
        for md in batch["metadata"]:
            a = normalize_author(md)
            keep.append(a is not None and counts.get(a, 0) >= MIN_ROWS)
        return keep

    filtered_train = raw["train"].filter(
        keep_batch, batched=True, num_proc=os.cpu_count() or 1
    )
    print("Number of rows after filtering ", filtered_train.num_rows)

    train_val = filtered_train.train_test_split(
        test_size=0.10, 
        seed=42, 
        shuffle=True,           
    )
    dataset = DatasetDict({
        'train': train_val['train'],
        'test': train_val['test'],
    })

    template_path = Path(prompt_template_path)
    raw_template = template_path.read_text(encoding="utf-8")

    '''In the make_map_fn, each data field should consist of the following 5 fields:
        data_source: The name of the dataset. To index the corresponding reward function in the RewardModel
        prompt: This field should be constructed in the format of huggingface chat_template. The tokenizer in RLHFDataset will apply chat template and tokenize the prompt.
        ability: Define the task category.
        reward_model: Currently, we only utilize the ground_truth field during evaluation. The ground_truth is computed by the extract_solution function. 
        NOTED that the implementation of the corresponding reward function should align with this extracted ground_truth.
        extra_info: Record some information of the current prompt. Not use for now.
    '''

    def make_map_fn(split):
        def process_fn(example, idx):
            post = example.pop("prompt")[0]["content"]
            #post = fix_tags(post)

            comment_history = ""
            for i, comment in enumerate(example["metadata"]["user_history"]["comments"]):
                comment_history += f"Comment {i+1} in subreddit r/{comment['subreddit']}: "
                comment_history += f"{comment['content']} \n"

            response = example.pop("completion")
            '''if args.response_only:
                response = fix_tags(response)
                response = add_missing_closing_tags(response)
                response = extract_response_block(response)  # if response is true, only take response
            else:
                response = fix_tags(response)
                response = add_missing_closing_tags(response)'''

            user_prompt = "From subreddit " + example["conv_id"] + ": " + post
            values = {
                "name": example["metadata"]["comment_author"],
                "description": example["user_persona"],
                "comment_history": comment_history,
                "platform": "Reddit",
                "memory" : None
            }
            
            system_content = fix_tags(raw_template.format(**values))    # print this out to check it's correct

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_content,
                    },
                    {
                        "role": "user",
                        "name": example["metadata"]["post_author"],
                        "content": user_prompt,
                    },
                ],
                "ability": "generation",
                "reward_model": {"style": "custom", "ground_truth": response},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "name": example["metadata"]["comment_author"],
                    "description": example["user_persona"],
                    "post": post
                    #"media_source": example["character"]["media_source"]
                },
            }
            return data

        return process_fn

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    src_cols = dataset['train'].column_names
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=src_cols)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=src_cols)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(args.local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    test_2p_path = os.path.join(local_dir, 'test_2p.parquet')
    write_fraction(test_dataset, test_2p_path, frac=0.02, seed=42)

    if hdfs_dir:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)