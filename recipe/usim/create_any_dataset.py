import re
import os
import datasets
import json, glob
import polars as pl
from datasets import Dataset, DatasetDict

from verl.utils.hdfs_io import copy, makedirs
import argparse
from pathlib import Path
from collections import Counter
from datasets import Features, Value, Sequence

'''
    Verl data processing
    NOTES:
        - parquet works but jsonl hasn't been tested 
        - ASSUME input parquet is already processed into train/seen/unseen etc. in process_raw.py
        - CHECK if we want to add more in extra_info
        - Not tested for belief
    TODO:
        - Add persona and comment history
'''


TARGET_FEATURES = Features({
    "data_source": Value("string"),
    "prompt": [{  
        "role": Value("string"),
        "content": Value("string"),
        "name": Value("string"),
    }],
    "ability": Value("string"),
    "reward_model": {
        "style": Value("string"),
        "ground_truth": Value("string"),
    },
    "extra_info": {
        "split": Value("string"),
        "index": Value("int64"),
        "name": Value("string"),
        "post": Value("string"),
    },
})


class DatasetMapper:
    """
    `DatasetMapper` is a helper to map the original dataset to fit our training formats.
    Essentially, we need
        - comment_history:
        - poster_id: the user name/id who made the post
        - post: the post content
        - post_prompt: besides `post` content itself, it includes more details like category/subreddit.
        - responsor_id: the user name/id who made the response
        - response: the response content
    """

    def __init__(self, raw_template: str, data_source: str, tag_path=None):
        self.platform = None
        self.raw_template = raw_template
        self.data_source = data_source
        if tag_path is not None:
            dset = datasets.load_dataset("json", data_files=tag_path, split="train")
            self.tag_dict = {row["index"]: row["tags"] for row in dset}
        else:
            self.tag_dict = None

    def get_comment_history(self, example) -> str:
        raise NotImplementedError()

    def get_poster_id(self, example) -> str:
        raise NotImplementedError()

    def get_post(self, example) -> str:
        raise NotImplementedError()

    def get_post_prompt(self, example) -> str:
        raise NotImplementedError()

    def get_responsor_id(self, example) -> str:
        raise NotImplementedError()

    def get_response_ts(self, metadata: dict) -> int:
        raise NotImplementedError()

    def get_response(self, example) -> str:
        raise NotImplementedError()

    def normalize_author(self, metadata: dict) -> str | None:
        raise NotImplementedError()


    def make_map_fn_sft(self, split):
        def process_fn(example, idx):
            user_prompt = self.get_post_prompt(example)

            response = self.get_response(example)

            if self.tag_dict is not None:
                tag = self.tag_dict[idx]
                response = f"<tag>{tag}</tag><response>{response}</response>"

            responsor_id = self.get_responsor_id(example)

            values = {
                "persona": '',#example["user_persona"]
                "name": responsor_id,
                #"description": example["character"]["description"],
                "platform": "Reddit",
                "memory": None
            }
            
            system_content = self.raw_template.format(**values)
            return {
            "messages": [                  
                {"role": "system", "name": None, "content": system_content},
                {"role": "user",  "name": self.get_poster_id(example), "content": user_prompt},
            ],
            "generation": response,  
            "name": responsor_id,                 
            "split": split,
            "index": idx,
            "extra_info": {
                "split": split,
                "index": idx,
                #"description": ", "#example["user_persona"],
                #"post": "", #post,
                #"media_source": "Reddit" #example["character"]["media_source"]
            },
        }
        return process_fn

    def make_map_fn(self, split):
        # FIXME: this comment is copied from previous code and may be out-of-date.
        """In the make_map_fn, each data field should consist of the following 5 fields:
        data_source: The name of the dataset. To index the corresponding reward function in the RewardModel
        prompt: This field should be constructed in the format of huggingface chat_template. The tokenizer in RLHFDataset will apply chat template and tokenize the prompt.
        ability: Define the task category.
        reward_model: Currently, we only utilize the ground_truth field during evaluation. The ground_truth is computed by the extract_solution function.
        NOTED that the implementation of the corresponding reward function should align with this extracted ground_truth.
        extra_info: Record some information of the current prompt. Not use for now.
        """
        assert self.platform is not None, "PLATFORM must be defined in subclass"

        def process_fn(example, idx):
            post = self.get_post(example)
            comment_history = "" #self.get_comment_history(example)
            response = self.get_response(example)
            """if args.response_only:
                response = fix_tags(response)
                response = add_missing_closing_tags(response)
                response = extract_response_block(response)  # if response is true, only take response
            else:
                response = fix_tags(response)
                response = add_missing_closing_tags(response)"""
            user_prompt = self.get_post_prompt(example)
            poster_id = self.get_poster_id(example)
            responsor_id = self.get_responsor_id(example)

            values = {
                "name": poster_id,
                "persona": "",#example["user_persona"],
                "comment_history": comment_history,
                "platform": self.platform,
                "memory": None,
            }

            system_content = fix_tags(self.raw_template.format(**values))  # print this out to check it's correct

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "name": None,
                        "content": system_content,
                    },
                    {
                        "role": "user",
                        "name": poster_id,
                        "content": user_prompt,
                    },
                ],
                "ability": "generation",
                "reward_model": {"style": "custom", "ground_truth": response},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "name": responsor_id,
                    #"description": example["user_persona"],
                    "post": post,
                    # "media_source": example["character"]["media_source"]
                },
            }
            return data

        return process_fn


class RedditMapper(DatasetMapper):
    def __init__(self, raw_template: str, data_source: str, tag_path):
        super().__init__(raw_template, data_source, tag_path)
        self.platform = "Reddit"

    def get_comment_history(self, example):
        comment_history = ""
        for i, comment in enumerate(example["metadata"]["user_history"]["comments"]):
            comment_history += f"Comment {i+1} in subreddit r/{comment['subreddit']}: "
            comment_history += f"{comment['content']} \n"
        return comment_history

    def get_poster_id(self, example):
        # [EDITED]
        return example["metadata"]["author_fullname"]

    def get_post(self, example):
        return example["prompt"][0]["content"]

    def get_post_prompt(self, example):
        return "From subreddit " + example["conv_id"] + ": " + self.get_post(example)

    def get_responsor_id(self, example):
        # [EDITED]
        return example["user_id"]

    def get_response_ts(self, metadata: dict) -> int:
        # [EDITED] fixed from comment_created_utc --> created_utc
        res = int(metadata["created_utc"])
        return res

    def get_response(self, example):
        return example["completion"]

    def normalize_author(self, metadata: dict) -> str | None:
        a = (metadata or {}).get("comment_author")
        if not isinstance(a, str):
            return None
        a = a.strip()
        # exclude "[deleted]" (case-insensitive)
        if a.lower() == "[deleted]":
            return None
        return a


class AmazonReviewMapper(DatasetMapper):
    def __init__(self, raw_template: str, data_source: str):
        super().__init__(raw_template, data_source)
        self.platform = "Amazon"

    def get_comment_history(self, example):
        comment_history = ""
        for i, comment in enumerate(example["metadata"]["user_history"]["reviews"]):
            comment_history += f"Review {i+1} in product {example['parent_asin']}: "
            comment_history += f"{comment['title']}: {comment['text']} \n"
        return comment_history

    def get_poster_id(self, example):
        return example["metadata"]["item_store"]

    def get_post(self, example):
        return example["prompt"][0]["content"]

    def get_post_prompt(self, example):
        return "From Amazon Review: " + self.get_post(example)

    def get_responsor_id(self, example):
        return example["metadata"]["review_user_id"]

    def get_response_ts(self, metadata: dict) -> int:
        return int(metadata["review_timestamp"])

    def get_response(self, example):
        return example["completion"]

    def normalize_author(self, metadata: dict) -> str | None:
        a = (metadata or {}).get("review_user_id")
        if not isinstance(a, str):
            return None
        a = a.strip()
        return a


# WARNING: Make sure to change prompt template if doing response_only
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["reddit", "amazon"], required=True)
    parser.add_argument("--jsonl_path", type=str, default=None)
    parser.add_argument("--parquet_path", type=str, default=None)
    parser.add_argument("--local_dir")
    parser.add_argument("--hf_repo", default="snap-stanford/filtered_subreddit_users")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--data_source", default="user-sim/generation")
    # parser.add_argument('--prompt_template', default='recipe/usim/character_template.txt')
    parser.add_argument("--response_only", action="store_true", default=False)
    parser.add_argument("--persona", action="store_true", default=False)  # add the persona
    parser.add_argument("--past_comments", action="store_true", default=False)
    parser.add_argument("--sft", action="store_true", default=False)
    parser.add_argument("--tag_path", default="/dfs/project/kgrlm/common/llm_twin/data/reddit_debug_verl/index_tags.jsonl")
    parser.add_argument("--split", default=None)

    args = parser.parse_args()
    if args.local_dir is None:
        # FIXME: when this is finalized, change "data.new" back to "data"
        args.local_dir = f"/dfs/project/kgrlm/common/llm_twin/data.new/{args.dataset}/rl_real_{args.dataset}_filtered"

    # If persona is true: character_template.txt
    # If past_comments is true: character_template_past_comments.txt
    # If neither persona and past_comments is true: character_template_bare.txt
    '''if args.persona:
        prompt_template_path = "./recipe/usim/character_templates/character_template.txt"
    elif args.past_comments:
        prompt_template_path = "./recipe/usim/character_templates/character_template_past_comments.txt"
    else:
        prompt_template_path = "./recipe/usim/character_templates/character_template_bare.txt"'''

    prompt_template_path = "./recipe/usim/character_templates/character_template_tag.txt"

    # CHANGE ARGS
    data_source = args.data_source

    # Load prompt template
    template_path = Path(prompt_template_path)
    raw_template = template_path.read_text(encoding="utf-8")

    # Create dataset mapper
    if args.dataset == "reddit":
        MapperClass = RedditMapper
    elif args.dataset == "amazon":
        MapperClass = AmazonReviewMapper
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    mapper = MapperClass(raw_template, data_source, args.tag_path)

    if args.jsonl_path:
        # TODO: Fix this part for any dataset.
        paths = sum((glob.glob(p.strip()) for p in args.jsonl_path.split(",")), [])

        def gen(paths):
            for p in paths:
                with open(p, "r", encoding="utf-8") as f:
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
            gen, gen_kwargs={"paths": paths}, cache_dir=os.path.join(args.local_dir, "hf-cache")  # isolates cache
        )
        raw = DatasetDict({"train": raw_train})
        print("USING JSONL PATH")
    else:
        raw = Dataset.from_parquet(args.parquet_path)

    # 3). Map for each example
    # --- Map a single split (no splitting logic) ---
    # Assume `raw` is a single hf Dataset loaded from args.parquet_path
    if args.split:
        split_name = args.split
    else:
        split_name = Path(args.parquet_path).stem 

    print(f'Mapping Dataset "{split_name}": {len(raw)} rows')
    src_cols = raw.column_names
    map_fn_raw = mapper.make_map_fn(split_name) if not args.sft else mapper.make_map_fn_sft(split_name)

    if args.sft:
        features = SFT_FEATURES
    else:
        features = TARGET_FEATURES

    def map_fn_wrap(ex, idx):
        out = map_fn_raw(ex, idx)
        if args.sft:
            msgs = out.get("messages", [])
            out["messages"] = [{
                "role": str(m.get("role", "")),
                "content": str(m.get("content", "")),
                "name": "" if m.get("name") in (None, "None") else str(m.get("name"))
            } for m in msgs]

            # keep other fields consistent with features
            ei = out.get("extra_info") or {}
            out["extra_info"] = {"split": str(ei.get("split", out.get("split", ""))),
                                "index": int(ei.get("index", out.get("index", idx)))}
            out["split"] = str(out.get("split", ""))
            out["index"] = int(out.get("index", idx))
            out["name"] = "" if out.get("name") is None else str(out["name"])
            out["generation"] = "" if out.get("generation") is None else str(out["generation"])
        return out


    if args.sft:
        mapped_ds = raw.map(
            function=map_fn_wrap,
            with_indices=True,
            remove_columns=raw.column_names,
            load_from_cache_file=False,
            num_proc=1,
        )
    else:
        mapped_ds = raw.map(
            function=map_fn_wrap,
            with_indices=True,
            remove_columns=raw.column_names,
            features=TARGET_FEATURES,
            load_from_cache_file=False,
            num_proc=1,
        )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)

    # Write a small preview JSON (up to 10 rows)
    # [Comment out]
    example_path = os.path.join(local_dir, f"{split_name}.example.json")
    mapped_ds.select(range(min(10, len(mapped_ds)))).to_json(example_path)
    print(f'Wrote preview to {example_path}')

    # Write the mapped parquet
    out_path = os.path.join(local_dir, f"{split_name}.parquet")
    mapped_ds.to_parquet(out_path)
    print(f'Wrote "{split_name}" with {len(mapped_ds)} rows to {out_path}')

    # Optional HDFS copy
    if hdfs_dir:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

