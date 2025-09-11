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
from datasets import Features, Value

"""
    Data preprocessing
    ASSUMPTION: response only tag still includes the response tokens.....?
    Original HF dataset only has train
    Check: DO we want to add anything else in extra_info?
    Remove columns that we don't use (metadata etc.)
"""


def extract_response_block(text):
    m = re.search(r"<response>\s*(.*?)\s*</response>", text, flags=re.DOTALL)
    return m.group(1) if m else None


def fix_tags(s):
    if not isinstance(s, str):
        return s
    s = s.replace("<\\belief>", "</belief>").replace("<\\response>", "</response>")
    s = re.sub(r"<\\\s*belief>", "</belief>", s)
    s = re.sub(r"<\\\s*response>", "</response>", s)
    return s


BELIEF_OPEN = re.compile(r"<\s*belief\s*>", re.I)
BELIEF_CLOSE = re.compile(r"<\s*/\s*belief\s*>", re.I)
RESP_OPEN = re.compile(r"<\s*response\s*>", re.I)
RESP_CLOSE = re.compile(r"<\s*/\s*response\s*>", re.I)


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

    def __init__(self, raw_template: str, data_source: str):
        self.platform = None
        self.raw_template = raw_template
        self.data_source = data_source

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
            comment_history = self.get_comment_history(example)
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
                "description": example["user_persona"],
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
                    "description": example["user_persona"],
                    "post": post,
                    # "media_source": example["character"]["media_source"]
                },
            }
            return data

        return process_fn


class RedditMapper(DatasetMapper):
    def __init__(self, raw_template: str, data_source: str):
        super().__init__(raw_template, data_source)
        self.platform = "Reddit"

    def get_comment_history(self, example):
        comment_history = ""
        for i, comment in enumerate(example["metadata"]["user_history"]["comments"]):
            comment_history += f"Comment {i+1} in subreddit r/{comment['subreddit']}: "
            comment_history += f"{comment['content']} \n"
        return comment_history

    def get_poster_id(self, example):
        return example["metadata"]["post_author"]

    def get_post(self, example):
        return example["prompt"][0]["content"]

    def get_post_prompt(self, example):
        return "From subreddit " + example["conv_id"] + ": " + self.get_post(example)

    def get_responsor_id(self, example):
        return example["metadata"]["comment_author"]

    def get_response_ts(self, metadata: dict) -> int:
        res = int(metadata["comment_created_utc"])
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
        return "From Amazon Review " + example["metadata"]["item_main_category"] + ": " + self.get_post(example)

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
    parser.add_argument("--local_dir")
    parser.add_argument("--hf_repo", default="snap-stanford/filtered_subreddit_users")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--data_source", default="user-sim/generation")
    # parser.add_argument('--prompt_template', default='recipe/usim/character_template.txt')
    parser.add_argument("--response_only", action="store_true", default=False)
    parser.add_argument("--persona", action="store_true", default=False)  # add the persona
    parser.add_argument("--past_comments", action="store_true", default=False)

    args = parser.parse_args()
    if args.local_dir is None:
        # FIXME: when this is finalized, change "data.new" back to "data"
        args.local_dir = f"/dfs/project/kgrlm/common/llm_twin/data.new/{args.dataset}/rl_real_{args.dataset}_filtered"

    # If persona is true: character_template.txt
    # If past_comments is true: character_template_past_comments.txt
    # If neither persona and past_comments is true: character_template_bare.txt
    if args.persona:
        prompt_template_path = "./recipe/usim/character_templates/character_template.txt"
    elif args.past_comments:
        prompt_template_path = "./recipe/usim/character_templates/character_template_past_comments.txt"
    else:
        prompt_template_path = "./recipe/usim/character_templates/character_template_bare.txt"

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
    mapper = MapperClass(raw_template, data_source)

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
        raw = datasets.load_dataset(args.hf_repo, "default")

    print("Number of rows before filtering ", raw.num_rows)

    MIN_ROWS = 50  # adjust if needed
    UNSEEN_FRACTION = 0.1
    SEEN_TEST_FRACTION = 0.1
    VAL_FRACTION = 0.1
    #                →→→→→→→→→ Users
    #                   1-UNSEEN_FRACTION      | UNSEEN_FRACTION
    #   ↓          |---------------------------+--------|
    #   ↓          |                           |        |
    #   ↓          |                           |        |
    #   sort       |                           |        |
    #   by         |                           |        |
    #   timestamp  |                           |        |
    #              |                           |        |
    #              |---------------------------+        |
    #              |    VAL_FRACTION           |        |
    #              |---------------------------+        |
    #              |    SEEN_TEST_FRACTION     |        |
    #              |------------------------------------|

    info = raw["train"].map(
        lambda row: {
            "responsor_id": mapper.normalize_author(row["metadata"]),
            "timestamp": mapper.get_response_ts(row["metadata"]),
        }
    )
    responser_ids = info["responsor_id"]
    timestamps = info["timestamp"]
    raw_df = raw["train"].to_polars()
    # Add a column "reponsor_id" for convenience, and filter out null (e.g., it is "[deleted]" and was normalized to None)
    raw_df = raw_df.with_columns(
        pl.Series(responser_ids).alias("responsor_id"), pl.Series(timestamps).alias("timestamp")
    ).filter(pl.col("responsor_id").is_not_null())
    # Filter out responsors with < MIN_ROWS rows
    raw_df = raw_df.filter(pl.len().over("responsor_id") >= MIN_ROWS)
    by_responsor = raw_df.partition_by("responsor_id", maintain_order=True, as_dict=True, include_key=True)
    print(f"Total valid responsors: {len(by_responsor)}")
    print(f"Total valid responses: {len(raw_df)}")

    # 1). Split the dataset into seen/unseen datasets (sort is for stable sampling)
    unseen_responsor_ids = set(
        raw_df.select("responsor_id").unique().sort("responsor_id").sample(fraction=UNSEEN_FRACTION, seed=42).to_series()
    )
    unseen_test_df = raw_df.filter(pl.col("responsor_id").is_in(unseen_responsor_ids))
    seen_df = raw_df.filter(~pl.col("responsor_id").is_in(unseen_responsor_ids))
    print(f"By splitting seen/unseen responsors, we got seen size {len(seen_df)}, unseen size {len(unseen_test_df)}")

    # 2). For each seen responsor, split into train/val and seen_test datasets
    seen_df_per_user = seen_df.partition_by("responsor_id", maintain_order=True, as_dict=True, include_key=True)
    seen_test_df_per_user = []
    seen_train_val_df_per_user = []
    for (responsor_id,), df in seen_df_per_user.items():
        df = df.sort("timestamp")
        n = len(df)
        seen_test_n = int(round(n * SEEN_TEST_FRACTION))
        assert (
            seen_test_n > 0
        ), f"Given reasonable {MIN_ROWS=}, {SEEN_TEST_FRACTION=}, each user should have at least one row in seen test set. Adjust them if not."
        seen_test_df_per_user.append(df[n - seen_test_n :])
        seen_train_val_df_per_user.append(df[: n - seen_test_n])
    seen_test_df = pl.concat(seen_test_df_per_user)
    train_val_df = pl.concat(seen_train_val_df_per_user)
    print(f"Partitioned seen ({len(seen_df)}) into train_val ({len(train_val_df)}) and seen_test ({len(seen_test_df)})")

    train_val_ds = Dataset.from_polars(train_val_df)

    train_val_ds = train_val_ds.train_test_split(
        test_size=VAL_FRACTION,
        seed=42,
        shuffle=True,
    )
    dataset: dict[str, Dataset] = {
        "train": train_val_ds["train"],
        "val": train_val_ds["test"],
        "seen_test": Dataset.from_polars(seen_test_df),
        "unseen_test": Dataset.from_polars(unseen_test_df),
    }

    # 3). Map for each example
    mapped_dataset: dict[str, Dataset] = {}
    for ds_name, ds in dataset.items():
        print(f"""Mapping Dataset "{ds_name}": {len(ds)} rows""")
        src_cols = ds.column_names
        mapped_ds = ds.map(function=mapper.make_map_fn(ds_name), with_indices=True, remove_columns=src_cols)
        mapped_dataset[ds_name] = mapped_ds

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(args.local_dir, exist_ok=True)
    for ds_name, ds in mapped_dataset.items():
        out_path = os.path.join(local_dir, f"{ds_name}.parquet")
        ds.to_parquet(out_path)
        print(f'Wrote "{ds_name}" with {len(ds)} rows to {out_path}')

    if hdfs_dir:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
