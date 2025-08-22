import re
import os
import datasets
from datasets import DatasetDict

from verl.utils.hdfs_io import copy, makedirs
import argparse
from pathlib import Path

'''
    Data preprocessing
    ASSUMPTION: response only tag still includes the response tokens.....?
    Original HF dataset only has train
    Check: DO we want to add anything else in extra_info?
    Remove columns that we don't use (metadata etc.)
'''

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
    parser.add_argument('--local_dir', default='/lfs/ampere4/0/echoi1/digitial-human-lm/data/reddit')
    parser.add_argument('--hf_repo', default='snap-stanford/synthetic_subreddit')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_source', default='user-sim/generation')
    parser.add_argument('--prompt_template', default='recipe/usim/character_template.txt')
    parser.add_argument('--response_only', action='store_true', default=False)

    args = parser.parse_args()

    # CHANGE ARGS
    data_source = args.data_source 

    raw = datasets.load_dataset(args.hf_repo, 'default')     
    train_val = raw['train'].train_test_split(
        test_size=0.10, 
        seed=42, 
        shuffle=True,           
    )
    dataset = DatasetDict({
        'train': train_val['train'],
        'test': train_val['test'],
    })

    template_path = Path(args.prompt_template)
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
            post = example.pop("post")
            post = fix_tags(post)
            
            response = example.pop("response")
            if args.response_only:
                response = fix_tags(response)
                response = add_missing_closing_tags(response)
                response = extract_response_block(response)  # if response is true, only take response
            else:
                response = fix_tags(response)
                response = add_missing_closing_tags(response)

            user_prompt = post
            values = {
                "name": example["character"]["name"],
                "description": example["character"]["description"],
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
                        "name": "author of the post",
                        "content": user_prompt,
                    },
                ],
                "ability": "generation",
                "reward_model": {"style": "custom", "ground_truth": response},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "name": example["character"]["name"],
                    "description": example["character"]["description"],
                    "media_source": example["character"]["media_source"]
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

    if hdfs_dir:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)