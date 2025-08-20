from datasets import load_dataset
from pprint import pprint
import random, os

LOCAL_DIR = "/lfs/ampere4/0/echoi1/digitial-human-lm/data/reddit"  # your --local_dir
ds = load_dataset(
    "parquet",
    data_files={
        "train": os.path.join(LOCAL_DIR, "train.parquet"),
        "validation": os.path.join(LOCAL_DIR, "test.parquet"),
    },
)

print(ds)                     # split sizes
print(ds["train"].features)   # column schema

for i in random.sample(range(len(ds["train"])), k=3):
    row = ds["train"][i]
    pprint({k: (row[k] if k != "prompt" else row["prompt"][:2]) for k in row})
