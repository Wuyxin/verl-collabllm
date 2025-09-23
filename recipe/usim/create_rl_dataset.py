import re
import os
import json
import glob
from datasets import Dataset, DatasetDict
from pathlib import Path
import argparse

def create_aita_rl_dataset(jsonl_path, output_dir, system_prompt_path):
    """
    Create RL dataset from AITA JSONL file for GRPO training.
    
    Args:
        jsonl_path: Path to AITA dataset JSONL file
        output_dir: Directory to save train/test parquet files
        system_prompt_path: Path to system prompt template
    """
    
    # Load system prompt template
    with open(system_prompt_path, 'r', encoding='utf-8') as f:
        system_prompt_template = f.read()
    
    # Load JSONL data
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples from {jsonl_path}")
    
    # Process examples into RL format
    def process_example(example, idx):
        # Extract AITA post content from the prompt
        aita_post = example["prompt"][0]["content"]
        
        # Extract user persona
        user_persona = example["user_persona"]
        
        # Extract ground truth response (the actual comment)
        ground_truth_response = example["completion"]
        
        # Extract existing tag if available
        existing_tag = example.get("tag", "")
        
        # Create system prompt with persona
        system_content = system_prompt_template.format(persona=user_persona)
        
        # Create the expected completion format with tag and response
        expected_completion = f"<tag>\n{existing_tag}\n</tag>\n<response>\n{ground_truth_response}\n</response>"
        
        # Create the data structure expected by VERL
        data = {
            "data_source": "aita-rl-training",
            "prompt": [
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user", 
                    "content": aita_post,
                },
            ],
            "ability": "generation",
            "reward_model": {
                "style": "custom", 
                "ground_truth": expected_completion
            },
            "extra_info": {
                "split": "train",  # Will be updated for test split
                "index": idx,
                "original_post": aita_post,
                "user_persona": user_persona,
                "existing_tag": existing_tag,
                "post_id": example["metadata"].get("post_id", ""),
                "comment_id": example["metadata"].get("comment_id", ""),
            },
        }
        
        return data
    
    # Process all examples
    processed_examples = [process_example(ex, idx) for idx, ex in enumerate(examples)]
    
    # Create dataset and split into train/test
    full_dataset = Dataset.from_list(processed_examples)
    
    # Split into train/test (90/10 split)
    train_test_split = full_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    
    # Update split info in extra_info
    def update_split_info(batch, split_name):
        for item in batch["extra_info"]:
            item["split"] = split_name
        return batch
    
    train_dataset = train_test_split["train"].map(
        lambda batch: update_split_info(batch, "train"), 
        batched=True
    )
    test_dataset = train_test_split["test"].map(
        lambda batch: update_split_info(batch, "test"), 
        batched=True
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    
    print(f"Saved {len(train_dataset)} training examples to {train_path}")
    print(f"Saved {len(test_dataset)} test examples to {test_path}")
    
    # Also create a small test subset for quick validation
    test_2p_path = os.path.join(output_dir, "test_2p.parquet")
    n_test_2p = max(1, int(len(test_dataset) * 0.02))
    test_2p = test_dataset.shuffle(seed=42).select(range(n_test_2p))
    test_2p.to_parquet(test_2p_path)
    print(f"Saved {len(test_2p)} examples ({len(test_2p)/len(test_dataset):.2%}) to {test_2p_path}")
    
    return train_path, test_path, test_2p_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create RL dataset from AITA JSONL for GRPO training")
    parser.add_argument(
        '--jsonl_path', 
        type=str, 
        default='/dfs/project/kgrlm/common/llm_twin/reddit/tagged/aita_tagged.jsonl',
        help='Path to AITA dataset JSONL file'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='/dfs/project/kgrlm/common/llm_twin/reddit/tagged',
        help='Output directory for train/test parquet files'
    )
    parser.add_argument(
        '--system_prompt_path',
        type=str,
        default='/dfs/project/kgrlm/akhatua/digitial-human-lm/verl/recipe/usim/rl_tag_template.txt',
        help='Path to system prompt template'
    )
    
    args = parser.parse_args()
    
    print("Creating AITA RL dataset...")
    print(f"Input JSONL: {args.jsonl_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"System prompt: {args.system_prompt_path}")
    
    train_path, test_path, test_2p_path = create_aita_rl_dataset(
        args.jsonl_path, 
        args.output_dir, 
        args.system_prompt_path
    )
    
    print("\nDataset creation completed!")
    print(f"Train: {train_path}")
    print(f"Test: {test_path}")
    print(f"Test 2%: {test_2p_path}")
