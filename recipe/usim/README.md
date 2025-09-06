## Training User Simulator

### 0. Preliminaries
Before starting, make sure you're logged in to the following:

- [x] `GitHub`
- [x] `wandb`: `wandb login`
- [x] `Hugging Face`: `huggingface-cli login`

Also export your OpenAI or Anthropic API key:

```bash
export OPENAI_API_KEY={YOUR_KEY}
export ANTHROPIC_API_KEY={YOUR_KEY}
```

---

### 1. Pull the Latest Code

```bash
cd verl
git pull
```

---

### 2. Generate the Dataset

```bash
PYTHONPATH=. python3 recipe/usim/create_dataset.py \
  --local_dir {YOUR_WORKING_DIR}/data/reddit/persona \
  --persona
```

---

### 3. Configure the Training Script

Open `verl/recipe/usim/train_grpo.sh` and modify the following fields:

- **a) Set the working directory:**

  ```bash
  EXP_NAME={A_NEW_EXP_NAME}
  WORKING_DIR={YOUR_WORKING_DIR}
  ```

- **b) Set GPU resources (default is 8 GPUs):**

  ```bash
  NUM_GPUS=16  # Adjust based on availability, will probably need at least 16 to train a 72B model
  ```

- **c) Set batch size:**
  ```bash
  BATCH_SIZE=64  # Try 128 if possible
  ```

- **c) Set micro batch size:**

  ```bash
  MICRO_BATCH_SIZE=1  # Try >=2 if possible
  ```

Then run 
```bash
sh recipe/usim/train_grpo.sh
```

---

### Common problems:
- Out-of-memory: Increase `NUM_GPUS`; Reduce `MICRO_BATCH_SIZE`; Reduce `actor_rollout_ref.rollout.gpu_memory_utilization`; Reduce ` actor_rollout_ref.rollout.n`
