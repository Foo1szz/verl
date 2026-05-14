# complementary_labels

Project workspace for MedQA complementary-label experiments with verl.

## Data preprocessing

The current preprocessing script uses the full-option MedQA files:

- `/data/common/datasets/MedQA_Oringin/train.jsonl`
- `/data/common/datasets/MedQA_Oringin/dev.jsonl`
- `/data/common/datasets/MedQA_Oringin/test.jsonl`

It does not use the `4_options` directory.

Run:

```bash
conda run --no-capture-output -n verl \
  python complementary_labels/data_preprocess/prepare_medqa_full_options.py
```

Outputs are written to:

```text
~/data/complementary_labels/medqa_full_options/
```

For verl RL training/evaluation, use:

```text
data.train_files=/data/home/yanghj/data/complementary_labels/medqa_full_options/train.parquet
data.val_files=/data/home/yanghj/data/complementary_labels/medqa_full_options/test.parquet
data.prompt_key=prompt
```

`dev.parquet` is also generated for optional validation or analysis.

Create the reproducible 5k RL training subset:

```bash
conda run --no-capture-output -n verl \
  python complementary_labels/data_preprocess/sample_medqa_train_subset.py \
  --num-samples 5000 \
  --seed 42
```

The subset is written to:

```text
~/data/complementary_labels/medqa_full_options_train_5k/train.parquet
```

For RL training on the 5k subset, use:

```text
data.train_files=/data/home/yanghj/data/complementary_labels/medqa_full_options_train_5k/train.parquet
data.val_files=/data/home/yanghj/data/complementary_labels/medqa_full_options/test.parquet
data.prompt_key=prompt
```

Create a complementary-label version of the full train split. Each question
keeps the original correct answer and adds one randomly sampled incorrect
option as `reward_model.complementary_label`:

```bash
conda run --no-capture-output -n verl \
  python complementary_labels/data_preprocess/create_medqa_complementary_labels.py \
  --seed 42
```

The generated train file is:

```text
/data/home/yanghj/data/complementary_labels/medqa_full_options_complementary_labels/train.parquet
```

`reward_model.ground_truth` is still the correct option. The complementary
label is also mirrored in `extra_info.complementary_label` for debugging.

## Evaluation

Evaluate `/data/common/LLMs/Qwen2.5-3B` on the processed MedQA test set with
1024 input tokens, 512 generated tokens, and greedy decoding:

```bash
conda run --no-capture-output -n verl \
  python complementary_labels/evaluation/evaluate_medqa_qwen.py \
  --data-file ~/data/complementary_labels/medqa_full_options/test.parquet \
  --model-path /data/common/LLMs/Qwen2.5-3B \
  --output-dir complementary_labels/results/qwen2_5_3b_medqa_test \
  --max-input-length 1024 \
  --max-new-tokens 512 \
  --batch-size 1
```

For a smoke test:

```bash
conda run --no-capture-output -n verl \
  python complementary_labels/evaluation/evaluate_medqa_qwen.py --limit 2
```

The script writes `predictions.jsonl`, `metrics.json`, and `run_config.json`.

## GRPO training

Train `/data/common/LLMs/Qwen2.5-3B` on the reproducible 5k MedQA subset:

```bash
conda activate verl
bash complementary_labels/training/run_qwen2_5_3b_medqa_grpo.sh
```

The training script uses:

```text
train_file=/data/home/yanghj/data/complementary_labels/medqa_full_options_train_5k/train.parquet
val_file=/data/home/yanghj/data/complementary_labels/medqa_full_options/dev.parquet
reward_file=complementary_labels/rewards/medqa_reward.py
```

Default reward is `1` only when the output has valid format and the answer is
correct; all other cases receive `0`. The answer is judged only inside
`<answer>...</answer>`.

TTRL majority-vote training uses no dataset label for the reward. For each
prompt, it samples 8 responses, chooses the unique most frequent extracted
answer when it appears at least twice, and rewards strict-format responses that
match that pseudo label:

```bash
conda activate verl
bash complementary_labels/training/run_qwen2_5_3b_medqa_ttrl.sh
```
