# GRPO Training

Default training target:

- Model: `/data/common/LLMs/Qwen2.5-3B`
- Train data: `/data/home/yanghj/data/complementary_labels/medqa_full_options_train_5k/train.parquet`
- Validation data: `/data/home/yanghj/data/complementary_labels/medqa_full_options/dev.parquet`
- Reward: `1` only when the output has valid format and the answer is correct; otherwise `0`

Run from the verl repo root:

```bash
conda activate verl
bash complementary_labels/training/run_qwen2_5_3b_medqa_grpo.sh
```

Useful overrides:

```bash
TOTAL_EPOCHS=1 TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 \
  bash complementary_labels/training/run_qwen2_5_3b_medqa_grpo.sh
```

The answer is judged only inside `<answer>...</answer>`. Invalid format receives
the same `0` reward as a wrong answer.

## TTRL majority-vote training

TTRL training does not use the dataset ground-truth label for reward. With
`ROLLOUT_N=8`, the reward manager groups the 8 responses for each prompt by
`uid`, extracts answer letters, and uses a most frequent answer as the pseudo
label only when it appears at least `MIN_VOTE_COUNT=2` times. If multiple
answers tie for the highest count, one of them is selected randomly with the
reproducible `TIE_BREAK_SEED`. A response receives reward `1` only when it
follows the strict format and matches that pseudo label; otherwise it receives
`0`.

Run from the verl repo root:

```bash
conda activate verl
bash complementary_labels/training/run_qwen2_5_3b_medqa_ttrl.sh
```

Useful SwanLab metrics:

```text
critic/score/mean
reward_extra/acc/mean
reward_extra/pseudo_label_available/mean
reward_extra/pseudo_label_matches_gold/mean
```

## Complementary-label reward experiments

The complementary-label train split is:

```text
/data/home/yanghj/data/complementary_labels/medqa_full_options_complementary_labels/train.parquet
```

For plain GRPO, the reward is `-1` when the extracted answer is the sampled
complementary label and `0` otherwise.

For complementary-label TTRL, the reward is `-1` when the extracted answer is
the complementary label; otherwise, if the answer matches the majority-vote
pseudo label and the output has strict format, the reward is `1`; all remaining
cases receive `0`. If the majority-vote pseudo label is the complementary
label, this prompt group does not use TTRL positive reward; it falls back to the
plain complementary-label rule where the complementary label gets `-1` and all
other labels get `0`.

If the output format is wrong but the complementary label can still be parsed,
the reward is still `-1`. This prevents the model from avoiding the negative
signal by breaking the required output format. Positive TTRL reward still
requires strict format.

Run the two complementary-label experiments serially:

```bash
sbatch complementary_labels/slurm/run/train_qwen25_3b_complementary_grpo_then_ttrl.sh
```
