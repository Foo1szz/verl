#!/usr/bin/env bash
# Smoke-test GRPO on one L40 node before submitting the full P6000 run.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# L40 nodes are CUDA-only, but the cluster environment may export ROCm/HIP
# visibility variables. verl rejects mixed CUDA/HIP/ROCR visibility settings.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES
unset RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES

export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=${SAVE_FREQ:--1}
export TEST_FREQ=${TEST_FREQ:-1}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}

export LOGGER=${LOGGER:-'["console","swanlab"]'}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_3b_grpo_l40_smoke}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${HOME}/checkpoints/complementary_labels_l40_smoke"}

mkdir -p "${CHECKPOINT_DIR}/swanlab"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"

bash "${SCRIPT_DIR}/run_qwen2_5_3b_medqa_grpo.sh" \
    data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
    "$@"
