#!/usr/bin/env bash
# Entropy-weighted TTRL training on the MedQA complementary-label train split.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${PROJECT_DIR}/.." && pwd)

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

MODEL_PATH=${MODEL_PATH:-/data/common/LLMs/Qwen2.5-3B}
TRAIN_FILE=${TRAIN_FILE:-/data/home/yanghj/data/complementary_labels/medqa_full_options_complementary_labels/train.parquet}
VAL_FILE=${VAL_FILE:-/data/home/yanghj/data/complementary_labels/medqa_full_options/dev.parquet}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
INFER_BACKEND=${INFER_BACKEND:-vllm}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-16}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.65}
MIN_VOTE_COUNT=${MIN_VOTE_COUNT:-2}
TIE_BREAK_SEED=${TIE_BREAK_SEED:-42}
PSEUDO_BETA=${PSEUDO_BETA:-2.0}
PSEUDO_EPS=${PSEUDO_EPS:-1e-8}

PPO_EPOCHS=${PPO_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-80}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
SAVE_FREQ=${SAVE_FREQ:-80}
TEST_FREQ=${TEST_FREQ:-4}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
LOGGER=${LOGGER:-'["console","swanlab"]'}

PROJECT_NAME=${PROJECT_NAME:-complementary_labels_medqa}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_3b_cl_entropy_ttrl}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${HOME}/checkpoints/complementary_labels_cl_entropy_ttrl"}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-"${CHECKPOINT_DIR}/swanlab"}
mkdir -p "${CHECKPOINT_DIR}/swanlab"

cd "${REPO_ROOT}"

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}"
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}"
)

ROLLOUT=(
    actor_rollout_ref.rollout.name="${INFER_BACKEND}"
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.n="${ROLLOUT_N}"
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
)

REWARD=(
    reward.num_workers=1
    reward.reward_manager.source=importlib
    reward.reward_manager.name=MedQAComplementaryEntropyWeightedTTRLRewardManager
    reward.reward_manager.module.path="${PROJECT_DIR}/rewards/medqa_complementary_entropy_weighted_ttrl_reward.py"
    reward.custom_reward_function.path="${PROJECT_DIR}/rewards/medqa_complementary_entropy_weighted_ttrl_reward.py"
    reward.custom_reward_function.name=compute_score
    +reward.custom_reward_function.reward_kwargs.min_vote_count="${MIN_VOTE_COUNT}"
    +reward.custom_reward_function.reward_kwargs.require_format_for_positive=True
    +reward.custom_reward_function.reward_kwargs.tie_break_seed="${TIE_BREAK_SEED}"
    +reward.custom_reward_function.reward_kwargs.beta="${PSEUDO_BETA}"
    +reward.custom_reward_function.reward_kwargs.eps="${PSEUDO_EPS}"
    +reward.custom_reward_function.reward_kwargs.use_token_entropy=True
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger="${LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.val_before_train="${VAL_BEFORE_TRAIN}"
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.max_actor_ckpt_to_keep=1
    trainer.max_critic_ckpt_to_keep=1
    trainer.log_val_generations=0
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
