#!/usr/bin/env bash

set -xeuo pipefail

if [ "$#" -lt 8 ]; then
    echo "Usage: bash GRPO.sh TRAIN_FILE MODEL_PATH PROJECT_NAME EXPERIMENT_NAME ALGO_NAME MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH CHECKPOINT_DIR [VAL_FILES]" >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${VERL_ROOT}/.." && pwd)

TRAIN_FILE=$1
MODEL_PATH=$2
PROJECT_NAME=$3
EXPERIMENT_NAME=$4
ALGO_NAME=$5
MAX_PROMPT_LENGTH=$6
MAX_RESPONSE_LENGTH=$7
CHECKPOINT_DIR=$8
shift 8

if [ "$#" -gt 0 ] && [[ "$1" != *=* ]] && [[ "$1" != --* ]]; then
    VAL_FILES=$1
    shift
else
    VAL_FILES="['${REPO_ROOT}/data/gsm8k/test.parquet','${REPO_ROOT}/data/math/test.parquet']"
fi

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
INFER_BACKEND=${INFER_BACKEND:-vllm}

# Hyperparameters from arXiv-2601.22595v1, Section "RL Algorithm Hyperparameter".
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}

# Operational defaults not specified by the paper.
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}
PPO_EPOCHS=${PPO_EPOCHS:-1}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:--1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
LOGGER=${LOGGER:-'["console","wandb"]'}
RESUME_MODE=${RESUME_MODE:-auto}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-2}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
DATA_SHUFFLE=${DATA_SHUFFLE:-True}
TRAINER_BALANCE_BATCH=${TRAINER_BALANCE_BATCH:-True}

ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"${CHECKPOINT_DIR}/rollouts"}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"${CHECKPOINT_DIR}/validation"}
export WANDB_DIR=${WANDB_DIR:-"${CHECKPOINT_DIR}/wandb"}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-"${CHECKPOINT_DIR}/swanlab"}
mkdir -p "${CHECKPOINT_DIR}" "${ROLLOUT_DATA_DIR}" "${VALIDATION_DATA_DIR}" "${WANDB_DIR}" "${SWANLAB_LOG_DIR}"

if [ "${ALGO_NAME}" != "grpo" ]; then
    echo "This script is configured for GRPO. Got ALGO_NAME=${ALGO_NAME}" >&2
    exit 1
fi

cd "${VERL_ROOT}"

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle="${DATA_SHUFFLE}"
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}"
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.optim.warmup_style=constant
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
    actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE}"
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}"
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}"
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
)

ROLLOUT=(
    actor_rollout_ref.rollout.name="${INFER_BACKEND}"
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}"
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    actor_rollout_ref.rollout.val_kwargs.n=1
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}"
)

REWARD=(
    reward.reward_manager.name=naive
    reward.custom_reward_function.path=null
    reward.custom_reward_function.name=compute_score
)

TRAINER=(
    trainer.balance_batch="${TRAINER_BALANCE_BATCH}"
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
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
    trainer.resume_mode="${RESUME_MODE}"
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}"
    trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP}"
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
