#!/usr/bin/env bash

set -euo pipefail
set -x

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONDA_SH="${CONDA_SH:-/data/software/anaconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-verl-train}"

if [[ -f "${CONDA_SH}" ]]; then
    # Auto-activate the env unless the user already did it.
    # If your shell setup is different, override CONDA_SH or activate manually.
    source "${CONDA_SH}"
    conda activate "${ENV_NAME}"
fi

cd "${PROJECT_ROOT}"

# Assumption: gpu04 maps to CUDA device index 4 on this machine.
# If not, override at runtime, e.g. CUDA_VISIBLE_DEVICES=0 bash ...
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export VERL_VLLM_WEIGHT_SYNC_USE_SHM="${VERL_VLLM_WEIGHT_SYNC_USE_SHM:-1}"
# Avoid auto-connecting to an old Ray cluster recorded in /tmp/ray/ray_current_cluster.
unset RAY_ADDRESS

mkdir -p logs

TRAIN_FILE="${TRAIN_FILE:-/mnt/sharedata/ssd_large/users/hanjie/verl-train-noise/processed/clean/train.jsonl}"
# verl currently still constructs a val dataloader even if validation is disabled.
# Keep a valid file here, but validation itself is turned off below.
# Note: the provided eval file uses data_source=AIME_2025, which does not match
# verl's built-in lowercase "aime*" reward routing. If you later enable validation,
# we should add a custom reward function or normalize the data_source first.
VAL_FILE="${VAL_FILE:-/mnt/sharedata/ssd_large/users/hanjie/verl-train-noise/processed/evals/aime_2025.jsonl}"
MODEL_PATH="${MODEL_PATH:-/mnt/sharedata/ssd_large/common/LLMs/Qwen2.5-1.5B-Instruct}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-2}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"

PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

LR="${LR:-3e-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-20}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.10}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}"
ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB="${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB:-1024}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-1}"

NOW="$(date +%Y%m%d_%H%M%S)"
PROJECT_NAME="${PROJECT_NAME:-verl_local_math_grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2_5_1_5b_full_grpo_gpu04_${NOW}}"
LOG_PATH="${LOG_PATH:-logs/${EXPERIMENT_NAME}.log}"

if [[ "${ROLLOUT_BACKEND}" == "vllm" ]]; then
    ROLLOUT_ARGS=(
        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.tensor_model_parallel_size=1
        actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
        actor_rollout_ref.rollout.n="${ROLLOUT_N}"
        actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
        actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
        actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}"
        actor_rollout_ref.rollout.enable_chunked_prefill=False
        actor_rollout_ref.rollout.load_format=safetensors
        actor_rollout_ref.rollout.free_cache_engine=True
        actor_rollout_ref.rollout.enforce_eager=False
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB}"
        actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_NUM_WORKERS}"
    )
elif [[ "${ROLLOUT_BACKEND}" == "hf" ]]; then
    ROLLOUT_ARGS=(
        actor_rollout_ref.rollout.name=hf
        actor_rollout_ref.rollout.tensor_model_parallel_size=1
        actor_rollout_ref.rollout.top_k=0
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
        actor_rollout_ref.rollout.n="${ROLLOUT_N}"
        actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
        actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_NUM_WORKERS}"
    )
else
    echo "Unsupported ROLLOUT_BACKEND=${ROLLOUT_BACKEND}. Use vllm or hf." >&2
    exit 1
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +ray_kwargs.ray_init.address=local \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    "${ROLLOUT_ARGS[@]}" \
    "$@" 2>&1 | tee "${LOG_PATH}"
