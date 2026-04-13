#!/usr/bin/env bash
# dependency: vllm==0.18.0, vllm-ascend@<54879467>, transformers@<cc7ab9be>
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIRED_ENV_PATH="/home/zmz/envs/qwen35-t29-lite"

if [[ "${VIRTUAL_ENV:-}" != "${REQUIRED_ENV_PATH}" ]]; then
  echo "This script must run from ${REQUIRED_ENV_PATH}" >&2
  echo "Current VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>}" >&2
  echo "The old shared env /shared/envs/qwen35 is deprecated for active bring-up." >&2
  exit 2
fi

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/ascend/env.qwen35_npu.sh"

# Ascend: `use_remove_padding=True` can segfault in conv backward (see KNOWN_ISSUES §4).
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-False}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-1}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-1}"
ENABLE_SLEEP_MODE="${ENABLE_SLEEP_MODE:-False}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-8192}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-['console']}"

PROJECT_NAME="${PROJECT_NAME:-GRPO-Qwen3_5}"
EXP_NAME="${EXP_NAME:-GRPO-Qwen3_5-4B}"
ENGINE="${ENGINE:-vllm}"
RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"
MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/models/Qwen3.5-4B}"
CKPTS_DIR="${CKPTS_DIR:-${RAY_DATA_HOME}/ckpts/${PROJECT_NAME}/${EXP_NAME}}"
TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/geo3k/train.parquet}"
TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/data/geo3k/test.parquet}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-16}"
FSDP_SIZE="${FSDP_SIZE:-16}"
GEN_TP="${GEN_TP:-2}"
SP_SIZE="${SP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.40}"
ROLLOUT_N="${ROLLOUT_N:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"

mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${CKPTS_DIR}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/ascend/check_qwen35_npu_env.py"

start_time="$(date +%Y%m%d_%H%M%S)"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.image_key=images \
  data.shuffle=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.freeze_vision_tower="${FREEZE_VISION_TOWER}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
  actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
  actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.actor.fsdp_config.offload_policy=True \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size="${SP_SIZE}" \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.offload_policy=True \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  actor_rollout_ref.rollout.ignore_eos=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_NUM_WORKERS}" \
  +actor_rollout_ref.rollout.enable_sleep_mode="${ENABLE_SLEEP_MODE}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  algorithm.use_kl_in_reward=False \
  reward.num_workers="${REWARD_NUM_WORKERS}" \
  trainer.use_legacy_worker_impl=auto \
  trainer.critic_warmup=0 \
  "trainer.logger=${TRAINER_LOGGERS}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.balance_batch=False \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.resume_from_path=checkpoints/ \
  trainer.val_before_train=True \
  trainer.save_freq=5 \
  trainer.test_freq=5 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  "$@" 2>&1 | tee "${ROOT_DIR}/logs/qwen3_5-4b-${start_time}.log"
