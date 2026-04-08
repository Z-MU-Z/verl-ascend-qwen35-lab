#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ASCEND_TOOLKIT_ENV="${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ASCEND_ATB_ENV="${ASCEND_ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"

# Shared storage layout for the Huawei two-host lab.
export SHARED_ROOT="${SHARED_ROOT:-/shared}"
export SHARED_ENV_ROOT="${SHARED_ENV_ROOT:-${SHARED_ROOT}/envs/qwen35}"
export SHARED_WEIGHTS_ROOT="${SHARED_WEIGHTS_ROOT:-${SHARED_ROOT}/weights}"
export SHARED_DATA_ROOT="${SHARED_DATA_ROOT:-${SHARED_ROOT}/data}"
export SHARED_CKPT_ROOT="${SHARED_CKPT_ROOT:-${SHARED_ROOT}/ckpts}"
export SHARED_LOG_ROOT="${SHARED_LOG_ROOT:-${SHARED_ROOT}/logs}"

# Keep the logical training home consistent across hosts.
export RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"
export MODEL_ROOT="${MODEL_ROOT:-${RAY_DATA_HOME}/models}"
export DATA_ROOT="${DATA_ROOT:-${RAY_DATA_HOME}/data}"

mkdir -p "${SHARED_ENV_ROOT}" "${SHARED_WEIGHTS_ROOT}" "${SHARED_DATA_ROOT}" "${SHARED_CKPT_ROOT}" "${SHARED_LOG_ROOT}"

# Common first-run bring-up settings based on the Qwen3.5 Ascend lab notes.
export QWEN35_ENV_STACK="${QWEN35_ENV_STACK:-pr5682}"
export QWEN35_USE_REMOVE_PADDING="${QWEN35_USE_REMOVE_PADDING:-0}"
export QWEN35_ULYSSES_SP_SIZE="${QWEN35_ULYSSES_SP_SIZE:-1}"
export QWEN35_WRAP_LAYER="${QWEN35_WRAP_LAYER:-Qwen3_5DecoderLayer}"

# Pull in the host Ascend runtime first when it is available so
# `torch_npu` and `vllm` can resolve the required shared libraries.
if [[ -f "${ASCEND_TOOLKIT_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_TOOLKIT_ENV}"
fi

if [[ -f "${ASCEND_ATB_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_ATB_ENV}"
fi

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/ascend/env.qwen35_npu.sh"

if [[ -f "${SHARED_ENV_ROOT}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${SHARED_ENV_ROOT}/bin/activate"
fi

cat <<EOF
[qwen35_shared]
repo=${ROOT_DIR}
shared_env=${SHARED_ENV_ROOT}
shared_weights=${SHARED_WEIGHTS_ROOT}
shared_data=${SHARED_DATA_ROOT}
shared_ckpts=${SHARED_CKPT_ROOT}
shared_logs=${SHARED_LOG_ROOT}
ray_data_home=${RAY_DATA_HOME}
python=$(command -v python || command -v python3)
EOF
