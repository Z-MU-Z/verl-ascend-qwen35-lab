#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

BOOTSTRAP_BUNDLE_ROOT="${BOOTSTRAP_BUNDLE_ROOT:-/shared/bootstrap_bundle}"
BUNDLE_DIST_DIR="${BUNDLE_DIST_DIR:-${BOOTSTRAP_BUNDLE_ROOT}/dist}"
BUNDLE_PYTHON_DIR="${BUNDLE_PYTHON_DIR:-${BOOTSTRAP_BUNDLE_ROOT}/python}"

ASCEND_TOOLKIT_ENV="${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ASCEND_ATB_ENV="${ASCEND_ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
SHARED_ENV_ROOT="${SHARED_ENV_ROOT:-/shared/envs/qwen35}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BOOTSTRAP_UPGRADE_BUILD_TOOLS="${BOOTSTRAP_UPGRADE_BUILD_TOOLS:-1}"
INSTALL_BASE_WHEELS_FROM_BUNDLE="${INSTALL_BASE_WHEELS_FROM_BUNDLE:-1}"
ALLOW_TORCH_FALLBACK_DEBUG="${ALLOW_TORCH_FALLBACK_DEBUG:-1}"

TRANSFORMERS_TARBALL="${TRANSFORMERS_TARBALL:-${BUNDLE_DIST_DIR}/transformers-cc7ab9be.tar.gz}"
VLLM_WHEEL_GLOB="${VLLM_WHEEL_GLOB:-${BUNDLE_DIST_DIR}/vllm-0.18.0-*.whl}"
VLLM_ASCEND_TARBALL="${VLLM_ASCEND_TARBALL:-${BUNDLE_DIST_DIR}/vllm-ascend-54879467.tar.gz}"
CATLASS_TARBALL="${CATLASS_TARBALL:-${BUNDLE_DIST_DIR}/catlass-src.tar.gz}"

WORK_ROOT="${WORK_ROOT:-/tmp/qwen35_fallback_bootstrap}"
VLLM_ASCEND_SRC_DIR="${WORK_ROOT}/vllm-ascend-54879467-src"
CATLASS_SRC_DIR="${WORK_ROOT}/catlass-src"
HELPER_BIN_DIR="${WORK_ROOT}/helper-bin"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

maybe_install_wheel_from_bundle() {
  local pattern="$1"
  local matches=()
  shopt -s nullglob
  matches=(${pattern})
  shopt -u nullglob
  if ((${#matches[@]} > 0)); then
    "${PYTHON_BIN}" -m pip install "${matches[@]}"
  fi
}

echo "[bootstrap_new_npu_machine_fallback]"
echo "repo=${ROOT_DIR}"
echo "bootstrap_bundle=${BOOTSTRAP_BUNDLE_ROOT}"
echo "bundle_dist=${BUNDLE_DIST_DIR}"
echo "bundle_python=${BUNDLE_PYTHON_DIR}"

if ! command -v npu-smi >/dev/null 2>&1; then
  echo "Missing required command: npu-smi" >&2
  exit 1
fi

if [[ ! -f "${ASCEND_TOOLKIT_ENV}" || ! -f "${ASCEND_ATB_ENV}" ]]; then
  echo "Missing Ascend runtime env scripts." >&2
  exit 1
fi

require_file "${TRANSFORMERS_TARBALL}"
require_file "${VLLM_ASCEND_TARBALL}"
require_file "${CATLASS_TARBALL}"

VLLM_WHEEL="$(compgen -G "${VLLM_WHEEL_GLOB}" | head -n 1 || true)"
if [[ -z "${VLLM_WHEEL}" ]]; then
  echo "Missing required vllm wheel matching: ${VLLM_WHEEL_GLOB}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ASCEND_TOOLKIT_ENV}"
# shellcheck disable=SC1090
source "${ASCEND_ATB_ENV}"

if [[ ! -d "${SHARED_ENV_ROOT}" ]]; then
  "${PYTHON_BIN}" -m venv "${SHARED_ENV_ROOT}"
fi

# shellcheck disable=SC1091
source "${SHARED_ENV_ROOT}/bin/activate"
PYTHON_BIN="${SHARED_ENV_ROOT}/bin/python"

if [[ "${BOOTSTRAP_UPGRADE_BUILD_TOOLS}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
fi

if [[ "${INSTALL_BASE_WHEELS_FROM_BUNDLE}" == "1" ]]; then
  maybe_install_wheel_from_bundle "${BUNDLE_PYTHON_DIR}/torch-*.whl"
  maybe_install_wheel_from_bundle "${BUNDLE_PYTHON_DIR}/torch_npu-*.whl"
  maybe_install_wheel_from_bundle "${BUNDLE_PYTHON_DIR}/triton_ascend-*.whl"
  maybe_install_wheel_from_bundle "${BUNDLE_PYTHON_DIR}/triton-ascend-*.whl"
fi

"${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements-npu.txt"
"${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}" --no-deps
"${PYTHON_BIN}" -m pip install setuptools-scm vcs-versioning
"${PYTHON_BIN}" -m pip install "${TRANSFORMERS_TARBALL}" "${VLLM_WHEEL}"

rm -rf "${WORK_ROOT}"
mkdir -p "${VLLM_ASCEND_SRC_DIR}" "${CATLASS_SRC_DIR}" "${HELPER_BIN_DIR}"
tar -xzf "${VLLM_ASCEND_TARBALL}" -C "${VLLM_ASCEND_SRC_DIR}" --strip-components=1
tar -xzf "${CATLASS_TARBALL}" -C "${CATLASS_SRC_DIR}" --strip-components=1

PREPARE_ARGS=(
  --source-dir "${VLLM_ASCEND_SRC_DIR}"
  --catlass-source-dir "${CATLASS_SRC_DIR}"
  --helper-bin-dir "${HELPER_BIN_DIR}"
  --helper-python3 /usr/bin/python3.9
  --helper-llvm-objdump /usr/local/Ascend/cann-8.5.0/tools/ccec_compiler/bin/llvm-objdump
)
if [[ "${ALLOW_TORCH_FALLBACK_DEBUG}" == "1" ]]; then
  PREPARE_ARGS+=(--allow-torch-fallback-debug)
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/ascend/prepare_vllm_ascend_source.py" "${PREPARE_ARGS[@]}"

export SOC_VERSION="${SOC_VERSION:-ascend910b4}"
export PATH="${HELPER_BIN_DIR}:$PATH"
"${PYTHON_BIN}" -m pip install -v --no-build-isolation --no-deps "${VLLM_ASCEND_SRC_DIR}"

bash "${ROOT_DIR}/scripts/ascend/verify_new_npu_machine.sh"
