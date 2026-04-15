#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TARGET_ENV_PATH="${TARGET_ENV_PATH:-/home/zmz/envs/qwen35-t29-lite}"
SEED_PYTHON="${SEED_PYTHON:-/shared/envs/qwen35/bin/python}"

ASCEND_TOOLKIT_ENV="${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ASCEND_ATB_ENV="${ASCEND_ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"

BOOTSTRAP_BUNDLE_ROOT="${BOOTSTRAP_BUNDLE_ROOT:-/home/zmz/bootstrap_bundle}"
BUNDLE_DIST_DIR="${BUNDLE_DIST_DIR:-${BOOTSTRAP_BUNDLE_ROOT}/dist}"
BUNDLE_PYTHON_DIR="${BUNDLE_PYTHON_DIR:-${BOOTSTRAP_BUNDLE_ROOT}/python}"
SHARED_DIST_DIR="${SHARED_DIST_DIR:-/shared/dist}"

PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-1000}"

TORCH_SPEC="${TORCH_SPEC:-torch==2.9.0}"
TORCH_NPU_SPEC="${TORCH_NPU_SPEC:-torch-npu==2.9.0}"
NUMPY_SPEC="${NUMPY_SPEC:-numpy==1.26.4}"
PYYAML_SPEC="${PYYAML_SPEC:-PyYAML==6.0.3}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.24.0}"
SETUPTOOLS_SPEC="${SETUPTOOLS_SPEC:-setuptools==80.9.0}"

TRANSFORMERS_TARBALL="${TRANSFORMERS_TARBALL:-}"
VLLM_WHEEL_GLOB="${VLLM_WHEEL_GLOB:-}"
VLLM_ASCEND_TARBALL="${VLLM_ASCEND_TARBALL:-}"
CATLASS_TARBALL="${CATLASS_TARBALL:-}"

WORK_ROOT="${WORK_ROOT:-/tmp/qwen35_t29_lite_clone}"
VLLM_ASCEND_SRC_DIR="${WORK_ROOT}/vllm-ascend-54879467-src"
CATLASS_SRC_DIR="${WORK_ROOT}/catlass-src"
HELPER_BIN_DIR="${WORK_ROOT}/helper-bin"

pick_first_existing() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -e "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

pick_first_glob() {
  local pattern
  for pattern in "$@"; do
    if [[ -z "${pattern}" ]]; then
      continue
    fi
    local matches=()
    shopt -s nullglob
    matches=(${pattern})
    shopt -u nullglob
    if ((${#matches[@]} > 0)); then
      printf '%s\n' "${matches[0]}"
      return 0
    fi
  done
  return 1
}

maybe_install_wheel_from_bundle() {
  local spec_name="$1"
  shift
  local wheel_glob
  for wheel_glob in "$@"; do
    local matches=()
    shopt -s nullglob
    matches=(${wheel_glob})
    shopt -u nullglob
    if ((${#matches[@]} > 0)); then
      python -m pip install --no-index "${matches[0]}"
      return 0
    fi
  done
  return 1
}

ensure_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

if [[ ! -x "${SEED_PYTHON}" ]]; then
  echo "Missing seed python: ${SEED_PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${ASCEND_TOOLKIT_ENV}" ]]; then
  echo "Missing Ascend toolkit env: ${ASCEND_TOOLKIT_ENV}" >&2
  exit 1
fi

if [[ ! -f "${ASCEND_ATB_ENV}" ]]; then
  echo "Missing Ascend ATB env: ${ASCEND_ATB_ENV}" >&2
  exit 1
fi

TRANSFORMERS_TARBALL="$(pick_first_existing \
  "${TRANSFORMERS_TARBALL}" \
  "${BUNDLE_DIST_DIR}/transformers-cc7ab9be.tar.gz" \
  "${SHARED_DIST_DIR}/transformers-cc7ab9be.tar.gz")"
VLLM_WHEEL_GLOB="$(pick_first_glob \
  "${VLLM_WHEEL_GLOB}" \
  "${BUNDLE_DIST_DIR}/vllm-0.18.0-*.whl" \
  "${SHARED_DIST_DIR}/vllm-0.18.0-*.whl")"
VLLM_ASCEND_TARBALL="$(pick_first_existing \
  "${VLLM_ASCEND_TARBALL}" \
  "${BUNDLE_DIST_DIR}/vllm-ascend-54879467.tar.gz" \
  "${SHARED_DIST_DIR}/vllm-ascend-54879467.tar.gz")"
CATLASS_TARBALL="$(pick_first_existing \
  "${CATLASS_TARBALL}" \
  "${BUNDLE_DIST_DIR}/catlass-src.tar.gz" \
  "${SHARED_DIST_DIR}/catlass-src.tar.gz")"

ensure_file "${TRANSFORMERS_TARBALL}"
ensure_file "${VLLM_WHEEL_GLOB}"
ensure_file "${VLLM_ASCEND_TARBALL}"
ensure_file "${CATLASS_TARBALL}"

mkdir -p "$(dirname "${TARGET_ENV_PATH}")"
"${SEED_PYTHON}" -m venv "${TARGET_ENV_PATH}"
# shellcheck disable=SC1090
source "${TARGET_ENV_PATH}/bin/activate"

python -m pip install --upgrade pip wheel "${SETUPTOOLS_SPEC}"

if ! maybe_install_wheel_from_bundle torch \
  "${BUNDLE_PYTHON_DIR}/torch-2.9.0-*.whl" \
  "${BUNDLE_PYTHON_DIR}/torch-2.9.0+cpu-*.whl"; then
  python -m pip install -i "${PIP_INDEX_URL}" --default-timeout="${PIP_DEFAULT_TIMEOUT}" "${TORCH_SPEC}"
fi

if ! maybe_install_wheel_from_bundle torch_npu \
  "${BUNDLE_PYTHON_DIR}/torch_npu-2.9.0-*.whl" \
  "${BUNDLE_PYTHON_DIR}/torch-npu-2.9.0-*.whl"; then
  python -m pip install -i "${PIP_INDEX_URL}" --default-timeout="${PIP_DEFAULT_TIMEOUT}" "${TORCH_NPU_SPEC}"
fi

if ! maybe_install_wheel_from_bundle numpy "${BUNDLE_PYTHON_DIR}/numpy-1.26.4-*.whl"; then
  python -m pip install -i "${PIP_INDEX_URL}" --default-timeout="${PIP_DEFAULT_TIMEOUT}" "${NUMPY_SPEC}"
fi

if ! maybe_install_wheel_from_bundle PyYAML "${BUNDLE_PYTHON_DIR}/PyYAML-6.0.3-*.whl"; then
  python -m pip install -i "${PIP_INDEX_URL}" --default-timeout="${PIP_DEFAULT_TIMEOUT}" "${PYYAML_SPEC}"
fi

if ! maybe_install_wheel_from_bundle torchvision "${BUNDLE_PYTHON_DIR}/torchvision-0.24.0-*.whl"; then
  python -m pip install -i "${PIP_INDEX_URL}" --default-timeout="${PIP_DEFAULT_TIMEOUT}" --no-deps "${TORCHVISION_SPEC}"
else
  :
fi

_verl_restore_errexit=0
_verl_restore_nounset=0
_verl_restore_pipefail=0
if [[ $- == *e* ]]; then
  _verl_restore_errexit=1
  set +e
fi
if [[ $- == *u* ]]; then
  _verl_restore_nounset=1
  set +u
fi
if set -o | grep -q '^pipefail[[:space:]]\+on$'; then
  _verl_restore_pipefail=1
  set +o pipefail
fi
# shellcheck disable=SC1090
source "${ASCEND_TOOLKIT_ENV}"
# shellcheck disable=SC1090
source "${ASCEND_ATB_ENV}"
if [[ "${_verl_restore_pipefail}" == "1" ]]; then
  set -o pipefail
fi
if [[ "${_verl_restore_nounset}" == "1" ]]; then
  set -u
fi
if [[ "${_verl_restore_errexit}" == "1" ]]; then
  set -e
fi
unset _verl_restore_errexit _verl_restore_nounset _verl_restore_pipefail

python - <<'PY'
import torch, torch_npu
print("torch =", torch.__version__)
print("torch_npu =", torch_npu.__version__)
print("npu available =", torch.npu.is_available())
print("npu count =", torch.npu.device_count())
PY

python -m pip install -i "${PIP_INDEX_URL}" --default-timeout="${PIP_DEFAULT_TIMEOUT}" -r "${ROOT_DIR}/requirements-npu.txt"
python -m pip install -e "${ROOT_DIR}" --no-deps
python -m pip install setuptools-scm vcs-versioning
python -m pip install -i "${PIP_INDEX_URL}" --default-timeout="${PIP_DEFAULT_TIMEOUT}" "${TRANSFORMERS_TARBALL}" "${VLLM_WHEEL_GLOB}"

rm -rf "${WORK_ROOT}"
mkdir -p "${VLLM_ASCEND_SRC_DIR}" "${CATLASS_SRC_DIR}" "${HELPER_BIN_DIR}"
tar -xzf "${VLLM_ASCEND_TARBALL}" -C "${VLLM_ASCEND_SRC_DIR}" --strip-components=1
tar -xzf "${CATLASS_TARBALL}" -C "${CATLASS_SRC_DIR}" --strip-components=1

python "${ROOT_DIR}/scripts/ascend/prepare_vllm_ascend_source.py" \
  --source-dir "${VLLM_ASCEND_SRC_DIR}" \
  --catlass-source-dir "${CATLASS_SRC_DIR}" \
  --helper-bin-dir "${HELPER_BIN_DIR}" \
  --helper-python3 /usr/bin/python3.9 \
  --helper-llvm-objdump /usr/local/Ascend/cann-8.5.0/tools/ccec_compiler/bin/llvm-objdump

export PATH="${HELPER_BIN_DIR}:$PATH"
export SOC_VERSION="${SOC_VERSION:-ascend910b4}"
python -m pip install -v --no-build-isolation --no-deps "${VLLM_ASCEND_SRC_DIR}"

source "${ROOT_DIR}/scripts/ascend/env.qwen35_npu.sh"
python "${ROOT_DIR}/scripts/ascend/check_qwen35_npu_env.py"

python - <<'PY'
import torch, torch_npu, transformers, vllm, vllm_ascend, ray, tensordict

print("torch =", torch.__version__)
print("torch_npu =", torch_npu.__version__)
print("transformers =", transformers.__version__)
print("vllm =", vllm.__version__)
print("vllm_ascend =", getattr(vllm_ascend, "__version__", "installed"))
print("ray =", ray.__version__)
print("tensordict =", tensordict.__version__)
print("npu available =", torch.npu.is_available())
print("npu count =", torch.npu.device_count())
PY

echo
echo "Bootstrap finished."
echo "Target env: ${TARGET_ENV_PATH}"
echo "Activate with:"
echo "  source ${TARGET_ENV_PATH}/bin/activate"
