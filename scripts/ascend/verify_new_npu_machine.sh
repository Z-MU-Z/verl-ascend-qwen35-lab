#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ASCEND_TOOLKIT_ENV="${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ASCEND_ATB_ENV="${ASCEND_ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
SHARED_ENV_ROOT="${SHARED_ENV_ROOT:-/shared/envs/qwen35}"
PYTHON_BIN="${PYTHON_BIN:-${SHARED_ENV_ROOT}/bin/python}"

if ! command -v npu-smi >/dev/null 2>&1; then
  echo "Missing required command: npu-smi" >&2
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

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ASCEND_TOOLKIT_ENV}"
# shellcheck disable=SC1090
source "${ASCEND_ATB_ENV}"

export TORCH_DEVICE_BACKEND_AUTOLOAD=0

echo "[verify_new_npu_machine]"
echo "repo=${ROOT_DIR}"
echo "python=${PYTHON_BIN}"
echo "ascend_toolkit_env=${ASCEND_TOOLKIT_ENV}"
echo "ascend_atb_env=${ASCEND_ATB_ENV}"

"${PYTHON_BIN}" - <<'PY'
mods = {}
for name in ["torch", "torch_npu", "transformers", "vllm", "vllm_ascend"]:
    try:
        module = __import__(name)
        mods[name] = getattr(module, "__version__", "installed")
    except Exception as exc:
        mods[name] = f"IMPORT_FAIL: {exc}"
print(mods)

failed = {name: value for name, value in mods.items() if str(value).startswith("IMPORT_FAIL:")}
if failed:
    raise SystemExit(f"Import verification failed: {failed}")
PY
