#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BOOTSTRAP_UPGRADE_BUILD_TOOLS="${BOOTSTRAP_UPGRADE_BUILD_TOOLS:-1}"

PIP_INSTALL_ARGS=()
if [[ -n "${PIP_INDEX_URL:-}" ]]; then
  PIP_INSTALL_ARGS+=(-i "${PIP_INDEX_URL}")
fi
if [[ -n "${PIP_EXTRA_INDEX_URL:-}" ]]; then
  PIP_INSTALL_ARGS+=(--extra-index-url "${PIP_EXTRA_INDEX_URL}")
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${CREATE_VENV:-0}" == "1" ]]; then
  VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  PYTHON_BIN="${VENV_DIR}/bin/python"
fi

echo "Using Python: ${PYTHON_BIN}"
echo "Repo root: ${ROOT_DIR}"
if ((${#PIP_INSTALL_ARGS[@]} > 0)); then
  echo "Pip install args: ${PIP_INSTALL_ARGS[*]}"
fi

cat <<'EOF'
This bootstrap installs the user-space Python dependencies for the Ascend lab.
It does not install CANN, torch, or torch_npu. Prepare those in the cluster image first.
EOF

if [[ "${BOOTSTRAP_UPGRADE_BUILD_TOOLS}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install "${PIP_INSTALL_ARGS[@]}" --upgrade pip setuptools wheel
else
  echo "Skipping pip/setuptools/wheel upgrade because BOOTSTRAP_UPGRADE_BUILD_TOOLS=${BOOTSTRAP_UPGRADE_BUILD_TOOLS}"
fi

"${PYTHON_BIN}" -m pip install "${PIP_INSTALL_ARGS[@]}" -r "${ROOT_DIR}/requirements-npu.txt"
"${PYTHON_BIN}" -m pip install "${PIP_INSTALL_ARGS[@]}" -e "${ROOT_DIR}" --no-deps
"${PYTHON_BIN}" -m pip install "${PIP_INSTALL_ARGS[@]}" \
  "git+https://github.com/huggingface/transformers.git@cc7ab9be" \
  "vllm==0.18.0" \
  "git+https://github.com/vllm-project/vllm-ascend.git@54879467"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/ascend/check_qwen35_npu_env.py"

echo
echo "Bootstrap finished."
echo "Load recommended env vars with:"
echo "  source ${ROOT_DIR}/scripts/ascend/env.qwen35_npu.sh"
