#!/usr/bin/env bash
# Multinode Qwen3.5 + Ray launcher for the Huawei-provided XPoints container.
set -euo pipefail

REMOTE_SSH=""
BOOTSTRAP_BASENAME="bootstrap_remote_qwen35_xpoints_container.sh"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3.5-xpoints}"
INSPECT_FORMAT='{{.Name}}|{{.State.Running}}'
DOCKER_BIN="${DOCKER_BIN:-docker}"
SUDO_BIN="${SUDO_BIN:-sudo}"
XPOINTS_ROOT="${XPOINTS_ROOT:-/shared/zmz/code2/XPoints}"
VLLM_ROOT="${VLLM_ROOT:-/vllm-workspace/vllm}"
VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT:-/vllm-workspace/vllm-ascend}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${XPOINTS_ROOT}/examples/grpo_trainer/run_qwen3_5_4b_vllm_fsdp_npu_container_clean.sh}"
LOG_DIR="${LOG_DIR:-${XPOINTS_ROOT}/logs}"
HEAD_RAY_PORT="${HEAD_RAY_PORT:-6379}"
HEAD_DASHBOARD_PORT="${HEAD_DASHBOARD_PORT:-8265}"
NUM_NODES="${NUM_NODES:-2}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
NUM_NPUS_PER_NODE="${NUM_NPUS_PER_NODE:-${NUM_GPUS_PER_NODE}}"
SOCKET_IFNAME="${SOCKET_IFNAME:-}"
LOCAL_SOCKET_IFNAME="${LOCAL_SOCKET_IFNAME:-}"
REMOTE_SOCKET_IFNAME="${REMOTE_SOCKET_IFNAME:-}"

FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-True}"
ENABLE_SLEEP_MODE="${ENABLE_SLEEP_MODE:-False}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-False}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"

usage() {
  echo "Usage: $0 --remote-ssh <user@host> [training overrides ...]" >&2
  exit 1
}

resolve_bootstrap_helper() {
  local candidate=""
  local git_common_dir=""
  local main_checkout_root=""
  local -a candidates=(
    "${REPO_ROOT}/scripts/ascend/${BOOTSTRAP_BASENAME}"
  )

  if git_common_dir="$(git -C "${REPO_ROOT}" rev-parse --git-common-dir 2>/dev/null)"; then
    main_checkout_root="$(cd "${git_common_dir}/.." && pwd)"
    candidates+=("${main_checkout_root}/scripts/ascend/${BOOTSTRAP_BASENAME}")
  fi

  if [[ "$(basename "$(dirname "${REPO_ROOT}")")" == ".worktrees" ]]; then
    main_checkout_root="$(cd "${REPO_ROOT}/../.." && pwd)"
    candidates+=("${main_checkout_root}/scripts/ascend/${BOOTSTRAP_BASENAME}")
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

is_exact_missing_container_error() {
  local probe_output="$1"
  [[ "${probe_output}" == *"No such container: ${CONTAINER_NAME}"* ]] || \
    [[ "${probe_output}" == *"No such object: ${CONTAINER_NAME}"* ]]
}

parse_container_probe_output() {
  local location="$1"
  local probe_output="$2"
  local normalized_output=""
  local container_name=""
  local running_state=""

  normalized_output="$(printf '%s\n' "${probe_output}" | awk 'NF { line=$0 } END { print line }')"

  IFS='|' read -r container_name running_state <<< "${normalized_output}"
  running_state="${running_state//[$'\r\n ']}"

  if [[ "${container_name}" != "/${CONTAINER_NAME}" ]]; then
    echo "error: unexpected ${location} probe result for '${CONTAINER_NAME}': ${probe_output}" >&2
    return 1
  fi

  case "${running_state}" in
    true)
      printf 'running\n'
      ;;
    false)
      printf 'not_running\n'
      ;;
    *)
      echo "error: unexpected ${location} probe result for '${CONTAINER_NAME}': ${probe_output}" >&2
      return 1
      ;;
  esac
}

probe_local_container_state() {
  local probe_output=""
  if ! probe_output="$(
    "${SUDO_BIN}" -n "${DOCKER_BIN}" container inspect --format "${INSPECT_FORMAT}" "${CONTAINER_NAME}" 2>&1
  )"; then
    if is_exact_missing_container_error "${probe_output}"; then
      printf 'not_running\n'
      return 0
    fi

    echo "error: local docker probe failed for '${CONTAINER_NAME}': ${probe_output}" >&2
    return 1
  fi

  parse_container_probe_output "local docker" "${probe_output}"
}

probe_remote_container_state() {
  local probe_output=""
  if ! probe_output="$(
    ssh -o BatchMode=yes \
      "$REMOTE_SSH" "${SUDO_BIN} -n ${DOCKER_BIN} container inspect --format \"${INSPECT_FORMAT}\" '${CONTAINER_NAME}'" 2>&1
  )"; then
    if is_exact_missing_container_error "${probe_output}"; then
      printf 'not_running\n'
      return 0
    fi

    echo "error: remote container probe failed for '${CONTAINER_NAME}' on '${REMOTE_SSH}': ${probe_output}" >&2
    return 1
  fi

  parse_container_probe_output "remote" "${probe_output}"
}

container_bash_local() {
  local script="$1"
  "${SUDO_BIN}" -n "${DOCKER_BIN}" exec "${CONTAINER_NAME}" bash -lc "${script}"
}

container_bash_remote() {
  local script="$1"
  local remote_cmd=""
  remote_cmd="$(printf "%q -n %q exec %q bash -lc %q" "${SUDO_BIN}" "${DOCKER_BIN}" "${CONTAINER_NAME}" "${script}")"
  ssh -o BatchMode=yes "${REMOTE_SSH}" "${remote_cmd}"
}

check_local_container_path() {
  local path="$1"
  container_bash_local "test -e '$path'"
}

check_remote_container_path() {
  local path="$1"
  container_bash_remote "test -e '$path'"
}

detect_head_ip() {
  local remote_host="$1"
  python3 - "$remote_host" <<'PY'
import socket
import sys

remote_host = sys.argv[1]
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect((remote_host, 22))
    print(s.getsockname()[0])
finally:
    s.close()
PY
}

detect_remote_node_ip() {
  local remote_host="$1"
  python3 - "$remote_host" <<'PY'
import socket
import sys

print(socket.gethostbyname(sys.argv[1]))
PY
}

detect_container_socket_ifname_local() {
  container_bash_local "
set -euo pipefail
iface=\"\$(awk '\$2 == \"00000000\" && \$1 != \"lo\" { print \$1; exit }' /proc/net/route)\"
if [[ -z \"\${iface:-}\" ]]; then
  iface=\"\$(ls /sys/class/net | grep -Ev '^(lo|docker.*|veth.*|virbr.*)$' | head -n 1)\"
fi
if [[ -z \"\${iface:-}\" ]]; then
  echo 'error: failed to determine local SOCKET_IFNAME inside container' >&2
  exit 1
fi
printf '%s\n' \"\$iface\"
"
}

detect_container_socket_ifname_remote() {
  container_bash_remote "
set -euo pipefail
iface=\"\$(awk '\$2 == \"00000000\" && \$1 != \"lo\" { print \$1; exit }' /proc/net/route)\"
if [[ -z \"\${iface:-}\" ]]; then
  iface=\"\$(ls /sys/class/net | grep -Ev '^(lo|docker.*|veth.*|virbr.*)$' | head -n 1)\"
fi
if [[ -z \"\${iface:-}\" ]]; then
  echo 'error: failed to determine remote SOCKET_IFNAME inside container' >&2
  exit 1
fi
printf '%s\n' \"\$iface\"
"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-ssh)
      REMOTE_SSH="${2:-}"
      shift 2
      ;;
    -h | --help)
      usage
      ;;
    *)
      break
      ;;
  esac
done

if [[ -z "$REMOTE_SSH" ]]; then
  echo "error: --remote-ssh is required" >&2
  usage
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- Prerequisites: local head container is expected to be running in docker ---
local_container_state="$(probe_local_container_state)" || exit 1
if [[ "${local_container_state}" != "running" ]]; then
  echo "error: local head container '${CONTAINER_NAME}' is not running" >&2
  exit 1
fi

# --- Remote: bootstrap only when the exact worker container is missing or
# stopped. Probe failures must fail closed before later stages.
remote_container_state="$(probe_remote_container_state)" || exit 1
if [[ "${remote_container_state}" != "running" ]]; then
  BOOTSTRAP="$(resolve_bootstrap_helper)" || {
    echo "error: could not find ${BOOTSTRAP_BASENAME} in worktree or main checkout" >&2
    exit 1
  }

  remote_user=""
  remote_host="$REMOTE_SSH"
  if [[ "$REMOTE_SSH" == *"@"* ]]; then
    remote_user="${REMOTE_SSH%@*}"
    remote_host="${REMOTE_SSH#*@}"
  fi

  if [[ -n "$remote_user" ]]; then
    REMOTE_HOST="$remote_host" REMOTE_USER="$remote_user" bash "$BOOTSTRAP"
  else
    REMOTE_HOST="$remote_host" bash "$BOOTSTRAP"
  fi
fi

mkdir -p "${LOG_DIR}"

if ! check_local_container_path "${XPOINTS_ROOT}/verl"; then
  echo "error: local container cannot see XPoints verl at: ${XPOINTS_ROOT}/verl" >&2
  exit 1
fi

if ! check_remote_container_path "${XPOINTS_ROOT}/verl"; then
  echo "error: remote container cannot see XPoints verl at: ${XPOINTS_ROOT}/verl" >&2
  exit 1
fi

if ! check_local_container_path "${TRAIN_SCRIPT}"; then
  echo "error: local container cannot see train script at: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

if ! check_remote_container_path "${TRAIN_SCRIPT}"; then
  echo "error: remote container cannot see train script at: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

remote_host="${REMOTE_SSH##*@}"
HEAD_IP="${HEAD_IP:-$(detect_head_ip "${remote_host}")}"
REMOTE_NODE_IP="${REMOTE_NODE_IP:-$(detect_remote_node_ip "${remote_host}")}"

if [[ -z "${HEAD_IP}" ]]; then
  echo "error: failed to determine head IP for remote host '${remote_host}'" >&2
  exit 1
fi

if [[ -z "${REMOTE_NODE_IP}" ]]; then
  echo "error: failed to determine remote node IP for host '${remote_host}'" >&2
  exit 1
fi

if [[ -n "${SOCKET_IFNAME}" && -z "${LOCAL_SOCKET_IFNAME}" ]]; then
  LOCAL_SOCKET_IFNAME="${SOCKET_IFNAME}"
fi

if [[ -z "${LOCAL_SOCKET_IFNAME}" ]]; then
  LOCAL_SOCKET_IFNAME="$(detect_container_socket_ifname_local)"
fi

if [[ -z "${REMOTE_SOCKET_IFNAME}" ]]; then
  REMOTE_SOCKET_IFNAME="$(detect_container_socket_ifname_remote)"
fi

session_name="qwen35_multinode_$(date +%Y%m%d_%H%M%S)"
outer_log="${LOG_DIR}/${session_name}.outer.log"

build_network_env_cmd() {
  local socket_ifname="$1"
  cat <<EOF
export SOCKET_IFNAME='${socket_ifname}'
export GLOO_SOCKET_IFNAME=\"\${GLOO_SOCKET_IFNAME:-\${SOCKET_IFNAME}}\"
export HCCL_SOCKET_IFNAME=\"\${HCCL_SOCKET_IFNAME:-\${SOCKET_IFNAME}}\"
export NCCL_SOCKET_IFNAME=\"\${NCCL_SOCKET_IFNAME:-\${SOCKET_IFNAME}}\"
export TP_SOCKET_IFNAME=\"\${TP_SOCKET_IFNAME:-\${SOCKET_IFNAME}}\"
EOF
}

local_network_env_cmd="$(build_network_env_cmd "${LOCAL_SOCKET_IFNAME}")"
remote_network_env_cmd="$(build_network_env_cmd "${REMOTE_SOCKET_IFNAME}")"
socket_ifname_map="${HEAD_IP}=${LOCAL_SOCKET_IFNAME},${REMOTE_NODE_IP}=${REMOTE_SOCKET_IFNAME}"

local_ray_head_cmd="
set -euo pipefail
${local_network_env_cmd}
ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address='${HEAD_IP}' --port='${HEAD_RAY_PORT}' --dashboard-host='0.0.0.0' --dashboard-port='${HEAD_DASHBOARD_PORT}' --num-gpus='${NUM_GPUS_PER_NODE}' --disable-usage-stats
"

remote_ray_worker_cmd="
set -euo pipefail
${remote_network_env_cmd}
ray stop --force >/dev/null 2>&1 || true
ray start --address='${HEAD_IP}:${HEAD_RAY_PORT}' --num-gpus='${NUM_GPUS_PER_NODE}' --resources='{\"NPU\": ${NUM_NPUS_PER_NODE}}' --disable-usage-stats
"

container_bash_local "${local_ray_head_cmd}"
container_bash_remote "${remote_ray_worker_cmd}"

train_cmd="
set -euo pipefail
cd '${XPOINTS_ROOT}'
${local_network_env_cmd}
export VERL_ROOT='${XPOINTS_ROOT}'
export PYTHONPATH='${VLLM_ROOT}:${VLLM_ASCEND_ROOT}:'\"\${PYTHONPATH:-}\"
export RAY_ADDRESS='auto'
export VERL_SOCKET_IFNAME_MAP='${socket_ifname_map}'
export FREEZE_VISION_TOWER='${FREEZE_VISION_TOWER}'
export ENABLE_SLEEP_MODE='${ENABLE_SLEEP_MODE}'
export FREE_CACHE_ENGINE='${FREE_CACHE_ENGINE}'
export USE_REMOVE_PADDING='${USE_REMOVE_PADDING}'
unset PYTORCH_NPU_ALLOC_CONF
unset PYTORCH_ALLOC_CONF
bash '${TRAIN_SCRIPT}' trainer.nnodes='${NUM_NODES}' trainer.n_gpus_per_node='${NUM_GPUS_PER_NODE}' \"\$@\"
"

echo "Multinode launcher"
echo "  Remote SSH: ${REMOTE_SSH}"
echo "  Container: ${CONTAINER_NAME}"
echo "  Head IP: ${HEAD_IP}"
echo "  Local socket ifname: ${LOCAL_SOCKET_IFNAME}"
echo "  Remote node IP: ${REMOTE_NODE_IP}"
echo "  Remote socket ifname: ${REMOTE_SOCKET_IFNAME}"
echo "  Ray head: ${HEAD_IP}:${HEAD_RAY_PORT}"
echo "  Train script: ${TRAIN_SCRIPT}"
echo "  Outer log: ${outer_log}"

container_bash_local "${train_cmd}" "$@" 2>&1 | tee "${outer_log}"
