#!/usr/bin/env bash

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_INTRA_PCIE_ENABLE="${HCCL_INTRA_PCIE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export ACTOR_FA_ENABLE="${ACTOR_FA_ENABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# vllm_ascend (e.g. camem allocator) imports CANN pyACL as `acl`. If callers set
# `PYTHONPATH=/path/to/verl` after sourcing ascend-toolkit/set_env.sh, they wipe
# the CANN site-packages entry and vLLM EngineCore worker procs die with
# ModuleNotFoundError: No module named 'acl'. Prepend the usual CANN path when
# it is missing.
_verl_ascend_py_sp=""
if [[ -n "${ASCEND_HOME:-}" && -d "${ASCEND_HOME}/python/site-packages" ]]; then
  _verl_ascend_py_sp="${ASCEND_HOME}/python/site-packages"
elif [[ -d "/usr/local/Ascend/cann-8.5.0/python/site-packages" ]]; then
  _verl_ascend_py_sp="/usr/local/Ascend/cann-8.5.0/python/site-packages"
fi
if [[ -n "${_verl_ascend_py_sp}" ]]; then
  case ":${PYTHONPATH:-}:" in
    *":${_verl_ascend_py_sp}:"*) ;;
    *) export PYTHONPATH="${_verl_ascend_py_sp}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
fi
unset _verl_ascend_py_sp
