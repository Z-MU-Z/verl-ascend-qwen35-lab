#!/usr/bin/env bash

# Single-node lab bring-up on `.36/.37` should not enable both intra-PCIE and
# intra-ROCE at once; the fallback timeline captured HCCL init failures until
# ROCE was forced off for this topology.
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-0}"
export HCCL_INTRA_PCIE_ENABLE="${HCCL_INTRA_PCIE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export ACTOR_FA_ENABLE="${ACTOR_FA_ENABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# vLLM worker startup on the isolated `torch 2.9` line may import
# `torch_npu.op_plugin.atb._atb_ops`, which in turn needs both the torch/torch_npu
# shared libraries and the host ATB runtime on `LD_LIBRARY_PATH`.
_verl_prepend_path_once() {
  local _target="$1"
  local _current="${LD_LIBRARY_PATH:-}"
  if [[ -z "${_target}" || ! -d "${_target}" ]]; then
    return 0
  fi
  case ":${_current}:" in
    *":${_target}:"*) ;;
    *) export LD_LIBRARY_PATH="${_target}${_current:+:${_current}}" ;;
  esac
}

_verl_python_prefix="${VIRTUAL_ENV:-${CONDA_PREFIX:-}}"
if [[ -n "${_verl_python_prefix}" ]]; then
  for _verl_torch_lib in \
    "${_verl_python_prefix}"/lib/python*/site-packages/torch/lib \
    "${_verl_python_prefix}"/lib/python*/site-packages/torch_npu/lib; do
    if [[ -d "${_verl_torch_lib}" ]]; then
      _verl_prepend_path_once "${_verl_torch_lib}"
    fi
  done
fi

for _verl_atb_lib in \
  /usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_1/lib \
  /usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib; do
  _verl_prepend_path_once "${_verl_atb_lib}"
done

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
unset _verl_atb_lib
unset _verl_ascend_py_sp
unset _verl_python_prefix
unset _verl_torch_lib
unset -f _verl_prepend_path_once
