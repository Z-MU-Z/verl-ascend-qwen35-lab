# Qwen3.5 Shared Environment TODO

> For this lab, follow the PR #5682 bring-up matrix first. Do not optimize or broaden versions before the first stable run.

## Goal

Build a two-host Huawei Ascend environment where:

- host-local NPU runtime prerequisites are aligned on `172.20.117.36` and `172.20.117.37`
- user-space training environment is shared as much as possible from `/shared`
- model, data, checkpoints, logs, and repo code use the same paths on both machines
- first validation follows the Qwen3.5 bring-up assumptions captured in this repo

## Current Findings

- `172.20.117.36` and `172.20.117.37` both expose:
  - `/usr/local/Ascend/ascend-toolkit/set_env.sh`
  - `/usr/local/Ascend/nnal/atb/set_env.sh`
  - `npu-smi`
- Both hosts currently default to `/usr/bin/python3` and Python `3.9.9`.
- `172.20.117.36` currently has only partial system packages in default Python:
  - `torch==2.8.0`
  - `transformers==4.57.6`
- `172.20.117.37` currently has no relevant training packages in default Python.
- Both hosts currently miss the lab-critical runtime pieces in default Python:
  - `torch_npu`
  - `vllm`
  - `ray`
  - `tensordict`
  - `triton_ascend`
- Shared venv has been created at `/shared/envs/qwen35` and is visible from both hosts.
- The shared wrapper script `scripts/ascend/env.qwen35_shared.sh` resolves to the same shared Python path on both hosts:
  - `/shared/envs/qwen35/bin/python`
- Current likely training communication NIC on both hosts is `enp61s0f0`.
- Current host IPs on that NIC are:
  - `172.20.117.36`
  - `172.20.117.37`

## Pitfalls Already Seen

- The default `python3` environments on the two hosts are not equally provisioned.
- The generic Ascend quickstart package versions differ from the PR #5682 bring-up pins, so this lab must prefer the PR-specific matrix first.
- A shared virtualenv under `/shared` is feasible because both hosts use the same system Python path, but that does not replace host-local `CANN` and `torch_npu`.
- `/shared` is shared for user-space assets, but on host `172.20.117.37` it is still backed by local disk. Capacity and ownership must still be checked before long runs.
- Pulling into the shared repo can be blocked by untracked files left behind by earlier `rsync`-based syncs. Compare file hashes first, then remove only the exact conflicting files before `git pull`.
- Shared NFS worktrees can briefly show stale `D` states or miss just-created files until metadata catches up. Re-check before assuming a real deletion.
- Shell quoting for remote `awk` one-liners is easy to break; prefer simpler commands when probing host IP and interface state over SSH.
- The default pip source can stall even on small bootstrap steps like upgrading `pip/setuptools/wheel`. Prefer setting `PIP_INDEX_URL` explicitly for this lab, and be ready to skip build-tool upgrades during bootstrap if the environment already has a working pip.
- The current shared venv was first created from system `python3.9.9`, but local `pyproject.toml` requires `Python >=3.10`. That means `pip install -e .` will always fail until the shared interpreter itself is rebuilt on top of Python 3.10+.

## Source Of Truth

- Upstream baseline: PR `#5682`
- Local repo baseline: `README.md`, `docs/ascend_qwen35_lab/RUNBOOK.md`, `docs/ascend_qwen35_lab/KNOWN_ISSUES.md`
- Required local workflow:
  1. change locally
  2. validate locally
  3. commit and push locally
  4. remote machines run `git pull`
  5. only then continue remote execution

## Shared Vs Host-Local Split

### Shared under `/shared`

- repo checkout path
- Python virtual environment
- model weights
- datasets
- checkpoints
- logs

### Host-local on each machine

- CANN installation
- `/usr/local/Ascend/ascend-toolkit/set_env.sh`
- `/usr/local/Ascend/nnal/atb/set_env.sh`
- `torch_npu`
- network variables such as `MASTER_ADDR`, `HCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`

## TODO

### Phase 1: Lock The Required Environment Matrix

- [ ] Record the PR-first package matrix in one place:
  - `transformers@cc7ab9be`
  - `vllm==0.18.0`
  - `vllm-ascend@54879467`
  - `ulysses_sequence_parallel_size=1`
  - `use_remove_padding=False` for first bring-up
  - explicit `Qwen3_5DecoderLayer` wrap policy
- [ ] Mark the generic Ascend quickstart versions as reference-only for this lab, not the first validation target.

### Phase 2: Align Host-Local Base Prerequisites

- [ ] On `172.20.117.36`, verify:
  - Python version
  - CANN availability
  - `torch_npu`
  - `torch`
  - current shell activation path
- [ ] On `172.20.117.37`, verify the same items.
- [ ] Decide whether both hosts will use:
  - the same system Python
  - or the same host-local Python bootstrap path before entering the shared venv
- [ ] Because local `verl` requires `Python >=3.10`, do not continue with a shared `3.9` virtualenv. First provision one shared Python `3.10+` interpreter path that both hosts can enter.
- [ ] Do not proceed until both machines have compatible host-local NPU prerequisites.

### Phase 3: Standardize Shared Paths

- [ ] Keep repo path fixed at `/home/zmz/code/verl-ascend-qwen35-lab`
- [ ] Keep shared storage roots fixed:
  - `/shared/envs/qwen35`
  - `/shared/weights`
  - `/shared/data`
  - `/shared/ckpts`
  - `/shared/logs`
- [ ] Ensure both machines resolve the same logical shortcuts:
  - `~/code`
  - `~/verl/models`
  - `~/verl/data`

### Phase 4: Build The Shared Python Environment

- [ ] Create the shared virtualenv at `/shared/envs/qwen35`
  - It must be built from Python `3.10+`, not the host default `3.9.9`.
- [ ] Activate it from both machines and confirm `python -V` and `which python` are identical
- [ ] Install user-space dependencies into the shared venv:
  - `requirements-npu.txt`
  - editable repo install
  - pinned `transformers`
  - pinned `vllm`
  - pinned `vllm-ascend`
- [ ] Verify the shared venv from both machines with:
  - `python scripts/ascend/check_qwen35_npu_env.py`

### Phase 5: Add Host-Local Activation Wrapper

- [ ] Add one shared env script for common settings and paths
- [ ] Add one host-local env script for per-machine network settings
- [ ] Shared script should cover:
  - `RAY_DATA_HOME`
  - shared venv activation
  - shared model/data/ckpt/log paths
  - PR-specific safe defaults
- [ ] Host-local script should cover:
  - `MASTER_ADDR`
  - `HCCL_SOCKET_IFNAME`
  - `GLOO_SOCKET_IFNAME`
  - node IP derivation if needed

Current repo helpers:

- Shared wrapper: `scripts/ascend/env.qwen35_shared.sh`
- Host-local template: `scripts/ascend/env.qwen35_host.sh.example`

### Phase 6: Verify Single-Node Bring-Up First

- [ ] Run `scripts/ascend/check_qwen35_npu_env.py` from the shared venv on both machines
- [ ] Run a single-node smoke test first on one host
- [ ] Prefer `Qwen3.5-4B` for the first smoke run after download completes
- [ ] Keep first-run settings conservative:
  - `use_remove_padding=False`
  - `ulysses_sequence_parallel_size=1`
  - low `gpu_memory_utilization`
  - low batch sizes
- [ ] Save logs and exact command lines

### Phase 7: Prepare Two-Host Execution

- [ ] Pick the master host and freeze `MASTER_ADDR`
- [ ] Determine the correct communication NIC on each machine
- [ ] Start Ray head on the master node
- [ ] Start Ray worker on the second node
- [ ] Confirm Ray reports the expected total NPU resources before training

### Phase 8: Run Two-Host Validation

- [ ] Start with a short multi-node smoke pass, not a long training job
- [ ] Verify:
  - both nodes join the Ray cluster
  - both nodes see the shared model/data paths
  - trainer can initialize distributed resources
  - vLLM starts with the pinned stack
- [ ] Only after that, raise batch size, rollout count, or memory utilization

### Phase 9: Document The Stable Procedure

- [ ] Write the final activation commands for both hosts
- [ ] Write the exact start order:
  - source host-local env
  - source shared env
  - activate shared venv
  - verify environment
  - start Ray
  - run training script
- [ ] Record residual caveats:
  - which settings are still experimental
  - which values came from PR discussion
  - which values were validated on this cluster

## Execution Order

1. Phase 2
2. Phase 3
3. Phase 4
4. Phase 5
5. Phase 6
6. Phase 7
7. Phase 8
8. Phase 9

Do not start multi-node training before single-node smoke validation passes.
