# Qwen3.5 Ascend Runbook

## Goal

Bring up `Qwen3.5 + FSDP + GRPO` on Huawei Ascend NPU with the upstream `verl` baseline already merged in [PR #5682](https://github.com/verl-project/verl/pull/5682).

Baseline frozen in this repo:

- Upstream repo: `verl-project/verl`
- Frozen commit: `4045d67063052dcb800c918c107b8d5a87046006`
- Why this commit: it is the merge commit for the Qwen3.5 FSDP GRPO support PR

## Current status

The lab currently has two distinct states:

- final target state:
  - `transformers@cc7ab9be`
  - `vllm==0.18.0`
  - `vllm-ascend@54879467`
  - `torch==2.10`
  - `triton==3.6`
- current fallback state:
  - shared env under `/shared/envs/qwen35`
  - `Python 3.10.20`
  - `torch 2.8.0`
  - `torch_npu 2.8.0.post2`
  - `vllm 0.18.0`
  - `vllm_ascend 0.0.0`

Important:

- the fallback state is only the last known importable debug environment
- it is not the PR `#5682` validation matrix
- the real blocker is still the missing `torch 2.10`-compatible `torch_npu` package, wheel bundle, or prebuilt image for this cluster
- as of `2026-04-09`, the public `torch-npu` README and package index now show a documented `torch 2.9.0` + `torch-npu 2.9.0` path for `CANN 8.5.0`; this does not replace the final `2.10` target matrix, but it is now the highest-priority intermediate validation candidate

## Current NPU reference baseline

For current Ascend bring-up, treat the public `2.8.x` line as the usable reference baseline:

- `CANN 8.5.0.B160`
- `torch_npu 2.8.0`
- `transformers@8e26f7e`

That public progress report came from issue `#5441`, which is consistent with the cluster behavior already seen in this lab: the public NPU line is still centered around `2.8.x`, while the PR `#5682` `2.10` matrix remains a target state rather than a proven public drop-in baseline.

Additional current note:

- the newer public `torch-npu` package metadata now exposes `2.9.0`
- the current `torch-npu` README also documents:
  - `CANN 8.5.0`
  - `torch 2.9.0`
  - `torch-npu 2.9.0`
  - branch `v2.9.0-7.3.0`
- for this lab, that makes `2.9.0` the preferred next isolated validation path before any attempt to force the final `2.10` matrix

## Recommended bring-up path right now

1. On each host, prepare the host-local Ascend runtime first:

```bash
cp scripts/ascend/env.qwen35_host.sh.example scripts/ascend/env.qwen35_host.sh
# edit scripts/ascend/env.qwen35_host.sh for the current host
source scripts/ascend/env.qwen35_host.sh
```

2. Activate the shared lab environment:

```bash
source scripts/ascend/env.qwen35_shared.sh
python scripts/ascend/check_qwen35_npu_env.py
```

The shared activation script should source the host Ascend runtime automatically when those `set_env.sh` files exist. If imports still fail, confirm both `/usr/local/Ascend/ascend-toolkit/set_env.sh` and `/usr/local/Ascend/nnal/atb/set_env.sh` are present and readable on that host.

3. Re-run the shared-env import check on both `172.20.117.36` and `172.20.117.37` and record the output before changing anything else.

Latest known fallback import status:

- `.36`: `torch`, `torch_npu`, `transformers`, `vllm`, and `vllm_ascend` all import from `/shared/envs/qwen35`
- `.37`: `torch`, `torch_npu`, `transformers`, `vllm`, and `vllm_ascend` all import from `/shared/envs/qwen35`
- for the latest `.36` agent handoff, including the validated isolated `torch 2.9.0` + `torch-npu 2.9.0` path and staged wheel locations, see `docs/ascend_qwen35_lab/HANDOFF_20260409.md`
- for the full fallback debugging timeline from shared-env recovery through the latest single-node smoke failure, see `docs/ascend_qwen35_lab/FALLBACK_DEBUG_TIMELINE_20260408.md`

4. If the cluster still only exposes only the old `2.8.x` line in practice, stop here and do not treat the fallback env as final validation.

At that point the next step is not another blind reinstall and not a default smoke run. The next step is to obtain one of:

- a `torch 2.10`-compatible `torch_npu` wheel
- an official image with the matching stack already built
- a Huawei-provided Python environment that reproduces the PR matrix

If the host can now actually install public `torch 2.9.0` + `torch-npu 2.9.0`, prefer validating that pair in a fresh isolated env before revisiting fallback `2.8.x` smoke retries.

5. Only after that missing runtime piece exists, rebuild the shared env and then install the pinned user-space stack.

If the user explicitly wants to continue fallback-line debugging before that missing runtime piece exists, prepare the extracted `vllm-ascend` source locally first and then sync it through git-backed repo changes:

```bash
python3 scripts/ascend/prepare_vllm_ascend_source.py \
  --source-dir /path/to/extracted/vllm-ascend-54879467-src \
  --helper-bin-dir /tmp/vllm-ascend-helper-bin \
  --helper-python3 /usr/bin/python3.9 \
  --helper-llvm-objdump /usr/local/Ascend/cann-8.5.0/tools/ccec_compiler/bin/llvm-objdump
```

Use `--allow-torch-fallback-debug` only for explicit fallback debugging. Do not treat that switch as validation of the final target matrix.

## Fallback debug retry template

When the user explicitly wants one more fallback-line diagnostic pass on `.36`, keep the repo as the source of truth and use a repeatable command sequence instead of re-editing `/tmp` by hand:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /shared/tools/miniforge3/etc/profile.d/conda.sh
conda activate /shared/envs/qwen35

rm -rf /tmp/vllm-ascend-54879467-src /tmp/vllm-ascend-helper-bin
mkdir -p /tmp/vllm-ascend-54879467-src
tar -xzf /shared/dist/vllm-ascend-54879467.tar.gz -C /tmp/vllm-ascend-54879467-src --strip-components=1

python3 scripts/ascend/prepare_vllm_ascend_source.py \
  --source-dir /tmp/vllm-ascend-54879467-src \
  --helper-bin-dir /tmp/vllm-ascend-helper-bin \
  --helper-python3 /usr/bin/python3.9 \
  --helper-llvm-objdump /usr/local/Ascend/cann-8.5.0/tools/ccec_compiler/bin/llvm-objdump

export PATH=/tmp/vllm-ascend-helper-bin:$PATH
/shared/envs/qwen35/bin/python -m pip install -v --no-build-isolation --no-deps /tmp/vllm-ascend-54879467-src
```

If the goal is only to probe how far the fallback `torch 2.8.x` line can go, rerun the prepare step with `--allow-torch-fallback-debug`. Keep that as a diagnostic-only branch, not a final validation path.

That diagnostic path has already been proven far enough to install `vllm_ascend` on `.36`. The next decision is no longer "can the package be built?" but "do we run a smoke on fallback, or stop and wait for the proper `torch 2.10` stack?"

## When to use the bootstrap script

Use `scripts/ascend/bootstrap_qwen35_npu_env.sh` only after both of these are true:

- the base runtime already has a compatible `torch` + `torch_npu` pair
- the host can either reach the required sources, or you have mirrored the pinned packages locally

Remember:

- the bootstrap script does not install `CANN`, `torch`, or `torch_npu`
- the cluster has already shown unreliable GitHub access for `pip install git+...`
- for this lab, local tarballs under `/shared/dist` are safer than assuming remote GitHub access
- `scripts/ascend/prepare_vllm_ascend_source.py` is the local reproducible way to carry the currently known `vllm-ascend` source patches forward; do not rely on ad-hoc edits under `/tmp` as the source of truth
- `/home/zmz/bootstrap_bundle/dist` on `.36` now collects the current hard packages for reuse on a new machine

## Smoke test prerequisites

Do not start the smoke script until:

- both hosts can import `torch`, `torch_npu`, and `transformers` from the intended shared env
- the env is no longer in the fallback `torch 2.8` debug-only state
- model and dataset paths are exported explicitly

Example exports:

```bash
export MODEL_PATH=/path/to/Qwen3.5-27B
export TRAIN_FILE=/path/to/train.parquet
export TEST_FILE=/path/to/test.parquet
```

Then start the safer smoke test:

```bash
bash scripts/ascend/run_qwen35_27b_npu_smoke.sh
```

## Why the smoke script differs from upstream example

The upstream NPU example is preserved in `examples/grpo_trainer/run_qwen3_5_27b_vllm_fsdp_npu.sh`, but this lab script changes the first-run defaults to reduce bring-up risk:

- `use_remove_padding=False`
- explicit `Qwen3_5DecoderLayer` wrap policy
- `ulysses_sequence_parallel_size=1`
- smaller rollout and batch defaults
- one epoch by default

These are not final performance settings. They are meant to answer one question first: can the stack start cleanly and enter training on your NPU cluster.

## Current 4B retry shape after the `convolution_backward` segfault

For the current `Qwen3.5-4B` geo3k retry on the isolated `torch 2.9` line, prefer a text-only smoke shape that minimizes the chance of re-hitting known worker-count and logger issues before we learn whether freezing the vision tower avoids the NPU backward crash:

```bash
FREEZE_VISION_TOWER=True \
TRAIN_BATCH_SIZE=1 \
PPO_MINI_BATCH_SIZE=1 \
PPO_MICRO_BATCH_SIZE=1 \
ROLLOUT_AGENT_NUM_WORKERS=1 \
REWARD_NUM_WORKERS=1 \
ROLLOUT_MAX_MODEL_LEN=4096 \
GPU_MEM_UTIL=0.2 \
TRAINER_LOGGERS="['console']" \
bash examples/grpo_trainer/run_qwen3_5_4b_vllm_fsdp_npu.sh
```

Notes:

- `FREEZE_VISION_TOWER=True` is the main hypothesis test for the current `aclnnConvolutionBackwardGetWorkspaceSize` segfault.
- `TRAIN_BATCH_SIZE=1` only works cleanly when `ROLLOUT_AGENT_NUM_WORKERS=1`; otherwise the agent loop can still fail earlier in `DataProto.chunk`.
- `TRAINER_LOGGERS="['console']"` avoids taking a hard dependency on `tensorboard` for this smoke.
- Start with `ROLLOUT_MAX_MODEL_LEN=4096`; if rollout needs more context, retry at `8192`.
- If this still segfaults in the same C++ stack, treat that as evidence against the current VL-tagged checkpoint on this NPU stack and escalate with the full crash stack plus package matrix.

## Suggested test sequence

1. Reconfirm the fallback shared env is still importable on both hosts.
2. Resolve the missing `torch 2.10`-compatible `torch_npu` source.
3. Rebuild `/shared/envs/qwen35` on the final PR matrix.
4. Run `scripts/ascend/check_qwen35_npu_env.py` again from both hosts.
5. Run a single-node smoke pass first.
6. Prefer a smaller Qwen3.5 checkpoint for a first launch sanity check if model download or memory pressure is still uncertain.
7. Run the 27B smoke script once the environment itself is stable.
8. Only after that, tune memory and throughput.
9. Only after that, evaluate precision alignment and multi-node scaling.

## Results logging

Copy the template in `results/templates/session_report.md` for each cluster run.
