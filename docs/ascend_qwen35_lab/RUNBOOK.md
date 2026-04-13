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
ENABLE_SLEEP_MODE=False \
ROLLOUT_MAX_MODEL_LEN=4096 \
GPU_MEM_UTIL=0.2 \
TRAINER_LOGGERS="['console']" \
bash examples/grpo_trainer/run_qwen3_5_4b_vllm_fsdp_npu.sh
```

Notes:

- `FREEZE_VISION_TOWER=True` is the main hypothesis test for the current `aclnnConvolutionBackwardGetWorkspaceSize` segfault.
- `TRAIN_BATCH_SIZE=1` only works cleanly when `ROLLOUT_AGENT_NUM_WORKERS=1`; otherwise the agent loop can still fail earlier in `DataProto.chunk`.
- For the current single-node `.36/.37` fallback line, keep `HCCL_INTRA_PCIE_ENABLE=1` and `HCCL_INTRA_ROCE_ENABLE=0`; enabling both caused early HCCL init failure during ref-model startup.
- Also source `scripts/ascend/env.qwen35_npu.sh` in the isolated `torch 2.9` env before the smoke retry; it now restores the runtime `LD_LIBRARY_PATH` entries needed by `torch_npu.op_plugin.atb._atb_ops` (`site-packages/torch/lib`, `site-packages/torch_npu/lib`, and the host `libatb.so` directories under `/usr/local/Ascend/nnal/atb/...`).
- Keep `ENABLE_SLEEP_MODE=False` on the current isolated `torch 2.9` line; otherwise `vllm_ascend/device_allocator/camem.py` can still enter the pluggable allocator path and fail with `TypeError: 'NoneType' object is not callable` after `vllm_ascend_C` import falls back.
- After disabling sleep mode, the next visible warnings on `.36` moved to `vLLMHttpServer`: missing Triton symbol path `triton.language.target_info` and unrecognized Qwen3.5 `rope_parameters` keys (`mrope_interleaved`, `mrope_section`). Keep the archived log under `/home/zmz/verl/log_archive/qwen35_4b_freezevis_singlecard_nosleepdefault2_20260411_210157.log` when comparing future retries.
- In that archived run, the final failure was deeper than the warnings: `EngineCore` exited with `torch._dynamo.exc.Unsupported: Import failure`, debug context `module_name: vllm_ascend.vllm_ascend_C`, on the path through `vllm_ascend/ops/layernorm.py -> enable_custom_op()`. Treat that import path as the current blocker before retrying larger training shapes.
- After the custom-op fallback/import fixes, the later `.36` archived run `/home/zmz/verl/log_archive/qwen35_4b_freezevis_singlecard_customopoff_20260411_232308.log` failed in a different place: `CheckpointManager.sleep_replicas()` still reached `vLLMHttpServer.sleep()`, and `vllm_ascend/device_allocator/camem.py` rejected `expandable_segments:True` with `AssertionError: Expandable segments are not compatible with memory pool.` For this NPU line, `enable_sleep_mode=False` should be treated as "do not enter rollout sleep at all."
- After patching `sleep()` to skip that path (`7d312155`), the next `.36` rerun `/home/zmz/verl/log_archive/qwen35_4b_freezevis_singlecard_nosleepskip_20260413_101327.log` progressed further into rollout resume, but then failed in `wake_up()`: `resume(tags=["weights"]) -> vLLMHttpServer.wake_up() -> vllm_ascend/worker/worker.py -> CaMemAllocator`, with the same `AssertionError: Expandable segments are not compatible with memory pool.` On this fallback line, `enable_sleep_mode=False` should therefore imply skipping both `sleep()` and `wake_up()`.
- `TRAINER_LOGGERS="['console']"` avoids taking a hard dependency on `tensorboard` for this smoke.
- Start with `ROLLOUT_MAX_MODEL_LEN=4096`; if rollout needs more context, retry at `8192`.
- If this still segfaults in the same C++ stack, treat that as evidence against the current VL-tagged checkpoint on this NPU stack and escalate with the full crash stack plus package matrix.

## SSH and remote lab paths (for the next agent)

### Workflow (see also repo `AGENTS.md`)

1. Edit and commit in **this** repository on your laptop (source of truth).
2. `git push` from the laptop.
3. SSH to the lab host and `cd` to the checkout below, then `git pull --ff-only`.
4. Only then run training, smoke tests, or install packages into a **non-shared** venv if policy requires it.

Do not make the remote machine the authoritative copy of code unless the user explicitly asks for a temporary hotfix (then mirror back to git immediately).

### Hosts and SSH

- Lab IPs referenced in docs: **`172.20.117.36`** (`.36`) and **`172.20.117.37`** (`.37`).
- Many developers use an **`~/.ssh/config` Host alias** (e.g. **`huawei36`**) pointing at `.36`. Verify with your local config; there is no guarantee the alias name is the same on every machine.
- On this lab laptop, confirmed aliases are **`huawei36`** → `zmz@172.20.117.36` and **`huawei37`** → `zmz@172.20.117.37`.
- **Agent / IDE caveat:** terminal commands may run in an environment that **cannot route** to the lab VLAN. If `ssh …` returns **`Connection timed out`**, the failure is often **the agent runner’s network**, not broken SSH keys on your laptop. Quick probe:  
  `ssh -o BatchMode=yes -o ConnectTimeout=15 huawei36 'echo OK'`

Copy-paste connect + sync:

```bash
ssh huawei36
cd /home/zmz/code/verl-ascend-qwen35-lab
git pull --ff-only
```

### Paths on the remote hosts

| Item | Typical path | Notes |
|------|----------------|------|
| Git checkout | `/shared/zmz/code/verl-ascend-qwen35-lab` **or** `/home/zmz/code/verl-ascend-qwen35-lab` | Use whichever exists; prior logs used `/shared/zmz/...`. |
| Shared fallback env (`2.8` line) | `/shared/envs/qwen35` | Do not mutate for `2.9` experiments per handoff policy. |
| Isolated `2.9` venv (example on `.36`) | `/home/zmz/envs/qwen35-t29-lite` | **Host-local** under `/home/zmz/...` — not visible on `.37` unless recreated. |
| Shared dataset / artifact roots | `/shared` | e.g. data and checkpoints under `/shared/...`; layout varies by job. |
| Model weights (example) | `/shared/weights/Qwen3.5-4B` | Confirm on host. |
| Long-running smoke logs | `/tmp/grpo_*.log` | Or a path you choose for `nohup … > log`. |
| Ascend runtime | `/usr/local/Ascend/ascend-toolkit/set_env.sh` | Also source `/usr/local/Ascend/nnal/atb/set_env.sh` when the stack expects it. |

Shortcuts from `AGENTS.md` (when applicable on cluster accounts): shared code **`/home/zmz/code`**, data **`/home/zmz/verl/data`**, models **`/home/zmz/verl/models`** — reconcile with actual `MODEL_PATH` / `RAY_DATA_HOME` in your shell.

### Doc and script pointers in this repo

- `docs/ascend_qwen35_lab/HANDOFF_20260409_FOR_37.md` — isolated `2.9` env, geo3k smoke pip deps, Hydra knobs (`agent.num_workers`, `max_model_len`, etc.).
- `docs/ascend_qwen35_lab/KNOWN_ISSUES.md` — torchvision / `PYTHONPATH`+`acl` / `convolution_backward` segfault and related issues.
- `scripts/ascend/env.qwen35_npu.sh` — ensures CANN `python/site-packages` stays on `PYTHONPATH` for `vllm_ascend` workers.

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
