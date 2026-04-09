# Qwen3.5 Ascend Fallback Debug Timeline

## Scope

This document records the fallback-line debugging timeline on `huawei36` and related shared-env work over the last few days, from restoring an importable shared environment to pushing the single-node smoke run deeper into `vllm` startup.

This is a diagnostic timeline for the current fallback stack only:

- shared env: `/shared/envs/qwen35`
- `Python 3.10.20`
- `torch 2.8.0`
- `torch_npu 2.8.0.post2`
- `transformers 5.3.0.dev0`
- `vllm 0.18.0`
- `vllm_ascend 0.0.0` built from `vllm-ascend@54879467`

It is not validation of the final PR `#5682` target matrix.

## Bottom Line

As of `2026-04-08`:

- the fallback shared environment is importable on `.36` and `.37`
- `vllm_ascend` is installed into the shared env through a patched fallback build flow
- the single-node smoke on `.36` progressed past:
  - basic import checks
  - Ray startup
  - dataset filtering
  - FSDP worker creation
  - `vllm_ascend` plugin activation
  - `vLLMHttpServer` launch
- the deepest confirmed blocker in the latest smoke run is now:
  - `AttributeError: torch._functorch.config.autograd_cache_normalize_inputs does not exist`
- current interpretation:
  - the fallback `torch 2.8.x` line is now deep enough to expose a more intrinsic `torch` / `vllm` / compile-path mismatch
  - the environment is no longer failing first on missing Python packages, missing shared libraries, or the earlier HCCL setting mistakes

## Timeline

### Stage 0: Restore a coherent fallback environment

Observed state:

- the public Ascend package line available on the cluster stayed on `torch_npu 2.8.x`
- pairing public `torch_npu 2.8.x` with `torch 2.10` was not importable
- the shared environment therefore had to be restored to a coherent fallback stack instead of the final PR matrix

Working fallback baseline recovered:

- `/shared/envs/qwen35`
- `Python 3.10.20`
- `torch 2.8.0`
- `torch_npu 2.8.0.post2`
- `vllm 0.18.0`

Implication:

- this restored debugging continuity, but it did not validate PR `#5682`
- all later smoke findings below should be read as fallback-line diagnostics, not final-stack proof

### Stage 1: `vllm-ascend` source bundle was not self-contained

First active blocker:

- `/shared/dist/vllm-ascend-54879467.tar.gz` was not reliably self-contained for direct fallback installation
- the extracted tree could miss `csrc/third_party/catlass`
- `git submodule update --init --recursive` was not a reliable recovery path in the tarball-based remote build flow

Root cause:

- the tarball path depended on submodule content that was not guaranteed to be present

Action taken:

- recovered the missing `catlass` content
- moved the repeatable recovery path into local repo tooling instead of ad-hoc remote edits

Outcome:

- the original "missing source bundle" problem stopped being the active blocker
- later fallback installation work switched to a reproducible prepared-source path

### Stage 2: `vllm-ascend@54879467` needed source-level patching on the fallback line

After `catlass` was present, the active install chain became a stack of source and toolchain mismatches.

Observed issues:

- hardcoded CANN include roots such as:
  - `include/experiment/platform`
  - `include/experiment/slog`
- actual `.36` host headers under `CANN 8.5.0.B160` were instead located at:
  - `include/aclnn/opdev/platform.h`
  - `include/toolchain/slog.h`
- helper scripts in the custom-op stage behaved differently depending on which `python3` they resolved first
- `setup.py` used bare `python3 -m pip show torch-npu`, which mis-resolved `TORCH_NPU_PATH` when the shell `PATH` had been adjusted
- `CMakeLists.txt` hard-required `PyTorch 2.9.0` while the fallback env remained on `torch 2.8.0`
- a later host-stub stage failed with:
  - `FileNotFoundError: [Errno 2] No such file or directory: 'llvm-objdump'`

Root cause:

- the fallback stack did not match the upstream package and toolchain assumptions encoded in `vllm-ascend@54879467`
- both interpreter lookup and helper-binary lookup had to be made explicit to keep the build deterministic

Action taken:

- captured the known source edits in `scripts/ascend/prepare_vllm_ascend_source.py`
- patched CANN include roots
- patched `setup.py` probing to use the active interpreter
- added explicit helper-bin shims for:
  - `python3`
  - `llvm-objdump`
- added an explicit fallback debug gate for the temporary `torch` version relaxation

Outcome:

- `.36` successfully built and installed `vllm_ascend` into `/shared/envs/qwen35`
- the built package carried `libvllm_ascend_kernels.so`
- the shared env became importable on both `.36` and `.37`

### Stage 3: Start the minimal single-node smoke on `.36`

Smoke intent:

- use `Qwen3.5-4B`
- use single-node `8` visible NPUs on `.36`
- run `1` training step
- keep first-run settings conservative

Reasoning:

- the question at this stage was no longer "can the environment import"
- the question became "how far can the fallback line get into actual training and rollout initialization"

### Stage 4: First remote probing hit shell-quoting breakage

Observed error:

- remote embedded Python checks failed with a shell parse error like:
  - `bad pattern`

Root cause:

- complex Python snippets embedded inside SSH commands were over-fragile under shell quoting

Action taken:

- replaced the brittle embedded block with simpler `python -c` checks

Outcome:

- import and environment probing became reliable enough to continue smoke debugging

### Stage 5: Smoke failed on the first batch-size assertion

Observed error:

- `AssertionError: real_train_batch_size (4) must be divisible by minimal possible batch size (8)`

Root cause:

- for single-node `8`-card FSDP, `real_train_batch_size = train_batch_size * rollout.n` needed to satisfy the minimal FSDP batch constraint
- the initial minimal smoke settings used `train_batch_size=4` and `rollout.n=1`

Action taken:

- increased `TRAIN_BATCH_SIZE` from `4` to `8`

Outcome:

- the run moved past the first configuration constraint

### Stage 6: Smoke then failed on normalized `ppo_mini_batch_size`

Observed error:

- `AssertionError: ppo_mini_batch_size 0 should be larger than 0 after normalization`

Root cause:

- `FSDPWorker` normalizes `ppo_mini_batch_size` by the device mesh size
- with `8` NPUs visible, `ppo_mini_batch_size=4` normalized down to `0`

Action taken:

- increased `PPO_MINI_BATCH_SIZE` from `4` to `8`

Outcome:

- the run moved past the second FSDP batch constraint

### Stage 7: HCCL environment default was wrong for this single-node fallback pass

Observed error:

- `HCCL_INTRA_PCIE_ENABLE or HCCL_INTRA_ROCE_ENABLE is invalid`
- failure surfaced during FSDP reference-model initialization and the first distributed broadcast

Root cause:

- the repo default in `scripts/ascend/env.qwen35_npu.sh` exported:
  - `HCCL_INTRA_PCIE_ENABLE=1`
  - `HCCL_INTRA_ROCE_ENABLE=1`
- for the current single-node fallback run on `.36`, that combination was not accepted

Action taken:

- reran the smoke with:
  - `HCCL_INTRA_PCIE_ENABLE=1`
  - `HCCL_INTRA_ROCE_ENABLE=0`

Outcome:

- the smoke clearly advanced beyond the previous HCCL failure point
- the old HCCL error stopped being the active blocker for this run shape

### Stage 8: `torchvision` was missing at runtime

Observed error:

- `ModuleNotFoundError: No module named 'torchvision'`
- failure surfaced through the Qwen3.5 vision-model import chain in `vllm`

Root cause:

- `torchvision` had not been installed into the fallback shared environment

Action taken:

- installed `torchvision==0.23.0` into `/shared/envs/qwen35` with `--no-deps`

Outcome:

- the earlier Qwen vision import chain started working
- the smoke progressed to deeper `vllm_ascend` runtime paths

### Stage 9: `vllm_ascend` runtime dependencies were incomplete

Once the run progressed past `torchvision`, a second layer of Python-side runtime dependencies surfaced.

Observed errors over successive retries:

- `ModuleNotFoundError: No module named 'numba'`
- `ModuleNotFoundError: No module named 'scipy'`
- later import probing showed more potentially missing fallback-line dependencies such as:
  - `quart`
  - `arctic_inference`
  - `torchaudio`

Root cause:

- the fallback installation path had produced an importable `vllm_ascend`, but not all of its practical runtime dependencies had been present in the shared env

Action taken:

- installed `numba==0.61.2`
- retried and continued import tracing
- installed `scipy` after confirming it was the next front-most missing dependency

Operational pitfall seen during `scipy` install:

- the default pip source stalled during the large wheel download
- switching to the Tsinghua mirror gave a reliable confirmation path

Outcome:

- `policy_flashlb` import eventually succeeded
- the frontier moved from pure Python missing-package errors into mixed runtime and backend initialization issues

### Stage 10: False `libhccl.so` failures were amplified by a shell-expansion mistake

Observed error:

- repeated import failures like:
  - `ImportError: libhccl.so: cannot open shared object file: No such file or directory`

Initial interpretation:

- it looked like the host Ascend runtime had not been activated correctly for child Python processes

Actual root cause:

- two things were true at once:
  - `torch_npu` genuinely required the full Ascend runtime plus explicit shared-library visibility
  - some remote debug commands were incorrectly constructing `LD_LIBRARY_PATH` because `$LD_LIBRARY_PATH` was expanded locally instead of on the remote shell

Actions taken:

- confirmed the actual `libhccl.so` location:
  - `/usr/local/Ascend/cann-8.5.0/lib64/libhccl.so`
- sourced both host runtime scripts first:
  - `/usr/local/Ascend/ascend-toolkit/set_env.sh`
  - `/usr/local/Ascend/nnal/atb/set_env.sh`
- prepended these shared-library roots explicitly:
  - `/shared/envs/qwen35/lib/python3.10/site-packages/torch/lib`
  - `/shared/envs/qwen35/lib/python3.10/site-packages/torch_npu/lib`
  - `/shared/envs/qwen35/lib/python3.10/site-packages/vllm_ascend`
- fixed remote command construction so `\$LD_LIBRARY_PATH` expanded on the remote host, not locally

Outcome:

- `ldd` now resolved `libhccl.so` and the relevant `torch_npu` dependencies correctly
- `torch_npu` became importable in the same runtime shape later used for the smoke

### Stage 11: `setuptools 82.0.1` broke `pkg_resources` for `torchair`

Observed error:

- `ModuleNotFoundError: No module named 'pkg_resources'`
- the error surfaced from `torch_npu.dynamo.torchair`

Root cause:

- the shared env had `setuptools 82.0.1`
- in that state, `pkg_resources` was no longer present for this runtime path
- `torchair` still imported `pkg_resources`

Action taken:

- downgraded `setuptools` to `80.9.0`

Outcome:

- `pkg_resources` import returned
- `vllm_ascend.worker.worker` became directly importable

### Stage 12: `vllm_ascend` worker import succeeded, but `vllm` still reported Triton drift

Observed warning-level errors during import:

- `Failed to import Triton kernels`
- `No module named 'triton.language.target_info'`

Interpretation:

- this showed another version drift between the fallback line and what `vllm` expected
- but at this point it was not the immediate fatal error for `vllm_ascend.worker.worker` import

Outcome:

- `policy_flashlb` import succeeded
- `vllm_ascend.worker.worker` import succeeded
- the smoke was able to advance into deeper runtime startup

### Stage 13: Latest smoke run reached `vLLMHttpServer` and `EngineCore` startup

Validated progress in the latest run:

- import checks passed
- Ray startup passed
- configuration validation passed
- dataset filtering completed
- dataloaders were created
- `WorkerDict` initialization started
- `vllm_ascend` platform plugin activated
- `vLLMHttpServer` launched
- `EngineCore` processes started and entered memory / executor initialization

This was a clear improvement over earlier failures that died in:

- config validation
- FSDP batch checks
- HCCL init
- missing Python dependencies
- missing `pkg_resources`
- missing `libhccl.so`

### Stage 14: Current deepest blocker is now a fallback-line `torch` / compile-path mismatch

Observed fatal error in the latest smoke run:

- `AttributeError: torch._functorch.config.autograd_cache_normalize_inputs does not exist`

Where it fails:

- inside `vllm` `EngineCore` startup
- during worker-side memory determination / compilation backend setup

Interpretation:

- the fallback line now reaches a deeper compatibility boundary between:
  - `torch 2.8.0`
  - `vllm 0.18.0`
  - the current `vllm` compilation backend path
- this is more intrinsic than the earlier missing-package and missing-library issues

Current status:

- the smoke still exits non-zero
- but it now fails much later than before, after most early environment and worker bring-up issues have already been cleared

## Confirmed Workarounds And Lessons

The following were confirmed useful during fallback debugging on `.36`:

- source both host runtime scripts before using the shared env:
  - `/usr/local/Ascend/ascend-toolkit/set_env.sh`
  - `/usr/local/Ascend/nnal/atb/set_env.sh`
- keep the smoke on single-node `8`-card settings that satisfy FSDP batch constraints:
  - `TRAIN_BATCH_SIZE=8`
  - `PPO_MINI_BATCH_SIZE=8`
- for this single-node fallback pass, use:
  - `HCCL_INTRA_PCIE_ENABLE=1`
  - `HCCL_INTRA_ROCE_ENABLE=0`
- explicitly prepend library roots when validating imports or running the smoke:
  - `torch/lib`
  - `torch_npu/lib`
  - `vllm_ascend`
- do not trust complex SSH one-liners until quoting is simplified and remote variable expansion is verified
- on this fallback line, `vllm_ascend` runtime needed at least:
  - `torchvision`
  - `numba`
  - `scipy`
- `setuptools<81` is currently required because `torchair` still imports `pkg_resources`
- default pip sources can stall; keep an explicit mirror ready for fallback debugging

## What Is Still Not Solved

- the final PR `#5682` matrix is still blocked by the missing `torch 2.10`-compatible `torch_npu`
- the fallback line still shows Triton and compile-path drift warnings
- the deepest active smoke failure is now:
  - `torch._functorch.config.autograd_cache_normalize_inputs` missing during `vllm` engine startup

## Post-Timeline Update: `2.9.0` became the next candidate line

After the main fallback sequence above was written down, one additional finding changed the immediate next-step recommendation:

- the public `torch-npu` package index now exposes `2.9.0`
- the current `torch-npu` README also documents a `CANN 8.5.0` pairing with:
  - `torch 2.9.0`
  - `torch-npu 2.9.0`
  - branch `v2.9.0-7.3.0`

Interpretation:

- this does not prove the lab cluster can run the full PR `#5682` matrix
- it also does not remove the final need for `torch 2.10`
- but it does explain why `vllm-ascend@54879467` carrying a hard `PyTorch 2.9.0` gate may reflect an intended upstream floor rather than a random local mismatch
- because of that, the best next experiment is no longer "keep stretching `torch 2.8.x`"
- the best next experiment is "validate `torch 2.9.0` + `torch-npu 2.9.0` in a fresh isolated env first"

## Recommended Use Of This Timeline

Use this document for:

- handoff across sessions
- explaining why the fallback line is "useful for diagnosis" but not a final validation result
- quickly identifying which earlier blockers were already resolved and should not be re-debugged first

Do not use this document as the only source of truth for the final target stack. For that, keep following:

- `docs/ascend_qwen35_lab/RUNBOOK.md`
- `docs/ascend_qwen35_lab/KNOWN_ISSUES.md`
- `docs/ascend_qwen35_lab/SHARED_ENV_TODO.md`
