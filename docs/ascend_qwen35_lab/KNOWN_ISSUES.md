# Qwen3.5 Ascend Known Issues

This file records the current blockers and sharp edges for `Qwen3.5 + FSDP + GRPO` on Ascend NPU, with source links back to the upstream discussion.

For the fallback-line debugging sequence on `.36`, including the path from shared-env recovery through the latest single-node smoke failure, see `docs/ascend_qwen35_lab/FALLBACK_DEBUG_TIMELINE_20260408.md`.

## Upstream state

- The support landed through [PR #5682](https://github.com/verl-project/verl/pull/5682).
- The older NPU-focused draft [PR #5434](https://github.com/verl-project/verl/pull/5434) was closed and should not be used as the testing baseline.

## Known issues

### 1. Public `torch_npu` does not currently satisfy the PR matrix

The PR-reported working stack is pinned to:

- `vllm==0.18.0`
- `torch==2.10`
- `triton==3.6`
- `transformers` at commit `cc7ab9be`
- `vllm-ascend` at commit `54879467`

But the public Ascend mirror currently exposed during cluster bring-up only yielded `torch_npu 2.8.x`, which failed to import when paired with `torch 2.10`:

```text
libtorch_npu.so: undefined symbol: _ZNK3c1010TensorImpl20is_contiguous_customENS_12MemoryFormatE
```

Current lab implication:

- the shared env is temporarily restored to an importable fallback state based on `torch 2.8.0` + `torch_npu 2.8.0.post2`
- that fallback is for continued debugging only
- `/shared/envs/qwen35` is now deprecated for active bring-up and should not be used for current `.36` retries
- the required current retry env is `/home/zmz/envs/qwen35-t29-lite`
- it does not count as validation of the PR `#5682` matrix
- for public-facing status, the current Ascend reference baseline should still be described as the `2.8.x` line
- as of `2026-04-09`, the public `torch-npu` package index and README now also expose a documented `2.9.0` line for `CANN 8.5.0`
- that makes `torch 2.9.0` + `torch-npu 2.9.0` the preferred intermediate validation candidate for this cluster
- it still does not replace the final PR target of `torch 2.10`

Sources:
- https://github.com/verl-project/verl/issues/5441
- https://github.com/verl-project/verl/pull/5682#issuecomment-4133604414
- https://github.com/verl-project/verl/pull/5682#issuecomment-4146330454
- local cluster observation while testing public `torch_npu 2.8.x`
- https://pypi.org/project/torch-npu/

### 2. Ulysses sequence parallel is not supported yet

Use `ulysses_sequence_parallel_size=1` for now.

Source:
- https://github.com/verl-project/verl/pull/5682#issuecomment-4133596736

### 3. The working package matrix is tightly pinned

The PR author reported:

- `vllm==0.18.0`
- `torch=2.10`
- `triton=3.6`
- `transformers` pinned to commit `cc7ab9be`
- `vllm-ascend` pinned to commit `54879467`

Sources:
- https://github.com/verl-project/verl/pull/5682#issuecomment-4133604414
- https://github.com/verl-project/verl/pull/5682#issuecomment-4146330454

### 4. `use_remove_padding=True` is risky on NPU

Multiple users reported FlashAttention or tiling failures on Ascend when `use_remove_padding=True`.
The lab smoke script therefore starts with `use_remove_padding=False`.

Source:
- https://github.com/verl-project/verl/pull/5682#issuecomment-4152705666

**Update (`2026-04-10`, `.36`, `torch_npu 2.9` + Qwen3.5-4B GRPO):** training can reach `trainer.fit()` → `_update_actor`, then **crash the FSDP `WorkerDict` with `SIGSEGV`** during autograd on **`convolution_backward` / `aclnnConvolutionBackwardGetWorkspaceSize`** (vision tower on a VL-tagged Qwen3.5 checkpoint). Ray then surfaces **`ActorUnavailableError: Socket closed`**. **`actor_rollout_ref.model.use_remove_padding=False` did not remove this crash** in the lab repro. Next things to try: **`actor_rollout_ref.actor.freeze_vision_tower=True`** for text-only datasets (geo3k), a **text-only base** checkpoint if available, or escalate to **Huawei** with the C++ stack (CANN 8.5 + `torch_npu` 2.9). `examples/grpo_trainer/run_qwen3_5_4b_vllm_fsdp_npu.sh` now defaults `use_remove_padding` to **False** via `USE_REMOVE_PADDING` for safer NPU bring-up.

### 5. Explicit FSDP wrap policy may be required

Some users could only start training after adding:

```text
+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']"
+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']"
```

Source:
- https://github.com/verl-project/verl/pull/5682#issuecomment-4154412727

### 6. Activation offload is not a safe default

There is an upstream report that `model.enable_activation_offload=True` can fail on this path.
Do not enable it for the first bring-up pass.

Source:
- https://github.com/verl-project/verl/pull/5682#issuecomment-4149745276

### 7. Newer `transformers` versions can introduce fresh breakage

There is an upstream discussion showing version drift can move the failure point from model forward into the vLLM startup path.
For reproducibility, keep the pinned `transformers` commit during first validation.

Source:
- https://github.com/verl-project/verl/pull/5682#issuecomment-4151822613

### 8. `vllm-ascend@54879467` fallback-line debugging currently needs source-level patching

On the current lab fallback stack, the failure is not a single issue. The build chain observed on `.36` currently includes:

- CANN include paths in the extracted source that do not match `CANN 8.5.0.B160`
- `setup.py` using bare `python3` when probing `torch-npu`, which can mis-resolve `TORCH_NPU_PATH`
- a hard `PyTorch 2.9.0` CMake gate even though the fallback environment is still on `torch 2.8.x`
- a later host-stub phase that can still lose `llvm-objdump` unless the tool lookup is made explicit

For reproducible debugging, keep those edits in the local helper:

- `scripts/ascend/prepare_vllm_ascend_source.py`

Do not treat ad-hoc edits inside remote `/tmp/.../vllm-ascend-*` trees as the authoritative patch source.

Current status of that chain:

- resolved for the current fallback install path on `.36`
- `vllm_ascend` is now installed in `/shared/envs/qwen35`
- both `.36` and `.37` can import `vllm_ascend` from the shared env
- the previously missing `libvllm_ascend_kernels.so` is no longer the active blocker for this fallback line

What still matters:

- this remains a patched fallback install path, not an upstream-clean final matrix
- the environment is still on `torch 2.8.x`, not the target `torch 2.10` line
- this shared fallback env is deprecated for active bring-up; current retries must run from `/home/zmz/envs/qwen35-t29-lite`
- smoke and training validation are still pending
- the fallback smoke now dies in `vllm` engine startup on:
  - `AttributeError: torch._functorch.config.autograd_cache_normalize_inputs does not exist`
- that failure is consistent with the fallback `torch 2.8.x` line being too old for the current `vllm` compile path

### 9. vLLM Qwen3.5 rollout imports `torchvision` even for text-only GRPO

Observed on `.36` with isolated `torch 2.9.0` env (`qwen35-t29-lite`) and `vllm==0.18.0` when starting `vLLMHttpServer`:

- Surface error: `ValidationError` for `VllmConfig`, message like `Model architectures ['Qwen3_5ForConditionalGeneration'] failed to be inspected`.
- Root cause in logs: `ModuleNotFoundError: No module named 'torchvision'` while importing `transformers.models.qwen2_vl.video_processing_qwen2_vl` (pulled in through vLLM’s `qwen3_5` → VL helper modules).

**Install policy**

- Do **not** run bare `pip install torchvision` on this stack: resolver may pull a `torchvision` that wants a **newer `torch` (e.g. 2.11)** and break the `torch_npu` pair.
- For `torch==2.9.0`, install the matching vision build with **`--no-deps`** (PyTorch wiki pairs **2.9.0** with **torchvision 0.24.0**).

Example (after venv + `source /usr/local/Ascend/ascend-toolkit/set_env.sh` so imports work):

```bash
pip install "torchvision==0.24.0" --no-deps --timeout 300 --retries 15
```

If downloads from PyPI time out, stage wheels on the host or use the lab mirror policy in `HANDOFF_20260409_FOR_37.md`.

**Do not replace `PYTHONPATH` after sourcing CANN**

If you run `export PYTHONPATH=/path/to/verl-ascend-qwen35-lab` after `ascend-toolkit/set_env.sh`, you remove CANN’s `python/site-packages` from `PYTHONPATH`. Then vLLM’s `vllm_ascend` worker fails with `ModuleNotFoundError: No module named 'acl'`. Prefer `export PYTHONPATH=/path/to/repo:$PYTHONPATH`, or rely on `scripts/ascend/env.qwen35_npu.sh` (it prepends the usual CANN site-packages when missing).

**Related env knob**

- If you lower `actor_rollout_ref.rollout.max_num_batched_tokens` via env/script, keep it **≥ `max_num_seqs`** (often `1024` in defaults) or vLLM raises `SchedulerConfig` validation errors.

### 10. Single-node `.36/.37` fallback runs can fail early if both intra-PCIE and intra-ROCE are enabled

Observed on `.36` (`2026-04-10`) during the single-node `Qwen3.5-4B` freeze-vision smoke:

- Failure surfaced in ref-model init / HCCL bring-up, not in the earlier `convolution_backward` path.
- Error shape:
  - `RuntimeError: createHCCLCommOrigin ... HcclGetRootInfo(&hcclID)`
  - `ERR02200 DIST call hccl api failed`
  - `Config_Error_Invalid_Environment_Variable(EI0001): Environment variable [HCCL_INTRA_PCIE_ENABLE or HCCL_INTRA_ROCE_ENABLE] is invalid`

For this single-node lab topology, the known-good fallback is:

```bash
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
```

`scripts/ascend/env.qwen35_npu.sh` now defaults to that combination for local bring-up. If a host needs a different fabric policy, override explicitly in the shell before running the smoke script.

### 11. Isolated `torch 2.9` runs on `.36` can still fail in vLLM worker init if ATB libraries are not on `LD_LIBRARY_PATH`

Observed on `.36` (`2026-04-11`) after the local fallback patch for missing `torch.ops._C_ascend.npu_gemma_rms_norm` was committed in `4890034a` and the job advanced past the earlier Qwen3.5 layernorm failure:

- the previous `AttributeError: '_OpNamespace' '_C_ascend' object has no attribute 'npu_gemma_rms_norm'` no longer reproduced in the rerun
- the next stable blocker moved to `torch_npu.op_plugin.atb._atb_ops`
- surface error in the vLLM worker:
  - `OSError: libatb.so: cannot open shared object file: No such file or directory`
- when only `libatb.so` was added, another missing dependency also surfaced:
  - `libtorch_python.so`

Direct host validation on `.36` showed the import succeeds once `LD_LIBRARY_PATH` includes all of:

- the active venv `site-packages/torch/lib`
- the active venv `site-packages/torch_npu/lib`
- `/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_1/lib`
- `/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib`

Current lab action:

- `scripts/ascend/env.qwen35_npu.sh` now prepends those runtime library directories when they exist
- keep using that script after `ascend-toolkit/set_env.sh` so remote smoke retries do not depend on ad-hoc manual exports
- if a host uses a different ATB layout, record the exact replacement path in the handoff before retrying

### 12. Isolated `torch 2.9` single-card retries can still die in `camem.py` unless `enable_sleep_mode=False`

Observed on `.36` (`2026-04-11`) after the run had already moved past:

- the missing `npu_gemma_rms_norm` op
- the missing `libatb.so` / `libtorch_python.so` visibility issue
- the earlier batch-shape validation failures

On the single-card `freeze_vision_tower=True` retry, the new fatal stack was:

```text
vllm_ascend/device_allocator/camem.py
TypeError: 'NoneType' object is not callable
```

Root cause from the imported `camem.py`:

- `vllm_ascend_C` failed to import because `libvllm_ascend_kernels.so` was not available in that isolated env
- the module logged `Sleep mode will be disabled`
- but `get_pluggable_allocator()` still called `init_module(...)` unconditionally
- because `init_module` had been set to `None`, worker init failed later during the sleep-mode allocator path

Current lab implication:

- for this fallback `torch 2.9` line, **do not rely on rollout sleep mode**
- use `actor_rollout_ref.rollout.enable_sleep_mode=False`
- `examples/grpo_trainer/run_qwen3_5_4b_vllm_fsdp_npu.sh` now defaults `ENABLE_SLEEP_MODE=False` for the safer NPU smoke shape
- if someone wants sleep mode later, first verify that `vllm_ascend_C` and `libvllm_ascend_kernels.so` both import cleanly in the target env

**Latest update (`2026-04-11`, `.36`, archived log `qwen35_4b_freezevis_singlecard_nosleepdefault2_20260411_210157.log`):**

- with `enable_sleep_mode=False`, the run **did not** reproduce the earlier `camem.py` / `NoneType` crash
- the job advanced again through `WorkerDict` and reached **`vLLMHttpServer`**
- the latest visible warnings then shifted to:
  - `Failed to import Triton kernels ... No module named 'triton.language.target_info'`
  - `Unrecognized keys in rope_parameters for rope_type='default': {'mrope_interleaved', 'mrope_section'}`
- the actual fatal exit was later in `EngineCore`, not at the warning site:
  - `torch._dynamo.exc.Unsupported: Import failure`
  - debug context: `module_name: vllm_ascend.vllm_ascend_C`
  - stack crossed `vllm_ascend/ops/layernorm.py -> enable_custom_op()` during the Qwen3.5 dummy/profile run

Current interpretation:

- `camem` sleep-mode allocator is no longer the active blocker on this line
- the active blocker moved deeper into the **vLLM / vllm-ascend custom-op import path**
- the Triton and rope warnings are still useful correlation signals, but they were **not** the final terminating frame in this run
- no newer `SIGSEGV` or `ActorUnavailableError` was captured in that archived log segment after the sleep-mode fix

### 13. `enable_sleep_mode=False` can still reach `camem` via `sleep_replicas()` and then fail on `expandable_segments:True`

Observed on `.36` in archived log:

- `/home/zmz/verl/log_archive/qwen35_4b_freezevis_singlecard_customopoff_20260411_232308.log`

Latest confirmed behavior:

- the run advanced past the earlier missing `npu_gemma_rms_norm` and custom-op import blockers
- `vLLMHttpServer` completed graph capture and `EngineCore` launched
- the later failure happened during `CheckpointManager.sleep_replicas()` rather than during model profile warmup

Fatal stack signature:

```text
AssertionError: Expandable segments are not compatible with memory pool.
```

Key path:

- `verl/checkpoint_engine/base.py -> sleep_replicas()`
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py -> sleep()`
- `vllm_ascend/worker/worker.py -> sleep`
- `vllm_ascend/device_allocator/camem.py -> CaMemAllocator`

Root cause:

- this repo's `vLLMHttpServer.__init__` sets `set_expandable_segments(True)` when `enable_sleep_mode=False`
- but `sleep()` was still callable later through checkpoint-engine replica management
- on Ascend, `vllm_ascend` `CaMemAllocator` rejects `expandable_segments:True` when entering the memory-pool sleep path

Lab implication:

- for the current fallback NPU line, `enable_sleep_mode=False` must also imply **skip rollout sleep calls**
- otherwise the job can still die in `camem.py` even after avoiding the earlier `NoneType` and custom-op startup failures

**Follow-up update (`2026-04-13`, `.36`, archived log `qwen35_4b_freezevis_singlecard_nosleepskip_20260413_101327.log`):**

- after locally patching `vLLMHttpServer.sleep()` to no-op when `enable_sleep_mode=False`, the run advanced further:
  - `filter dataset len: 2101`
  - `filter dataset len: 601`
  - rollout worker launched, weights loaded, `AgentLoopManager` started, and training moved to `Training from scratch`
- the same allocator conflict then reappeared one step later in **`wake_up()`**, not `sleep()`

Updated fatal path:

- `verl/workers/fsdp_workers.py -> rollout_mode()`
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py -> resume(tags=["weights"])`
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py -> collective_rpc("wake_up")`
- `vllm_ascend/worker/worker.py -> wake_up`
- `vllm_ascend/device_allocator/camem.py -> CaMemAllocator`

Updated implication:

- for this fallback NPU line, `enable_sleep_mode=False` must imply **skip both `sleep()` and `wake_up()`**
- guarding `vLLMHttpServer.wake_up()` alone is insufficient, because `ServerAdapter.resume()` can still dispatch `collective_rpc("wake_up")` whenever only `free_cache_engine=True`
- otherwise `CaMemAllocator` can still be re-entered during rollout resume and hit the same `expandable_segments:True` assertion

### 14. Local fix timeline for the current `.36` fallback line

- `4890034a`: fallback for missing `torch.ops._C_ascend.npu_gemma_rms_norm`
- `a252524d`: restore Ascend/ATB runtime library paths for the isolated env
- `34e118b3` + `7c2c2984`: force `ENABLE_SLEEP_MODE=False` via the Qwen3.5 4B smoke script
- `55f59261`: record the deeper `vllm_ascend.vllm_ascend_C` import-failure root cause in local docs
- `a8fc03cf`: disable the late vllm-ascend custom-op import before Dynamo/profile run
- `7d312155`: skip rollout `sleep()` when `enable_sleep_mode=False`
- `ebb36589`: skip wrapper-level `wake_up()` when `enable_sleep_mode=False`
- local follow-up after `ebb36589`: `ServerAdapter.resume()` / `release()` must also gate on `enable_sleep_mode`, not only `free_cache_engine`

Current status after the `ServerAdapter` follow-up:

- solved: missing `npu_gemma_rms_norm` startup failure
- solved: late custom-op import failure during profile/dummy run
- solved: `sleep()`-time `camem` allocator assertion
- solved: wrapper-level `wake_up()` entry when `enable_sleep_mode=False`
- solved locally: adapter-level `collective_rpc("wake_up")` / `collective_rpc("sleep")` should now be skipped when `enable_sleep_mode=False`
- still active: rerun confirmation on `.36` after syncing the local adapter guard

### 15. The required `/home/zmz/envs/qwen35-t29-lite` env on `.36` can still fail inside `EngineCore` if `_C_ascend.npu_causal_conv1d_custom` is absent

Observed on `.36` (`2026-04-13`) in archived log:

- `/home/zmz/verl/log_archive/qwen35_4b_freezevis_t29lite_n8_20260413_133024.log`

Confirmed runtime facts from the active lab env:

- `source /home/zmz/envs/qwen35-t29-lite/bin/activate`
- `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- `source /usr/local/Ascend/nnal/atb/set_env.sh`
- `hasattr(torch.ops, "_C_ascend") == True`
- `hasattr(torch.ops._C_ascend, "npu_gemma_rms_norm") == False`
- `hasattr(torch.ops._C_ascend, "npu_causal_conv1d_custom") == False`

Latest deterministic failure shape:

- the run no longer died at the older shared-env `torch._functorch.config.autograd_cache_normalize_inputs` error
- the run passed env checks, Ray startup, dataset filtering, and rollout worker construction
- the fatal error moved deeper into the vLLM execution path during the Qwen3.5 / Qwen3Next GatedDeltaNet prefill path
- final stack shape in the archived log:
  - `vllm/model_executor/models/qwen3_next.py -> gdn_attention_core`
  - `vllm_ascend/patch/worker/patch_qwen3_5.py`
  - `vllm_ascend/patch/worker/patch_qwen3_next.py`
  - `AttributeError: '_OpNamespace' '_C_ascend' object has no attribute 'npu_causal_conv1d_custom'`
  - `EngineCore encountered a fatal error`
  - `EngineDeadError`

Current interpretation:

- this is not a startup-only import error
- it happens during actual model execution / profile dummy run inside vLLM
- package source under `vllm_ascend` still contains Python/NPU `causal_conv1d` implementations, so the failure looks like a missing custom-op registration / exposure gap rather than proof that the math path is completely unavailable

Current local mitigation:

- add a local Python fallback that registers `_C_ascend.npu_causal_conv1d_custom` when the op is missing
- route that fallback to the already-imported `causal_conv1d_update(...)` path used by the shipped vllm-ascend Qwen patches
- keep the logs archived permanently so Huawei can compare the missing-op failure before and after the fallback
