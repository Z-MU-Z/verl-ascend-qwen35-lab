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
