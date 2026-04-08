# Qwen3.5 Ascend Known Issues

This file records the current blockers and sharp edges for `Qwen3.5 + FSDP + GRPO` on Ascend NPU, with source links back to the upstream discussion.

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

Sources:
- https://github.com/verl-project/verl/issues/5441
- https://github.com/verl-project/verl/pull/5682#issuecomment-4133604414
- https://github.com/verl-project/verl/pull/5682#issuecomment-4146330454
- local cluster observation while testing public `torch_npu 2.8.x`

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
