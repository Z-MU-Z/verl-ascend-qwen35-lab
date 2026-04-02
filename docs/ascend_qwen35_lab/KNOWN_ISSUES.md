# Qwen3.5 Ascend Known Issues

This file records the current blockers and sharp edges for `Qwen3.5 + FSDP + GRPO` on Ascend NPU, with source links back to the upstream discussion.

## Upstream state

- The support landed through [PR #5682](https://github.com/verl-project/verl/pull/5682).
- The older NPU-focused draft [PR #5434](https://github.com/verl-project/verl/pull/5434) was closed and should not be used as the testing baseline.

## Known issues

### 1. Ulysses sequence parallel is not supported yet

Use `ulysses_sequence_parallel_size=1` for now.

Source:
- https://github.com/verl-project/verl/pull/5682#issuecomment-4133596736

### 2. The working package matrix is tightly pinned

The PR author reported:

- `vllm==0.18.0`
- `torch=2.10`
- `triton=3.6`
- `transformers` pinned to commit `cc7ab9be`
- `vllm-ascend` pinned to commit `54879467`

Sources:
- https://github.com/verl-project/verl/pull/5682#issuecomment-4133604414
- https://github.com/verl-project/verl/pull/5682#issuecomment-4146330454

### 3. `use_remove_padding=True` is risky on NPU

Multiple users reported FlashAttention or tiling failures on Ascend when `use_remove_padding=True`.
The lab smoke script therefore starts with `use_remove_padding=False`.

Source:
- https://github.com/verl-project/verl/pull/5682#issuecomment-4152705666

### 4. Explicit FSDP wrap policy may be required

Some users could only start training after adding:

```text
+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']"
+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']"
```

Source:
- https://github.com/verl-project/verl/pull/5682#issuecomment-4154412727

### 5. Activation offload is not a safe default

There is an upstream report that `model.enable_activation_offload=True` can fail on this path.
Do not enable it for the first bring-up pass.

Source:
- https://github.com/verl-project/verl/pull/5682#issuecomment-4149745276

### 6. Newer `transformers` versions can introduce fresh breakage

There is an upstream discussion showing version drift can move the failure point from model forward into the vLLM startup path.
For reproducibility, keep the pinned `transformers` commit during first validation.

Source:
- https://github.com/verl-project/verl/pull/5682#issuecomment-4151822613
