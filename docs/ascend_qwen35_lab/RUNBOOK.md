# Qwen3.5 Ascend Runbook

## Goal

Bring up `Qwen3.5 + FSDP + GRPO` on Huawei Ascend NPU with the upstream `verl` baseline already merged in [PR #5682](https://github.com/verl-project/verl/pull/5682).

Baseline frozen in this repo:

- Upstream repo: `verl-project/verl`
- Frozen commit: `4045d67063052dcb800c918c107b8d5a87046006`
- Why this commit: it is the merge commit for the Qwen3.5 FSDP GRPO support PR

## Recommended first-run path

1. Prepare the cluster image with `CANN` and `torch_npu`.
2. Activate the Python environment you actually want to use on the cluster.
3. Run:

```bash
source scripts/ascend/env.qwen35_npu.sh
bash scripts/ascend/bootstrap_qwen35_npu_env.sh
```

4. Export the model and dataset locations:

```bash
export MODEL_PATH=/path/to/Qwen3.5-27B
export TRAIN_FILE=/path/to/train.parquet
export TEST_FILE=/path/to/test.parquet
```

5. Start the safer smoke test:

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

## Suggested test sequence

1. Environment report from `scripts/ascend/check_qwen35_npu_env.py`
2. Single-node smoke run with 27B
3. Full first epoch with logs collected
4. Only after that, tune memory and throughput
5. Only after that, evaluate precision alignment and scaling

## Results logging

Copy the template in `results/templates/session_report.md` for each cluster run.
