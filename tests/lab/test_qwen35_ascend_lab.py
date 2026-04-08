from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = ROOT / "scripts/ascend/run_qwen35_27b_npu_smoke.sh"
ISSUES_DOC = ROOT / "docs/ascend_qwen35_lab/KNOWN_ISSUES.md"
RUNBOOK_DOC = ROOT / "docs/ascend_qwen35_lab/RUNBOOK.md"
TODO_DOC = ROOT / "docs/ascend_qwen35_lab/TODO.md"


def test_run_script_pins_safe_smoke_defaults() -> None:
    content = RUN_SCRIPT.read_text()

    assert "actor_rollout_ref.model.use_remove_padding=False" in content
    assert "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1" in content
    assert "actor_rollout_ref.ref.ulysses_sequence_parallel_size=1" in content
    assert "Qwen3_5DecoderLayer" in content
    assert "transformers.git@cc7ab9be" in content
    assert "vllm==0.18.0" in content


def test_known_issues_doc_records_key_blockers() -> None:
    content = ISSUES_DOC.read_text()

    assert "# Qwen3.5 Ascend Known Issues" in content
    assert "https://github.com/verl-project/verl/issues/5441" in content
    assert "undefined symbol" in content
    assert "torch 2.10" in content
    assert "2.8.x" in content
    assert "Ulysses sequence parallel" in content
    assert "use_remove_padding=True" in content
    assert "Qwen3_5DecoderLayer" in content
    assert "https://github.com/verl-project/verl/pull/5682#issuecomment-4133596736" in content
    assert "https://github.com/verl-project/verl/pull/5682#issuecomment-4152705666" in content
    assert "prepare_vllm_ascend_source.py" in content
    assert "llvm-objdump" in content


def test_runbook_calls_out_matrix_gate_before_smoke() -> None:
    content = RUNBOOK_DOC.read_text()

    assert "# Qwen3.5 Ascend Runbook" in content
    assert "CANN 8.5.0.B160" in content
    assert "torch_npu 2.8.0" in content
    assert "missing `torch 2.10`-compatible `torch_npu`" in content
    assert "source scripts/ascend/env.qwen35_shared.sh" in content
    assert "Do not start the smoke script until" in content
    assert "prepare_vllm_ascend_source.py" in content
    assert "--helper-llvm-objdump" in content
    assert "export PATH=/tmp/vllm-ascend-helper-bin:$PATH" in content
    assert "--no-build-isolation --no-deps /tmp/vllm-ascend-54879467-src" in content


def test_todo_records_local_overlay_helper() -> None:
    content = TODO_DOC.read_text()

    assert "prepare_vllm_ascend_source.py" in content
    assert "helper-bin shims" in content
    assert "Convert the remote-only `vllm-ascend` source patches" in content
