from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = ROOT / "scripts/ascend/run_qwen35_27b_npu_smoke.sh"
ISSUES_DOC = ROOT / "docs/ascend_qwen35_lab/KNOWN_ISSUES.md"


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
    assert "Ulysses sequence parallel" in content
    assert "use_remove_padding=True" in content
    assert "Qwen3_5DecoderLayer" in content
    assert "https://github.com/verl-project/verl/pull/5682#issuecomment-4133596736" in content
    assert "https://github.com/verl-project/verl/pull/5682#issuecomment-4152705666" in content
