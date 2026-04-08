from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_DOC = ROOT / "docs/ascend_qwen35_lab/NEW_MACHINE_FALLBACK_BOOTSTRAP.md"
BOOTSTRAP_SCRIPT = ROOT / "scripts/ascend/bootstrap_new_npu_machine_fallback.sh"
VERIFY_SCRIPT = ROOT / "scripts/ascend/verify_new_npu_machine.sh"


def test_new_machine_bootstrap_doc_records_bundle_strategy() -> None:
    content = BOOTSTRAP_DOC.read_text()

    assert "# New Machine Fallback Bootstrap" in content
    assert "bootstrap_bundle/" in content
    assert "hard packages" in content
    assert "ordinary pip packages" in content
    assert "transformers-cc7ab9be.tar.gz" in content
    assert "vllm-ascend-54879467.tar.gz" in content
    assert "catlass" in content
    assert "huawei36" in content


def test_new_machine_bootstrap_script_uses_bundle_for_hard_packages() -> None:
    content = BOOTSTRAP_SCRIPT.read_text()

    assert "BOOTSTRAP_BUNDLE_ROOT" in content
    assert "BUNDLE_DIST_DIR" in content
    assert "prepare_vllm_ascend_source.py" in content
    assert "catlass" in content
    assert "requirements-npu.txt" in content
    assert "-e \"${ROOT_DIR}\" --no-deps" in content
    assert "transformers-cc7ab9be.tar.gz" in content
    assert "vllm-0.18.0" in content
    assert "vllm-ascend-54879467.tar.gz" in content
    assert "--no-build-isolation --no-deps" in content


def test_new_machine_verify_script_checks_base_runtime_and_imports() -> None:
    content = VERIFY_SCRIPT.read_text()

    assert "npu-smi" in content
    assert "ascend-toolkit/set_env.sh" in content
    assert "nnal/atb/set_env.sh" in content
    assert "torch_npu" in content
    assert "vllm_ascend" in content
    assert "TORCH_DEVICE_BACKEND_AUTOLOAD=0" in content
