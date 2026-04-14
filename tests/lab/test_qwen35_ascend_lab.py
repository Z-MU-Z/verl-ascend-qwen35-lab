from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = ROOT / "scripts/ascend/run_qwen35_27b_npu_smoke.sh"
RUN_4B_SCRIPT = ROOT / "examples/grpo_trainer/run_qwen3_5_4b_vllm_fsdp_npu.sh"
ENV_SCRIPT = ROOT / "scripts/ascend/env.qwen35_npu.sh"
CHECK_ENV_SCRIPT = ROOT / "scripts/ascend/check_qwen35_npu_env.py"
ISSUES_DOC = ROOT / "docs/ascend_qwen35_lab/KNOWN_ISSUES.md"
RUNBOOK_DOC = ROOT / "docs/ascend_qwen35_lab/RUNBOOK.md"
TODO_DOC = ROOT / "docs/ascend_qwen35_lab/TODO.md"
VLLM_ASYNC_SERVER = ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
VLLM_ROLLOUT = ROOT / "verl/workers/rollout/vllm_rollout/vllm_rollout.py"


def test_run_script_pins_safe_smoke_defaults() -> None:
    content = RUN_SCRIPT.read_text()

    assert "actor_rollout_ref.model.use_remove_padding=False" in content
    assert "Qwen3_5DecoderLayer" in content
    assert "transformers.git@cc7ab9be" in content
    assert "vllm==0.18.0" in content


def test_4b_run_script_disables_sleep_mode_for_safe_npu_smoke() -> None:
    content = RUN_4B_SCRIPT.read_text()

    assert 'REQUIRED_ENV_PATH="/home/zmz/envs/qwen35-t29-lite"' in content
    assert 'This script must run from ${REQUIRED_ENV_PATH}' in content
    assert 'ENABLE_SLEEP_MODE="${ENABLE_SLEEP_MODE:-False}"' in content
    assert '+actor_rollout_ref.rollout.enable_sleep_mode="${ENABLE_SLEEP_MODE}"' in content
    assert 'SP_SIZE="${SP_SIZE:-1}"' in content
    assert 'actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}"' in content
    assert 'actor_rollout_ref.ref.ulysses_sequence_parallel_size="${SP_SIZE}"' in content
    assert "Qwen3_5DecoderLayer" in content
    assert "transformers@<cc7ab9be>" in content
    assert "vllm==0.18.0" in content


def test_4b_run_script_bootstraps_ascend_runtime_and_cleans_repo_pythonpath() -> None:
    content = RUN_4B_SCRIPT.read_text()

    assert 'ASCEND_TOOLKIT_ENV="${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"' in content
    assert 'ASCEND_ATB_ENV="${ASCEND_ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"' in content
    assert "_verl_restore_errexit" in content
    assert "set +u" in content
    assert "set +e" in content
    assert "set +o pipefail" in content
    assert "_verl_restore_nounset" in content
    assert 'source "${ASCEND_TOOLKIT_ENV}"' in content
    assert 'source "${ASCEND_ATB_ENV}"' in content
    assert 'if [[ "${_verl_restore_errexit}" == "1" ]]; then' in content
    assert 'if [[ -n "${PYTHONPATH:-}" ]]; then' in content
    assert 'verl-ascend-qwen35-lab' in content
    assert 'export PYTHONPATH="${_verl_clean_pythonpath}"' in content
    assert 'import verl' in content
    assert 'verl.__file__' in content


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
    assert "libatb.so" in content
    assert "LD_LIBRARY_PATH" in content
    assert "camem.py" in content
    assert "enable_sleep_mode=False" in content
    assert "triton.language.target_info" in content
    assert "torch._dynamo.exc.Unsupported: Import failure" in content
    assert "vllm_ascend.vllm_ascend_C" in content
    assert "Expandable segments are not compatible with memory pool" in content
    assert "wake_up" in content
    assert "4890034a" in content
    assert "a8fc03cf" in content
    assert "7d312155" in content
    assert "ebb36589" in content
    assert 'collective_rpc("wake_up")' in content
    assert "/home/zmz/envs/qwen35-t29-lite" in content
    assert "/shared/envs/qwen35" in content
    assert "npu_causal_conv1d_custom" in content
    assert "qwen35_4b_freezevis_t29lite_n8_20260413_133024.log" in content
    assert "function' object has no attribute 'scalar'" in content
    assert "qwen35_4b_freezevis_t29lite_n8_causalconvfix_20260413_164416.log" in content
    assert "qwen35_4b_freezevis_t29lite_n8_refconvfix_20260413_173332.log" in content
    assert "global_step_5" in content


def test_env_script_exports_single_node_hccl_and_runtime_library_paths() -> None:
    content = ENV_SCRIPT.read_text()

    assert 'export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-0}"' in content
    assert 'export HCCL_INTRA_PCIE_ENABLE="${HCCL_INTRA_PCIE_ENABLE:-1}"' in content
    assert "site-packages/torch/lib" in content
    assert "site-packages/torch_npu/lib" in content
    assert "/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_1/lib" in content
    assert "LD_LIBRARY_PATH" in content


def test_runbook_calls_out_matrix_gate_before_smoke() -> None:
    content = RUNBOOK_DOC.read_text()

    assert "# Qwen3.5 Ascend Runbook" in content
    assert "CANN 8.5.0.B160" in content
    assert "torch_npu 2.8.0" in content
    assert "missing `torch 2.10`-compatible `torch_npu`" in content
    assert "source /home/zmz/envs/qwen35-t29-lite/bin/activate" in content
    assert "Do not start the smoke script until" in content
    assert "prepare_vllm_ascend_source.py" in content
    assert "--helper-llvm-objdump" in content
    assert "export PATH=/tmp/vllm-ascend-helper-bin:$PATH" in content
    assert "--no-build-isolation --no-deps /tmp/vllm-ascend-54879467-src" in content
    assert "ENABLE_SLEEP_MODE=False" in content
    assert "triton.language.target_info" in content
    assert "torch._dynamo.exc.Unsupported: Import failure" in content
    assert "vllm_ascend.vllm_ascend_C" in content
    assert "Expandable segments are not compatible with memory pool" in content
    assert "wake_up()" in content
    assert 'collective_rpc("wake_up")' in content
    assert "/home/zmz/envs/qwen35-t29-lite" in content
    assert "deprecated for active bring-up" in content
    assert "npu_causal_conv1d_custom" in content
    assert "t29-lite" in content
    assert "function' object has no attribute 'scalar'" in content
    assert "refconvfix" in content
    assert "Training Progress" in content


def test_check_env_script_requires_t29_lite_runtime() -> None:
    content = CHECK_ENV_SCRIPT.read_text()

    assert 'REQUIRED_ENV_PATH = "/home/zmz/envs/qwen35-t29-lite"' in content
    assert "ERROR: active Python is not the required lab env" in content


def test_vllm_async_server_skips_sleep_when_sleep_mode_disabled() -> None:
    content = VLLM_ASYNC_SERVER.read_text()

    assert "or not self.config.enable_sleep_mode" in content


def test_vllm_async_server_skips_wake_up_when_sleep_mode_disabled() -> None:
    content = VLLM_ASYNC_SERVER.read_text()

    assert "if self.node_rank != 0 or not self.config.enable_sleep_mode:" in content


def test_vllm_rollout_skips_resume_and_release_when_sleep_mode_disabled() -> None:
    content = VLLM_ROLLOUT.read_text()

    assert "if self.config.free_cache_engine and self.config.enable_sleep_mode:" in content


def test_todo_records_local_overlay_helper() -> None:
    content = TODO_DOC.read_text()

    assert "prepare_vllm_ascend_source.py" in content
    assert "helper-bin shims" in content
    assert "Convert the remote-only `vllm-ascend` source patches" in content
    assert "/home/zmz/envs/qwen35-t29-lite" in content
