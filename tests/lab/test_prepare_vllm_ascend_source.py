import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREPARE_SCRIPT = ROOT / "scripts/ascend/prepare_vllm_ascend_source.py"


def _write_fake_vllm_ascend_tree(root: Path) -> None:
    (root / "csrc/cmake").mkdir(parents=True)
    (root / "csrc/moe_grouped_matmul/op_host").mkdir(parents=True)
    (root / "csrc/utils").mkdir(parents=True)

    (root / "setup.py").write_text(
        """
import subprocess
import sys

        torch_npu_command = "python3 -m pip show torch-npu | grep '^Location:' | awk '{print $2}'"
        try:
            torch_npu_path = subprocess.check_output(torch_npu_command, shell=True).decode().strip()
            torch_npu_path += "/torch_npu"
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Retrieve torch version version failed: {e}")
""".strip()
        + "\n"
    )
    (root / "CMakeLists.txt").write_text(
        """
run_python(TORCH_VERSION
  "import torch; print(torch.__version__)" "Failed to locate torch path")
# check torch version is 2.9.0
if(NOT ${TORCH_VERSION} VERSION_EQUAL "2.9.0")
  message(FATAL_ERROR "Expected PyTorch version 2.9.0, but found ${TORCH_VERSION}")
endif()
""".strip()
        + "\n"
    )

    old_include_text = """
include_directories(/usr/local/Ascend/include/experiment/platform)
include_directories(/usr/local/Ascend/include/experiment/slog)
""".strip()
    for rel_path in [
        "csrc/cmake/config.cmake",
        "csrc/cmake/intf_pub.cmake",
        "csrc/moe_grouped_matmul/op_host/CMakeLists.txt",
        "csrc/utils/CMakeLists.txt",
    ]:
        (root / rel_path).write_text(old_include_text + "\n")


def _write_partial_include_tree(root: Path) -> None:
    _write_fake_vllm_ascend_tree(root)
    (root / "csrc/cmake/config.cmake").write_text(
        "include_directories(/usr/local/Ascend/include/experiment/platform)\n"
    )
    (root / "csrc/cmake/intf_pub.cmake").write_text(
        "include_directories(/usr/local/Ascend/include/experiment/platform)\n"
    )
    (root / "csrc/moe_grouped_matmul/op_host/CMakeLists.txt").write_text(
        "include_directories(/usr/local/Ascend/include/experiment/platform)\n"
    )
    (root / "csrc/utils/CMakeLists.txt").write_text(
        "include_directories(/usr/local/Ascend/include/experiment/slog)\n"
    )


def _run_prepare(source_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PREPARE_SCRIPT),
            "--source-dir",
            str(source_dir),
            *extra_args,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_prepare_script_patches_known_source_compatibility_issues(tmp_path: Path) -> None:
    source_dir = tmp_path / "vllm-ascend-src"
    _write_fake_vllm_ascend_tree(source_dir)

    result = _run_prepare(source_dir)

    assert result.returncode == 0, result.stderr
    assert "Patched 5 file(s)." in result.stdout

    setup_py = (source_dir / "setup.py").read_text()
    assert 'subprocess.check_output([sys.executable, "-m", "pip", "show", "torch-npu"])' in setup_py
    assert 'torch_npu_location = next(' in setup_py
    assert 'torch_npu_path = torch_npu_location + "/torch_npu"' in setup_py
    assert 'python3 -m pip show torch-npu' not in setup_py

    for rel_path in [
        "csrc/cmake/config.cmake",
        "csrc/cmake/intf_pub.cmake",
        "csrc/moe_grouped_matmul/op_host/CMakeLists.txt",
        "csrc/utils/CMakeLists.txt",
    ]:
        content = (source_dir / rel_path).read_text()
        assert "include/aclnn/opdev" in content
        assert "include/toolchain" in content
        assert "include/experiment/platform" not in content
        assert "include/experiment/slog" not in content

    cmake_lists = (source_dir / "CMakeLists.txt").read_text()
    assert 'VERSION_EQUAL "2.9.0"' in cmake_lists


def test_prepare_script_only_relaxes_torch_gate_with_explicit_flag(tmp_path: Path) -> None:
    source_dir = tmp_path / "vllm-ascend-src"
    _write_fake_vllm_ascend_tree(source_dir)

    result = _run_prepare(source_dir, "--allow-torch-fallback-debug")

    assert result.returncode == 0, result.stderr

    cmake_lists = (source_dir / "CMakeLists.txt").read_text()
    assert "temporary remote debug" in cmake_lists
    assert 'if(${TORCH_VERSION} VERSION_LESS "2.8.0")' in cmake_lists
    assert 'Expected PyTorch version >= 2.8.0' in cmake_lists
    assert 'VERSION_EQUAL "2.9.0"' not in cmake_lists


def test_prepare_script_fails_loudly_when_expected_snippet_is_missing(tmp_path: Path) -> None:
    source_dir = tmp_path / "vllm-ascend-src"
    _write_fake_vllm_ascend_tree(source_dir)
    (source_dir / "setup.py").write_text("import sys\n")

    result = _run_prepare(source_dir)

    assert result.returncode != 0
    assert "Expected snippet not found" in result.stderr


def test_prepare_script_handles_partial_include_replacements(tmp_path: Path) -> None:
    source_dir = tmp_path / "vllm-ascend-src"
    _write_partial_include_tree(source_dir)

    result = _run_prepare(source_dir)

    assert result.returncode == 0, result.stderr
    assert "Patched 5 file(s)." in result.stdout

    assert "include/aclnn/opdev" in (source_dir / "csrc/cmake/config.cmake").read_text()
    assert "include/aclnn/opdev" in (source_dir / "csrc/cmake/intf_pub.cmake").read_text()
    assert "include/aclnn/opdev" in (source_dir / "csrc/moe_grouped_matmul/op_host/CMakeLists.txt").read_text()
    assert "include/toolchain" in (source_dir / "csrc/utils/CMakeLists.txt").read_text()


def test_prepare_script_can_copy_catlass_cache_into_empty_target(tmp_path: Path) -> None:
    source_dir = tmp_path / "vllm-ascend-src"
    catlass_src = tmp_path / "catlass-src"
    target_catlass = source_dir / "csrc/third_party/catlass"
    _write_fake_vllm_ascend_tree(source_dir)
    target_catlass.mkdir(parents=True)
    (catlass_src / "include").mkdir(parents=True)
    (catlass_src / "include/catlass.hpp").write_text("// catlass\n")

    result = _run_prepare(source_dir, "--catlass-source-dir", str(catlass_src))

    assert result.returncode == 0, result.stderr
    assert (target_catlass / "include/catlass.hpp").read_text() == "// catlass\n"


def test_prepare_script_can_create_helper_bin_shims_for_remote_builds(tmp_path: Path) -> None:
    source_dir = tmp_path / "vllm-ascend-src"
    helper_dir = tmp_path / "helper-bin"
    helper_python = tmp_path / "python3.9"
    helper_objdump = tmp_path / "llvm-objdump"

    _write_fake_vllm_ascend_tree(source_dir)
    helper_python.write_text("#!/bin/sh\n")
    helper_objdump.write_text("#!/bin/sh\n")

    result = _run_prepare(
        source_dir,
        "--helper-bin-dir",
        str(helper_dir),
        "--helper-python3",
        str(helper_python),
        "--helper-llvm-objdump",
        str(helper_objdump),
    )

    assert result.returncode == 0, result.stderr
    assert (helper_dir / "python3").is_symlink()
    assert (helper_dir / "python3").resolve() == helper_python.resolve()
    assert (helper_dir / "llvm-objdump").is_symlink()
    assert (helper_dir / "llvm-objdump").resolve() == helper_objdump.resolve()
