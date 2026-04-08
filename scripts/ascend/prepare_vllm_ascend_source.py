#!/usr/bin/env python3
"""Apply local debug-only source patches to an extracted vllm-ascend tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


class PatchError(RuntimeError):
    """Raised when the expected upstream source layout has drifted."""


SETUP_OLD = """        torch_npu_command = "python3 -m pip show torch-npu | grep '^Location:' | awk '{print $2}'"
        try:
            torch_npu_path = subprocess.check_output(torch_npu_command, shell=True).decode().strip()
            torch_npu_path += "/torch_npu"
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Retrieve torch version version failed: {e}")
"""

SETUP_NEW = """        try:
            torch_npu_show = subprocess.check_output([sys.executable, "-m", "pip", "show", "torch-npu"]).decode().splitlines()
            torch_npu_location = next(line.split(":", 1)[1].strip() for line in torch_npu_show if line.startswith("Location:"))
            torch_npu_path = torch_npu_location + "/torch_npu"
        except Exception as e:
            raise RuntimeError(f"Retrieve torch_npu path failed: {e}")
"""

CMAKE_TORCH_GATE_OLD = """# check torch version is 2.9.0
if(NOT ${TORCH_VERSION} VERSION_EQUAL "2.9.0")
  message(FATAL_ERROR "Expected PyTorch version 2.9.0, but found ${TORCH_VERSION}")
endif()
"""

CMAKE_TORCH_GATE_NEW = """# temporary remote debug: allow torch 2.8.x+ to test how far build proceeds
if(${TORCH_VERSION} VERSION_LESS "2.8.0")
  message(FATAL_ERROR "Expected PyTorch version >= 2.8.0, but found ${TORCH_VERSION}")
endif()
"""

INCLUDE_REPLACEMENTS = {
    "include/experiment/platform": "include/aclnn/opdev",
    "include/experiment/slog": "include/toolchain",
}

INCLUDE_FILES = [
    "csrc/cmake/config.cmake",
    "csrc/cmake/intf_pub.cmake",
    "csrc/moe_grouped_matmul/op_host/CMakeLists.txt",
    "csrc/utils/CMakeLists.txt",
]


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if old not in text:
        raise PatchError(f"Expected snippet not found in {path}")
    updated = text.replace(old, new, 1)
    if updated == text:
        return False
    path.write_text(updated)
    return True


def replace_all(path: Path, replacements: dict[str, str]) -> bool:
    text = path.read_text()
    updated = text
    changed = False
    for old, new in replacements.items():
        if old not in updated:
            raise PatchError(f"Expected snippet not found in {path}: {old}")
        next_text = updated.replace(old, new)
        changed = changed or next_text != updated
        updated = next_text
    if changed:
        path.write_text(updated)
    return changed


def apply_patches(source_dir: Path, allow_torch_fallback_debug: bool) -> int:
    patched_files = 0

    if replace_once(source_dir / "setup.py", SETUP_OLD, SETUP_NEW):
        patched_files += 1

    for rel_path in INCLUDE_FILES:
        if replace_all(source_dir / rel_path, INCLUDE_REPLACEMENTS):
            patched_files += 1

    if allow_torch_fallback_debug and replace_once(
        source_dir / "CMakeLists.txt",
        CMAKE_TORCH_GATE_OLD,
        CMAKE_TORCH_GATE_NEW,
    ):
        patched_files += 1

    return patched_files


def create_helper_symlink(helper_bin_dir: Path, link_name: str, target: Path | None) -> None:
    if target is None:
        return
    if not target.exists():
        raise PatchError(f"Helper target does not exist: {target}")
    helper_bin_dir.mkdir(parents=True, exist_ok=True)
    link_path = helper_bin_dir / link_name
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Path to an extracted vllm-ascend source tree.",
    )
    parser.add_argument(
        "--allow-torch-fallback-debug",
        action="store_true",
        help="Relax the hard torch 2.9.0 CMake gate for fallback debug-only experiments.",
    )
    parser.add_argument(
        "--helper-bin-dir",
        type=Path,
        help="Optional directory where helper shims such as python3 and llvm-objdump should be created.",
    )
    parser.add_argument(
        "--helper-python3",
        type=Path,
        help="Optional path used to create a helper-bin python3 shim for remote builds.",
    )
    parser.add_argument(
        "--helper-llvm-objdump",
        type=Path,
        help="Optional path used to create a helper-bin llvm-objdump shim for remote builds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        patched_files = apply_patches(
            source_dir=args.source_dir.resolve(),
            allow_torch_fallback_debug=args.allow_torch_fallback_debug,
        )
        if args.helper_bin_dir is not None:
            helper_bin_dir = args.helper_bin_dir.resolve()
            create_helper_symlink(helper_bin_dir, "python3", args.helper_python3)
            create_helper_symlink(helper_bin_dir, "llvm-objdump", args.helper_llvm_objdump)
    except PatchError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Patched {patched_files} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
