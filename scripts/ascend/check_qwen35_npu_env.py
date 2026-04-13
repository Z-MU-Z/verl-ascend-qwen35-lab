#!/usr/bin/env python3

from __future__ import annotations

import importlib
import sys
from importlib.metadata import PackageNotFoundError, version

REQUIRED_ENV_PATH = "/home/zmz/envs/qwen35-t29-lite"

PINS = {
    "transformers": "git+https://github.com/huggingface/transformers.git@cc7ab9be",
    "vllm": "0.18.0",
    "vllm-ascend": "git+https://github.com/vllm-project/vllm-ascend.git@54879467",
}

OPTIONAL_PACKAGES = [
    "torch",
    "torch_npu",
    "transformers",
    "vllm",
    "ray",
    "tensordict",
    "triton_ascend",
]


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def module_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return "missing"

    return getattr(module, "__version__", "installed")


def main() -> int:
    active_python = sys.executable
    print(f"Required lab env: {REQUIRED_ENV_PATH}")
    print(f"Active Python: {active_python}")
    if not active_python.startswith(REQUIRED_ENV_PATH):
        print("ERROR: active Python is not the required lab env.")
        print("Refusing to continue with deprecated shared/fallback environments.")
        return 2

    print("Expected pins for this lab:")
    for name, pin in PINS.items():
        print(f"  - {name}: {pin}")

    print("\nDetected runtime packages:")
    for name in OPTIONAL_PACKAGES:
        print(f"  - {name}: {package_version(name)}")

    torch_version = module_version("torch")
    torch_npu_version = module_version("torch_npu")
    print("\nImported modules:")
    print(f"  - torch: {torch_version}")
    print(f"  - torch_npu: {torch_npu_version}")

    print("\nNotes:")
    print("  - This script does not install CANN or torch_npu.")
    print("  - If torch_npu is missing, prepare the cluster base image first.")
    print("  - If transformers or vllm do not match the pins above, reproducibility is lower.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
