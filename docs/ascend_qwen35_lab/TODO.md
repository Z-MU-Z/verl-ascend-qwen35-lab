# Ascend Qwen3.5 Test TODO

Current status:

- The shared env is currently back in a coherent fallback state:
  - `/shared/envs/qwen35`
  - `Python 3.10.20`
  - `torch 2.8.0`
  - `torch_npu 2.8.0.post2`
- On `2026-04-07`, `.36` was rechecked and the shared env also contains:
  - `vllm 0.18.0`
  - `xgrammar 0.1.33`
  - `setuptools-scm 10.0.5`
  - `vcs-versioning 1.1.1`
- `torch_npu` and `vllm` imports currently work only after the Ascend host runtime has been sourced first:
  - `/usr/local/Ascend/ascend-toolkit/set_env.sh`
  - `/usr/local/Ascend/nnal/atb/set_env.sh`
- This is not the final PR `#5682` validation stack.
- The main blocker is still the missing `torch 2.10`-compatible `torch_npu` source for the cluster.
- Secondary blocker: `/shared/dist/vllm-ascend-54879467.tar.gz` is not self-contained. Its build expects the missing `csrc/third_party/catlass` submodule and fails when `git submodule update` is unavailable.
- `catlass` has now been vendored successfully into a temporary source tree on `.36`, so that original tarball completeness issue is no longer the active blocker.
- New active blocker chain from `2026-04-07`:
  - `vllm-ascend@54879467` hardcodes CANN include paths such as `include/experiment/platform` and `include/experiment/slog`
  - the `.36` host currently exposes `CANN 8.5.0.B160`, where the matching headers are instead under:
    - `include/aclnn/opdev/platform.h`
    - `include/toolchain/slog.h`
  - a temporary source-only patch replacing those include roots allowed `bash csrc/build_aclnn.sh ...` to succeed on `.36`
  - the earlier "standalone succeeds, pip fails" differential is now narrower:
    - `build_aclnn.sh` can succeed from the shared env if helper scripts resolve `python3` from system `/usr/bin/python3.9`
    - leaving conda `python3.10` first in `PATH` still causes the custom-op stage to fail
  - after forcing system `python3` first in `PATH`, the pip path moved past the custom-op failure and exposed two more source issues:
    - `setup.py` hardcodes `python3 -m pip show torch-npu`, which mis-resolves `TORCH_NPU_PATH` to `/torch_npu` when `PATH` no longer points at the conda interpreter first
    - `CMakeLists.txt` hard-requires `PyTorch 2.9.0`, while the shared fallback env is still `torch 2.8.0+cpu`
  - after temporarily relaxing both of those source-only gates just to probe deeper, the main `vllm_ascend_C` build progressed substantially further and then failed in Ascend host-stub extraction with:
    - `FileNotFoundError: [Errno 2] No such file or directory: 'llvm-objdump'`
  - `.36` does have `llvm-objdump` under:
    - `/usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler/bin/llvm-objdump`
    - `/usr/local/Ascend/cann-8.5.0/tools/ccec_compiler/bin/llvm-objdump`
  - so the newest active debugging target is why that later build stage still loses `llvm-objdump` even though the Ascend env exposes it

## Immediate next actions

- [x] Confirm `/shared/dist/transformers-cc7ab9be.tar.gz` and `/shared/dist/vllm-ascend-54879467.tar.gz` are still present
- [x] Verify the shared env on `.36` still imports `torch`, `torch_npu`, `transformers`, and `vllm` after sourcing Ascend runtime env scripts
- [x] Obtain a complete `vllm-ascend@54879467` source bundle, or vendor the missing `catlass` tree into the extracted source before building on `.36`
- [ ] Install `vllm-ascend` into `/shared/envs/qwen35` in controlled mode and re-run the import check
- [x] Convert the remote-only `vllm-ascend` source patches into a local repo patch or reproducible local overlay before the next remote retry
  - local helper added: `scripts/ascend/prepare_vllm_ascend_source.py`
  - it can patch the extracted source tree for:
    - CANN include path compatibility
    - `setup.py` `torch_npu` path probing via `sys.executable`
    - optional fallback `torch` gate relaxation under an explicit debug flag
    - optional helper-bin shims for `python3` and `llvm-objdump`
- [ ] Capture the narrowest reproducible workaround for the `build_aclnn` helper-script `python3` dependency under the conda env
- [ ] Isolate why the later `vllm_ascend_C` build still loses `llvm-objdump` after the custom-op stage succeeds
- [ ] Decide whether the hard `torch 2.9.0` gate in `vllm-ascend@54879467` should be treated as expected upstream policy or as a patchable blocker for the current `torch 2.8` fallback line
- [ ] Run `scripts/ascend/check_qwen35_npu_env.py` from the shared env on both hosts and record the output
- [ ] Confirm cluster base image versions on both hosts: `CANN`, driver, firmware, `torch`, `torch_npu`
- [ ] Snapshot the current package state with `python -m pip list`
- [ ] Ask Huawei or the cluster owner for a `torch 2.10`-compatible `torch_npu` wheel, image, or prebuilt Python env

## After the blocker is resolved

- [ ] Rebuild `/shared/envs/qwen35` on top of the final PR matrix
- [ ] Verify the pinned Python stack installs cleanly with `scripts/ascend/bootstrap_qwen35_npu_env.sh` or equivalent local-tarball installs
- [ ] Run `scripts/ascend/check_qwen35_npu_env.py` again and save the output
- [ ] Run a single-node smoke test with safe defaults before any two-host work
- [ ] Prefer a smaller Qwen3.5 checkpoint for the first launch sanity check if model download or memory pressure is still uncertain
- [ ] Run the 27B smoke script with `use_remove_padding=False`

## After the first successful smoke

- [ ] Save the first successful log under `logs/`
- [ ] Fill in `results/templates/session_report.md` for the smoke run
- [ ] If startup fails, compare against `docs/ascend_qwen35_lab/KNOWN_ISSUES.md`
- [ ] If startup succeeds, run a longer 1-epoch pass with the same safe defaults
- [ ] Measure memory headroom before changing any performance knobs
- [ ] Try `gpu_memory_utilization` tuning only after a stable baseline exists
- [ ] Try `use_remove_padding=True` only as an explicit experiment, not as the default
- [ ] Try scaling beyond one node only after the single-node baseline is stable
- [ ] Record precision observations against a GPU reference run if available
- [ ] Open follow-up issues or patches back to upstream `verl` if new NPU-specific blockers appear
