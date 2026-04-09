# Ascend Qwen3.5 Test TODO

Current status:

- The shared env is currently back in a coherent fallback state:
  - `/shared/envs/qwen35`
  - `Python 3.10.20`
  - `torch 2.8.0`
  - `torch_npu 2.8.0.post2`
- On `2026-04-08`, the shared env on `.36` now contains:
  - `vllm 0.18.0`
  - `vllm_ascend 0.0.0`
  - `xgrammar 0.1.33`
  - `setuptools-scm 10.0.5`
  - `vcs-versioning 1.1.1`
- `torch_npu` and `vllm` imports currently work only after the Ascend host runtime has been sourced first:
  - `/usr/local/Ascend/ascend-toolkit/set_env.sh`
  - `/usr/local/Ascend/nnal/atb/set_env.sh`
- This is not the final PR `#5682` validation stack.
- The main blocker is still the missing `torch 2.10`-compatible `torch_npu` source for the cluster.
- As of `2026-04-09`, there is now a documented public intermediate path for `CANN 8.5.0`:
  - `torch 2.9.0`
  - `torch_npu 2.9.0`
- That `2.9.0` line does not replace the final `2.10` target, but it is now the preferred next validation path over continuing to stretch fallback `2.8.x`.
- On `2026-04-09`, an isolated `.36` validation env at `/home/zmz/envs/qwen35-t29-lite` confirmed:
  - `torch 2.9.0+cpu` imports
  - `torch_npu 2.9.0` imports after sourcing the Ascend host runtime
  - `torch.npu.is_available()` returns `True`
  - `torch.npu.device_count()` returns `8`
- Later on `2026-04-09`, that same isolated `.36` env also confirmed imports for:
  - `transformers 5.3.0.dev0`
  - `vllm 0.18.0`
  - `vllm_ascend 0.0.0`
- The detailed `.36` continuation handoff for this new `2.9` line is recorded in:
  - `docs/ascend_qwen35_lab/HANDOFF_20260409.md`
- That isolated `2.9` env also showed two undeclared runtime dependencies for minimal `torch_npu` import:
  - `PyYAML 6.0.3`
  - `numpy 1.26.4`
- The later `transformers` and `vllm_ascend` bring-up in that isolated env additionally required:
  - staged offline wheel installation for most `transformers` runtime dependencies
  - `setuptools-scm 10.0.5`
  - `vcs-versioning 1.1.1`
  - `tomli`
  - `pybind11`
- For `.36`, downloading the large `2.9` wheels directly on the host is currently less reliable than downloading them locally and uploading them into `/home/zmz/bootstrap_bundle/python/` for offline install.
- Secondary blocker history:
  - `/shared/dist/vllm-ascend-54879467.tar.gz` was not self-contained for direct fallback installation
  - its extracted `csrc/third_party/catlass` tree could be empty, so `git submodule update` on the tarball path was not a reliable recovery path
  - this is now worked around by the local source-prep helper plus a reusable `catlass` source tarball in `bootstrap_bundle`
- Resolved fallback install chain from `2026-04-08`:
  - `vllm-ascend@54879467` hardcodes CANN include paths such as `include/experiment/platform` and `include/experiment/slog`
  - the `.36` host currently exposes `CANN 8.5.0.B160`, where the matching headers are instead under:
    - `include/aclnn/opdev/platform.h`
    - `include/toolchain/slog.h`
  - source patching those include roots is now captured in `scripts/ascend/prepare_vllm_ascend_source.py`
  - the earlier "standalone succeeds, pip fails" differential was narrowed to:
    - `build_aclnn.sh` can succeed from the shared env if helper scripts resolve `python3` from system `/usr/bin/python3.9`
    - leaving conda `python3.10` first in `PATH` still causes the custom-op stage to fail
  - after forcing system `python3` first in `PATH`, the pip path exposed two more source issues:
    - `setup.py` hardcodes `python3 -m pip show torch-npu`, which mis-resolves `TORCH_NPU_PATH` to `/torch_npu` when `PATH` no longer points at the conda interpreter first
    - `CMakeLists.txt` hard-requires `PyTorch 2.9.0`, while the shared fallback env is still `torch 2.8.0+cpu`
  - the later host-stub phase also needed `llvm-objdump` injected explicitly even though `.36` already had it under:
    - `/usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler/bin/llvm-objdump`
    - `/usr/local/Ascend/cann-8.5.0/tools/ccec_compiler/bin/llvm-objdump`
  - with helper-bin shims for `python3` and `llvm-objdump`, plus `catlass` injection and the fallback debug gate, `.36` successfully built and installed `vllm_ascend`
  - the resulting wheel also carried `libvllm_ascend_kernels.so`, so that shared object is no longer the active blocker
- Current confirmed fallback state:
  - `.36` imports `torch`, `torch_npu`, `transformers`, `vllm`, and `vllm_ascend`
  - `.37` imports `torch`, `torch_npu`, `transformers`, `vllm`, and `vllm_ascend`
  - `/home/zmz/bootstrap_bundle/dist` on `.36` now contains:
    - `transformers-cc7ab9be.tar.gz`
    - `vllm-0.18.0-cp38-abi3-manylinux_2_31_aarch64.whl`
    - `vllm-ascend-54879467.tar.gz`
    - `catlass-src.tar.gz`

## Immediate next actions

- [x] Confirm `/shared/dist/transformers-cc7ab9be.tar.gz` and `/shared/dist/vllm-ascend-54879467.tar.gz` are still present
- [x] Verify the shared env on `.36` still imports `torch`, `torch_npu`, `transformers`, and `vllm` after sourcing Ascend runtime env scripts
- [x] Obtain a complete `vllm-ascend@54879467` source bundle, or vendor the missing `catlass` tree into the extracted source before building on `.36`
- [x] Install `vllm-ascend` into `/shared/envs/qwen35` in controlled mode and re-run the import check
- [x] Run `scripts/ascend/check_qwen35_npu_env.py` from the shared env on `.36` and record the output
- [x] Verify the fallback shared env imports on `.37`, including `vllm_ascend`
- [x] Convert the remote-only `vllm-ascend` source patches into a local repo patch or reproducible local overlay before the next remote retry
  - local helper added: `scripts/ascend/prepare_vllm_ascend_source.py`
  - it can patch the extracted source tree for:
    - CANN include path compatibility
    - `setup.py` `torch_npu` path probing via `sys.executable`
    - optional fallback `torch` gate relaxation under an explicit debug flag
    - optional helper-bin shims for `python3` and `llvm-objdump`
- [x] Capture the narrowest reproducible workaround for the `build_aclnn` helper-script `python3` dependency under the conda env
- [x] Isolate why the later `vllm_ascend_C` build still loses `llvm-objdump` after the custom-op stage succeeds
- [ ] Decide whether the hard `torch 2.9.0` gate in `vllm-ascend@54879467` should be treated as expected upstream policy or as a patchable blocker for the current `torch 2.8` fallback line
- [ ] Create a fresh isolated `2.9.0` validation env on `.36`; do not mutate `/shared/envs/qwen35` yet
- [x] In an isolated `.36` env, confirm minimal imports for:
  - `torch`
  - `torch_npu`
- [x] Record whether public `torch 2.9.0` + `torch_npu 2.9.0` really works against the current host runtime without extra private artifacts
- [x] In that isolated env, continue upward and confirm minimal imports for:
  - `transformers`
  - `vllm`
  - `vllm_ascend`
- [ ] Confirm cluster base image versions on both hosts: `CANN`, driver, firmware, `torch`, `torch_npu`
- [x] Snapshot the current package state with `python -m pip show` for key fallback packages on `.36`
- [ ] Ask Huawei or the cluster owner for a `torch 2.10`-compatible `torch_npu` wheel, image, or prebuilt Python env
- [ ] Decide whether to run a minimal fallback smoke now that `.36/.37` both import `vllm_ascend`, or stop and wait for the proper `torch 2.10` line

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
