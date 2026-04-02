# Ascend Qwen3.5 Test TODO

- [ ] Confirm cluster base image versions: CANN, torch, torch_npu, driver, firmware
- [ ] Run `scripts/ascend/check_qwen35_npu_env.py` and record the output
- [ ] Verify the pinned Python stack installs cleanly with `scripts/ascend/bootstrap_qwen35_npu_env.sh`
- [ ] Run the smoke script on 27B with `use_remove_padding=False`
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
