# New Machine Fallback Bootstrap

## Goal

Bring up the current Qwen3.5 Ascend fallback debug stack on a new NPU machine with the few hard-to-install artifacts staged ahead of time, while leaving ordinary pip packages to install from the configured indexes.

This document is for the current fallback line only:

- `torch 2.8.x`
- `torch_npu 2.8.x`
- `transformers@cc7ab9be`
- `vllm==0.18.0`
- `vllm-ascend@54879467`

It is not the final `torch 2.10` PR `#5682` validation matrix.

## Recommended staging model

Assemble the bundle on `huawei36` first, then copy it to the new machine.

Recommended layout:

```text
bootstrap_bundle/
  dist/
    transformers-cc7ab9be.tar.gz
    vllm-0.18.0-<platform>.whl
    vllm-ascend-54879467.tar.gz
    catlass-src.tar.gz
  python/
    # optional but strongly recommended on fresh hosts
    Miniforge3-<platform>.sh
    torch-2.8.0-<platform>.whl
    torch_npu-2.8.0.post2-<platform>.whl
    triton_ascend-<platform>.whl
```

Rules of thumb:

- put hard packages in `bootstrap_bundle/`
- keep ordinary pip packages out of the bundle unless the new machine cannot reach your package indexes
- keep the bundle small and focused on the things that were already shown to be brittle

## What counts as a hard package here

For this lab, the hard packages are:

- `transformers-cc7ab9be.tar.gz`
- `vllm-0.18.0-*.whl`
- `vllm-ascend-54879467.tar.gz`
- `catlass-src.tar.gz`

The base runtime wheels are optional in the bundle, but highly recommended for a new host:

- `torch 2.8.x`
- `torch_npu 2.8.x`
- `triton-ascend`

Ordinary pip packages can still be installed online from `requirements-npu.txt` and normal package indexes.

## Base machine assumptions

This bootstrap flow can verify the machine and optionally install Python wheels, but it does not try to fully automate system-side driver or firmware deployment.

The target machine should already have or be prepared to receive:

- Ascend driver and firmware
- CANN toolkit
- `npu-smi`
- a working Python 3.10 interpreter, or a Miniforge installer available in the bundle

Expected host runtime scripts:

- `/usr/local/Ascend/ascend-toolkit/set_env.sh`
- `/usr/local/Ascend/nnal/atb/set_env.sh`

## Bundle preparation on `huawei36`

Keep the bundle on `.36` while iterating, then copy it to the new machine when needed.

Example:

```bash
mkdir -p /home/zmz/bootstrap_bundle/dist /home/zmz/bootstrap_bundle/python

# Place the hard packages here.
ls -lh /home/zmz/bootstrap_bundle/dist
```

If the new host should reuse the same artifacts, copy the bundle over with `scp` or `rsync`.

## One-click fallback bootstrap

Use:

```bash
bash scripts/ascend/bootstrap_new_npu_machine_fallback.sh
```

The script will:

1. verify the Ascend host runtime basics
2. create or reuse the target Python environment
3. optionally install base wheels from `bootstrap_bundle/python/`
4. install ordinary pip packages from `requirements-npu.txt`
5. install the hard packages from `bootstrap_bundle/dist/`
6. extract and patch `vllm-ascend`
7. inject `catlass`
8. install `vllm_ascend`
9. run the verification script

## Verification

Run:

```bash
bash scripts/ascend/verify_new_npu_machine.sh
```

Expected success signals:

- `npu-smi` is available
- Ascend env scripts are present
- `torch`, `torch_npu`, `transformers`, `vllm`, and `vllm_ascend` import

## Important caveats

- This flow is for the fallback debug line, not the final PR matrix.
- `vllm-ascend` still needs source patching during fallback bring-up.
- If the base runtime wheels are missing from the bundle, the script can only verify them, not conjure them.
- If the machine cannot reach pip indexes, mirror more wheels into `bootstrap_bundle/`.
