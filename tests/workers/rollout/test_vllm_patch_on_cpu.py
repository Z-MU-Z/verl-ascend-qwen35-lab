import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_patch_module():
    module_path = Path(__file__).resolve().parents[3] / "verl" / "utils" / "vllm" / "patch.py"
    spec = importlib.util.spec_from_file_location("verl_utils_vllm_patch", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(self, name: str):
        self.name = name

    def float(self):
        return FakeTensor(f"{self.name}.float")

    def type_as(self, other):
        return FakeTensor(f"{self.name}.type_as({other.name})")


class FakeWeight:
    def __init__(self, name: str):
        self.name = name

    def float(self):
        return self

    def __radd__(self, other):
        return f"{other}+{self.name}"


def test_patch_vllm_ascend_gemma_rms_norm_fallback_uses_npu_rms_norm_when_op_missing():
    patch_module = _load_patch_module()
    calls = {}

    class FakeAscendGemmaRMSNorm:
        def __init__(self):
            self.weight = FakeWeight("weight")
            self.variance_epsilon = 1e-6

        def forward_oot(self, x, residual=None):
            return ("original", x, residual)

    def fake_npu_rms_norm(x, weight, epsilon):
        calls["x"] = x.name
        calls["weight"] = weight
        calls["epsilon"] = epsilon
        return FakeTensor("norm_out"), None

    fake_layernorm_module = SimpleNamespace(AscendGemmaRMSNorm=FakeAscendGemmaRMSNorm)
    fake_torch_module = SimpleNamespace(ops=SimpleNamespace(_C_ascend=SimpleNamespace()))
    fake_torch_npu_module = SimpleNamespace(npu_rms_norm=fake_npu_rms_norm)

    patched = patch_module.patch_vllm_ascend_gemma_rms_norm_fallback(
        layernorm_module=fake_layernorm_module,
        torch_module=fake_torch_module,
        torch_npu_module=fake_torch_npu_module,
    )

    assert patched is True

    module = FakeAscendGemmaRMSNorm()
    output = module.forward_oot(FakeTensor("hidden"))

    assert output.name == "norm_out.type_as(hidden)"
    assert calls == {
        "x": "hidden.float",
        "weight": "1.0+weight",
        "epsilon": 1e-6,
    }


def test_patch_vllm_ascend_gemma_rms_norm_fallback_preserves_residual_path():
    patch_module = _load_patch_module()

    class FakeAscendGemmaRMSNorm:
        def __init__(self):
            self.weight = FakeWeight("weight")
            self.variance_epsilon = 1e-6

        def forward_oot(self, x, residual=None):
            return ("original", x.name, residual)

    fake_layernorm_module = SimpleNamespace(AscendGemmaRMSNorm=FakeAscendGemmaRMSNorm)
    fake_torch_module = SimpleNamespace(ops=SimpleNamespace(_C_ascend=SimpleNamespace()))
    fake_torch_npu_module = SimpleNamespace(npu_rms_norm=lambda *args, **kwargs: None)

    patch_module.patch_vllm_ascend_gemma_rms_norm_fallback(
        layernorm_module=fake_layernorm_module,
        torch_module=fake_torch_module,
        torch_npu_module=fake_torch_npu_module,
    )

    module = FakeAscendGemmaRMSNorm()
    assert module.forward_oot(FakeTensor("hidden"), residual="residual") == ("original", "hidden", "residual")


def test_patch_vllm_ascend_gemma_rms_norm_fallback_skips_when_op_exists():
    patch_module = _load_patch_module()

    class FakeAscendGemmaRMSNorm:
        def forward_oot(self, x, residual=None):
            return ("original", x, residual)

    fake_layernorm_module = SimpleNamespace(AscendGemmaRMSNorm=FakeAscendGemmaRMSNorm)
    original_forward = FakeAscendGemmaRMSNorm.forward_oot
    fake_torch_module = SimpleNamespace(
        ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_gemma_rms_norm=object()))
    )
    fake_torch_npu_module = SimpleNamespace()

    patched = patch_module.patch_vllm_ascend_gemma_rms_norm_fallback(
        layernorm_module=fake_layernorm_module,
        torch_module=fake_torch_module,
        torch_npu_module=fake_torch_npu_module,
    )

    assert patched is False
    assert FakeAscendGemmaRMSNorm.forward_oot is original_forward
