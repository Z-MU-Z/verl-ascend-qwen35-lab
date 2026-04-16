import importlib.util
import logging
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


def test_patch_vllm_ascend_gemma_rms_norm_fallback_logs_when_enabled(caplog):
    patch_module = _load_patch_module()

    class FakeAscendGemmaRMSNorm:
        def __init__(self):
            self.weight = FakeWeight("weight")
            self.variance_epsilon = 1e-6

        def forward_oot(self, x, residual=None):
            return ("original", x, residual)

    fake_layernorm_module = SimpleNamespace(AscendGemmaRMSNorm=FakeAscendGemmaRMSNorm)
    fake_torch_module = SimpleNamespace(ops=SimpleNamespace(_C_ascend=SimpleNamespace()))
    fake_torch_npu_module = SimpleNamespace(npu_rms_norm=lambda *args, **kwargs: (FakeTensor("norm_out"), None))

    with caplog.at_level(logging.INFO):
        patched = patch_module.patch_vllm_ascend_gemma_rms_norm_fallback(
            layernorm_module=fake_layernorm_module,
            torch_module=fake_torch_module,
            torch_npu_module=fake_torch_npu_module,
        )

    assert patched is True
    assert "Applied Ascend Gemma RMSNorm fallback" in caplog.text
    assert "npu_gemma_rms_norm" in caplog.text


def test_patch_vllm_ascend_gemma_rms_norm_fallback_logs_when_native_op_exists(caplog):
    patch_module = _load_patch_module()

    class FakeAscendGemmaRMSNorm:
        def forward_oot(self, x, residual=None):
            return ("original", x, residual)

    fake_layernorm_module = SimpleNamespace(AscendGemmaRMSNorm=FakeAscendGemmaRMSNorm)
    fake_torch_module = SimpleNamespace(
        ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_gemma_rms_norm=object()))
    )
    fake_torch_npu_module = SimpleNamespace()

    with caplog.at_level(logging.INFO):
        patched = patch_module.patch_vllm_ascend_gemma_rms_norm_fallback(
            layernorm_module=fake_layernorm_module,
            torch_module=fake_torch_module,
            torch_npu_module=fake_torch_npu_module,
        )

    assert patched is False
    assert "Skipped Ascend Gemma RMSNorm fallback" in caplog.text
    assert "native op exists" in caplog.text


def test_patch_vllm_ascend_custom_op_disable_primes_cached_flag():
    patch_module = _load_patch_module()
    fake_utils_module = SimpleNamespace(_CUSTOM_OP_ENABLED=None)

    patched = patch_module.patch_vllm_ascend_custom_op_disable(utils_module=fake_utils_module)

    assert patched is True
    assert fake_utils_module._CUSTOM_OP_ENABLED is False


def test_patch_vllm_ascend_causal_conv1d_fallback_registers_python_op_when_missing():
    patch_module = _load_patch_module()
    calls = {}

    def fake_fallback_runner(
        x,
        weight,
        *,
        conv_state,
        bias_opt,
        query_start_loc,
        cache_indices,
        has_initial_state,
        num_accepted_tokens,
        activation_mode,
        pad_slot_id,
    ):
        calls["x"] = x
        calls["weight"] = weight
        calls["conv_state"] = conv_state
        calls["bias_opt"] = bias_opt
        calls["query_start_loc"] = query_start_loc
        calls["cache_indices"] = cache_indices
        calls["has_initial_state"] = has_initial_state
        calls["num_accepted_tokens"] = num_accepted_tokens
        calls["activation_mode"] = activation_mode
        calls["pad_slot_id"] = pad_slot_id
        return "fallback-output"

    fake_torch_module = SimpleNamespace(
        bool="bool_dtype",
        int32="int32_dtype",
        ops=SimpleNamespace(_C_ascend=SimpleNamespace()),
    )
    tensor_calls = []

    def fake_tensor(values, device=None, dtype=None):
        values = list(values)
        tensor_calls.append({"values": values, "device": device, "dtype": dtype})
        return FakeIndexTensor(values, device=device, dtype=dtype)

    fake_torch_module.tensor = fake_tensor

    patched = patch_module.patch_vllm_ascend_causal_conv1d_fallback(
        torch_module=fake_torch_module,
        fallback_runner=fake_fallback_runner,
    )

    assert patched is True
    assert hasattr(fake_torch_module.ops._C_ascend, "npu_causal_conv1d_custom")

    conv_state = FakeConvState()
    output = fake_torch_module.ops._C_ascend.npu_causal_conv1d_custom(
        FakeInputTensor("hidden", device="npu:0"),
        FakeTransposeTensor("weight_t"),
        conv_state=conv_state,
        bias_opt="bias",
        query_start_loc_opt=(0, 3, 5),
        cache_indices_opt=(7, 9),
        initial_state_mode_opt=(True, False),
        num_accepted_tokens_opt=[],
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=0,
    )

    assert output == "fallback-output"
    assert calls["x"].name == "hidden"
    assert calls["weight"].name == "weight_t"
    assert calls["conv_state"] is conv_state
    assert calls["bias_opt"] == "bias"
    assert calls["query_start_loc"].values == [0, 3, 5]
    assert calls["cache_indices"].values == [7, 9]
    assert calls["has_initial_state"].values == [True, False]
    assert calls["num_accepted_tokens"] is None
    assert calls["activation_mode"] == 1
    assert calls["pad_slot_id"] == -1
    assert conv_state.zeroed == [9]
    assert tensor_calls == [
        {"values": [7, 9], "device": "npu:0", "dtype": "int32_dtype"},
        {"values": [0, 3, 5], "device": "npu:0", "dtype": "int32_dtype"},
        {"values": [True, False], "device": "npu:0", "dtype": "bool_dtype"},
    ]


def test_patch_vllm_ascend_causal_conv1d_fallback_skips_when_op_exists():
    patch_module = _load_patch_module()
    original = object()
    fake_torch_module = SimpleNamespace(
        ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_causal_conv1d_custom=original))
    )

    patched = patch_module.patch_vllm_ascend_causal_conv1d_fallback(
        torch_module=fake_torch_module,
        fallback_runner=lambda *args, **kwargs: None,
    )

    assert patched is False
    assert fake_torch_module.ops._C_ascend.npu_causal_conv1d_custom is original


def test_patch_vllm_ascend_causal_conv1d_fallback_logs_when_enabled(caplog):
    patch_module = _load_patch_module()
    fake_torch_module = SimpleNamespace(
        bool="bool_dtype",
        int32="int32_dtype",
        ops=SimpleNamespace(_C_ascend=SimpleNamespace()),
    )
    fake_torch_module.tensor = lambda *args, **kwargs: None

    with caplog.at_level(logging.INFO):
        patched = patch_module.patch_vllm_ascend_causal_conv1d_fallback(
            torch_module=fake_torch_module,
            fallback_runner=lambda *args, **kwargs: "fallback-output",
        )

    assert patched is True
    assert "Applied Ascend causal_conv1d fallback" in caplog.text
    assert "npu_causal_conv1d_custom" in caplog.text


def test_patch_vllm_ascend_causal_conv1d_fallback_logs_when_native_op_exists(caplog):
    patch_module = _load_patch_module()
    original = object()
    fake_torch_module = SimpleNamespace(
        ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_causal_conv1d_custom=original))
    )

    with caplog.at_level(logging.INFO):
        patched = patch_module.patch_vllm_ascend_causal_conv1d_fallback(
            torch_module=fake_torch_module,
            fallback_runner=lambda *args, **kwargs: None,
        )

    assert patched is False
    assert "Skipped Ascend causal_conv1d fallback" in caplog.text
    assert "native op exists" in caplog.text


class FakeTransposeTensor:
    def __init__(self, name: str):
        self.name = name

    def transpose(self, dim0, dim1):
        return f"{self.name}.transpose({dim0},{dim1})"


class FakeInputTensor:
    def __init__(self, name: str, device: str):
        self.name = name
        self.device = device


class FakeMask:
    def __init__(self, values):
        self.values = list(values)

    def any(self):
        return any(self.values)


class FakeIndexTensor:
    def __init__(self, values, device=None, dtype=None):
        self.values = list(values)
        self.device = device
        self.dtype = dtype

    def numel(self):
        return len(self.values)

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __invert__(self):
        return FakeMask([not value for value in self.values])

    def __getitem__(self, item):
        if isinstance(item, FakeMask):
            return FakeIndexTensor([value for value, keep in zip(self.values, item.values) if keep])
        return self.values[item]


class FakeConvState:
    def __init__(self):
        self.zeroed = []

    def transpose(self, dim0, dim1):
        return f"conv_state.transpose({dim0},{dim1})"

    def __setitem__(self, item, value):
        assert value == 0
        self.zeroed.extend(item.values)
