# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To support different vLLM versions, we add the model into SUPPORTED_MOE_MODELS separately to avoid triggering
# unsupported issues.
SUPPORTED_MOE_MODELS = []

try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM

    SUPPORTED_MOE_MODELS.append(DeepseekV2ForCausalLM)
    SUPPORTED_MOE_MODELS.append(DeepseekV3ForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.mixtral import MixtralForCausalLM

    SUPPORTED_MOE_MODELS.append(MixtralForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeLLMForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3NextForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration

    SUPPORTED_MOE_MODELS.append(KimiVLForConditionalGeneration)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3_5MoeForCausalLM)
except ImportError:
    pass


def patch_vllm_ascend_gemma_rms_norm_fallback(layernorm_module=None, torch_module=None, torch_npu_module=None):
    """Fallback to torch_npu.npu_rms_norm when vllm_ascend expects a missing Gemma custom op."""
    if layernorm_module is None:
        try:
            import vllm_ascend.ops.layernorm as layernorm_module
        except ImportError:
            return False

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return False

    if torch_npu_module is None:
        try:
            import torch_npu as torch_npu_module
        except ImportError:
            return False

    ascend_ops = getattr(getattr(torch_module, "ops", None), "_C_ascend", None)
    if ascend_ops is None or hasattr(ascend_ops, "npu_gemma_rms_norm"):
        return False

    ascend_gemma_rms_norm = getattr(layernorm_module, "AscendGemmaRMSNorm", None)
    if ascend_gemma_rms_norm is None:
        return False

    original_forward = ascend_gemma_rms_norm.forward_oot
    if getattr(original_forward, "_verl_npu_gemma_fallback", False):
        return True

    def forward_oot_with_fallback(self, x, residual=None):
        if residual is not None:
            return original_forward(self, x, residual=residual)
        normalized, _ = torch_npu_module.npu_rms_norm(x.float(), 1.0 + self.weight.float(), self.variance_epsilon)
        return normalized.type_as(x)

    forward_oot_with_fallback._verl_npu_gemma_fallback = True
    ascend_gemma_rms_norm.forward_oot = forward_oot_with_fallback
    return True


def patch_vllm_ascend_custom_op_disable(utils_module=None):
    """Prime vllm-ascend custom-op cache to avoid first import under Dynamo."""
    if utils_module is None:
        try:
            import vllm_ascend.utils as utils_module
        except ImportError:
            return False

    if not hasattr(utils_module, "_CUSTOM_OP_ENABLED"):
        return False

    utils_module._CUSTOM_OP_ENABLED = False
    return True


def _build_causal_conv1d_ref_runner(torch_module, causal_conv1d_ref):
    def run_causal_conv1d_ref_fallback(
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
        del num_accepted_tokens
        if query_start_loc is None or cache_indices is None:
            raise RuntimeError("causal_conv1d custom-op fallback requires varlen query_start_loc and cache_indices")

        native_weight = weight.transpose(0, 1).contiguous()
        native_conv_state = conv_state.transpose(-1, -2).contiguous()
        activation = "silu" if activation_mode == 1 else None
        outputs = []

        for idx in range(len(cache_indices)):
            cache_index = cache_indices[idx]
            if cache_index == pad_slot_id:
                continue

            start = query_start_loc[idx]
            end = query_start_loc[idx + 1]
            seq = x[start:end].transpose(0, 1).unsqueeze(0)
            initial_states = None
            if has_initial_state is None or has_initial_state[idx]:
                initial_states = native_conv_state[cache_index].unsqueeze(0)

            out, final_state = causal_conv1d_ref(
                seq,
                native_weight,
                bias=bias_opt,
                initial_states=initial_states,
                return_final_states=True,
                activation=activation,
            )
            native_conv_state[cache_index].copy_(final_state.squeeze(0))
            outputs.append(out.squeeze(0).transpose(0, 1))

        if not outputs:
            return x
        return torch_module.cat(outputs, dim=0)

    return run_causal_conv1d_ref_fallback


def patch_vllm_ascend_causal_conv1d_fallback(torch_module=None, causal_conv1d_module=None, fallback_runner=None):
    """Install a Python fallback for missing npu_causal_conv1d_custom in torch.ops._C_ascend."""
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return False

    ascend_ops = getattr(getattr(torch_module, "ops", None), "_C_ascend", None)
    if ascend_ops is None or hasattr(ascend_ops, "npu_causal_conv1d_custom"):
        return False

    if getattr(ascend_ops, "_verl_npu_causal_conv1d_fallback", False):
        return True

    if fallback_runner is None:
        if causal_conv1d_module is None:
            try:
                import vllm_ascend.ops.triton.mamba.causal_conv1d as causal_conv1d_module
            except ImportError:
                return False

        causal_conv1d_ref = getattr(causal_conv1d_module, "causal_conv1d_ref", None)
        if causal_conv1d_ref is None:
            return False
        fallback_runner = _build_causal_conv1d_ref_runner(torch_module, causal_conv1d_ref)

    def _to_tensor_1d(values, *, device, dtype):
        if values is None:
            return None
        if hasattr(values, "numel"):
            return values
        if len(values) == 0:
            return None
        return torch_module.tensor(list(values), device=device, dtype=dtype)

    def npu_causal_conv1d_custom_fallback(
        x,
        weight,
        *,
        conv_state,
        bias_opt=None,
        query_start_loc_opt=(),
        cache_indices_opt=(),
        initial_state_mode_opt=(),
        num_accepted_tokens_opt=(),
        activation_mode=0,
        pad_slot_id=-1,
        run_mode=0,
    ):
        del run_mode
        conv_state_indices = _to_tensor_1d(cache_indices_opt, device=x.device, dtype=torch_module.int32)
        query_start_loc = _to_tensor_1d(query_start_loc_opt, device=x.device, dtype=torch_module.int32)
        has_initial_state = _to_tensor_1d(initial_state_mode_opt, device=x.device, dtype=torch_module.bool)
        num_accepted_tokens = _to_tensor_1d(num_accepted_tokens_opt, device=x.device, dtype=torch_module.int32)

        if conv_state_indices is not None and has_initial_state is not None:
            missing_initial_state = ~has_initial_state
            if missing_initial_state.any():
                conv_state[conv_state_indices[missing_initial_state]] = 0

        return fallback_runner(
            x,
            weight,
            conv_state=conv_state,
            bias_opt=bias_opt,
            query_start_loc=query_start_loc,
            cache_indices=conv_state_indices,
            has_initial_state=has_initial_state,
            num_accepted_tokens=num_accepted_tokens,
            activation_mode=activation_mode,
            pad_slot_id=pad_slot_id,
        )

    npu_causal_conv1d_custom_fallback._verl_npu_causal_conv1d_fallback = True
    setattr(ascend_ops, "npu_causal_conv1d_custom", npu_causal_conv1d_custom_fallback)
    setattr(ascend_ops, "_verl_npu_causal_conv1d_fallback", True)
    return True


def patch_vllm_moe_model_weight_loader(model):
    # this is a work around to load the weight of vllm fused moe model
    # it is from a bug from vllm 0.8.2
    # all the weights are supposed to have a weight_loader, but the moe weights
    # do not have a weight_loader, so we need to patch it
    # (True, 'model.embed_tokens.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.bias')
    # (True, 'model.layers.0.self_attn.o_proj.weight')
    # (True, 'model.layers.0.mlp.gate.weight')
    # (True, 'model.layers.0.mlp.shared_expert.gate_up_proj.weight')
    # (True, 'model.layers.0.mlp.shared_expert.down_proj.weight')
    # (False, 'model.layers.0.mlp.shared_expert_gate.weight')   use default
    # (False, 'model.layers.0.input_layernorm.weight')          use default
    # (False, 'model.layers.0.post_attention_layernorm.weight') use default
    # (False, 'model.layers.0.mlp.experts.w13_weight')          use mlp.experts.weight_loader
    # (False, 'model.layers.0.mlp.experts.w2_weight')          use mlp.experts.weight_loader

    # Early return if no MOE models are supported
    if not SUPPORTED_MOE_MODELS:
        return

    original_model_type = type(model)
    if hasattr(model, "runnable") and "ACLGraphWrapper" in str(original_model_type):
        model = model.runnable
        original_model_type = type(model)

    # Define MLP attribute mapping for different model types
    MLP_ATTR_MAPPING = {}
    try:
        from vllm.model_executor.models.mixtral import MixtralForCausalLM

        MLP_ATTR_MAPPING[MixtralForCausalLM] = "block_sparse_moe"
    except ImportError:
        pass

    DEFAULT_MLP_ATTR = "mlp"

    # Get inner model (either model.model or model.language_model)
    inner_model = getattr(model, "model", None) or getattr(model, "language_model", None)
    if inner_model is None:
        raise ValueError("The provided model does not have a valid 'model' or 'language_model' attribute.")

    if not isinstance(model, tuple(SUPPORTED_MOE_MODELS)) and not isinstance(inner_model, tuple(SUPPORTED_MOE_MODELS)):
        return

    # TODO(@leisuzz): class Qwen3MoeLLMForCausalLM is not available if VLLM version < 0.11.0,
    # will update the 'if statement' with 'isinstance' when verl commonly use VLLM version >= 0.11.0
    if type(inner_model).__name__ in ("Qwen3MoeLLMForCausalLM", "Qwen3_5MoeForCausalLM"):
        inner_model = inner_model.model  # Reassign inner_model in Qwen3-vl

    for layer_idx, layer in enumerate(inner_model.layers):
        mlp_attr = MLP_ATTR_MAPPING.get(original_model_type, DEFAULT_MLP_ATTR)

        mlp = getattr(layer, mlp_attr, None)
        if not mlp:
            continue

        experts = getattr(mlp, "experts", None)
        if not experts or not hasattr(experts, "weight_loader"):
            continue

        # Patch the weight loaders
        for name, param in mlp.named_parameters():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = experts.weight_loader
