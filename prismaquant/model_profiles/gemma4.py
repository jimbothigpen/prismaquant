"""Gemma 4 profile (Google's multimodal family — text + vision + audio).

Covers:
  - Gemma4ForConditionalGeneration (multimodal MoE + dense, all sizes)
  - Gemma4ForCausalLM (text-only)

Almost entirely vLLM-metadata-derived — Gemma 4 has a clean
`packed_modules_mapping` (`qkv_proj`, `gate_up_proj`) and a standard
`hf_to_vllm_mapper` that matches Qwen3.5/3.6's body-prefix convention.
No MTP heads (not in vLLM's speculative registry at this vLLM version),
so PrismaQuant doesn't need a custom MTP forward builder.

Source passthrough prefixes cover the three modality towers (vision,
audio, and their embedding projectors) — these pass through as BF16
until we wire real multimodal calibration, matching the Qwen3.6 visual
encoder policy.

Minimal size: ~30 lines. Everything else inherits from base.
"""
from __future__ import annotations

from .base import ModelProfile


class Gemma4Profile(ModelProfile):

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type in {"gemma4", "gemma4_text"}:
            return True
        for arch in architectures:
            if arch.startswith("Gemma4"):
                return True
        return False

    @property
    def name(self) -> str:
        return "gemma4"

    def vllm_architecture_class(self) -> str:
        # `Gemma4ForConditionalGeneration` exposes the full multimodal
        # prefix map (vision_tower, audio_tower, embed_vision,
        # embed_audio, language_model). Auto-derived
        # `fused_sibling_group` and `to_vllm_internal_name` inherit
        # from base — no overrides needed.
        return "Gemma4ForConditionalGeneration"

    # `on_disk_expert_qname` intentionally NOT overridden: vLLM's
    # `Gemma4TextModel.load_weights` already runs a substring remap
    # `.experts.{id}.{proj}` → `.moe.experts.{id}.{proj}` (see
    # `vllm.model_executor.models.gemma4.py:1554`). Emitting the HF
    # naming (no `.moe.`) lets vLLM's own remap path land the per-expert
    # tensors correctly on `FusedMoE.w13_weight` / `w2_weight`.
    # Overriding to inject `.moe.` ourselves produces a double `.moe.`
    # after vLLM's remap runs — verified experimentally.

    def export_tensor_name(self, model_qname: str) -> str:
        """Keep body/expert export keys in recipe form.

        Gemma 4's vLLM weight iterator performs its own body and
        `.experts` -> `.moe.experts` remaps. Source lookup still uses the
        declarative `recipe_to_source` rules, but export must not pre-apply
        those remaps or vLLM sees doubled `.moe.` prefixes.
        """
        if (
            model_qname.startswith("model.layers.")
            or model_qname.startswith("model.embed_tokens")
            or model_qname.startswith("model.norm")
        ):
            return model_qname
        return super().export_tensor_name(model_qname)

    # === streaming-probe hooks (Gemma-4 multi-layer-type rope + per_layer_input) ===

    def init_rotaries(self, rotary, cfg, device, dtype) -> bool:
        """Gemma-4 has multi-layer-type rope: cfg.rope_parameters is a
        dict-of-dicts keyed by layer_type (e.g. full_attention,
        sliding_attention). Mirrors transformers.Gemma4TextRotaryEmbedding.__init__:
        registers <layer_type>_inv_freq + <layer_type>_attention_scaling per type,
        plus <layer_type>_original_inv_freq clones. Also registers a None-keyed
        alias so callers that bypass the layer's layer_type propagation
        (e.g. probe paths) don't crash."""
        import torch as _torch
        rope_params = getattr(cfg, "rope_parameters", None)
        if not (isinstance(rope_params, dict) and rope_params
                and all(isinstance(v, dict) for v in rope_params.values()
                        if v is not None)):
            return False
        try:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        except Exception:
            return False
        first = None
        for layer_type, params in rope_params.items():
            if params is None:
                continue
            rope_type = params.get("rope_type", "default")
            kwargs = {"device": device, "layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                kwargs["head_dim_key"] = "global_head_dim"
            if rope_type != "default":
                fn = ROPE_INIT_FUNCTIONS.get(rope_type)
                if fn is None:
                    continue
                inv_freq, scale = fn(cfg, **kwargs)
            else:
                inv_freq, scale = rotary.compute_default_rope_parameters(cfg, **kwargs)
            inv_freq = inv_freq.to(dtype=_torch.float32, device=device)
            rotary.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            rotary.register_buffer(f"{layer_type}_original_inv_freq",
                                   inv_freq.clone(), persistent=False)
            setattr(rotary, f"{layer_type}_attention_scaling", scale)
            if first is None:
                first = (inv_freq, scale)
        if first is None:
            return False
        rotary.register_buffer("None_inv_freq", first[0], persistent=False)
        setattr(rotary, "None_attention_scaling", first[1])
        rotary.register_buffer("inv_freq", first[0], persistent=False)
        if hasattr(rotary, "original_inv_freq"):
            rotary.register_buffer("original_inv_freq",
                                   first[0].clone(), persistent=False)
        rotary.attention_scaling = first[1]
        # Cache hidden_size_per_layer_input for extra_layer_kwargs to use later.
        text_cfg = getattr(cfg, "text_config", None) or cfg
        h_per = getattr(text_cfg, "hidden_size_per_layer_input", None)
        Gemma4Profile._h_per_layer_input = int(h_per) if h_per else None
        # Cache num_hidden_layers + num_kv_shared_layers for shared_kv_states.
        n_layers = getattr(text_cfg, "num_hidden_layers", None)
        Gemma4Profile._num_hidden_layers = int(n_layers) if n_layers else 0
        # === KV zeros synthesis (Gemma-4 kv-shared layers) ===
        # Cache shape parameters for synthesizing zero K/V tensors that
        # let kv-shared layers (last num_kv_shared_layers of the model)
        # unpack shared_kv_states without crashing. Hessian for q_proj on
        # those layers will be slightly biased (attention output is zero,
        # so downstream sees no info from these layers) — acceptable.
        Gemma4Profile._head_dim = int(getattr(text_cfg, "head_dim", 0) or 0)
        Gemma4Profile._global_head_dim = int(getattr(text_cfg, "global_head_dim", 0) or 0)
        Gemma4Profile._num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", 0) or 0)
        global_kv = getattr(text_cfg, "num_global_key_value_heads", None)
        Gemma4Profile._num_global_kv_heads = int(global_kv) if global_kv else Gemma4Profile._num_kv_heads
        Gemma4Profile._attn_k_eq_v = bool(getattr(text_cfg, "attention_k_eq_v", False))
        Gemma4Profile._layer_types = list(getattr(text_cfg, "layer_types", []) or [])

        return True

    def extra_layer_kwargs(self, *, input_ids=None) -> dict:
        """Gemma-4 decoder layers consume a per-layer additive embedding
        ("per_layer_input", shape [B, T, hidden_size_per_layer_input]).
        The probe runs each layer in isolation; rather than precomputing
        and slicing the model-level per_layer_inputs (which would require
        keeping the per-layer embedding modules resident + threading layer_idx
        through the call sites), we pass an all-ones tensor. This makes the
        per_layer_input multiplication a no-op and slightly biases Hessians
        for per_layer_input_gate / per_layer_projection toward over-allocation
        — a conservative trade-off that yields valid probe data for everything
        else (attn, main MLP, MoE experts). For full fidelity, replace this
        with the proper sliced computation — see modeling_gemma4.py:1632."""
        if input_ids is None:
            return {}
        # Use class-level state because profile_from_model returns fresh instances.
        H_per = getattr(Gemma4Profile, "_h_per_layer_input", None)
        if H_per is None:
            return {}
        import torch as _torch
        B, T = input_ids.shape
        per_layer_input = _torch.ones(
            (B, T, H_per),
            dtype=_torch.bfloat16,
            device=input_ids.device,
        )
        kw = {"per_layer_input": per_layer_input}
        # Synthesize zero K/V tuples per layer so kv-shared layers
        # (which read shared_kv_states[kv_shared_layer_index] and unpack)
        # don't crash. Non-shared layers will overwrite their slot during
        # the forward pass; shared layers read either the populated slot
        # (if a non-shared sibling has run earlier in this batch — which
        # in prismaquant's sequential probe loop is always the case) or
        # the zero placeholder.
        n_layers = getattr(Gemma4Profile, "_num_hidden_layers", 0)
        if n_layers > 0:
            layer_types = getattr(Gemma4Profile, "_layer_types", [])
            num_kv_heads = getattr(Gemma4Profile, "_num_kv_heads", 0)
            num_global_kv_heads = getattr(Gemma4Profile, "_num_global_kv_heads", num_kv_heads)
            head_dim = getattr(Gemma4Profile, "_head_dim", 0)
            global_head_dim = getattr(Gemma4Profile, "_global_head_dim", head_dim) or head_dim
            attn_k_eq_v = getattr(Gemma4Profile, "_attn_k_eq_v", False)
            sk = []
            for i in range(n_layers):
                lt = layer_types[i] if i < len(layer_types) else "sliding_attention"
                is_sliding = lt == "sliding_attention"
                use_alt = attn_k_eq_v and not is_sliding
                kvh = num_global_kv_heads if use_alt else num_kv_heads
                hd = global_head_dim if (not is_sliding and global_head_dim) else head_dim
                if kvh > 0 and hd > 0:
                    z = _torch.zeros((B, kvh, T, hd),
                                     dtype=_torch.bfloat16,
                                     device=input_ids.device)
                    sk.append((z, z))
                else:
                    sk.append(None)
            kw["shared_kv_states"] = sk
        return kw

