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

    def init_rotaries(self, rotary, cfg, device, dtype) -> bool:
        """Gemma 4's text rotary is multi-layer-type: it registers one
        ``<layer_type>_inv_freq`` buffer per entry in ``config.layer_types``,
        with *mixed* rope types (e.g. ``sliding_attention``=default,
        ``full_attention``=proportional). The generic single-rope fallback in
        ``_init_rotary_inplace`` calls ``compute_default_rope_parameters(cfg,
        device)`` with no ``layer_type`` → ``KeyError: None`` on
        ``config.rope_parameters[layer_type]`` (issue #6).

        Re-run the rotary's own ``__init__`` on the real device: it rebuilds
        every ``<layer_type>_inv_freq`` / ``<layer_type>_attention_scaling``
        with the correct per-type rope init function (proportional / linear /
        default, plus any per-type kwargs). A hand-rolled
        ``compute_default_rope_parameters`` loop would silently apply the
        *default* formula to the proportional layer and produce wrong
        frequencies."""
        if getattr(rotary, "layer_types", None) is None:
            return False
        if getattr(cfg, "rope_parameters", None) is None:
            return False
        try:
            type(rotary).__init__(rotary, cfg, device=device)
        except Exception:
            return False
        # --- C3 graft (sync 2026-06-12): scaffolding cache-population. ---
        # This block was formerly the tail of the fork's own init_rotaries
        # (71faaa8). The fork's redundant hand-rolled per-type inv_freq loop is
        # dropped — upstream's type(rotary).__init__ above rebuilds every
        # <layer_type>_inv_freq with the correct per-type rope init function —
        # but these structural-metadata caches are load-bearing and kept. Every
        # value derives purely from cfg/text_config (NOT from any local of the
        # dropped loop), so it is valid on the normal setup path. Populated only
        # after a successful rotary init (matching the fork's original
        # caches-iff-success lifecycle). extra_layer_kwargs reads these to supply
        # per_layer_input and synthesize zero K/V for kv-shared Gemma-4 layers;
        # if _h_per_layer_input is unset it returns {} for every layer.
        text_cfg = getattr(cfg, "text_config", None) or cfg
        h_per = getattr(text_cfg, "hidden_size_per_layer_input", None)
        Gemma4Profile._h_per_layer_input = int(h_per) if h_per else None
        n_layers = getattr(text_cfg, "num_hidden_layers", None)
        Gemma4Profile._num_hidden_layers = int(n_layers) if n_layers else 0
        Gemma4Profile._head_dim = int(getattr(text_cfg, "head_dim", 0) or 0)
        Gemma4Profile._global_head_dim = int(getattr(text_cfg, "global_head_dim", 0) or 0)
        Gemma4Profile._num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", 0) or 0)
        global_kv = getattr(text_cfg, "num_global_key_value_heads", None)
        Gemma4Profile._num_global_kv_heads = int(global_kv) if global_kv else Gemma4Profile._num_kv_heads
        Gemma4Profile._attn_k_eq_v = bool(getattr(text_cfg, "attention_k_eq_v", False))
        Gemma4Profile._layer_types = list(getattr(text_cfg, "layer_types", []) or [])
        # Clear the per-layer-inputs cache: it's per-probe-pass and stale state
        # across init_rotaries calls would silently misroute slices.
        Gemma4Profile._per_layer_inputs_cache = None
        Gemma4Profile._per_layer_inputs_cache_key = None
        return True

    # ------------------------------------------------------------
    # Cross-layer KV sharing.  Gemma4's last `num_kv_shared_layers`
    # attention layers have no k/v_proj — they reuse the K/V computed by
    # the last non-shared layer of their `layer_type`, passed via a
    # `shared_kv_states` dict the model forward threads through every layer.
    # ------------------------------------------------------------
    def new_forward_pass_state(self) -> dict:
        return {"shared_kv_states": {}}

    def capture_forward_pass_state(self, pass_state: dict):
        """Snapshot Gemma4's integer-indexed shared K/V states to CPU."""
        skv = (pass_state or {}).get("shared_kv_states") or {}
        out = {}
        for layer_idx, kv in skv.items():
            try:
                key = int(layer_idx)
                out[key] = tuple(t.detach().to("cpu") for t in kv)
            except Exception as exc:
                raise RuntimeError(
                    f"Gemma4 shared_kv_states[{layer_idx!r}] could not be "
                    "captured as a CPU tensor tuple"
                ) from exc
        return out

    def isolated_layer_pass_state(self, captured, layer) -> dict:
        """For an isolated (phase-3) layer forward: a shared layer needs its
        source layer's captured K/V (the attention moves them to the right device
        itself); a non-shared layer just needs a writable dict to store into.
        Always returns a `shared_kv_states` dict so the layer never sees
        `None`."""
        attn = getattr(layer, "self_attn", None)
        if getattr(attn, "is_kv_shared_layer", False) and captured:
            source_idx = getattr(attn, "kv_shared_layer_index", None)
            if source_idx is not None:
                kv = captured.get(int(source_idx))
                if kv is not None:
                    return {"shared_kv_states": {int(source_idx): kv}}
        return {"shared_kv_states": {}}

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

    def head_resident_extra_prefixes(self, root) -> list[str]:
        """Three modules drive the proper per_layer_input computation in
        `extra_layer_kwargs`:
          - `embed_tokens_per_layer` — token-identity component (large;
             vocab_size_per_layer_input * num_layers * h_per_layer; e.g.
             ~4.7 GB at BF16 on gemma-4-E2B-it, vocab 256K, 35 layers,
             h_per=256).
          - `per_layer_model_projection` — context-aware Linear (small).
          - `per_layer_projection_norm` — RMSNorm (tiny).
        Without these resident, `extra_layer_kwargs` falls back to an
        all-ones synthetic per_layer_input, which biases Hessians for
        `per_layer_input_gate` / `per_layer_projection` toward
        over-allocation.

        Memory note: the embed_tokens_per_layer cost is significant on
        small-VRAM systems. high-VRAM machines (MI300X 192 GB, etc.)
        absorb it trivially; for tighter VRAM budgets a future
        optimization could materialize → precompute → free instead of
        keeping the full table resident. The current head-resident
        approach is the simplest correct path."""
        inner = getattr(root, "model", None)
        if inner is not None and hasattr(inner, "embed_tokens_per_layer"):
            return [
                "model.embed_tokens_per_layer.",
                "model.per_layer_model_projection.",
                "model.per_layer_projection_norm.",
            ]
        if hasattr(root, "embed_tokens_per_layer"):
            return [
                "embed_tokens_per_layer.",
                "per_layer_model_projection.",
                "per_layer_projection_norm.",
            ]
        return []

    @staticmethod
    def _maybe_compute_per_layer_input(base_model, input_ids, layer_idx):
        """Compute the proper Gemma-4 per_layer_input slice for `layer_idx`
        using `base_model.get_per_layer_inputs` + `project_per_layer_inputs`
        (mirrors `modeling_gemma4.py:Gemma4TextModel.forward` lines
        ~1632-1635).

        Caches the full `[B, T, num_layers, hidden_per_layer]` tensor on
        the class so we only run the (potentially large) embed_tokens_per_layer
        lookup once per probe pass. Cache key: id(input_ids) + shape, which
        invalidates correctly because the probe loop reuses one input_ids
        tensor across every layer.

        Returns the slice tensor on success; `None` if any of:
          - env `PRISMAQUANT_GEMMA4_DISABLE_PROPER_PLI=1` (manual opt-out)
          - base_model lacks the per-layer modules (not Gemma 4)
          - per_layer modules are still meta-device (head-resident not
            applied yet, or extra prefixes unset)
          - any unexpected error raised during the computation
        Caller falls back to the synthetic all-ones tensor.

        Manual opt-out: set `PRISMAQUANT_GEMMA4_DISABLE_PROPER_PLI=1` in
        the environment to force the synthetic-ones fallback. Useful when
        the ~4.7 GB embed_tokens_per_layer head-resident cost is
        prohibitive (small-VRAM systems) and the over-allocation bias
        on per_layer_input_gate / per_layer_projection is acceptable."""
        import os as _os
        if _os.environ.get("PRISMAQUANT_GEMMA4_DISABLE_PROPER_PLI") == "1":
            return None
        import torch as _torch
        cache_key = (
            id(input_ids), tuple(input_ids.shape),
            input_ids.device, input_ids.dtype,
        )
        cached = getattr(Gemma4Profile, "_per_layer_inputs_cache", None)
        cached_key = getattr(Gemma4Profile, "_per_layer_inputs_cache_key", None)
        if cached is None or cached_key != cache_key:
            if not (hasattr(base_model, "embed_tokens_per_layer")
                    and hasattr(base_model, "per_layer_model_projection")
                    and hasattr(base_model, "per_layer_projection_norm")
                    and hasattr(base_model, "get_per_layer_inputs")
                    and hasattr(base_model, "project_per_layer_inputs")
                    and hasattr(base_model, "embed_tokens")):
                return None
            try:
                # Refuse if any of the modules' weights are still meta —
                # head-resident materialization hasn't covered them.
                etp_w = base_model.embed_tokens_per_layer.weight
                if etp_w.device.type == "meta":
                    return None
                with _torch.no_grad():
                    inputs_embeds = base_model.embed_tokens(input_ids)
                    pli_token = base_model.get_per_layer_inputs(
                        input_ids, inputs_embeds)
                    pli = base_model.project_per_layer_inputs(
                        inputs_embeds, pli_token)
                Gemma4Profile._per_layer_inputs_cache = pli
                Gemma4Profile._per_layer_inputs_cache_key = cache_key
                cached = pli
            except Exception:
                return None
        if cached is None or layer_idx is None:
            return None
        if layer_idx < 0 or layer_idx >= cached.size(2):
            return None
        return cached[:, :, layer_idx, :].contiguous()

    def extra_layer_kwargs(self, *, input_ids=None, base_model=None,
                           layer_idx=None) -> dict:
        """Gemma-4 decoder layers consume a per-layer additive embedding
        ("per_layer_input", shape [B, T, hidden_size_per_layer_input]).

        Preferred path: when `base_model` and `layer_idx` are provided
        AND the per-layer modules are head-resident (see
        `head_resident_extra_prefixes`), compute the proper slice via
        `_maybe_compute_per_layer_input`. This makes the Hessians for
        `per_layer_input_gate` / `per_layer_projection` reflect what
        these Linears actually see at inference time.

        Fallback (back-compat): pass an all-ones tensor. This makes the
        per_layer_input multiplication a no-op and biases Hessians for
        the two per-layer Linears toward over-allocation. Used when the
        caller doesn't pass base_model/layer_idx, or when the per-layer
        modules aren't resident (e.g., custom probe configurations that
        don't honor head_resident_extra_prefixes)."""
        if input_ids is None:
            return {}
        # Use class-level state because profile_from_model returns fresh instances.
        H_per = getattr(Gemma4Profile, "_h_per_layer_input", None)
        if H_per is None:
            return {}
        import torch as _torch
        B, T = input_ids.shape
        per_layer_input = None
        if base_model is not None and layer_idx is not None:
            per_layer_input = Gemma4Profile._maybe_compute_per_layer_input(
                base_model, input_ids, layer_idx)
        if per_layer_input is None:
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

