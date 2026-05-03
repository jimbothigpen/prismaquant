"""Gemma 3 profile (Google's multimodal family — text + SigLIP vision).

Covers:
  - Gemma3ForConditionalGeneration (multimodal: text + vision)
  - Gemma3ForCausalLM (text-only)

Gemma 3 has the same multi-layer-type rope pattern as Gemma 4 (full_attention
+ sliding_attention, dispatched per layer via `config.layer_types`), but
without Gemma 4's per_layer_input feature, kv-shared layers, or MoE block.
The streaming probe needed two adjustments to handle Gemma 3:

  1. `stage_text_only_promote_inner_model_type=True` so AutoConfig resolves
     to `Gemma3TextConfig` after staging — the outer `Gemma3Config` has no
     `text_config` (we strip it) and falls back to a 26-layer default that
     mismatches the actual checkpoint.

  2. `init_rotaries` registers `<layer_type>_inv_freq` per layer_type so
     `Gemma3RotaryEmbedding.forward(hidden, position_ids, layer_type)` finds
     a real on-device buffer instead of a leftover meta-device tensor.

Without (2), the probe forward fails inside transformers' `apply_rotary_pos_emb`
with `RuntimeError: Tensor on device cpu is not on the expected device meta!`.
"""
from __future__ import annotations

from .base import ModelProfile


class Gemma3Profile(ModelProfile):

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type in {"gemma3", "gemma3_text"}:
            return True
        for arch in architectures:
            if arch.startswith("Gemma3"):
                return True
        return False

    @property
    def name(self) -> str:
        return "gemma3"

    def vllm_architecture_class(self) -> str:
        return "Gemma3ForConditionalGeneration"

    # ------------------------------------------------------------
    # Source passthrough — vision tower stays BF16 for v1.
    # ------------------------------------------------------------
    def source_passthrough_prefixes(self) -> tuple[str, ...]:
        return (
            "model.vision_tower.",
            "model.multi_modal_projector.",
        )

    def stage_text_only_strip_keys(self) -> tuple[str, ...]:
        return (
            "vision_config",
            "image_token_index",
            "boi_token_index",
            "eoi_token_index",
            "mm_tokens_per_image",
        )

    def visual_config_key(self) -> str:
        return "vision_config"

    def visual_layer_prefix(self) -> str:
        # SigLIP vision encoder under the multimodal umbrella.
        return "model.vision_tower.vision_model.encoder.layers"

    def live_to_recipe_name(self, live_qname: str) -> str:
        """Multimodal load wraps body Linears under
        `model.language_model.layers.X.*`; the text-only probe sees flat
        `model.layers.X.*`. Strip the language_model. wrapper to align.
        Also handles the unsloth repackage's reversed `language_model.model.X.`
        layout."""
        if live_qname.startswith("model.language_model."):
            live_qname = "model." + live_qname[len("model.language_model."):]
        elif live_qname.startswith("language_model.model."):
            live_qname = "model." + live_qname[len("language_model.model."):]
        return live_qname

    def checkpoint_to_live_name(self, ckpt_key: str, *,
                                multimodal: bool = False) -> str | None:
        """Override base to handle unsloth's reversed prefix convention.

        Official HF gemma-3 multimodal checkpoints store body weights
        under `model.language_model.layers.X.*`; the unsloth repackage
        uses `language_model.model.layers.X.*` (vision_tower at top
        level, language_model wrapping the inner Gemma3TextModel).
        Both must collapse to flat `model.X.*` for the text-only probe
        whose staged Gemma3ForCausalLM skeleton expects that layout.
        """
        if ckpt_key.endswith(".weight_scale_inv"):
            return None
        # Drop visual/audio/multimodal-projector keys when running text-only.
        # Cover both the official `model.<vision_tower|visual>.X` location
        # AND the unsloth top-level `vision_tower.X` / `multi_modal_projector.X`.
        visual_drop_prefixes = (
            "model.visual.",
            "model.vision_tower.",
            "model.audio_tower.",
            "model.embed_vision.",
            "model.embed_audio.",
            "model.multi_modal_projector.",
            "vision_tower.",
            "audio_tower.",
            "multi_modal_projector.",
            "mtp.",
        )
        if any(ckpt_key.startswith(p) for p in visual_drop_prefixes):
            if multimodal and not ckpt_key.startswith("mtp."):
                return ckpt_key  # multimodal load keeps these
            return None
        if not multimodal:
            if ckpt_key.startswith("model.language_model."):
                return "model." + ckpt_key[len("model.language_model."):]
            if ckpt_key.startswith("language_model.model."):
                return "model." + ckpt_key[len("language_model.model."):]
            if ckpt_key.startswith("language_model."):
                # Strip just the wrapper. Inner top-level pieces (lm_head)
                # remain at top level rather than nesting under "model.".
                return ckpt_key[len("language_model."):]
        return ckpt_key

    def stage_text_only_promote_inner_model_type(self) -> bool:
        # `Gemma3ForCausalLM.config: Gemma3TextConfig`. We need
        # `model_type: gemma3_text` on the staged config so AutoConfig
        # picks Gemma3TextConfig — its `__post_init__` builds `layer_types`
        # from `sliding_window_pattern`, which the streaming probe and the
        # multi-layer-type rope path both depend on. With model_type left
        # at "gemma3" (the outer multimodal config), AutoConfig resolves to
        # `Gemma3Config` whose `text_config` we just stripped, so it falls
        # back to default values (26 layers etc.) — completely wrong.
        return True

    # ------------------------------------------------------------
    # Multi-layer-type rope (mirrors Gemma 4's pattern, but Gemma 3 has
    # no `global_head_dim` quirk and no `rope_type=proportional` case).
    # ------------------------------------------------------------
    def init_rotaries(self, rotary, cfg, device, dtype) -> bool:
        """Gemma-3 RotaryEmbedding expects per-layer-type buffers
        (`<layer_type>_inv_freq` + `<layer_type>_attention_scaling`).
        `Gemma3RotaryEmbedding.__init__` builds these from
        `cfg.rope_parameters` — a dict keyed by layer_type. Replicate that
        construction here so the rotary module's buffers live on `device`
        with `dtype` instead of remaining on meta after the streaming
        skeleton load.

        Also registers a None-keyed alias so callers that bypass the
        layer's layer_type propagation (older probe paths) don't crash."""
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
        # None-keyed alias for legacy callers that drop the layer_type kwarg.
        rotary.register_buffer("None_inv_freq", first[0], persistent=False)
        setattr(rotary, "None_attention_scaling", first[1])
        # Generic fallbacks expected by some transformers helpers.
        rotary.register_buffer("inv_freq", first[0], persistent=False)
        if hasattr(rotary, "original_inv_freq"):
            rotary.register_buffer("original_inv_freq",
                                   first[0].clone(), persistent=False)
        rotary.attention_scaling = first[1]
        return True
