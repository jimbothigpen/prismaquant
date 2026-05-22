# Fork notes — `jimbothigpen/prismaquant`

> **This is a downstream fork of [`RobTand/prismaquant`](https://github.com/RobTand/prismaquant)
> with patches needed to support [`prismaquant-llama`](https://github.com/jimbothigpen/prismaquant-llama),
> a llama.cpp / GGUF-targeting adapter for the prismaquant Bayesian
> mixed-precision allocator.** The upstream `prismaquant` package targets
> vLLM + compressed-tensors; our fork extends its `incremental_probe` and
> `streaming_model` paths to handle architecture patterns that the
> llama.cpp ecosystem encounters routinely but the vLLM-targeted
> upstream doesn't yet support.

> **Disclosure — this is vibe-coded.** I'm an enthusiast, not a
> programmer. Every line of code, doc, and commit message in this
> fork's patch series was written with [Claude Code](https://claude.com/claude-code)
> doing the actual implementation; I drive the design decisions,
> review changes, and decide what ships. This is disclosed up front
> because the volume of activity here would otherwise be misleading —
> assume AI-assisted unless explicitly stated otherwise. Issues and
> PRs are still welcome; just calibrate expectations accordingly. The
> mathematical core of prismaquant (the closed-form Δloss surrogate,
> the streaming-probe architecture, the allocator) is
> [RobTand](https://github.com/RobTand/prismaquant)'s work, not mine —
> this fork is just the compatibility patches needed to run that work
> on a couple more architectures.

The upstream README ([`README.md`](README.md)) covers the project's
mathematical foundations and vLLM-targeted use cases — read that first.

## What this fork adds

A series of focused patches on `main` that enable `incremental_probe`
to work on architectures we ran into while building
[`prismaquant-llama`](https://github.com/jimbothigpen/prismaquant-llama).
Two clusters of patches:

### Generic / defensive (apply to any model with these patterns)

These are bug fixes / robustness improvements that benefit any user,
not specific to gemma-4 or NemotronH:

| Patch | File | Why |
|---|---|---|
| `llm_config → text_config` rename | `sensitivity_probe.py` | InternVL/NemotronH-Nano-Omni V3 wrap their LM config inside `llm_config`; `stage_text_only` was checking only for `text_config` |
| `backbone.layers` fallback | `layer_streaming.py` | NemotronH (and other Mamba-2 hybrids) expose decoder layers under `backbone.layers`, not `model.layers` |
| `embeddings` prefix in head-resident | `layer_streaming.py` | NemotronH names its token embedding `embeddings`, not `embed_tokens` |
| `embed_tokens` / `embeddings` dual lookup | `incremental_probe.py` | mirror of the above for the runtime forward call site |
| Defensive `.get()` on `autoscale_diag` | `streaming_model.py` | older diag dict shape lacks some keys; one-off KeyError → silent default |
| Multi-layer-type rope fallback | `streaming_model.py` | when `cfg.rope_parameters` is a dict-of-dicts (DSv4, Gemma3, Gemma-4), `rope_init_fn(cfg, device)` raises KeyError(None); fall back to per-type buffer registration |
| Per-layer `position_embeddings` for multi-layer-type rope | `incremental_probe.py` | recompute cos/sin per layer when layers have different rope_theta or partial_rotary_factor |
| Orphan-tensor skip in `_fast_install` | `layer_streaming.py` | Gemma-4 saves K-norm/V-norm/etc. weights for kv-shared layers but `Gemma4TextAttention` doesn't allocate the attrs; `transformers.from_pretrained` ignores silently — we mirror that |

### Architecture-specific (Gemma-4 streaming-probe profile)

The existing `Gemma4Profile` in upstream covered only vLLM allocator
metadata (MoE expert regex, multimodal tower passthrough). To run the
streaming probe path on Gemma-4, the profile needs to override
`init_rotaries`, `extra_layer_kwargs`, and inject synthetic K/V
tensors for kv-shared layers. See
[`docs/gemma4-profile.md`](https://github.com/jimbothigpen/prismaquant-llama/blob/main/docs/gemma4-profile.md)
in the prismaquant-llama repo for the detailed breakdown of all seven
patches and tradeoffs.

### Architecture-specific (Gemma 3 profile)

Gemma 3 has no `Gemma3Profile` in upstream — `DefaultProfile` was used,
which fails in two ways: (1) `stage_text_only` leaves `model_type=gemma3`
(the multimodal outer type), so `AutoConfig` resolves to a 26-layer default
config instead of the 34-layer text config; (2) `Gemma3RotaryEmbedding`
expects per-layer-type buffers (`<layer_type>_inv_freq`) registered by an
`init_rotaries` hook — without it the skeleton's rotary stays on `meta`
and the first decoder forward crashes in `apply_rotary_pos_emb`.

| Patch | File | Why |
|---|---|---|
| Gemma 3 streaming-probe profile | `model_profiles/gemma3.py` | New `Gemma3Profile`: promotes inner `model_type` to `gemma3_text` so `AutoConfig` picks `Gemma3TextConfig` (builds `layer_types` from `sliding_window_pattern`); implements `init_rotaries` to register per-layer-type `inv_freq` / `attention_scaling` from `cfg.rope_parameters` (sliding vs full attention types) |
| unsloth `language_model` wrapper unwrap | `model_profiles/gemma3.py` | Unsloth gemma-3 uses inverted nesting (`language_model.model.X`) vs HF official (`model.language_model.X`); `checkpoint_to_live_name` override rewrites and drops vision tower / projector keys so head tensors materialize correctly |

## Branch policy

`main` IS the patch series — this fork's reason to exist is to carry
the patches, so there's no point in keeping a "clean upstream mirror"
branch. We periodically `git fetch upstream && git rebase upstream/main`
to stay current with `RobTand/prismaquant`. PRs upstream will shrink
the fork's footprint over time.

If you need the unpatched upstream, use the `upstream` remote
(`https://github.com/RobTand/prismaquant`) directly.

## Install for prismaquant-llama users

```bash
pip install git+https://github.com/jimbothigpen/prismaquant.git
```

Or for development:

```bash
git clone https://github.com/jimbothigpen/prismaquant.git
cd prismaquant
pip install -e .
```

## Validated arches via `prismaquant-llama`

- `unsloth/gemma-3-4b-it` (multi-layer-type rope, sliding-window attention, partial rotary factor) — `Gemma3Profile`; unsloth checkpoint wrapping handled by `checkpoint_to_live_name` override
- `google/gemma-4-E4B-it` (iSWA hybrid, partial rotary, kv-sharing) — required all 8 generic + 4 gemma-specific patches
- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` (Mamba-2 + MoE hybrid) — generic patches sufficient through Stage C config + layer-list resolution; deeper Mamba-2-specific blockers tracked separately
- `unsloth/gpt-oss-20b-BF16` (MoE) — `DefaultProfile` path; no patches needed beyond the generic NemotronH support already merged

## Upstreaming roadmap

The 8 generic patches are clear bug fixes and good candidates for
upstream PRs to `RobTand/prismaquant`. We're holding those PRs until
prismaquant-llama is publicly released and the patches have a few weeks
of in-use validation. Until then, this fork is the install target.

The 4 architecture-specific Gemma-4 patches are larger and may not
match upstream's vLLM-first scope; those are likely to live on the
fork long-term.

## License

Same as upstream — see [`LICENSE`](LICENSE).
