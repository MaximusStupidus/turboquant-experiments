# Part 2 Attempt 1: VibeVoice-Realtime-0.5B (BLOCKED)

**Status:** Paused. Code preserved under `speech-tts-improvements/vibevoice/` for future revival.
**Date blocked:** 2026-04-17
**Reason:** VibeVoice's internal cache-compatibility shim (`MockCacheLayer`) is incomplete for current transformers versions.

---

## What worked

- Launched AWS g5.2xlarge (A10G 23GB) with PyTorch Deep Learning AMI
- Loaded `microsoft/VibeVoice-Realtime-0.5B` successfully
- Confirmed the architecture matches our design:
  - 24 total layers (4 base LM + 20 TTS LM)
  - 14 Q heads, 2 KV heads (GQA 7:1)
  - head_dim=64, hidden_size=896
- 4 separate KV caches (lm, tts_lm, neg_lm, neg_tts_lm) with shapes:
  - `lm`: 4 layers, prefilled with 108 tokens, shape `(1, 2, 108, 64)` bfloat16
  - `tts_lm`: 20 layers, 316 tokens
  - `neg_lm`: 4 layers, 1 token
  - `neg_tts_lm`: 20 layers, 1 token
- Wrote a shim that successfully converts the pre-built `.pt` voice prompts
  from the old `DynamicCache` format (tensors in `__dict__`) to the new
  `.layers`-based format via `DynamicCache.from_legacy_cache()`.

## What broke (the cascade)

Each fix revealed the next layer of broken compatibility:

| # | Error | Root cause | Fix applied | Result |
|---|---|---|---|---|
| 1 | `No module named 'transformers.models.qwen2.tokenization_qwen2_fast'` | transformers 4.55+ removed this module, VibeVoice needs it | Pin `transformers>=4.51,<4.55` | Fixed |
| 2 | `'class' is already used by a Transformers model` | VibeVoice calls `AutoModel.register()` at import time without `exist_ok=True` | Monkey-patch `register()` to set `exist_ok=True` by default | Fixed |
| 3 | `IndexError: list index out of range` in `self.layers[layer_idx]` | Loaded cache has `key_cache` / `value_cache` in `__dict__` but empty `.layers` (new transformers stores data in `.layers`) | Read raw `__dict__`, rebuild via `DynamicCache.from_legacy_cache()` | Fixed |
| 4 | `'MockCacheLayer' object has no attribute 'is_compileable'` | VibeVoice's `MockCacheLayer` compat shim doesn't implement all methods the new transformers expects | Monkey-patch class attr to `False` | Fixed |
| 5 | `'MockCacheLayer' object has no attribute 'keys'` | `MockCacheLayer` also missing `.keys` / `.values` attrs that new transformers' property-based `key_cache` expects | **NOT FIXED** — would require substantial rewrite | Blocked |

## Root cause (the deeper problem)

There is **no single transformers version** where both these hold:
- Has `qwen2.tokenization_qwen2_fast` (VibeVoice's text tokenizer needs it) — removed in 4.55
- Has OLD list-based `DynamicCache.key_cache` (VibeVoice's internal shim was written against) — migrated to a property-based layer API around 4.50

In the narrow window [4.51, 4.55), the tokenizer exists but the cache API has already moved on. VibeVoice includes a `MockCacheLayer` compat shim that was supposed to bridge these, but it was never finished — it lacks `is_compileable`, `keys`, `values`, and `get_mask_sizes()`, all of which the new transformers cache code expects on each layer object.

## What would unblock this

**Not a quick patch.** A real fix requires rewriting `MockCacheLayer` in
`vibevoice/modular/modeling_vibevoice_streaming_inference.py` to properly
implement the new transformers layer protocol:

- `keys` and `values` properties returning the underlying legacy tensors
- `is_compileable` attribute
- `get_mask_sizes(cache_position)` method
- `.update(key_states, value_states, cache_kwargs)` that dispatches correctly

And possibly updates to `_ensure_cache_has_layers()` and
`_update_model_kwargs_for_generation()` in the same file.

**Estimated scope:** 3-4 hours of focused work. Could be contributed back to
Microsoft's VibeVoice repo as a PR.

## How to revive later

1. Check the preserved code under `speech-tts-improvements/vibevoice/`:
   - `scripts/00_test_vibevoice.py` — test script with all 4 monkey-patches we applied
   - `scripts/01_inspect_voice_prompt.py` — cache structure inspector
   - `scripts/02_inspect_model_structure.py` — model layer inspector
   - `scripts/03_inspect_raw_cache.py` — raw `__dict__` dumper

2. Fork Microsoft's VibeVoice repo

3. Rewrite `MockCacheLayer` in
   `vibevoice/modular/modeling_vibevoice_streaming_inference.py`

4. Run our `00_test_vibevoice.py` against the forked version —
   should work end-to-end

5. Submit PR back to Microsoft

## Why we pivoted to Parler-TTS

The scientific question ("does TurboQuant work on autoregressive TTS?") is
equally well-answered by any autoregressive TTS model. Parler-TTS has:

- Active maintenance, works with current transformers
- Standard HuggingFace `DynamicCache` (no custom split-architecture cache)
- Text-description-based voice control (no pre-computed `.pt` files to
  reconstruct)
- Similar scale (880M vs 500M)

We preserve the VibeVoice path as unfinished future work.
