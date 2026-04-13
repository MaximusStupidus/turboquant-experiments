# Issue Log

Tracking problems encountered during the experiment, their diagnosis, and resolution.

---

## Issue 001: KIVI shows identical perplexity to fp16 baseline

**Date:** 2026-04-09
**Phase:** 3a
**Severity:** Critical — invalidates KIVI comparison numbers

**Symptom:** KIVI 4-bit and KIVI 2-bit both report perplexity = 6.57, identical to the fp16 baseline. This is impossible — 2-bit quantization must introduce some quality loss.

**Initial diagnosis (WRONG):** Originally thought the sliding-window eval doesn't use the cache because "nothing gets stored and read back." This was incorrect — transformers models call `cache.update()` during every forward pass including single-pass prefill, and both TurboQuant and KIVI dequantize on every `update()` call. Both should affect perplexity even in single-pass mode.

**Updated diagnosis (after reading source code):** Both `TurboQuantCache` and HF `QuantizedCache` follow the same pattern: quantize on store, dequantize on every read via `update()`. The model calls `update()` at each layer during prefill, so the attention layer receives dequantized (lossy) K/V values. TurboQuant clearly shows this effect (PPL 6.85/7.44/12.46 vs 6.57 baseline). KIVI shows zero effect (PPL 6.57/6.57), which means **something is preventing KIVI's quantization from actually engaging.**

**Likely root causes (not yet confirmed):**
1. The `optimum-quanto` backend may not be actually quantizing — the object is created without error but the quantize step is a no-op due to a config/version mismatch.
2. The `residual_length` parameter may default to something ≥ 2048, meaning the entire window fits in the residual buffer and nothing ever gets quantized.
3. A subtle API issue where the QuantizedCache is created but the model creates a separate internal cache and uses that instead.

**Investigation needed:** Add debug logging inside the KIVI factory to check (a) whether `_quantized_keys` is ever populated (non-empty), and (b) what the residual_length is set to.

**Status:** Open — root cause not yet confirmed. TurboQuant numbers are valid. KIVI numbers are invalid.

---

## Issue 002: Runtime estimate was wildly off (2-3 min vs 45-90 min predicted)

**Date:** 2026-04-09
**Phase:** 3a
**Severity:** Low — wrong estimate, not a bug

**Symptom:** The full 5-config sweep completed in 2-3 minutes. The plan estimated 45-90 minutes.

**Diagnosis:** The perplexity evaluation uses single-pass forward computation per window (not autoregressive generation). A single forward pass through 2048 tokens on an A100 takes ~50ms. With ~60 windows per config and 5 configs, total compute is roughly 60 × 5 × 0.05s ≈ 15 seconds for perplexity, plus ~30 seconds for throughput measurement per config. Total: ~3-4 minutes. The 45-90 minute estimate assumed autoregressive token-by-token evaluation, which is what we actually need (see Issue 001). Once we fix the eval to be truly autoregressive, the runtime will increase significantly — likely to 15-30 minutes per config.

**Status:** Will self-resolve when Issue 001 is fixed (autoregressive eval will naturally take longer).

---

## Issue 003: Peak memory higher with TurboQuant than fp16

**Date:** 2026-04-09
**Phase:** 3a
**Severity:** Low — expected at short context, needs investigation at long context

**Symptom:** TurboQuant configs show 19.19 GB peak memory vs 18.42 GB for fp16 baseline. KV cache compression should *reduce* memory, not increase it.

**Diagnosis:** At a 2048-token sliding window, the fp16 KV cache is only ~256 MB (small relative to the 16 GB model weights). TurboQuant adds random projection matrices per layer (~a few hundred MB total). At short context, the projection matrix overhead exceeds the KV cache savings. At longer contexts (8k, 32k, 128k tokens), the KV cache becomes much larger and the savings from compression should outweigh the fixed overhead of the projection matrices. This needs to be verified with a longer-context evaluation.

**Status:** Open — needs long-context evaluation to confirm the crossover point.

---

## Issue 004: Community turboquant package API mismatch

**Date:** 2026-04-09
**Phase:** 3a
**Severity:** Resolved

**Symptom:** `TurboQuantCache(key_bits=4, value_bits=2)` raised TypeError. The package uses a single `bits` parameter, not separate key/value bits.

**Fix:** Changed factory to `TurboQuantCache(bits=N)`.

**Status:** Resolved.

---

## Issue 005: numpy.trapz removed in NumPy 2.0+

**Date:** 2026-04-09
**Phase:** 3a
**Severity:** Resolved

**Symptom:** `turboquant` package internally calls `np.trapz` which was removed in NumPy 2.0 (renamed to `np.trapezoid`).

**Fix:** Added monkey-patch at the top of the sweep script: `if not hasattr(np, "trapz"): np.trapz = np.trapezoid`

**Status:** Resolved.

---

## Issue 006: QuantizedCache API mismatch

**Date:** 2026-04-09
**Phase:** 3a
**Severity:** Resolved

**Symptom:** `QuantizedCache(cache_config={"backend": "quanto", ...})` raised TypeError. The current transformers version takes `backend` as a direct parameter.

**Fix:** Changed to `QuantizedCache(backend="quanto", nbits=N, config=model_config)`.

**Status:** Resolved.

---

## Issue 007: optimum-quanto not installed

**Date:** 2026-04-09
**Phase:** 3a
**Severity:** Resolved

**Symptom:** QuantizedCache raised ImportError asking for `optimum-quanto`. Our pyproject.toml had `quanto>=0.2.0` but the transformers integration expects the `optimum-quanto` package.

**Fix:** `uv pip install optimum-quanto` on the pod. Need to update pyproject.toml to use `optimum-quanto` instead of `quanto`.

**Status:** Resolved (pod-level fix applied; pyproject.toml update pending).
