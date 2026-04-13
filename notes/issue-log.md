# Issue Log

Tracking problems encountered during the experiment, their diagnosis, and resolution.

---

## Issue 001: KIVI shows identical perplexity to fp16 baseline

**Date:** 2026-04-09
**Phase:** 3a
**Severity:** Critical — invalidates KIVI comparison numbers

**Symptom:** KIVI 4-bit and KIVI 2-bit both report perplexity = 6.57, identical to the fp16 baseline. This is impossible — 2-bit quantization must introduce some quality loss.

**Diagnosis 1 (WRONG):** Originally thought the sliding-window eval doesn't use the cache because "nothing gets stored and read back." Incorrect — transformers calls `cache.update()` during every forward pass.

**Diagnosis 2 (WRONG):** After reading source code, thought both TurboQuant and KIVI dequantize on `update()` and return lossy values. The source code *appeared* to show this pattern for both.

**Diagnosis 3 (CONFIRMED — root cause found via debug_kivi.py v4):** KIVI's `QuantizedCache.update()` method **returns the original unmodified fp16 values**, not the dequantized lossy values. The quantized representation is stored internally (for memory savings), but the values returned to the attention layer are the exact originals. This means the attention computation is always fp16-precise, and perplexity is bit-identical to the baseline.

This is **by design**: KIVI's purpose is to reduce memory during long-context *serving*, where cached values from step N are read back at step N+K via `_dequantize`. In a single forward pass, the values are used immediately (the return value of `update()`), so the quantized copy is never read back and has no effect on computation.

TurboQuant, by contrast, modifies the K/V values *on write* (project→quantize→dequantize→return lossy values), so the attention layer sees lossy values even in single-pass mode. This is why TurboQuant shows a perplexity effect and KIVI does not.

**Implications:**
1. Our perplexity eval (single-pass per window) **cannot measure KIVI's quality impact**. Autoregressive token-by-token generation is required.
2. KIVI's memory numbers (18.06 GB, 18.00 GB) ARE valid — the cache IS stored in fewer bits.
3. TurboQuant's perplexity numbers (6.85, 7.44, 12.46) ARE valid.
4. The current comparison is **not apples-to-apples** — TurboQuant is penalized (showing loss) while KIVI is not (showing zero loss), even though both would show loss in real autoregressive use.

**Fix needed:** Rewrite perplexity eval to use autoregressive generation (token-by-token), where each step reads the dequantized cache from previous steps. This is the only way to measure KIVI's quality tradeoff and produce a fair comparison.

**Status:** Root cause confirmed. Fix pending — needs autoregressive eval rewrite.

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
