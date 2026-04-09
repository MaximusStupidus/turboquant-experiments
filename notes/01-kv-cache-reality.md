# Phase 1 — KV Cache Inspection: What I Saw

## Results summary

Ran `01_inspect_kv_cache.py` on Llama-3.1-8B-Instruct (NousResearch mirror) on an A100-SXM4-80GB.

- **Config matched expectations exactly:** 32 layers, 32 Q heads, 8 KV heads, head_dim=128, fp16. All consistent with the foundations doc.
- **Weight memory:** 8.03B parameters, 16.06 GB in fp16. No surprises.
- **KV cache shape:** `(1, 8, 20, 128)` — batch=1, 8 KV heads, 20 tokens, head_dim=128. Exactly as predicted.
- **Per-token KV cost:** 131,072 bytes (128 KB). Matches the formula `2 × 32 × 8 × 128 × 2 = 131,072` from the foundations doc.
- **KV = weights crossover:** ~122,532 tokens. At max context (131k), the KV cache is 17.18 GB — **larger than the 16.06 GB model itself** (107%).
- **K/V distributions:** K vectors had std ~1.5–1.9 across layers; V vectors started tiny at layer 0 (std=0.04) and grew to std=0.58 by layer 31. Distributions change significantly with depth.
- **Outlier channels detected:** None, at the 10x threshold, across all three inspected layers.

## On the outlier result

The inspection found no outlier channels at the 10x threshold across all three layers (first, middle, last). This does *not* invalidate the TurboQuant premise — it means our measurement was too narrow to surface the phenomenon. Three reasons: (1) the prompt was only 20 tokens, and outlier channels in KV caches are known to emerge more strongly at longer context lengths as certain channels accumulate extreme values across many tokens; (2) a 10x threshold is conservative — real outliers in production settings are often 3–5x, which would still wreck naive int4 quantization even if they don't hit 10x; (3) we only tested one prompt. **For phase 2, this means we should: run the baseline eval at multiple context lengths (512, 2048, 8192 minimum), lower the outlier threshold to 3x and 5x in addition to 10x, and examine per-channel variance across the full eval dataset rather than a single prompt.** The absence of outliers at 20 tokens / 10x threshold is a floor, not a ceiling — it tells us where outliers *aren't*, not where they are.
