# Phase 3 — Results Interpretation

## The comparison table (autoregressive eval, 2048 tokens)

| Config | PPL | vs baseline | tok/s | Peak Mem |
|---|---:|---:|---:|---:|
| fp16 (baseline) | 5.51 | — | 38.2 | 15.22 GB |
| KIVI 4-bit | 5.55 | +0.04 | 29.4 | 15.07 GB |
| KIVI 2-bit | 7.86 | +2.35 | 29.5 | 15.04 GB |
| TurboQuant 4-bit | 5.54 | +0.03 | 34.4 | 15.39 GB |
| TurboQuant 3-bit | 5.66 | +0.15 | 34.4 | 15.39 GB |
| TurboQuant 2-bit | 6.13 | +0.62 | 34.5 | 15.39 GB |

## The headline: TurboQuant beats KIVI at aggressive compression

At 4-bit, both methods are essentially tied (5.55 vs 5.54) — moderate compression doesn't stress either method enough for the difference to matter.

At 2-bit, TurboQuant wins decisively: 6.13 vs 7.86. KIVI's perplexity degraded by +2.35 (model is noticeably worse), while TurboQuant degraded by only +0.62 (still very usable). **The random projection trick saves 1.73 perplexity points at 2-bit.** This is the central thesis of the TurboQuant paper validated on real data: the more aggressive the compression, the more the outlier-flattening projection helps.

## Why this happens (connecting back to foundations)

At 4-bit (16 quantization buckets), there's enough precision to represent most K/V values accurately even when some channels are 3-5x larger than others. At 2-bit (only 4 buckets), precision is so scarce that outlier channels eat the entire bucket budget, crushing the precision for the majority of "normal" channels. TurboQuant's random projection flattens the distribution before quantization, ensuring every channel gets a fair share of the available precision. This is exactly the mechanism we studied in Phase 0 (the "smear into the soup" effect) and visualized in `outlier_demo.py`.

## Sweet spot: TurboQuant 3-bit

PPL = 5.66 (+0.15 from baseline) at ~5x KV cache compression. Almost no quality loss. This is the recommended default from the community package and the data backs it up — it sits at the knee of the quality-compression curve where you get most of the savings for almost none of the cost.

## Throughput surprise: KIVI is slowest

Expected TurboQuant to be slower (projection overhead). Instead, KIVI is the slowest at 29.4 tok/s (23% slower than fp16), while TurboQuant is 34.4 tok/s (10% slower). The `optimum-quanto` backend's dequantization path appears to have significant overhead that outweighs its memory bandwidth savings. TurboQuant's overhead is smaller, possibly because the random projection is a simple matrix multiply that GPUs are highly optimized for.

## Memory: similar at short context

All configs show similar peak memory (15.0–15.4 GB) because at 2048 tokens the KV cache is only ~256 MB — tiny relative to the 15 GB model weights. KIVI does show slightly lower memory (15.04 GB) than TurboQuant (15.39 GB) because TurboQuant stores projection matrices as fixed overhead. **The real memory story would emerge at 32k+ tokens** where the KV cache dominates — we haven't measured this yet but it's a natural follow-up.

## Issue 001 and what we learned from debugging it

Our first attempt at this comparison produced invalid KIVI numbers (identical perplexity to fp16) because the sliding-window eval method processes all tokens in a single forward pass. KIVI's `QuantizedCache.update()` returns the original fp16 values to the attention layer (storing quantized copies internally for memory savings). The quantization only affects computation when cached values are read back on a *subsequent* generation step — which never happens in single-pass eval.

The fix was switching to autoregressive eval (token-by-token generation), where each step reads the quantized cache from all previous steps. This is also how LLMs run in production, making the results more realistic.

**Lesson learned:** eval methodology must match the deployment scenario. A benchmark that doesn't exercise the code path you're testing will produce misleading results, regardless of how carefully the measurement code is written.

## What I'd do differently

1. **Test at longer contexts** (8k, 32k tokens) where KV cache memory dominates and the compression savings are much more dramatic.
2. **Test on diverse prompts/datasets** — WikiText-2 is a standard benchmark but it's all Wikipedia text. Different text distributions might stress the quantization differently.
3. **Profile where the throughput overhead comes from** — is KIVI slow because of quanto's dequantization? Is TurboQuant's overhead from the projection matmul? Understanding this would help optimize both methods.
