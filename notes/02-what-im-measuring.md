# Phase 2 — What I'm Measuring and Why

## The four metrics

### 1. Perplexity (WikiText-2)

Perplexity measures how surprised the model is by real text. A perplexity of 6.57 means the model is, on average, choosing between ~7 equally likely next tokens. Lower is better. We measure it using a sliding window (2048 tokens wide, 512 stride) over 32k tokens from the WikiText-2 test split — a standard benchmark that makes our numbers directly comparable to published results.

**What changes mean:** A perplexity increase of 0.1–0.3 from TurboQuant compression would be negligible — normal measurement noise. An increase of 0.5–1.0 is noticeable but likely acceptable for the memory savings. An increase above 2.0 means the compression is too aggressive and the model's language ability is degrading. If perplexity doubles, the model is broken.

### 2. Peak GPU memory

The maximum GPU memory used during evaluation. Our baseline is 18.42 GB — that's 14.96 GB of model weights plus ~3.5 GB for the KV cache and activations during a 2048-token window.

**What changes mean:** When TurboQuant compresses the KV cache, the weight memory stays at 14.96 GB (we're not quantizing weights), but the KV portion shrinks. So the peak memory drop tells us the *real-world savings* from KV cache compression. If TurboQuant achieves 4x KV compression, the KV contribution drops from ~3.5 GB to ~0.9 GB, and peak memory goes from 18.42 GB to ~15.9 GB. At longer context lengths where KV dominates, the savings would be much more dramatic.

### 3. Tokens per second (generation throughput)

How fast the model generates new tokens. Baseline: 37.9 tok/s on an A100 80GB, measured as the median of 3 runs of 128-token greedy generation.

**What changes mean:** Smaller KV cache means less data the GPU has to read during each attention step. If attention is memory-bandwidth-bound (which it often is during generation), throughput should *increase* with TurboQuant. However, the random projection itself adds a small computation cost at each step, which could offset the bandwidth savings. Whether throughput goes up or down depends on which effect dominates — the experiment will tell us.

### 4. Extended outlier analysis

Per-channel value statistics for K and V tensors at three context lengths (512, 2048, 8192) and three outlier thresholds (3x, 5x, 10x), across three layers (first, middle, last).

**What we found:** No outliers at 10x at any context length — Llama-3.1-8B does not have the extreme outlier channels reported in older models. But at 3x, there are 3–13 outlier channels per layer, and they are remarkably consistent across context lengths (same channels appear regardless of input length, suggesting it's a structural property of the model). V values at early layers are tiny (std=0.03) while late layers are much larger (std=0.6), meaning a uniform quantization strategy across all layers may not be optimal.

**What this means for TurboQuant:** The 3–5x outlier channels, while not extreme, are still enough to degrade naive int4 quantization. TurboQuant's random projection should flatten these into a more uniform distribution. The phase 3 experiment will test whether this actually produces better perplexity than naive quantization at the same compression ratio, or whether the moderate outlier levels in this model make the difference marginal.

## What "TurboQuant is working" looks like

- Perplexity stays within 0.5 of baseline (6.57) at 4x KV compression
- Peak GPU memory drops measurably
- Throughput stays the same or improves
- Perplexity at the same compression ratio is better than naive int8/int4 quantization (i.e., TurboQuant's outlier handling actually helps)

## What "TurboQuant is broken" looks like

- Perplexity jumps above 10 at any compression level
- Perplexity is *worse* than naive quantization at the same compression ratio (meaning the projection overhead hurts more than the outlier flattening helps)
- Memory savings are smaller than expected (implementation overhead eating the compression gains)
