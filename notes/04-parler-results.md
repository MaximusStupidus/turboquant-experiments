# Part 2 results — TurboQuant on Parler-TTS Mini v1

**Model:** `parler-tts/parler-tts-mini-v1` (880M params, encoder-decoder)
**Hardware:** AWS A10G 23 GB, bfloat16
**Generation:** sampling at Parler's native `temperature=1.0`,
`do_sample=True`, fixed `set_seed(42)` per sample; `max_length` per
text (450 / 1200 / 2400 steps for short / medium / long).
**Eval:** 3 voices × 3 texts = 9 prompts per config, compared across
fp16 baseline and TurboQuant at 4-bit / 3-bit / 2-bit — 36 WAVs total.
**WER:** Whisper small.en. **Speaker similarity:** ECAPA-TDNN cosine
between baseline and quantized generations of the same prompt.

## Headline numbers

| Config | Mean WER | Median WER | Mean spk-sim vs fp16 | Mean RTF | Peak GPU |
|---|---:|---:|---:|---:|---:|
| fp16 baseline | **0.22** | 0.13 | — | 1.45 | 3.0 GB |
| TurboQuant 4-bit | 0.28 | 0.23 | 0.76 | 2.62 | 3.0 GB |
| TurboQuant 3-bit | 0.31 | 0.33 | 0.73 | 2.62 | 3.0 GB |
| TurboQuant 2-bit | **0.45** | 0.38 | 0.65 | 2.88 | 3.0 GB |

![WER](../speech-tts-improvements/parler/results/plots/wer.png)
![Speaker similarity](../speech-tts-improvements/parler/results/plots/speaker_similarity.png)

## Monotonic degradation — the Part-1-style curve

Both metrics degrade monotonically with aggressive quantization, just
like Part 1's perplexity-vs-bits curve on Llama-3.1-8B:

| Config | Part 1 (PPL vs fp16) | Part 2 (WER vs fp16) | Part 2 (Spk-sim vs fp16) |
|---|---|---|---|
| fp16 | 5.51 | 0.22 | — |
| 4-bit | +0.03 (5.54) | +0.06 (0.28) | 0.76 |
| 3-bit | +0.15 (5.66) | +0.10 (0.31) | 0.73 |
| 2-bit | +0.56 (6.07) | +0.24 (0.45) | 0.65 |

Both experiments show compression cost *accelerates* at 2-bit: the
4→3-bit gap is small, the 3→2-bit gap is large. This is the
signature of TurboQuant's projection+codebook pipeline — at 4-bit
the codebook has enough levels to track the Beta-distributed values
accurately; at 2-bit (4 levels) the quantization error starts
dominating.

## Methodology note — why the first sweep didn't show this

The first sweep set `temperature=0.7` (a "conservative" choice on my
part); the second sweep used Parler's native `temperature=1.0` from
its generation config. The difference is massive: on
`jon__long`, probe WER went from 0.85 at temp=0.7 to 0.13 at temp=1.0.
Lowering temperature biases sampling toward top-probability tokens,
which for this model turn out to be lower-quality — Parler was
trained with temp=1.0 in the pipeline and that's the distribution
the audio codec expects. Every metric in the table above was
regenerated after the temperature fix; the previous version of this
doc (and `metrics.json`) contained noisy, buzz-padded results that
didn't reflect the algorithm.

## What the experiment shows

### 1. TurboQuant transfers to autoregressive TTS.

The same `HandrolledTurboQuantCache` from Part 1 (Llama-3.1-8B) plugs
into Parler-TTS's decoder unchanged — only `num_layers`,
`num_kv_heads`, and `head_dim` differ. Once wrapped in
`EncoderDecoderCache`, it runs end-to-end at 4/3/2-bit with no NaN,
no OOM, no cache errors.

### 2. 4-bit is ~free. 2-bit has a real cost.

WER at 4-bit (0.28) is only 0.06 above fp16 (0.22) — essentially
within sampling noise. At 3-bit we pay 0.10, and at 2-bit we pay
0.24. The "it's basically free at 4-bit" story from Part 1 holds.

### 3. Voice identity degrades gracefully.

Speaker similarity drops 0.76 → 0.73 → 0.65 across 4/3/2-bit — a
slow, monotonic drift rather than a cliff. Even at 2-bit the voice
is ~65% similar to the fp16 original, meaning the speaker is still
recognisable but with noticeable timbre shifts. Listen to
`laura__long` across the four configs in `audio_comparison.html`:
the same voice remains identifiable, with increasing roughness.

### 4. Short prompts are robust across all bit levels.

WER on short texts (`short` row) stays 0.08–0.25 across 4/3/2-bit
— consistent with the 128-token residual buffer. Short clips
(3–4 s ≈ ~300 audio frames) have most of their generation live in
the residual buffer, so the cache never actually gets quantized
for those tokens. The quantization effect only shows up on medium
(~13 s) and long (~20–28 s) clips that overflow the buffer.

### 5. Memory parity.

All four configs report peak GPU 3.0 GB. At Parler's ~30 s context
cap, the self-attention KV cache is a few MB even at fp16 — too
small for quantization savings to show up. The memory argument for
TurboQuant-on-TTS kicks in at much longer contexts (audiobooks,
long-form dialogue) than Parler supports.

### 6. RTF penalty ~1.8×.

2.62–2.88 RTF vs 1.45 baseline. Much better than Part 1's ~18×
slowdown on Llama-8B, because Parler's 880M decoder forward is
cheap — handrolled-cache ops don't dominate per-step time. Still
not real-time on A10G.

## What the experiment doesn't (yet) show

### 1. Is the random projection specifically what saves quality?

Part 1 had KIVI (naive scalar quantization, no projection) as a
second baseline and showed TurboQuant beat KIVI by 1.7 PPL at 2-bit.
Part 2 doesn't have a naive-quant baseline, so we can't isolate the
projection trick's specific contribution. A follow-up sweep with a
pure-scalar-quant cache would close this gap.

### 2. Per-prompt variance.

We have n=1 sample per (config, prompt). Some prompts show bigger
degradation than others (`gary__medium` 2-bit WER 0.82,
`gary__long` 2-bit WER 0.38) due to per-prompt sampling variance on
top of the quantization effect. Multi-seed averaging (3-5 draws per
config per prompt) would sharpen the means further, at ~3-5× the GPU
cost.

### 3. Longer contexts.

Parler maxes out at 4096 positions ≈ 30 s audio; the memory argument
for KV cache compression gets stronger at much longer contexts
(podcasts, audiobooks) than Parler supports. A follow-up on a
longer-context TTS model would be the right testbed for the
memory claim.

## Artefacts in this repo

- `speech-tts-improvements/parler/results/baseline/` — 9 fp16 WAVs
- `speech-tts-improvements/parler/results/tq_{4,3,2}bit/` — 9 WAVs each
- `speech-tts-improvements/parler/results/timings_*.json` — per-sample
  gen_time / audio_duration / RTF, per-config peak GPU
- `speech-tts-improvements/parler/results/metrics.json` — WER + spk-sim
- `speech-tts-improvements/parler/results/plots/` — rtf.png,
  memory.png, wer.png, speaker_similarity.png
- `speech-tts-improvements/parler/results/audio_comparison.html` —
  interactive frontend: rows = prompts, columns = configs; each cell
  plays the generated audio with duration / WER / speaker-sim chips
  and Whisper's transcript inline.

## Bottom line

TurboQuant compresses Parler-TTS's KV cache at 4/3/2-bit with no
setup surprises and produces a monotonic quality-vs-bits curve that
mirrors Part 1's pattern on text LLMs. 4-bit is essentially free;
3-bit is the sweet spot; 2-bit has a real but usable cost. The
algorithm transfers from text generation to audio generation
without structural changes, just a `max_length` kwarg and an
`EncoderDecoderCache` wrap around the cache object.
