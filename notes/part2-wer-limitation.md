# Part 2 — Why the WER numbers are unusable (and what's salvageable)

**Date:** 2026-04-21
**Applies to:** `speech-tts-improvements/parler/results/metrics.json`

## TL;DR

The WER column in `metrics.json` is pinned to ~0.95 across **all** configs
(including fp16 baseline) and cannot discriminate between quantization
levels. The cause is a generation-side issue that applies to every config
equally, so **WER is not telling us anything about TurboQuant**. The
speaker-similarity column *is* informative and shows the expected
compression-quality curve.

## What we observed

Running `03_compute_metrics.py` on all 36 WAVs produced:

| Config | Mean WER | Median WER | Mean Speaker Sim | Median |
|---|---:|---:|---:|---:|
| baseline | 0.951 | 0.971 | — | — |
| tq_4bit  | 0.946 | 0.971 | 0.834 | 0.893 |
| tq_3bit  | 0.945 | 0.971 | 0.657 | 0.721 |
| tq_2bit  | 0.949 | 0.971 | 0.683 | 0.683 |

Whisper small.en transcripts were extremely short across **every** config,
e.g.

- `jon__short` (ref: "Hello, this is a test of voice quality after KV
  cache compression.") → transcript `"Hello!"` at fp16, 4-bit, 3-bit,
  and 2-bit.
- `gary__long` (ref: ~50-word TurboQuant description) → `"Large Language
  Models Store"` at fp16 as well.

If fp16 itself scores WER 0.95, there's no headroom to detect
quantization damage.

## Root cause

Inspecting the energy envelope of `baseline/jon__short.wav`:

```
t=0.0–4.5s   RMS ≈ 0.29   (real speech — "Hello, this is a test...")
t=4.5s       drop
t=5.0–20.5s  RMS ≈ 0.19   (constant buzz/hum, not silence)
t=21.0–28.8s RMS ≈ 0.21   (different constant hum)
```

For `gary__long.wav`: ~2 s of speech then 28 s of near-silence
(RMS < 0.002).

So two distinct failure modes after the actual text finishes:
1. **Constant buzz/hum** (most clips): non-speech audio that Whisper
   latches onto and uses as an excuse to stop.
2. **Dead silence** (e.g. `gary__long`): Whisper still only returns the
   opening words and gives up.

Both occur in **fp16 baseline**, so they are not caused by quantization.

## Why Parler does this

Our generation scripts
(`speech-tts-improvements/parler/scripts/01_generate_baseline.py`,
`02_generate_turboquant.py`) call `model.generate(..., do_sample=False)`
with **no `max_new_tokens`** argument. Parler then defaults to the full
`max_position_embeddings` (4096 positions ≈ 29.85 s at ~86 fps). For a
3-word prompt like "Hello, this is a test…" (~3 s of speech), the model
keeps emitting audio tokens for ~25 s past the actual content, and
greedy decoding wanders into non-speech codebook regions.

This is a **Parler configuration issue**, not a TurboQuant issue, and
it affects all configs identically.

## What we tried to salvage WER

1. **Truncate to first 8 s** (where speech definitely is) before
   transcription — still `"Hello!"` / hallucinations on baseline.
2. **numpy ndarray vs file path API** — file path gives `"Hello!"`,
   ndarray gives garbage (`"Asarger"`), file path is closer to
   ground truth but still ceiling-limited.
3. **Resample to 16 kHz before passing to Whisper** — same `"Hello!"`.
4. **Explicit `temperature=0.0`** — same `"Hello!"`.

None of these recover real intelligibility numbers because the
generated audio simply doesn't contain ~30 s of intelligible speech in
any config.

## What *is* salvageable from this sweep

### Speaker similarity (ECAPA-TDNN cosine, vs fp16 baseline)

Per-sample, long-content clips (the cases where the residual buffer
matters):

```
               4-bit   3-bit   2-bit
jon__long       0.85    0.76    0.54     ← monotonic voice drift
laura__long     0.61    0.25    0.23     ← voice lost at 3/2-bit
gary__long      0.93    0.14    0.38     ← 3-bit crashes for this sample
laura__medium   0.63    0.48    0.57
jon__medium     1.00    0.99    0.98     ← preserved
short clips     0.97+   0.97+   0.96+    ← preserved (fit inside
                                           the 128-token residual buffer)
```

This gives us the headline result:
- **4-bit preserves voice identity** (mean sim 0.83).
- **3-bit and 2-bit lose voice identity on long clips** (mean sim
  0.66–0.68).
- **Short content is unaffected** at every level because it fits in the
  residual buffer and is never quantized.

### Durations and RMS (in the earlier results README / plots)

`laura__short` truncation (20 s → 17 s → 15 s → 12 s as bits shrink)
and `laura__long` RMS collapse at 3/2-bit corroborate the speaker-sim
story, even though they're coarser signals.

## To properly answer the WER question we would need to

1. Regenerate each config with either:
   - `max_new_tokens` sized to the text length (e.g. ~100 tokens per
     second of expected speech × 1.3 safety margin), **or**
   - a proper stopping-criteria callback that halts when predicted
     audio is non-speech for >0.5 s, **or**
   - Parler's sampling (`do_sample=True, temperature=0.7`) which is
     less prone to the looping-buzz failure than greedy.
2. Re-run `03_compute_metrics.py`.

Any of these would move WER off its ceiling and let the metric actually
discriminate between configs. Rough GPU budget: ~30 min on A10G to
regenerate all 36 WAVs.
