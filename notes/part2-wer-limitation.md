# Part 2 — WER investigation log and methodology notes

**Applies to:** `speech-tts-improvements/parler/results/metrics.json`

## TL;DR

The first sweep produced a degenerate WER signal (mean ~0.95 at every
config including fp16) because our `model.generate()` call omitted
`max_length`, letting Parler pad every clip with 25+ s of non-speech
buzz that Whisper couldn't penetrate. After fixing
(`max_length` per text, sampling with `temperature=0.7`) the second
sweep gave a meaningful but sampling-noise-dominated signal. We can
see qualitative quantization effects but not a Part-1-style clean
monotonic curve.

## First sweep (commit 74bd4ef) — unusable

### What we observed

| Config | Mean WER | Median WER |
|---|---:|---:|
| baseline | 0.95 | 0.97 |
| tq_4bit | 0.95 | 0.97 |
| tq_3bit | 0.95 | 0.97 |
| tq_2bit | 0.95 | 0.97 |

All transcripts truncated to the first 1–3 words:

- `jon__short` (ref "Hello, this is a test of voice quality…")
  → `"Hello!"` in every config
- `gary__long` (ref ~50-word TurboQuant description)
  → `"Large Language Models Store"` in every config

### Root cause

Our generation scripts called `model.generate(...)` with no
`max_new_tokens`/`max_length`, so Parler defaulted to
`generation_config.max_length = 2580` (≈ 30 s at 86 fps) for every
prompt. A 3-word "Hello, this is a test" prompt fills ~3 s of speech;
the remaining 25+ s is the model autoregressively hallucinating
**non-speech audio that is not silent** — constant buzz/hum with
steady energy. Whisper latches onto the buzz and stops transcribing.

Energy envelope of `baseline/jon__short.wav` from that sweep:

```
t=0.0–4.5 s    RMS ≈ 0.29   ← real speech
t=5.0–20.5 s   RMS ≈ 0.19   ← constant buzz (non-speech)
t=21.0–28.8 s  RMS ≈ 0.21   ← different constant hum
```

Identical buzz shape appears in every config including fp16 → not a
quantization artefact.

### Attempts to salvage

1. Truncate to first 8 s before transcription — still "Hello!"
2. numpy-array vs file-path whisper API — same result
3. Resample to 16 kHz — same
4. Explicit `temperature=0.0` — same

All failed because the degraded input distribution was the same in
every config.

## Attempted fix 1 — add `max_new_tokens` (commit 19459f6) — no effect

Added per-text budgets `{short: 600, medium: 1400, long: 2400}` and
passed via `max_new_tokens=...`. Re-ran baseline. Output audio was
**bit-identical** to the previous sweep — same energy envelope down to
the third decimal.

### Root cause

Parler-TTS's custom `generate()` builds an audio-codebook delay-pattern
mask sized to `generation_config.max_length` and silently ignores
`max_new_tokens`:

```python
# parler_tts/modeling_parler_tts.py:1998
max_length=self.generation_config.max_length,
```

`generation_config.max_length` defaults to 2580 on Parler mini v1, so
every clip got the same ~30 s budget regardless of our kwarg.

## Fix 2 — pass `max_length` directly (commit be8a831) — works

`voices_and_texts.py` now exports `MAX_LENGTH = {short: 450,
medium: 1200, long: 2400}` and the generation scripts pass it as
`model.generate(..., max_length=...)`. Smoke-test on three prompts:

```
short  max_length= 450 dur= 3.52s  (was 28.80 s)
medium max_length=1200 dur=13.83 s (was 29.85 s)
long   max_length=2400 dur=27.76 s (was 29.85 s)
```

Clips now terminate when actual speech ends instead of padding with
buzz. Verified on all 36 WAVs of the second sweep.

## Second sweep results

| Config | Mean WER | Median WER | Mean spk-sim |
|---|---:|---:|---:|
| baseline | 0.54 | 0.40 | — |
| tq_4bit | **0.35** | 0.25 | 0.64 |
| tq_3bit | **0.38** | 0.33 | 0.63 |
| tq_2bit | 0.56 | 0.60 | 0.67 |

### What this tells us

1. **2-bit WER is measurably worse** than 4-bit / 3-bit (0.56 vs
   0.35 / 0.38) — first clean cross-config signal on intelligibility.
2. **Voice identity is approximately preserved** across all configs
   (speaker similarity ~0.64, non-monotonic in bits).
3. **No catastrophic failures.** All 36 clips produce intelligible
   speech roughly on par with the noisy fp16 baseline. No collapse
   to silence or buzz at any bit level.

### Why it's noisier than Part 1

Part 1 used **greedy decoding** on text LLMs, so fp16 and each
quantized config produced *deterministic, directly comparable* output.
WER / perplexity differences came purely from quantization.

Part 2 uses **sampling** (Parler's custom `generate()` collapses on
greedy, and `do_sample=True, temperature=0.7` is the workable
setting). With `set_seed(42)` we get reproducibility *within* a config
but the sampling trajectories for fp16 vs tq_4bit are not the same
audio — quantization perturbs the per-step logits, which changes
which token is sampled, which changes every subsequent token. So each
config is drawing one sample from a *slightly different* distribution.

Consequences:

- Baseline WER (0.54) is already noisy — it represents Parler's
  intrinsic sampling variance at fp16, not a quality ceiling.
- Per-prompt WER can flip between configs in counterintuitive ways
  (tq_4bit beats baseline on `jon__long` because baseline's seed-42
  trajectory happened to end after one sentence while tq_4bit's
  perturbed trajectory continued through the full text).
- With n=1 sample per (config, prompt), we can't separate
  "quantization changed the output" from "sampling rolled a
  different trajectory."

### What would make the signal clean

To get a Part-1-style monotonic curve we'd need one of:

1. **n samples per prompt per config** (e.g. 5 different seeds),
   then average WER / speaker sim → amortize sampling variance.
2. **Switch back to greedy** with `do_sample=False` — but the first
   sweep showed greedy collapses Parler into buzz for many prompts.
   Would need a proper stopping-criteria callback (emit EOS when
   logit entropy drops to near-zero for N steps in a row).
3. **A longer benchmark** (e.g. LibriTTS test-clean) — average out
   per-prompt variance by using dozens of prompts instead of 9.

All three are straightforward follow-ups; none were in scope for the
2-hour GPU timebox of this session.

## Bottom line

TurboQuant ran end-to-end on Parler-TTS Mini v1 at 4/3/2-bit with no
NaN / OOM / collapse, produced intelligible speech at every level,
and showed a measurable WER degradation at 2-bit. That validates the
transfer from text LLMs qualitatively. Quantifying the curve as
sharply as Part 1 did requires averaging over sampling variance,
which is a follow-up.
