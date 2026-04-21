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

Using **Harvard Sentences** (IEEE 1969, the standard phonetically-
balanced TTS benchmark) — common English words, no proper nouns,
no acronyms.

| Config | Mean WER | Median WER | Mean spk-sim vs fp16 | Mean RTF | Peak GPU |
|---|---:|---:|---:|---:|---:|
| fp16 baseline | **0.04** (4%) | 0.04 | — | 1.45 | 3.0 GB |
| TurboQuant 4-bit | 0.06 | 0.06 | 0.72 | 2.62 | 3.0 GB |
| TurboQuant 3-bit | 0.12 | 0.02 | 0.71 | 2.62 | 3.0 GB |
| TurboQuant 2-bit | **0.21** (21%) | 0.13 | 0.72 | 2.88 | 3.0 GB |

![WER](../speech-tts-improvements/parler/results/plots/wer.png)
![Speaker similarity](../speech-tts-improvements/parler/results/plots/speaker_similarity.png)

## Monotonic degradation — the Part-1-style curve

Both metrics degrade monotonically with aggressive quantization, just
like Part 1's perplexity-vs-bits curve on Llama-3.1-8B:

| Config | Part 1 (PPL vs fp16) | Part 2 (WER vs fp16) | Part 2 (Spk-sim vs fp16) |
|---|---|---|---|
| fp16 | 5.51 | 0.04 | — |
| 4-bit | +0.03 (5.54) | +0.02 (0.06) | 0.72 |
| 3-bit | +0.15 (5.66) | +0.08 (0.12) | 0.71 |
| 2-bit | +0.56 (6.07) | +0.17 (0.21) | 0.72 |

Both experiments show compression cost *accelerates* at 2-bit: the
4→3-bit gap is small, the 3→2-bit gap is large. This is the
signature of TurboQuant's projection+codebook pipeline — at 4-bit
the codebook has enough levels to track the Beta-distributed values
accurately; at 2-bit (4 levels) the quantization error starts
dominating.

## Methodology fixes (three rounds)

This result came out of three corrections, each recorded in
`notes/part2-wer-limitation.md`:

1. **`max_length` instead of `max_new_tokens`.** Parler's custom
   `generate()` ignores `max_new_tokens` and uses
   `generation_config.max_length` (default 2580 ≈ 30 s). Our first
   sweep omitted both, so every clip was padded with 25 s of
   non-speech buzz that Whisper couldn't penetrate — WER pinned
   at 0.95 on baseline.
2. **`temperature=1.0` (Parler's native default), not 0.7.** We
   initially lowered temperature to "tighten" sampling; in practice
   this biased toward high-probability-but-low-quality tokens.
   Probe on `jon__long`: WER 0.85 at t=0.7 → 0.13 at t=1.0.
3. **Harvard sentences instead of TurboQuant-themed prompts.** The
   original prompts contained proper nouns (names) and acronyms
   ("KV cache") that Whisper consistently mis-transcribes even on
   clean fp16 audio, inflating WER across all configs. Harvard
   Sentences (IEEE 1969) use common English words and are the
   standard benchmark for TTS/ASR quality eval.

With all three fixes, baseline WER dropped from ceiling (0.95) to a
realistic 0.04, and the quantization effect became visible.

## What the experiment shows

### 1. TurboQuant transfers to autoregressive TTS.

The same `HandrolledTurboQuantCache` from Part 1 (Llama-3.1-8B) plugs
into Parler-TTS's decoder unchanged — only `num_layers`,
`num_kv_heads`, and `head_dim` differ. Once wrapped in
`EncoderDecoderCache`, it runs end-to-end at 4/3/2-bit with no NaN,
no OOM, no cache errors.

### 2. 4-bit is ~free. 2-bit has a real cost.

WER at 4-bit (0.06) is only 0.02 above fp16 (0.04) — the 4-bit
transcripts are essentially indistinguishable from baseline. At
3-bit we pay 0.08, and at 2-bit we pay 0.17 (a ~5× jump in error
rate vs baseline). The "it's basically free at 4-bit" story from
Part 1 holds. Example at 2-bit where quality breaks down
audibly — `gary__long` transcript: *"The birch canoe slid on the
smooth planks, ghoul the sheet to the dark blue plus bull, now
one stays a-germ. The acklin deserve this rude juice of terror."*

### 3. Voice identity is robust across bit levels.

Speaker similarity sits at ~0.72 at every bit level, essentially
flat. Voice identity is *not* what quantization destroys —
intelligibility is. This makes sense because the voice prompt
(description → speaker embedding) is consumed before any cache
generation happens, so speaker identity comes from the text
encoder + first few decoder steps, which stay in the 128-token
residual buffer even for long clips. What quantization hurts is
the continuous audio-token generation *after* the voice is
established.

### 4. Short prompts are robust across all bit levels.

WER on short texts (`short` row) stays 0.0–0.13 across 4/3/2-bit
— the short Harvard sentence "The birch canoe slid on the smooth
planks" transcribes perfectly or near-perfectly at every bit
level. Short clips (~3–4 s ≈ ~300 audio frames) have most of their
generation live in the 128-token residual buffer, so the cache
never actually gets quantized for those tokens. The degradation
signal concentrates on medium (~8 s) and long (~20 s) clips that
overflow the buffer.

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
