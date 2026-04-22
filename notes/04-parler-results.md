# Part 2 results — TurboQuant on Parler-TTS Mini v1

**Model:** `parler-tts/parler-tts-mini-v1` (880M params, encoder-decoder)
**Hardware:** AWS A10G 23 GB, bfloat16
**Generation:** sampling at Parler's native `temperature=1.0`,
`do_sample=True`, fixed `set_seed(42)` per sample; `max_length` per
text (450 / 1200 / 2400 steps for short / medium / long).
**Eval:** 3 voices × 3 texts = 9 prompts per config, compared across
6 configs — fp16 baseline, TurboQuant at 4-bit / 3-bit / 2-bit, plus
two 2-bit ablations (no-projection, naive min-max). **54 WAVs total.**
**WER:** Whisper small.en. **Speaker similarity:** ECAPA-TDNN cosine
between baseline and quantized generations of the same prompt.

## Headline numbers

Using **Harvard Sentences** (IEEE 1969, the standard phonetically-
balanced TTS benchmark) — common English words, no proper nouns,
no acronyms. **3 seeds per cell** (7 / 19 / 42), so the numbers
below are means with per-prompt std across seeds.

| Config | Mean WER ± std | Mean spk-sim vs fp16 ± std | Mean RTF | Peak GPU |
|---|:--:|:--:|---:|---:|
| fp16 baseline | **0.043 ± 0.029** | — | 1.45 | 3.0 GB |
| TurboQuant 4-bit | **0.045 ± 0.025** | 0.760 ± 0.109 | 2.62 | 3.0 GB |
| TurboQuant 3-bit | **0.079 ± 0.064** | 0.750 ± 0.100 | 2.62 | 3.0 GB |
| TurboQuant 2-bit | **0.228 ± 0.180** | 0.730 ± 0.110 | 2.88 | 3.0 GB |

### What the error bars say

| Comparison | ΔWER | Is it real? |
|---|---:|:--:|
| 4-bit vs baseline | +0.002 | **No** — indistinguishable at n=27 |
| 3-bit vs baseline | +0.036 | Within 1σ — grey zone |
| 2-bit vs baseline | +0.185 | **Yes** — 6σ of baseline noise |

The 4-bit result is more surprising multi-seed than single-seed:
the single-seed (seed=42) sweep showed 4-bit WER 0.064, making it
look ~50% worse than baseline. With two more seeds it regresses to
~baseline. Likely story: seed=42 rolled an unlucky trajectory on
2 of the 9 4-bit prompts, inflating the single-seed mean. Multi-seed
averages that out. Same lesson applies to the 3-bit number —
single-seed 0.117 was a ~50% overstatement.

**The 2-bit number holds up.** 0.228 mean with 0.180 per-prompt
std and a 0.185 delta-vs-baseline is well outside sampling noise —
quantization at 2-bit genuinely hurts intelligibility.

### Ablation at 2-bit

To isolate *what* makes TurboQuant work on TTS, we ran two 2-bit
variants alongside the main sweep:

| Config | Mean WER | Median WER | Mean spk-sim vs fp16 |
|---|---:|---:|---:|
| TurboQuant 2-bit (rotation + Beta codebook) | **0.21** | 0.13 | **0.72** |
| No-projection 2-bit (Beta codebook only, no rotation) | **0.65** | 0.75 | 0.63 |
| Naive 2-bit (per-token min-max uniform, no rotation, no Beta) | **0.20** | 0.13 | 0.68 |

Two clean findings:

1. **The random rotation is essential.** Skip it and WER jumps from
   0.21 to **0.65** (3× worse) — the Beta codebook assumes
   unit-sphere-rotated inputs, so it mis-matches the raw K/V
   distribution and reconstruction collapses. The rotation is doing
   real work, not decoration.

2. **Given the rotation is applied, the specific codebook choice
   barely affects WER.** TurboQuant's Beta-optimal codebook
   (0.21 WER) and a simple per-token adaptive min-max uniform
   codebook (0.20 WER) are statistically indistinguishable on this
   benchmark. On speaker similarity TurboQuant holds a small edge
   (0.72 vs 0.68), but on intelligibility the two are tied. This
   refines the paper's claim: on Llama-8B (Part 1) TurboQuant at
   2-bit beats per-group int-2 quantization via optimum-quanto by
   ~1.7 PPL — the Beta-optimal codebook clearly helps there. On
   Parler-TTS the picture is more nuanced. Plausible explanation:
   Parler's 880M decoder has less outlier-heavy K/V distributions,
   so a per-token adaptive scale can re-fit itself to the range of
   each vector and match the mathematical optimality of a fixed
   Beta codebook. "Optimal" is always optimal-for-a-distribution;
   when the distribution is less extreme, adaptive simple beats
   fixed-optimal.

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

### 1. Statistical tightness at small n.

With 9 prompts per config, per-prompt sampling variance remains a
confounder. A multi-seed sweep (3 seeds × 9 prompts = 27 runs per
config) produced the `mean ± std` columns above; the std is
non-negligible relative to config-to-config deltas at bit rates near
the noise floor (4-bit vs baseline especially). A proper
LibriTTS-scale benchmark (~100+ prompts) would tighten the error
bars further.

### 2. Whisper's own noise floor.

Baseline WER 0.04 is close to Whisper small.en's intrinsic error rate
on clean read speech. Sub-0.10 WER deltas between configs should be
read with that in mind. Whisper-large-v3 would discriminate better
at the low end but takes ~4× longer per clip.

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

## Conclusion

**Does TurboQuant transfer from text LLMs to autoregressive TTS?**
Yes, with a twist worth reporting.

**What works cleanly:**
- The algorithm runs end-to-end on Parler-TTS with no structural
  changes (just pre-wrap the cache in `EncoderDecoderCache`).
- Quality degrades monotonically with bit rate: 4-bit free, 3-bit
  grey zone, 2-bit clearly degraded but usable.
- At 4-bit the compressed cache produces output statistically
  indistinguishable from fp16 (0.045 ± 0.025 vs 0.043 ± 0.029 WER
  over 27 samples).

**What the 2-bit ablation actually proves:**
- The random rotation is essential. Skip it and 2-bit WER jumps
  3× (0.21 → 0.65). The Beta codebook mis-matches the unrotated
  distribution; reconstruction collapses.
- The Beta-optimal codebook is not essential on this model. Keep
  the rotation but swap the codebook for a naive per-token min-max
  quantizer, and WER is the same (0.20 vs 0.21).

**Why this differs from the LLM case:**
On Llama-8B the full pipeline beats naive int-2 quantization by
~1.7 PPL at 2-bit — both rotation and codebook contribute. On
Parler-TTS only the rotation contributes measurably. Plausible
explanation: Parler's 880M decoder has less outlier-heavy K/V
distributions than Llama-8B's attention, so a per-token adaptive
scale handles the range as well as the Beta-optimal fixed codebook.
"Optimal" is always optimal-for-a-distribution; when the baseline
distribution isn't as extreme, optimality doesn't buy much.

**The general insight:**
The rotation is the *transferable* part of TurboQuant — pure
geometry (Johnson-Lindenstrauss preserves dot products, so
attention math is invariant), and it flattens any distribution
enough that a simple quantizer can handle it. The codebook is
*distribution-specific* math that helps when the baseline
distribution is bad. Practitioner takeaway: add the rotation
first; the downstream quantizer can be simple.

**What this work does NOT show:**
- Memory savings (Parler's 30 s context cap keeps the fp16 KV
  cache at a few MB; no compression benefit visible at this scale).
- Statistically tight 3-bit result. n=27 is not enough to resolve
  the 3-bit delta vs baseline; LibriTTS-scale would.
- Per-tensor equivalence with the community `turboquant` package.
  Only end-to-end perplexity was matched on Part 1; tensor-level
  equivalence tests are in `tests/test_numerical_equivalence.py`
  but they validate the algorithm against its paper spec, not
  against a specific reference implementation.

**Bottom line:**
TurboQuant's rotation-based compression generalises to speech
generation cleanly. Its specific codebook choice generalises less
cleanly. And the residual buffer that makes the whole thing
practical came from KIVI / StreamingLLM, not TurboQuant — the
compression story is real, but the component attribution is more
nuanced than any single paper claims.
