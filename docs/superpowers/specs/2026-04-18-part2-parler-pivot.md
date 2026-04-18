# Part 2 (Pivot) — TurboQuant on Parler-TTS Mini v1

**Status:** Active. Supersedes the VibeVoice-based Part 2 design (paused — see `notes/part2-vibevoice-blocked.md`).
**Date:** 2026-04-18
**Model:** `parler-tts/parler-tts-mini-v1` (HuggingFace)

---

## 1. Why pivot

VibeVoice-Realtime-0.5B is structurally the right target (autoregressive transformer with KV cache, small enough for A10G), but its internal cache-compatibility shim (`MockCacheLayer`) is incomplete and requires substantial maintenance work on Microsoft's abandoned repo to run on current transformers. That work is out of scope for an experiment whose question is "does TurboQuant work on autoregressive TTS." Any autoregressive TTS answers that question.

**Parler-TTS Mini v1 advantages:**
- 880M params — fits on A10G 23GB with massive headroom
- Standard HuggingFace `DynamicCache` — our existing `HandrolledTurboQuantCache` from Part 1 plugs in as a drop-in
- Text-description-based voice control — no pre-computed `.pt` files to reconstruct
- Encoder-decoder structure, but only the **decoder** has the growing KV cache we target
- Canonical inference example in the model card; well-documented

**The VibeVoice code stays.** It moves to `speech-tts-improvements/vibevoice/` with a note explaining what's blocked and what future work would unblock it.

## 2. Architecture

Encoder-decoder:
- **Encoder:** Flan-T5-large. Runs once per input (not autoregressive). Produces a fixed-size context for cross-attention. **Not our target.**
- **Decoder:** `ParlerTTSDecoder` — GPT-2-style autoregressive transformer with cross-attention to the T5 encoder.
- **Audio codec:** DAC (Descript Audio Codec), 9 parallel codebooks, 44.1kHz output. Runs inside `model.generate()`; emits raw waveform.

Decoder specs (mini-v1):

| Param | Value |
|---|---|
| num_hidden_layers | 24 |
| num_attention_heads | 16 |
| num_key_value_heads | 16 (no GQA — full MHA) |
| head_dim | 64 |
| hidden_size | 1024 |
| ffn_dim | 4096 |
| vocab_size | 1088 (1024 codes + 64 special) |
| num_codebooks | 9 (delay-pattern, **summed into single embedding per step — doesn't multiply sequence length**) |
| max_position_embeddings | 4096 (~30 sec of audio at 86 frames/sec) |

Cache object model: `EncoderDecoderCache(self_attention=DynamicCache, cross_attention=DynamicCache)`. The cross-attention cache is constant-sized (depends only on description length) and is not our target. The self-attention cache grows with generated audio and **is** our target.

## 3. KV cache injection (the clean part)

From the `ParlerTTSDecoder.forward()` source:

```python
if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
    past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
```

**Translation:** if we pass a single `DynamicCache` (or subclass) via `past_key_values=`, the decoder auto-wraps it into the self-attention half of the encoder-decoder cache. We never touch cross-attention.

Injection pattern:

```python
from language_model_improvements.handrolled_turboquant import HandrolledTurboQuantCache

cache = HandrolledTurboQuantCache(
    num_layers=24, num_kv_heads=16, head_dim=64,
    bits=4, device="cuda", dtype=torch.bfloat16,
    residual_length=128,
)

audio = model.generate(
    input_ids=description_ids,
    prompt_input_ids=text_ids,
    past_key_values=cache,  # auto-wrapped
    do_sample=False,
)
```

**Same `HandrolledTurboQuantCache` from Part 1.** Only `num_layers` and `num_kv_heads` change (24 & 16 instead of 32 & 8). The Beta codebook gets recomputed for head_dim=64 (same as VibeVoice would have been — no change from our current defaults, since Part 1 used head_dim=128 and our helper recomputes per call).

## 4. Experimental configs

Same shape as Part 1 but audio-focused:

| Config | Method | Purpose |
|---|---|---|
| fp16 (baseline) | No compression | Reference audio |
| Handrolled TQ 4-bit | Our from-scratch impl | Moderate compression |
| Handrolled TQ 3-bit | Our from-scratch impl | Sweet spot |
| Handrolled TQ 2-bit | Our from-scratch impl | Aggressive compression |

**No KIVI for Part 2.** KIVI was the naive baseline in Part 1 to prove the projection trick matters; we already have that result. Part 2 asks "does the same handrolled method that worked on text tokens also work on audio tokens?" — no need to re-run the KIVI comparison.

**No community TurboQuant** either. Their package is text-LLM-oriented; getting it to inject into Parler's `EncoderDecoderCache` wrapper would require yet more compat work. Our handrolled impl is what we validated in Part 1.

## 5. Test data

**Voices (3 fixed descriptions, all top-5 quality per model card):**

```python
VOICES = {
    "jon":   "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.",
    "laura": "Laura's voice is clear and calm, delivered at a moderate speed with a neutral pitch. The recording is very clear with no background noise.",
    "gary":  "Gary speaks with a slightly expressive tone at a moderate speed. The recording is very clear and close up, with no background noise.",
}
```

**Texts (3 prompts, varying length):**

```python
TEXTS = [
    "short": "Hello, this is a test of voice quality after KV cache compression.",
    "medium": "Hello, this message is from Dr. [redacted], [redacted], [redacted], and [redacted]. We are testing voice quality after KV cache compression on open source TurboQuant.",
    "long": "Large language models store a KV cache during text generation — cached Key and Value vectors from all previous tokens that the model reads back at each step to compute attention. At long context lengths, this cache can exceed the model weights in memory. TurboQuant compresses it.",
]
```

**Total audio files per config:** 3 voices × 3 texts = 9 WAVs per config. 4 configs × 9 = **36 WAVs total**.

**Determinism:** fix `torch.manual_seed(42)` before each generate. Use `do_sample=False` (greedy) for reproducibility (quality will be slightly lower but consistent across configs, which is what matters for comparisons).

## 6. Metrics

| Metric | What it measures | Tool | Runs on |
|---|---|---|---|
| **UTMOS** | Audio naturalness (1-5 MOS) | `speechmos` pip pkg or UTMOS GH repo | CPU (post-hoc) |
| **Speaker similarity** | Voice identity preservation | SpeechBrain ECAPA-TDNN on HF, cosine sim between baseline audio and quantized audio for the same (voice, text) pair | CPU (post-hoc) |
| **WER** | Intelligibility | OpenAI Whisper `small.en` transcription, `jiwer.wer(reference=text, hypothesis=transcript)` | CPU (post-hoc) |
| **RTF** | Generation speed / audio duration | Wall-clock timing inside generate | GPU (at runtime) |
| **Time-to-first-audio** | Latency | Time from generate() call start to first chunk available | GPU (at runtime) — but Parler doesn't stream in the basic API; TTFA here is just total generation time for the shortest clip |
| **Peak GPU memory** | Memory footprint | `torch.cuda.max_memory_allocated()` | GPU |

**Threshold for "TurboQuant works":**
- UTMOS drop ≤ 0.3 compared to baseline
- Speaker similarity ≥ 0.85 cosine to baseline
- WER absolute increase ≤ 5 percentage points

## 7. Scope and timebox

**2-hour hard timebox on GPU execution.** If we don't have working baseline audio + at least one TurboQuant config's audio by then, we stop and reassess.

**Out of scope for this iteration:**
- Torch compile optimization
- Streaming inference
- Parler-TTS Large v1 (3x model, diminishing returns for our question)
- Cross-attention cache quantization (small, constant-sized, not the bottleneck)
- Fine-tuning or training

## 8. Environment pins (AWS g5.2xlarge)

Recommended stack after parler-tts install:

```
torch (cu124 wheels fine on CUDA 13 driver, A10G)
transformers==4.46.1    # matches parler-tts pin, sidesteps all API drift
accelerate (any recent)
numpy<2                 # parler fine-tuning has numpy-2 issues, pin to be safe
soundfile
jiwer                   # WER
openai-whisper          # ASR for WER
speechbrain             # ECAPA-TDNN speaker embeddings
# TurboQuant is built into transformers 4.57+ as QuantizedCache, but since we
# pin to 4.46.1 we use our handrolled one only — simpler.
```

**AWS AMI gotchas to preempt** (learned from VibeVoice attempt):
- Deep Learning Base AMI Ubuntu 24.04 does NOT have `pip` pre-installed; `sudo apt install -y python3-pip python3-venv ffmpeg git build-essential` first.
- Use `uv` for our repo's environment (same as Part 1), install `parler-tts` into the uv venv via `uv pip install git+...`.

## 9. Pre-committed design decisions

1. **Use `HandrolledTurboQuantCache` only** (not community turboquant, not KIVI). Part 1 validated the handrolled impl against the community package; Part 2 reuses that artifact.

2. **Only quantize the self-attention cache** (the growing one). Cross-attention is constant and not the memory bottleneck.

3. **Greedy decoding** (`do_sample=False`, fixed seed) for reproducibility. Quality cost acceptable for comparative eval.

4. **Metrics computed post-hoc on CPU.** GPU generates WAVs; CPU (on laptop after sync) runs UTMOS + speaker sim + WER. This keeps GPU time minimal.

5. **Only if baseline-quality audio emerges, proceed to quantization.** If the baseline sounds like static, there's no point quantizing and measuring degradation.

## 10. Stopping conditions (clear criteria)

Stop and declare done when **any** of these is true:
- All 4 configs produce audio for all 9 (voice × text) combinations, plots generated, notes/04-parler-results.md written → **success**
- 2-hour GPU timebox expired → **partial** — commit what we have, write results note with caveats
- Baseline fp16 audio is garbage (e.g., model weights don't load, codec doesn't decode correctly) → **blocked** — document and park, same as VibeVoice
