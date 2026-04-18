# Parler-TTS experiments (Part 2 pivot)

Applies TurboQuant KV cache quantization to Parler-TTS Mini v1, the autoregressive decoder-only TTS model from HuggingFace.

See `docs/superpowers/specs/2026-04-18-part2-parler-pivot.md` for the full design and `docs/superpowers/plans/2026-04-18-part2-parler-impl.md` for the execution plan.

## What's here

- `voices_and_texts.py` — fixed test data (3 voices × 3 texts)
- `scripts/00_check_setup.py` — verify AWS environment
- `scripts/01_generate_baseline.py` — generate fp16 baseline audio (9 WAVs)
- `scripts/02_generate_turboquant.py` — sweep 4/3/2-bit TurboQuant (27 WAVs)
- `scripts/03_compute_metrics.py` — CPU-side metrics (WER, speaker similarity)
- `scripts/04_plot_results.py` — comparison plots

## Running

See the implementation plan for step-by-step instructions.

## Why we're using Parler (not VibeVoice)

VibeVoice was the original Part 2 target but has a cache-API compatibility
issue with current transformers that requires rewriting its internal
`MockCacheLayer`. See `speech-tts-improvements/vibevoice/` and
`notes/part2-vibevoice-blocked.md` for the preserved code + diagnosis.
