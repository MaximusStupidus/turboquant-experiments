"""Multi-seed sweep — tightens the Part 2 error bars.

The main sweep (scripts 01 + 02) uses seed 42 for every config +
prompt, giving a single sample per cell. Sampling variance can
confound the quantization signal at bit rates near the noise floor
(4-bit vs baseline especially).

This script re-runs the 4 main configs (baseline + TurboQuant at 4/3/2-bit)
across 3 seeds and writes all outputs under
`results/seed<s>/<config>/<prompt>.wav`, letting
`03_compute_metrics.py` compute mean + std across seeds.

We intentionally SKIP the ablations (noproj_2bit, naive_2bit) here —
those were one-shot sanity checks and their single-seed numbers are
already informative. Saving ~50% GPU time for the more valuable
multi-seed-on-main-configs.

Run on GPU:
    uv run --no-sync python speech-tts-improvements/parler/scripts/07_generate_multiseed.py

Expected duration on A10G: ~1 hour total (12 configs × 9 prompts × ~40 s).
"""
import os
import sys
import time
import json
import torch
import numpy as np
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "speech-tts-improvements", "parler"))

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed, EncoderDecoderCache, DynamicCache
from voices_and_texts import VOICES, TEXTS, MAX_LENGTH
from language_model_improvements.handrolled_turboquant import HandrolledTurboQuantCache

RESULTS_DIR = os.path.join(REPO_ROOT, "speech-tts-improvements/parler/results")
SEEDS = [7, 19]  # seed 42 already covered by scripts 01 + 02; add these two more.

print("=== Parler-TTS multi-seed sweep ===\n")

device = "cuda:0"
dtype = torch.bfloat16
print(f"Device: {device}, dtype: {dtype}\n")

print("Loading model...")
t0 = time.time()
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    torch_dtype=dtype,
).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
print(f"Loaded in {time.time() - t0:.1f}s\n")

decoder_cfg = model.config.decoder
num_layers = decoder_cfg.num_hidden_layers
num_kv_heads = decoder_cfg.num_key_value_heads
head_dim = decoder_cfg.hidden_size // decoder_cfg.num_attention_heads


def run_one(config_name: str, bits_or_none, seed: int):
    """Generate all 9 prompts at one config+seed. Writes to
    results/seed<seed>/<config_name>/<prompt>.wav."""
    out_dir = os.path.join(RESULTS_DIR, f"seed{seed}", config_name)
    os.makedirs(out_dir, exist_ok=True)
    timings = {"config": config_name, "seed": seed, "per_sample": {}}

    for voice_name, voice_desc in VOICES.items():
        for text_name, text in TEXTS.items():
            key = f"{voice_name}__{text_name}"
            print(f"  [seed={seed}] {config_name}/{key}")

            set_seed(seed)
            torch.manual_seed(seed)

            past_kv = None
            if bits_or_none is not None:
                inner = HandrolledTurboQuantCache(
                    num_layers=num_layers, num_kv_heads=num_kv_heads,
                    head_dim=head_dim, bits=bits_or_none, device=device,
                    dtype=dtype, residual_length=128, seed=42,
                )
                past_kv = EncoderDecoderCache(inner, DynamicCache())

            desc_ids = tokenizer(voice_desc, return_tensors="pt").input_ids.to(device)
            prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

            torch.cuda.synchronize()
            t_gen = time.time()
            kwargs = dict(
                input_ids=desc_ids, prompt_input_ids=prompt_ids,
                max_length=MAX_LENGTH[text_name],
                do_sample=True, temperature=1.0,
            )
            if past_kv is not None:
                kwargs["past_key_values"] = past_kv
            with torch.no_grad():
                audio = model.generate(**kwargs)
            torch.cuda.synchronize()
            gen_time = time.time() - t_gen

            audio_arr = audio.float().cpu().numpy().squeeze().astype(np.float32)
            audio_duration = len(audio_arr) / model.config.sampling_rate
            out_path = os.path.join(out_dir, f"{key}.wav")
            sf.write(out_path, audio_arr, model.config.sampling_rate)

            rtf = gen_time / audio_duration if audio_duration > 0 else float("inf")
            timings["per_sample"][key] = {
                "gen_time_sec": round(gen_time, 3),
                "audio_duration_sec": round(audio_duration, 3),
                "rtf": round(rtf, 3),
            }

    with open(os.path.join(RESULTS_DIR, f"timings_seed{seed}_{config_name}.json"), "w") as f:
        json.dump(timings, f, indent=2)


configs = [
    ("baseline", None),
    ("tq_4bit", 4),
    ("tq_3bit", 3),
    ("tq_2bit", 2),
]

t_total = time.time()
for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
    for config_name, bits in configs:
        run_one(config_name, bits, seed)

print(f"\nTotal elapsed: {(time.time() - t_total)/60:.1f} min")
print("Multi-seed sweep done.")
