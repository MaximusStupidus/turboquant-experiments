"""Ablation sweep at 2-bit to isolate what makes TurboQuant work on TTS.

Runs the 9 Harvard-sentence prompts through two additional cache configs,
both at 2-bit (where the quantization effect is largest):

  noproj_2bit: HandrolledTurboQuantCache with use_projection=False.
               Skip the random rotation; keep the Beta codebook,
               residual buffer, and everything else. Isolates the
               projection step's specific contribution to quality.

  naive_2bit:  NaiveQuantCache — per-channel min-max uniform
               quantization. No projection, no Beta codebook. Residual
               buffer matches TurboQuant (128 tokens) for a fair
               comparison. This is the KIVI-style baseline that Part 1
               showed TurboQuant beats by 1.7 PPL on Llama-8B.

The existing tq_2bit run (in results/tq_2bit/) stays as the "full
TurboQuant" reference. Three-way comparison at 2-bit:
  - results/tq_2bit/         full pipeline (projection + Beta codebook)
  - results/noproj_2bit/     no projection, same codebook + buffer
  - results/naive_2bit/      pure min-max uniform, no projection

If TurboQuant 2-bit beats BOTH, the full pipeline is justified.
If noproj_2bit == tq_2bit, the projection is doing nothing for TTS.
If naive_2bit == tq_2bit, neither the projection nor the Beta codebook
matter for TTS — any 2-bit quant of the KV cache is fine.

Run on GPU:
    uv run --no-sync python speech-tts-improvements/parler/scripts/06_generate_ablation.py
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
from language_model_improvements.handrolled_turboquant import (
    HandrolledTurboQuantCache,
    NaiveQuantCache,
)

RESULTS_DIR = os.path.join(REPO_ROOT, "speech-tts-improvements/parler/results")

print("=== Parler-TTS ablation sweep (2-bit only) ===\n")

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
print(f"Decoder: {num_layers} layers × {num_kv_heads} KV heads × head_dim={head_dim}\n")


def make_cache(kind: str):
    """Build a fresh cache for one generation."""
    if kind == "noproj":
        inner = HandrolledTurboQuantCache(
            num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
            bits=2, device=device, dtype=dtype,
            residual_length=128, seed=42, use_projection=False,
        )
    elif kind == "naive":
        inner = NaiveQuantCache(
            num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
            bits=2, device=device, dtype=dtype, residual_length=128,
        )
    else:
        raise ValueError(kind)
    return EncoderDecoderCache(inner, DynamicCache())


def run_config(kind: str, label: str):
    out_dir = os.path.join(RESULTS_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    timings = {"config": label, "bits": 2, "per_sample": {}}

    print(f"\n{'='*60}")
    print(f"CONFIG: {label} (kind={kind}, bits=2)")
    print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats()

    for voice_name, voice_desc in VOICES.items():
        for text_name, text in TEXTS.items():
            key = f"{voice_name}__{text_name}"
            print(f"  Generating: {key}")

            set_seed(42)
            torch.manual_seed(42)

            cache = make_cache(kind)

            desc_ids = tokenizer(voice_desc, return_tensors="pt").input_ids.to(device)
            prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

            torch.cuda.synchronize()
            t_gen = time.time()
            try:
                with torch.no_grad():
                    audio = model.generate(
                        input_ids=desc_ids,
                        prompt_input_ids=prompt_ids,
                        past_key_values=cache,
                        max_length=MAX_LENGTH[text_name],
                        do_sample=True,
                        temperature=1.0,
                    )
                torch.cuda.synchronize()
                gen_time = time.time() - t_gen

                audio_arr = audio.float().cpu().numpy().squeeze().astype(np.float32)
                audio_duration = len(audio_arr) / model.config.sampling_rate
                out_path = os.path.join(out_dir, f"{key}.wav")
                sf.write(out_path, audio_arr, model.config.sampling_rate)

                rtf = gen_time / audio_duration if audio_duration > 0 else float("inf")
                print(f"    {audio_duration:.2f}s audio in {gen_time:.2f}s (RTF={rtf:.2f})")
                timings["per_sample"][key] = {
                    "gen_time_sec": round(gen_time, 3),
                    "audio_duration_sec": round(audio_duration, 3),
                    "rtf": round(rtf, 3),
                    "success": True,
                }
            except Exception as e:
                print(f"    FAILED: {e}")
                timings["per_sample"][key] = {"success": False, "error": str(e)}

    timings["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    timings["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    with open(os.path.join(RESULTS_DIR, f"timings_{label}.json"), "w") as f:
        json.dump(timings, f, indent=2)

    print(f"  Peak GPU memory: {timings['peak_gpu_gb']} GB")


for kind, label in [("noproj", "noproj_2bit"), ("naive", "naive_2bit")]:
    run_config(kind, label)

print("\nAblation sweep done.")
