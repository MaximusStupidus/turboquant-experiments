"""Generate TurboQuant-compressed audio at 4-bit, 3-bit, 2-bit.

Uses our HandrolledTurboQuantCache from Part 1 (language-model-improvements/handrolled_turboquant.py).
The decoder will auto-wrap our cache into EncoderDecoderCache(ours, DynamicCache()).

Run on GPU:
    uv run python speech-tts-improvements/parler/scripts/02_generate_turboquant.py

Output: speech-tts-improvements/parler/results/tq_{4bit,3bit,2bit}/*.wav + timings
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

print("=== Parler-TTS TurboQuant sweep ===\n")

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

# Decoder config for cache
decoder_cfg = model.config.decoder
num_layers = decoder_cfg.num_hidden_layers
num_kv_heads = decoder_cfg.num_key_value_heads
head_dim = decoder_cfg.hidden_size // decoder_cfg.num_attention_heads
print(f"Decoder: {num_layers} layers × {num_kv_heads} KV heads × head_dim={head_dim}\n")


def run_config(bits: int, label: str):
    out_dir = os.path.join(RESULTS_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    timings = {"config": label, "bits": bits, "per_sample": {}}

    print(f"\n{'='*60}")
    print(f"CONFIG: {label} (bits={bits})")
    print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats()

    for voice_name, voice_desc in VOICES.items():
        for text_name, text in TEXTS.items():
            key = f"{voice_name}__{text_name}"
            print(f"  Generating: {key}")

            set_seed(42)
            torch.manual_seed(42)

            # Fresh cache per generation — new random projection matrices
            # (fixed seed inside HandrolledTurboQuantCache for reproducibility)
            inner_cache = HandrolledTurboQuantCache(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                bits=bits,
                device=device,
                dtype=dtype,
                residual_length=128,
                seed=42,
            )
            # Parler's prepare_inputs_for_generation expects EncoderDecoderCache;
            # without the wrap it dereferences an empty cache via legacy tuple
            # indexing. Wrap explicitly so get_seq_length() is used instead.
            cache = EncoderDecoderCache(inner_cache, DynamicCache())

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
                        temperature=0.7,
                    )
                torch.cuda.synchronize()
                gen_time = time.time() - t_gen

                audio_arr = audio.float().cpu().numpy().squeeze().astype(np.float32)
                audio_duration = len(audio_arr) / model.config.sampling_rate
                out_path = os.path.join(out_dir, f"{key}.wav")
                sf.write(out_path, audio_arr, model.config.sampling_rate)

                rtf = gen_time / audio_duration
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


for bits, label in [(4, "tq_4bit"), (3, "tq_3bit"), (2, "tq_2bit")]:
    run_config(bits, label)

print("\nAll configs done.")
