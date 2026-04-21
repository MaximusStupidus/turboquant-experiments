"""Generate fp16 baseline audio: 3 voices × 3 texts = 9 WAVs.

Run on GPU (AWS A10G):
    uv run python speech-tts-improvements/parler/scripts/01_generate_baseline.py

Output: speech-tts-improvements/parler/results/baseline/*.wav + timings.
"""
import os
import sys
import time
import json
import torch
import numpy as np
import soundfile as sf

# Add repo root + parler folder to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "speech-tts-improvements", "parler"))

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed
from voices_and_texts import VOICES, TEXTS, MAX_LENGTH

OUT_DIR = os.path.join(REPO_ROOT, "speech-tts-improvements/parler/results/baseline")
os.makedirs(OUT_DIR, exist_ok=True)

print("=== Parler-TTS baseline generation ===\n")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda:0" else torch.float32
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

# Report model info
decoder_cfg = model.config.decoder
print("=== Decoder config ===")
print(f"  num_hidden_layers    : {decoder_cfg.num_hidden_layers}")
print(f"  num_attention_heads  : {decoder_cfg.num_attention_heads}")
print(f"  num_key_value_heads  : {decoder_cfg.num_key_value_heads}")
print(f"  hidden_size          : {decoder_cfg.hidden_size}")
print(f"  head_dim             : {decoder_cfg.hidden_size // decoder_cfg.num_attention_heads}")
print(f"  sampling_rate        : {model.config.sampling_rate} Hz")
print()

timings = {"config": "fp16_baseline", "per_sample": {}}

torch.cuda.reset_peak_memory_stats() if device == "cuda:0" else None

for voice_name, voice_desc in VOICES.items():
    for text_name, text in TEXTS.items():
        key = f"{voice_name}__{text_name}"
        print(f"Generating: {key}")

        # Fix seed per-sample for reproducibility
        set_seed(42)
        torch.manual_seed(42)

        desc_ids = tokenizer(voice_desc, return_tensors="pt").input_ids.to(device)
        prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        if device == "cuda:0":
            torch.cuda.synchronize()
        t_gen = time.time()
        with torch.no_grad():
            audio = model.generate(
                input_ids=desc_ids,
                prompt_input_ids=prompt_ids,
                max_length=MAX_LENGTH[text_name],
                do_sample=True,
                temperature=1.0,
            )
        if device == "cuda:0":
            torch.cuda.synchronize()
        gen_time = time.time() - t_gen

        audio_arr = audio.float().cpu().numpy().squeeze().astype(np.float32)
        audio_duration = len(audio_arr) / model.config.sampling_rate

        out_path = os.path.join(OUT_DIR, f"{key}.wav")
        sf.write(out_path, audio_arr, model.config.sampling_rate)

        rtf = gen_time / audio_duration
        print(f"  {audio_duration:.2f}s audio in {gen_time:.2f}s (RTF={rtf:.2f})")
        timings["per_sample"][key] = {
            "gen_time_sec": round(gen_time, 3),
            "audio_duration_sec": round(audio_duration, 3),
            "rtf": round(rtf, 3),
        }

if device == "cuda:0":
    timings["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    timings["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

# Save timings
timings_dir = os.path.join(REPO_ROOT, "speech-tts-improvements/parler/results")
with open(os.path.join(timings_dir, "timings_baseline.json"), "w") as f:
    json.dump(timings, f, indent=2)

print(f"\nDone. {len(VOICES) * len(TEXTS)} files in {OUT_DIR}")
print(f"Peak GPU memory: {timings.get('peak_gpu_gb', 'N/A')} GB")
