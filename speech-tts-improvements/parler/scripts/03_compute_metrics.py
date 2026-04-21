"""Compute audio quality metrics on the generated WAVs.

Runs on CPU (laptop, after syncing results from AWS).

Metrics:
  - WER: transcribe with Whisper small.en, compare to input text
  - Speaker similarity: WavLM-based embedding cosine similarity vs baseline
  - (UTMOS skipped unless speechmos installed — documented as future work)

Usage:
    python3 speech-tts-improvements/parler/scripts/03_compute_metrics.py
"""
import os
import sys
import json
import numpy as np
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO_ROOT, "speech-tts-improvements", "parler"))

from voices_and_texts import VOICES, TEXTS

RESULTS_DIR = os.path.join(REPO_ROOT, "speech-tts-improvements/parler/results")
CONFIGS = ["baseline", "tq_4bit", "tq_3bit", "tq_2bit", "noproj_2bit", "naive_2bit"]

print("=== Computing audio quality metrics ===\n")

# Lazy imports with graceful fallbacks
try:
    import whisper
    whisper_model = whisper.load_model("small.en")
    print("Whisper loaded.\n")
except ImportError:
    print("whisper not installed. Install with: pip install openai-whisper")
    print("Skipping WER.")
    whisper_model = None

try:
    import jiwer
except ImportError:
    print("jiwer not installed. Install with: pip install jiwer")
    jiwer = None


def compute_wer(audio_path, reference_text):
    if whisper_model is None or jiwer is None:
        return None
    try:
        result = whisper_model.transcribe(audio_path, language="en", verbose=False)
        transcript = result["text"].strip()
        wer_score = jiwer.wer(reference_text.lower(), transcript.lower())
        return {"transcript": transcript, "wer": round(wer_score, 4)}
    except Exception as e:
        return {"error": str(e)}


def compute_speaker_similarity(audio_path, ref_audio_path):
    """Cosine similarity between speaker embeddings. Requires speechbrain."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        import torch
    except ImportError:
        return None

    if not hasattr(compute_speaker_similarity, "_model"):
        compute_speaker_similarity._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.expanduser("~/.speechbrain_cache"),
        )
    classifier = compute_speaker_similarity._model

    try:
        a, sr1 = sf.read(audio_path)
        b, sr2 = sf.read(ref_audio_path)
        # Resample if needed; speechbrain expects 16kHz
        import librosa
        if sr1 != 16000:
            a = librosa.resample(a.astype(np.float32), orig_sr=sr1, target_sr=16000)
        if sr2 != 16000:
            b = librosa.resample(b.astype(np.float32), orig_sr=sr2, target_sr=16000)

        emb_a = classifier.encode_batch(torch.tensor(a).unsqueeze(0)).squeeze()
        emb_b = classifier.encode_batch(torch.tensor(b).unsqueeze(0)).squeeze()
        sim = torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=0).item()
        return round(sim, 4)
    except Exception as e:
        return {"error": str(e)}


metrics = {config: {} for config in CONFIGS}

for config in CONFIGS:
    config_dir = os.path.join(RESULTS_DIR, config)
    if not os.path.isdir(config_dir):
        print(f"Skipping {config} — directory not found")
        continue

    print(f"\n=== {config} ===")
    for voice_name in VOICES:
        for text_name, text in TEXTS.items():
            key = f"{voice_name}__{text_name}"
            audio_path = os.path.join(config_dir, f"{key}.wav")
            if not os.path.exists(audio_path):
                continue

            entry = {}

            # WER
            wer_result = compute_wer(audio_path, text)
            if wer_result:
                entry["wer"] = wer_result

            # Speaker similarity vs baseline
            if config != "baseline":
                baseline_path = os.path.join(RESULTS_DIR, "baseline", f"{key}.wav")
                if os.path.exists(baseline_path):
                    sim = compute_speaker_similarity(audio_path, baseline_path)
                    if sim is not None:
                        entry["speaker_similarity_vs_baseline"] = sim

            metrics[config][key] = entry
            print(f"  {key}: {entry}")

with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved to {os.path.join(RESULTS_DIR, 'metrics.json')}")
