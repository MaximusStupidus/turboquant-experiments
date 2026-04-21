"""Aggregate WER + speaker similarity across the multi-seed sweep.

Reads:
  - results/<config>/<prompt>.wav           (seed 42, from scripts 01/02)
  - results/seed7/<config>/<prompt>.wav     (from script 07)
  - results/seed19/<config>/<prompt>.wav    (from script 07)

For each (config, prompt), computes WER against the reference text
via Whisper small.en and speaker similarity against the seed-42
fp16 baseline of the same prompt via ECAPA-TDNN. Then aggregates
across the 3 seeds per (config, prompt) into mean + std.

Writes:
  - results/metrics_multiseed.json    per-seed raw numbers
  - results/metrics_aggregated.json   mean ± std tables
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
SEEDS = [42, 7, 19]
CONFIGS = ["baseline", "tq_4bit", "tq_3bit", "tq_2bit"]

print("=== Multi-seed metrics aggregation ===\n")

try:
    import whisper
    import jiwer
    whisper_model = whisper.load_model("small.en")
    print("Whisper loaded.\n")
except ImportError as e:
    print(f"FATAL: {e}")
    sys.exit(1)


def compute_wer(audio_path: str, reference_text: str):
    try:
        result = whisper_model.transcribe(audio_path, language="en", verbose=False)
        transcript = result["text"].strip()
        wer = jiwer.wer(reference_text.lower(), transcript.lower())
        return {"transcript": transcript, "wer": round(wer, 4)}
    except Exception as e:
        return {"error": str(e)}


# Lazy speaker-similarity model
_spk_model = None
def compute_speaker_similarity(path_a: str, path_b: str):
    global _spk_model
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        import torch, librosa
    except ImportError:
        return None
    if _spk_model is None:
        _spk_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.expanduser("~/.speechbrain_cache"),
        )
    try:
        a, sr1 = sf.read(path_a)
        b, sr2 = sf.read(path_b)
        if sr1 != 16000:
            a = librosa.resample(a.astype(np.float32), orig_sr=sr1, target_sr=16000)
        if sr2 != 16000:
            b = librosa.resample(b.astype(np.float32), orig_sr=sr2, target_sr=16000)
        emb_a = _spk_model.encode_batch(torch.tensor(a).unsqueeze(0)).squeeze()
        emb_b = _spk_model.encode_batch(torch.tensor(b).unsqueeze(0)).squeeze()
        sim = torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=0).item()
        return round(sim, 4)
    except Exception as e:
        return {"error": str(e)}


def wav_path(seed: int, config: str, key: str) -> str:
    if seed == 42:
        return os.path.join(RESULTS_DIR, config, f"{key}.wav")
    return os.path.join(RESULTS_DIR, f"seed{seed}", config, f"{key}.wav")


raw = {}   # [seed][config][key] = {wer, speaker_similarity_vs_baseline}

for seed in SEEDS:
    raw[seed] = {}
    for config in CONFIGS:
        raw[seed][config] = {}
        for voice_name in VOICES:
            for text_name, text in TEXTS.items():
                key = f"{voice_name}__{text_name}"
                p = wav_path(seed, config, key)
                if not os.path.exists(p):
                    print(f"  missing: {p}")
                    continue
                entry = {}
                w = compute_wer(p, text)
                if w is not None and "wer" in w:
                    entry["wer"] = w["wer"]
                    entry["transcript"] = w.get("transcript", "")
                if config != "baseline":
                    # Compare to SAME-seed baseline so we're measuring
                    # quantization drift, not seed drift.
                    bp = wav_path(seed, "baseline", key)
                    if os.path.exists(bp):
                        sim = compute_speaker_similarity(p, bp)
                        if isinstance(sim, float):
                            entry["speaker_similarity"] = sim
                raw[seed][config][key] = entry
                wer_str = f"{entry.get('wer', '?'):.3f}" if isinstance(entry.get('wer'), float) else '?'
                sim_str = f"{entry.get('speaker_similarity', '?'):.3f}" if isinstance(entry.get('speaker_similarity'), float) else '-'
                print(f"  seed={seed:3d} {config:10s} {key:16s} wer={wer_str} sim={sim_str}")

with open(os.path.join(RESULTS_DIR, "metrics_multiseed.json"), "w") as f:
    json.dump(raw, f, indent=2)
print(f"\nwrote metrics_multiseed.json")

# ---- Aggregate across seeds ----
print("\n=== Aggregation (mean ± std across seeds) ===\n")
agg = {}
for config in CONFIGS:
    agg[config] = {}
    for voice_name in VOICES:
        for text_name in TEXTS:
            key = f"{voice_name}__{text_name}"
            wers, sims = [], []
            for seed in SEEDS:
                e = raw.get(seed, {}).get(config, {}).get(key, {})
                if isinstance(e.get("wer"), float):
                    wers.append(e["wer"])
                if isinstance(e.get("speaker_similarity"), float):
                    sims.append(e["speaker_similarity"])
            agg[config][key] = {
                "wer_mean": float(np.mean(wers)) if wers else None,
                "wer_std": float(np.std(wers, ddof=0)) if len(wers) >= 2 else 0.0,
                "wer_n": len(wers),
                "sim_mean": float(np.mean(sims)) if sims else None,
                "sim_std": float(np.std(sims, ddof=0)) if len(sims) >= 2 else 0.0,
                "sim_n": len(sims),
            }

# Per-config mean across all 9 prompts
print(f'{"config":12s} {"mean WER ± std":>18s} {"mean spk_sim ± std":>22s}')
for config in CONFIGS:
    wers = [agg[config][k]["wer_mean"] for k in agg[config]
            if isinstance(agg[config][k]["wer_mean"], float)]
    sims = [agg[config][k]["sim_mean"] for k in agg[config]
            if isinstance(agg[config][k]["sim_mean"], float)]
    wer_str = f"{np.mean(wers):.3f} ± {np.std(wers, ddof=0):.3f}" if wers else "—"
    sim_str = f"{np.mean(sims):.3f} ± {np.std(sims, ddof=0):.3f}" if sims else "—"
    print(f"{config:12s} {wer_str:>18s} {sim_str:>22s}")

with open(os.path.join(RESULTS_DIR, "metrics_aggregated.json"), "w") as f:
    json.dump(agg, f, indent=2)
print(f"\nwrote metrics_aggregated.json")
