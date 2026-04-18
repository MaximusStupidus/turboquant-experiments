"""Generate comparison plots for Part 2 Parler results."""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(REPO_ROOT, "speech-tts-improvements/parler/results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CONFIGS = ["baseline", "tq_4bit", "tq_3bit", "tq_2bit"]
LABELS = {"baseline": "fp16\n(baseline)", "tq_4bit": "TurboQuant\n4-bit", "tq_3bit": "TurboQuant\n3-bit", "tq_2bit": "TurboQuant\n2-bit"}
COLORS = {"baseline": "#4CAF50", "tq_4bit": "#FF9800", "tq_3bit": "#FB8C00", "tq_2bit": "#F57C00"}

# Load timings
timings = {}
for config in CONFIGS:
    path = os.path.join(RESULTS_DIR, f"timings_{config}.json")
    if os.path.exists(path):
        with open(path) as f:
            timings[config] = json.load(f)

# Load metrics (may not all exist)
metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
metrics = {}
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)


# Plot 1: mean RTF across configs
fig, ax = plt.subplots(figsize=(10, 6))
names, rtfs, colors = [], [], []
for config in CONFIGS:
    if config in timings:
        samples = timings[config].get("per_sample", {})
        rtf_values = [s.get("rtf", 0) for s in samples.values() if isinstance(s, dict) and "rtf" in s]
        if rtf_values:
            names.append(LABELS.get(config, config))
            rtfs.append(float(np.mean(rtf_values)))
            colors.append(COLORS.get(config, "#888"))

if names:
    bars = ax.bar(names, rtfs, color=colors, edgecolor="white")
    for b, v in zip(bars, rtfs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                f"{v:.2f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Mean Real-Time Factor (gen_time / audio_duration)")
    ax.set_title("Parler-TTS generation speed (lower is faster)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rtf.png"), dpi=120)
    plt.close()
    print(f"  saved: rtf.png")

# Plot 2: peak GPU memory
fig, ax = plt.subplots(figsize=(10, 6))
names, mems, colors = [], [], []
for config in CONFIGS:
    if config in timings and "peak_gpu_gb" in timings[config]:
        names.append(LABELS.get(config, config))
        mems.append(timings[config]["peak_gpu_gb"])
        colors.append(COLORS.get(config, "#888"))

if names:
    bars = ax.bar(names, mems, color=colors, edgecolor="white")
    for b, v in zip(bars, mems):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                f"{v:.1f} GB", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_title("Parler-TTS memory footprint")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "memory.png"), dpi=120)
    plt.close()
    print(f"  saved: memory.png")

# Plot 3: mean WER (if metrics exist)
if metrics:
    fig, ax = plt.subplots(figsize=(10, 6))
    names, wers, colors = [], [], []
    for config in CONFIGS:
        if config in metrics:
            wer_values = []
            for key, entry in metrics[config].items():
                if isinstance(entry, dict) and "wer" in entry and isinstance(entry["wer"], dict):
                    wer_values.append(entry["wer"].get("wer", 0))
            if wer_values:
                names.append(LABELS.get(config, config))
                wers.append(float(np.mean(wer_values)))
                colors.append(COLORS.get(config, "#888"))

    if names:
        bars = ax.bar(names, wers, color=colors, edgecolor="white")
        for b, v in zip(bars, wers):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                    f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("Mean WER (lower is better)")
        ax.set_title("Parler-TTS intelligibility")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "wer.png"), dpi=120)
        plt.close()
        print(f"  saved: wer.png")

# Plot 4: speaker similarity
if metrics:
    fig, ax = plt.subplots(figsize=(10, 6))
    names, sims, colors = [], [], []
    for config in CONFIGS:
        if config == "baseline":
            continue
        if config in metrics:
            sim_values = []
            for key, entry in metrics[config].items():
                if isinstance(entry, dict) and isinstance(entry.get("speaker_similarity_vs_baseline"), (int, float)):
                    sim_values.append(entry["speaker_similarity_vs_baseline"])
            if sim_values:
                names.append(LABELS.get(config, config))
                sims.append(float(np.mean(sim_values)))
                colors.append(COLORS.get(config, "#888"))

    if names:
        bars = ax.bar(names, sims, color=colors, edgecolor="white")
        for b, v in zip(bars, sims):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                    f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("Speaker similarity to baseline (cosine)")
        ax.set_title("Voice identity preservation under quantization")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "speaker_similarity.png"), dpi=120)
        plt.close()
        print(f"  saved: speaker_similarity.png")

print(f"\nAll plots in {PLOTS_DIR}")
