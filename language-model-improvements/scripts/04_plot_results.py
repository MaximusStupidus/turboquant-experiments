"""Phase 3 plotting: generate comparison charts from summary.json.

Produces three bar charts:
1. Perplexity across all configs (lower is better)
2. Peak GPU memory (lower is better — means compression is working)
3. Generation throughput in tokens/sec (higher is better)

Color coding:
- Green = fp16 baseline
- Blue = KIVI (naive quantization)
- Orange = TurboQuant

Usage:
    uv run python language-model-improvements/scripts/04_plot_results.py \
        --input language-model-improvements/results/summary.json \
        --output-dir language-model-improvements/results/plots
"""
import argparse
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def categorize_color(config_name):
    """Assign color based on method type."""
    if "baseline" in config_name.lower() or "fp16" in config_name.lower():
        return "#4CAF50"   # green
    elif "kivi" in config_name.lower():
        return "#2196F3"   # blue
    else:
        return "#FF9800"   # orange (TurboQuant)


def collect_data(summary):
    """Extract names, values, and colors from summary for plotting."""
    entries = []

    if summary["baseline"]:
        entries.append({
            "name": "fp16\n(baseline)",
            "ppl": summary["baseline"]["perplexity"]["value"],
            "mem_gb": summary["baseline"]["memory"]["peak_gpu_bytes"] / 1e9,
            "tok_s": summary["baseline"]["throughput"]["tokens_per_sec"],
            "color": "#4CAF50",
        })

    for r in summary["experiments"]:
        entries.append({
            "name": r["config_name"].replace(" ", "\n"),
            "ppl": r["perplexity"]["value"],
            "mem_gb": r["memory"]["peak_gpu_bytes"] / 1e9 if r["memory"]["peak_gpu_bytes"] else None,
            "tok_s": r["throughput"]["tokens_per_sec"],
            "color": categorize_color(r["config_name"]),
        })

    return entries


def plot_metric(entries, metric_key, ylabel, title, label_fmt, output_path, higher_is_better=False):
    """Generic bar chart for one metric."""
    # Filter entries with valid data for this metric
    valid = [e for e in entries if e[metric_key] is not None]
    if not valid:
        print(f"  skipped {title} — no valid data")
        return

    names = [e["name"] for e in valid]
    values = [e[metric_key] for e in valid]
    colors = [e["color"] for e in valid]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                label_fmt.format(val), ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        Patch(facecolor="#4CAF50", label="fp16 baseline"),
        Patch(facecolor="#2196F3", label="KIVI (naive quantization)"),
        Patch(facecolor="#FF9800", label="TurboQuant"),
    ]
    ax.legend(handles=legend_elements, loc="upper right" if higher_is_better else "upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  saved: {os.path.basename(output_path)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="language-model-improvements/results/summary.json")
    parser.add_argument("--output-dir", default="language-model-improvements/results/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary = load_summary(args.input)
    entries = collect_data(summary)

    print("Generating plots...")

    plot_metric(
        entries, "ppl",
        ylabel="Perplexity (WikiText-2)",
        title="Perplexity: fp16 vs KIVI vs TurboQuant\n(lower is better)",
        label_fmt="{:.2f}",
        output_path=os.path.join(args.output_dir, "perplexity_comparison.png"),
    )

    plot_metric(
        entries, "mem_gb",
        ylabel="Peak GPU Memory (GB)",
        title="Peak Memory: fp16 vs KIVI vs TurboQuant\n(lower is better)",
        label_fmt="{:.1f} GB",
        output_path=os.path.join(args.output_dir, "memory_comparison.png"),
    )

    plot_metric(
        entries, "tok_s",
        ylabel="Tokens / sec",
        title="Generation Throughput: fp16 vs KIVI vs TurboQuant\n(higher is better)",
        label_fmt="{:.1f}",
        output_path=os.path.join(args.output_dir, "throughput_comparison.png"),
        higher_is_better=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()
