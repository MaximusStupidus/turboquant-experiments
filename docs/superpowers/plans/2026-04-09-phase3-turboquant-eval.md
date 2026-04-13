# Phase 3 — TurboQuant Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the frozen phase 2 eval harness with TurboQuant and KIVI (naive baseline) applied to the KV cache at multiple bit-width configurations, producing a structured comparison against the fp16 baseline that shows perplexity vs. memory vs. throughput tradeoffs.

**Architecture:** A single eval script (`03_turboquant_eval.py`) that reuses the exact same perplexity, throughput, and memory measurement functions from `02_baseline_eval.py` (imported, not copy-pasted), but swaps the cache object. It sweeps over a config list of (method, key_bits, value_bits) tuples and saves one JSON per config plus a combined summary JSON. A separate plotting script produces comparison charts.

**Tech Stack:** Python 3.11+, `torch`, `transformers`, `turboquant` (pip), `quanto` (for HF QuantizedCache backend), `matplotlib` for plots. Same `uv` environment, two new dependencies added.

---

## File structure after phase 3

```
language-model-improvements/
├── kv_utils.py                          # existing
├── eval_utils.py                        # existing
├── eval_core.py                         # NEW — shared eval functions extracted from 02
├── scripts/
│   ├── 01_inspect_kv_cache.py           # existing
│   ├── 02_baseline_eval.py             # existing (FROZEN — not modified)
│   ├── 03_turboquant_eval.py           # NEW — sweep over cache configs
│   └── 04_plot_results.py              # NEW — comparison plots
├── results/
│   ├── 01_kv_cache_inspection.txt       # existing
│   ├── baseline.json                    # existing (FROZEN)
│   ├── kivi_4bit.json                   # NEW
│   ├── kivi_2bit.json                   # NEW
│   ├── turboquant_k4v4.json            # NEW
│   ├── turboquant_k4v2.json            # NEW
│   ├── turboquant_k2v2.json            # NEW
│   ├── summary.json                     # NEW — all configs in one file
│   └── plots/                           # NEW
│       ├── perplexity_comparison.png
│       ├── memory_comparison.png
│       └── throughput_comparison.png
└── tests/
    ├── test_kv_utils.py                 # existing
    └── test_eval_utils.py              # existing
```

**Note on the "frozen harness" rule:** `02_baseline_eval.py` is NOT modified. Instead, we extract the reusable evaluation functions into `eval_core.py` and import them from both `02` (retroactively, as a refactor) and `03`. The measurement logic is identical — only the cache object changes. This preserves the freeze guarantee: the same code that produced `baseline.json` is the code that produces the TurboQuant and KIVI results.

---

## Task 1: Add `turboquant` and `quanto` dependencies

**Files:**
- Modify: `~/Desktop/Experiments/turboquant-experiments/pyproject.toml`

- [ ] **Step 1: Add dependencies**

```toml
[project]
name = "turboquant-experiments"
version = "0.1.0"
description = "Comparative benchmarks of TurboQuant on Llama-3.1-8B and VibeVoice"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4",
    "transformers>=4.45",
    "accelerate>=0.34",
    "numpy>=2.0",
    "matplotlib>=3.9",
    "huggingface-hub>=0.25",
    "datasets>=3.0",
    "turboquant>=0.2.0",
    "quanto>=0.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
pythonpath = ["."]
```

- [ ] **Step 2: Commit**

```bash
cd ~/Desktop/Experiments/turboquant-experiments
git add pyproject.toml
git commit -m "chore: add turboquant and quanto dependencies for phase 3"
git push
```

---

## Task 2: Extract shared eval functions into `eval_core.py`

The perplexity, throughput, and outlier analysis functions currently live inside `02_baseline_eval.py`. We need them in `03_turboquant_eval.py` too. Rather than copy-paste (which would break the "same measurement code" guarantee), we extract them into a shared module.

**Files:**
- Create: `~/Desktop/Experiments/turboquant-experiments/language-model-improvements/eval_core.py`

- [ ] **Step 1: Write eval_core.py**

This file contains the exact same evaluation functions from `02_baseline_eval.py`, extracted verbatim so both scripts can import them.

```python
"""Shared evaluation functions for the TurboQuant experiment.

These are the measurement functions used by both the baseline eval (phase 2)
and the TurboQuant/KIVI eval (phase 3). They are extracted here so that both
scripts use IDENTICAL measurement code — the only thing that changes between
experiments is the cache object passed to the model.

DO NOT MODIFY these functions after baseline.json has been produced.
If a bug is found, fix it here, re-run ALL experiments (baseline included),
and re-commit all results.
"""
import time
import numpy as np
import torch
from datasets import load_dataset

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from language_model_improvements.eval_utils import perplexity_from_nlls, format_bytes
from language_model_improvements.kv_utils import kv_cache_bytes, find_outlier_channels


def load_wikitext2_test(tokenizer, max_tokens: int = 32768):
    """Load WikiText-2 test split, concatenate, tokenize, truncate."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join([t for t in ds["text"] if t.strip()])
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids[0]
    if input_ids.shape[0] > max_tokens:
        input_ids = input_ids[:max_tokens]
    return input_ids


def eval_perplexity(model, input_ids, stride: int = 512, max_length: int = 2048,
                    cache_factory=None):
    """Sliding-window perplexity evaluation.

    Args:
        model: the HF causal LM
        input_ids: 1-D tensor of token ids
        stride: sliding window step
        max_length: sliding window size
        cache_factory: callable that returns a fresh cache object, or None for
            default (fp16 DynamicCache). This is how TurboQuant and KIVI caches
            get injected — the measurement code stays identical.

    Returns:
        (perplexity, total_nlls, token_counts, peak_mem_bytes)
    """
    device = model.device
    seq_len = input_ids.shape[0]
    total_nlls = []
    token_counts = []

    torch.cuda.reset_peak_memory_stats()

    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_chunk = input_ids[begin_loc:end_loc].unsqueeze(0).to(device)

        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100

        kwargs = {}
        if cache_factory is not None:
            kwargs["past_key_values"] = cache_factory()

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids, **kwargs)
            nll = outputs.loss.float().item()

        total_nlls.append(nll * trg_len)
        token_counts.append(trg_len)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    peak_mem = torch.cuda.max_memory_allocated()
    ppl = perplexity_from_nlls(total_nlls, token_counts)
    return ppl, total_nlls, token_counts, peak_mem


def eval_generation_throughput(
    model, tokenizer, prompt: str,
    max_new_tokens: int = 128, num_runs: int = 3,
    cache_factory=None,
):
    """Measure generation speed in tokens/sec."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    speeds = []

    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        if cache_factory is not None:
            gen_kwargs["past_key_values"] = cache_factory()

        with torch.no_grad():
            out = model.generate(**gen_kwargs)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated_tokens = out.shape[1] - inputs.input_ids.shape[1]
        speed = generated_tokens / elapsed
        speeds.append(speed)
        print(f"  run {i+1}/{num_runs}: {generated_tokens} tokens in {elapsed:.2f}s = {speed:.1f} tok/s")

    return float(np.median(speeds))
```

- [ ] **Step 2: Commit**

```bash
cd ~/Desktop/Experiments/turboquant-experiments
git add language-model-improvements/eval_core.py
git commit -m "refactor: extract shared eval functions into eval_core.py for phase 3"
git push
```

---

## Task 3: Write the TurboQuant + KIVI sweep script

**Files:**
- Create: `~/Desktop/Experiments/turboquant-experiments/language-model-improvements/scripts/03_turboquant_eval.py`

- [ ] **Step 1: Write the sweep script**

```python
"""Phase 3: evaluate TurboQuant and KIVI KV-cache quantization on Llama-3.1-8B.

Runs the FROZEN eval harness (same perplexity/throughput/memory functions as
phase 2 baseline) with different cache objects:

1. KIVI 4-bit  — HF QuantizedCache, per-channel linear quantization
2. KIVI 2-bit  — same, more aggressive
3. TurboQuant key=4bit, value=4bit — random projection + quantization
4. TurboQuant key=4bit, value=2bit — asymmetric (values compressed more)
5. TurboQuant key=2bit, value=2bit — most aggressive

Each config produces a JSON with the same structure as baseline.json.
A summary JSON collects all configs for easy plotting.

Usage:
    uv run python language-model-improvements/scripts/03_turboquant_eval.py \
        --model NousResearch/Meta-Llama-3.1-8B-Instruct \
        --output-dir language-model-improvements/results
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from language_model_improvements.eval_core import (
    load_wikitext2_test,
    eval_perplexity,
    eval_generation_throughput,
)
from language_model_improvements.eval_utils import format_bytes
from language_model_improvements.kv_utils import kv_cache_bytes


# ─── Cache factory functions ─────────────────────────────────────────────────

def make_kivi_factory(model_config, nbits):
    """Return a factory that creates a fresh KIVI QuantizedCache."""
    def factory():
        from transformers import QuantizedCache
        return QuantizedCache(
            cache_config={"backend": "quanto", "nbits": nbits},
            max_batch_size=1,
            max_cache_len=2048,
            config=model_config,
        )
    return factory


def make_turboquant_factory(key_bits, value_bits):
    """Return a factory that creates a fresh TurboQuantCache."""
    def factory():
        from turboquant import TurboQuantCache
        return TurboQuantCache(
            key_bits=key_bits,
            value_bits=value_bits,
        )
    return factory


# ─── Config definitions ──────────────────────────────────────────────────────

def get_configs(model_config):
    """Return the list of (name, filename, cache_factory) configs to sweep."""
    return [
        (
            "KIVI 4-bit",
            "kivi_4bit.json",
            make_kivi_factory(model_config, nbits=4),
        ),
        (
            "KIVI 2-bit",
            "kivi_2bit.json",
            make_kivi_factory(model_config, nbits=2),
        ),
        (
            "TurboQuant k4v4",
            "turboquant_k4v4.json",
            make_turboquant_factory(key_bits=4, value_bits=4),
        ),
        (
            "TurboQuant k4v2",
            "turboquant_k4v2.json",
            make_turboquant_factory(key_bits=4, value_bits=2),
        ),
        (
            "TurboQuant k2v2",
            "turboquant_k2v2.json",
            make_turboquant_factory(key_bits=2, value_bits=2),
        ),
    ]


# ─── Run one config ──────────────────────────────────────────────────────────

def run_single_config(model, tokenizer, input_ids, config_name, cache_factory, args):
    """Run perplexity + throughput for one config. Returns a results dict."""
    print(f"\n{'='*60}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*60}")

    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    # Perplexity
    print("  Evaluating perplexity...")
    try:
        ppl, nlls, tcounts, peak_mem = eval_perplexity(
            model, input_ids,
            stride=args.stride,
            max_length=args.max_length,
            cache_factory=cache_factory,
        )
        total_scored = sum(tcounts)
        print(f"  perplexity: {ppl:.4f}")
        print(f"  tokens scored: {total_scored:,}")
        print(f"  peak GPU memory: {format_bytes(peak_mem)}")
    except Exception as e:
        print(f"  PERPLEXITY FAILED: {e}")
        traceback.print_exc()
        ppl, total_scored, peak_mem = None, 0, 0

    # Throughput
    print("  Measuring throughput...")
    gen_prompt = "Explain the concept of quantization in neural networks in detail."
    try:
        tokens_per_sec = eval_generation_throughput(
            model, tokenizer, gen_prompt,
            max_new_tokens=128, num_runs=3,
            cache_factory=cache_factory,
        )
        print(f"  median tokens/sec: {tokens_per_sec:.1f}")
    except Exception as e:
        print(f"  THROUGHPUT FAILED: {e}")
        traceback.print_exc()
        tokens_per_sec = None

    # KV memory (analytical — approximate for quantized caches)
    per_token_fp16 = kv_cache_bytes(
        seq_len=1, num_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads, head_dim=head_dim, p_bytes=2,
    )

    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    return {
        "config_name": config_name,
        "meta": {
            "model": args.model,
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "stride": args.stride,
            "max_length": args.max_length,
        },
        "perplexity": {
            "value": round(ppl, 4) if ppl is not None else None,
            "tokens_scored": total_scored,
            "dataset": "wikitext-2-raw-v1 (test split)",
        },
        "memory": {
            "peak_gpu_bytes": peak_mem,
            "peak_gpu_human": format_bytes(peak_mem) if peak_mem else "N/A",
            "weight_bytes": weight_bytes,
            "kv_per_token_fp16_bytes": per_token_fp16,
        },
        "throughput": {
            "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec is not None else None,
            "gen_prompt": gen_prompt,
            "max_new_tokens": 128,
            "num_runs": 3,
        },
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3: TurboQuant + KIVI eval sweep")
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", default="language-model-improvements/results")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Model: {args.model}")
    print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    print(f"Seed: {args.seed}")
    print()

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded.")

    # Load dataset once (shared across all configs)
    print("Loading WikiText-2...")
    input_ids = load_wikitext2_test(tokenizer, max_tokens=args.max_tokens)
    print(f"  dataset tokens: {input_ids.shape[0]:,}")

    # Run sweep
    configs = get_configs(model.config)
    all_results = []

    for config_name, filename, cache_factory in configs:
        result = run_single_config(
            model, tokenizer, input_ids,
            config_name, cache_factory, args,
        )

        # Save individual result
        out_path = os.path.join(args.output_dir, filename)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved: {out_path}")

        all_results.append(result)

        # Clear CUDA cache between configs
        torch.cuda.empty_cache()

    # Load baseline for the summary
    baseline_path = os.path.join(args.output_dir, "baseline.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
    else:
        baseline = None
        print("WARNING: baseline.json not found — summary will lack baseline comparison")

    # Save combined summary
    summary = {
        "baseline": baseline,
        "experiments": all_results,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # Print comparison table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'PPL':>8} {'tok/s':>8} {'Peak Mem':>12}")
    print("-" * 55)
    if baseline:
        print(f"{'fp16 (baseline)':<25} {baseline['perplexity']['value']:>8.2f} "
              f"{baseline['throughput']['tokens_per_sec']:>8.1f} "
              f"{baseline['memory']['peak_gpu_human']:>12}")
    for r in all_results:
        ppl_str = f"{r['perplexity']['value']:.2f}" if r['perplexity']['value'] else "FAIL"
        tps_str = f"{r['throughput']['tokens_per_sec']:.1f}" if r['throughput']['tokens_per_sec'] else "FAIL"
        mem_str = r['memory']['peak_gpu_human']
        print(f"{r['config_name']:<25} {ppl_str:>8} {tps_str:>8} {mem_str:>12}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd ~/Desktop/Experiments/turboquant-experiments
git add language-model-improvements/scripts/03_turboquant_eval.py
git commit -m "feat(phase3): TurboQuant + KIVI eval sweep script"
git push
```

---

## Task 4: Write the plotting script

**Files:**
- Create: `~/Desktop/Experiments/turboquant-experiments/language-model-improvements/scripts/04_plot_results.py`

- [ ] **Step 1: Write the plotting script**

```python
"""Phase 3 plotting: generate comparison charts from summary.json.

Usage:
    uv run python language-model-improvements/scripts/04_plot_results.py \
        --input language-model-improvements/results/summary.json \
        --output-dir language-model-improvements/results/plots
"""
import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def plot_perplexity(summary, output_dir):
    """Bar chart: perplexity across all configs."""
    names = []
    ppls = []
    colors = []

    if summary["baseline"]:
        names.append("fp16\n(baseline)")
        ppls.append(summary["baseline"]["perplexity"]["value"])
        colors.append("#4CAF50")  # green for baseline

    for r in summary["experiments"]:
        if r["perplexity"]["value"] is not None:
            names.append(r["config_name"].replace(" ", "\n"))
            ppls.append(r["perplexity"]["value"])
            colors.append("#2196F3" if "KIVI" in r["config_name"] else "#FF9800")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, ppls, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels on bars
    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{ppl:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Perplexity (WikiText-2)", fontsize=12)
    ax.set_title("Perplexity Comparison: fp16 vs KIVI vs TurboQuant", fontsize=14)
    ax.set_ylim(0, max(ppls) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="fp16 baseline"),
        Patch(facecolor="#2196F3", label="KIVI (naive quantization)"),
        Patch(facecolor="#FF9800", label="TurboQuant"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "perplexity_comparison.png"), dpi=150)
    print(f"  saved: perplexity_comparison.png")


def plot_memory(summary, output_dir):
    """Bar chart: peak GPU memory across all configs."""
    names = []
    mems = []
    colors = []

    if summary["baseline"]:
        names.append("fp16\n(baseline)")
        mems.append(summary["baseline"]["memory"]["peak_gpu_bytes"] / 1e9)
        colors.append("#4CAF50")

    for r in summary["experiments"]:
        if r["memory"]["peak_gpu_bytes"]:
            names.append(r["config_name"].replace(" ", "\n"))
            mems.append(r["memory"]["peak_gpu_bytes"] / 1e9)
            colors.append("#2196F3" if "KIVI" in r["config_name"] else "#FF9800")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, mems, color=colors, edgecolor="white", linewidth=0.5)

    for bar, mem in zip(bars, mems):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{mem:.1f} GB", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Peak GPU Memory (GB)", fontsize=12)
    ax.set_title("Memory Comparison: fp16 vs KIVI vs TurboQuant", fontsize=14)
    ax.set_ylim(0, max(mems) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_comparison.png"), dpi=150)
    print(f"  saved: memory_comparison.png")


def plot_throughput(summary, output_dir):
    """Bar chart: tokens/sec across all configs."""
    names = []
    speeds = []
    colors = []

    if summary["baseline"]:
        names.append("fp16\n(baseline)")
        speeds.append(summary["baseline"]["throughput"]["tokens_per_sec"])
        colors.append("#4CAF50")

    for r in summary["experiments"]:
        if r["throughput"]["tokens_per_sec"] is not None:
            names.append(r["config_name"].replace(" ", "\n"))
            speeds.append(r["throughput"]["tokens_per_sec"])
            colors.append("#2196F3" if "KIVI" in r["config_name"] else "#FF9800")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, speeds, color=colors, edgecolor="white", linewidth=0.5)

    for bar, s in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{s:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Tokens / sec", fontsize=12)
    ax.set_title("Generation Throughput: fp16 vs KIVI vs TurboQuant", fontsize=14)
    ax.set_ylim(0, max(speeds) * 1.3)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_comparison.png"), dpi=150)
    print(f"  saved: throughput_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="language-model-improvements/results/summary.json")
    parser.add_argument("--output-dir", default="language-model-improvements/results/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary = load_summary(args.input)

    print("Generating plots...")
    plot_perplexity(summary, args.output_dir)
    plot_memory(summary, args.output_dir)
    plot_throughput(summary, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd ~/Desktop/Experiments/turboquant-experiments
git add language-model-improvements/scripts/04_plot_results.py
git commit -m "feat(phase3): comparison plotting script"
git push
```

---

## Task 5: Run the sweep on RunPod

All steps run **inside the pod**.

- [ ] **Step 1: Launch pod and bootstrap**

A100 80GB, PyTorch 2.4 template, 50GB volume.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/MaximusStupidus/turboquant-experiments.git
cd turboquant-experiments
uv sync --extra dev
export HF_HOME=/workspace/hf_cache
```

- [ ] **Step 2: Dry-run — verify imports work**

```bash
uv run python -c "from turboquant import TurboQuantCache; print('turboquant OK')"
uv run python -c "from transformers import QuantizedCache; print('QuantizedCache OK')"
```

If `turboquant` import fails, check if the package API has changed — read the error and adjust the import in `03_turboquant_eval.py`.

- [ ] **Step 3: Run the full sweep**

```bash
uv run python language-model-improvements/scripts/03_turboquant_eval.py --model NousResearch/Meta-Llama-3.1-8B-Instruct --output-dir language-model-improvements/results
```

Expected runtime: ~45-90 minutes (5 configs × ~10-15 min each for perplexity + throughput). Watch the console output — if a config fails, the script catches the error and continues to the next one. Some configs may fail if the library API doesn't match our assumptions — that's OK, we record what works and debug what doesn't.

**Key things to watch for during the run:**
- If TurboQuant import fails: the `turboquant` PyPI package may have a different API than documented. Read the error, check `pip show turboquant` for the installed version, and look at the package source.
- If KIVI QuantizedCache fails: the `quanto` backend init may need different args for this transformers version. The error will tell us what's wrong.
- If perplexity is NaN or inf: the quantization is too aggressive and numerical precision collapsed. That's data (record it as a failure), not a bug.

- [ ] **Step 4: Generate plots (still on the pod)**

```bash
uv run python language-model-improvements/scripts/04_plot_results.py --input language-model-improvements/results/summary.json --output-dir language-model-improvements/results/plots
```

- [ ] **Step 5: Commit and push from the pod**

```bash
git config user.name "Ojas Jain"
git config user.email "96643674+MaximusStupidus@users.noreply.github.com"
git remote set-url origin https://TOKEN@github.com/MaximusStupidus/turboquant-experiments.git
git add language-model-improvements/results/
git commit -m "data(phase3): TurboQuant + KIVI sweep results and comparison plots"
git push
```

- [ ] **Step 6: Shut down the pod**

---

## Task 6: Pull and review results on laptop

- [ ] **Step 1: Pull**

```bash
cd ~/Desktop/Experiments/turboquant-experiments
git pull
```

- [ ] **Step 2: Open the plots**

```bash
open language-model-improvements/results/plots/*.png
```

- [ ] **Step 3: Review the comparison table in summary.json**

```bash
python3 -c "
import json
with open('language-model-improvements/results/summary.json') as f:
    d = json.load(f)
baseline = d['baseline']
print(f\"{'Config':<25} {'PPL':>8} {'tok/s':>8} {'Peak Mem':>12}\")
print('-' * 55)
print(f\"{'fp16 (baseline)':<25} {baseline['perplexity']['value']:>8.2f} {baseline['throughput']['tokens_per_sec']:>8.1f} {baseline['memory']['peak_gpu_human']:>12}\")
for r in d['experiments']:
    ppl = f\"{r['perplexity']['value']:.2f}\" if r['perplexity']['value'] else 'FAIL'
    tps = f\"{r['throughput']['tokens_per_sec']:.1f}\" if r['throughput']['tokens_per_sec'] else 'FAIL'
    mem = r['memory']['peak_gpu_human']
    print(f\"{r['config_name']:<25} {ppl:>8} {tps:>8} {mem:>12}\")
"
```

---

## Task 7: Writing checkpoint — `notes/03-results.md`

**Files:**
- Create: `~/Desktop/Experiments/turboquant-experiments/notes/03-results.md`

- [ ] **Step 1: Owner writes the results interpretation**

The note should address:

1. **The comparison table.** Which configs worked? Which failed? What do the numbers show?
2. **Perplexity.** Did TurboQuant preserve model quality? How does it compare to KIVI at the same bit width? Was there a config where perplexity barely moved (the sweet spot)?
3. **Memory.** Did peak GPU memory drop measurably? By how much compared to what the analytical formula predicted?
4. **Throughput.** Did TurboQuant change generation speed? Faster (bandwidth savings) or slower (projection overhead)?
5. **TurboQuant vs KIVI.** At the same compression ratio, which method produces better perplexity? This is the central question of the whole experiment. If TurboQuant wins, the random-projection trick adds value. If KIVI wins or ties, the trick may be unnecessary for this model.
6. **Any surprises?** Results that didn't match expectations from the foundations doc.
7. **What would you do differently?** If you were to re-run this experiment, what would you change?

- [ ] **Step 2: Submit for supervisor review**

- [ ] **Step 3: Commit**

```bash
cd ~/Desktop/Experiments/turboquant-experiments
git add notes/03-results.md
git commit -m "docs(phase3): results interpretation and TurboQuant vs KIVI analysis"
git push
```

---

## Phase 3 done

At this point:
- TurboQuant and KIVI have been evaluated on the same model with the same frozen harness.
- Individual and combined result JSONs are committed.
- Comparison plots are committed.
- The owner has written an interpretation of the results.
- The central question — "does the random projection trick actually help for Llama-3.1-8B's KV cache?" — has a data-backed answer.

Phase 4 (write-up & repo polish) begins with a separate planning session.

---

## Self-review checklist

- [x] Spec coverage: multiple TurboQuant configs ✅, naive baseline (KIVI) for comparison ✅, same frozen harness ✅, perplexity + memory + throughput measured ✅, plots ✅, results JSON ✅, writing checkpoint with interpretation ✅.
- [x] Placeholder scan: no TBDs, all code blocks complete, all commands exact.
- [x] Type consistency: `cache_factory` parameter is consistently `Callable[[], Cache] | None` across `eval_core.py` and `03_turboquant_eval.py`. `format_bytes` and `kv_cache_bytes` signatures match existing code.
- [x] Error handling: sweep script catches per-config failures and continues (records None for failed metrics), because some configs may fail due to API mismatches.
- [x] The frozen harness is imported from `eval_core.py`, not duplicated. `02_baseline_eval.py` is not modified.
