# Phase 2 — Baseline Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, re-runnable evaluation harness that measures perplexity, peak GPU memory, tokens/sec, and KV cache memory for Llama-3.1-8B in fp16 — the baseline against which all TurboQuant experiments will be compared. Also extends the outlier analysis from phase 1 to longer contexts and lower thresholds to properly characterize the KV cache distributions before quantization.

**Architecture:** A single eval script (`02_baseline_eval.py`) that takes a model name, a dataset slice, a seed, and context-length configs as arguments. It loads the model, runs perplexity evaluation on a fixed WikiText-2 test slice, measures generation throughput, tracks peak GPU memory, computes analytical KV cache memory, and runs extended outlier analysis. All results go to a single JSON file for easy downstream comparison. A separate pure-function module (`eval_utils.py`) handles perplexity computation and timing so it can be unit-tested without a GPU.

**Tech Stack:** Python 3.11+, `torch`, `transformers`, `datasets` (for WikiText-2), `numpy`, `pytest`. Same `uv`-managed environment as phase 1, with `datasets` added as a dependency.

---

## File structure after phase 2

```
language-model-improvements/
├── kv_utils.py                          # existing (from phase 1)
├── eval_utils.py                        # NEW — pure perplexity + timing helpers
├── scripts/
│   ├── 01_inspect_kv_cache.py           # existing
│   └── 02_baseline_eval.py             # NEW — the eval harness
├── results/
│   ├── 01_kv_cache_inspection.txt       # existing
│   └── baseline.json                    # NEW — fp16 baseline numbers
└── tests/
    ├── test_kv_utils.py                 # existing
    └── test_eval_utils.py              # NEW — unit tests for eval helpers
```

---

## Task 1: Add `datasets` dependency

**Files:**
- Modify: `~/Desktop/Experiments/turboquant-experiments/pyproject.toml`

- [ ] **Step 1: Add datasets to dependencies**

In `pyproject.toml`, add `"datasets>=3.0"` to the `dependencies` list:

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
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
pythonpath = ["."]
```

- [ ] **Step 2: Sync the environment**

Run: `cd ~/Desktop/Experiments/turboquant-experiments && uv sync --extra dev`
Expected: uv installs `datasets` and its dependencies. No errors.

- [ ] **Step 3: Verify datasets loads**

Run: `uv run python -c "from datasets import load_dataset; print('datasets OK')"`
Expected: `datasets OK`

- [ ] **Step 4: Commit**

```bash
cd ~/Desktop/Experiments/turboquant-experiments
git add pyproject.toml uv.lock
git commit -m "chore: add datasets dependency for WikiText-2 eval"
git push
```

---

## Task 2: TDD `eval_utils.py` — perplexity from log-likelihoods

**Files:**
- Create: `~/Desktop/Experiments/turboquant-experiments/language-model-improvements/eval_utils.py`
- Create: `~/Desktop/Experiments/turboquant-experiments/language-model-improvements/tests/test_eval_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `language-model-improvements/tests/test_eval_utils.py`:

```python
"""Unit tests for eval helpers — pure math, no GPU."""
import math
import numpy as np
import pytest
from language_model_improvements.eval_utils import (
    perplexity_from_nlls,
    format_bytes,
    timed,
)


def test_perplexity_from_nlls_perfect_model():
    """A model that assigns probability 1 to every token has perplexity 1."""
    nlls = [0.0, 0.0, 0.0]  # log(1) = 0
    token_counts = [10, 10, 10]
    assert perplexity_from_nlls(nlls, token_counts) == pytest.approx(1.0)


def test_perplexity_from_nlls_known_value():
    """Known: if avg NLL = ln(100) ≈ 4.605, perplexity = 100."""
    nll_per_token = math.log(100)
    nlls = [nll_per_token * 50]  # 50 tokens, total NLL = ln(100)*50
    token_counts = [50]
    assert perplexity_from_nlls(nlls, token_counts) == pytest.approx(100.0, rel=1e-4)


def test_perplexity_from_nlls_multiple_chunks():
    """Perplexity should be the same whether computed in one chunk or many."""
    nll_per_token = 2.5
    # One chunk: 100 tokens
    ppl_one = perplexity_from_nlls([nll_per_token * 100], [100])
    # Two chunks: 60 + 40 tokens
    ppl_two = perplexity_from_nlls(
        [nll_per_token * 60, nll_per_token * 40], [60, 40]
    )
    assert ppl_one == pytest.approx(ppl_two, rel=1e-6)


def test_perplexity_from_nlls_empty_raises():
    """No data should raise ValueError."""
    with pytest.raises(ValueError):
        perplexity_from_nlls([], [])


def test_format_bytes():
    assert format_bytes(0) == "0 B"
    assert format_bytes(1023) == "1023 B"
    assert format_bytes(1024) == "1.00 KB"
    assert format_bytes(1_048_576) == "1.00 MB"
    assert format_bytes(1_073_741_824) == "1.00 GB"
    assert format_bytes(16_060_522_496) == "14.96 GB"


def test_timed_returns_result_and_elapsed():
    """timed() should return (result, elapsed_seconds)."""
    result, elapsed = timed(lambda: sum(range(1000)))
    assert result == 499500
    assert elapsed > 0
    assert elapsed < 5  # should be near-instant
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Desktop/Experiments/turboquant-experiments && uv run pytest language-model-improvements/tests/test_eval_utils.py -v`
Expected: all tests fail with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Write the implementation**

Create `language-model-improvements/eval_utils.py`:

```python
"""Pure helpers for the eval harness.

No torch, no transformers — just math and formatting so these can be
unit-tested fast on CPU.
"""
import math
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def perplexity_from_nlls(
    total_nlls: list[float],
    token_counts: list[int],
) -> float:
    """Compute perplexity from accumulated negative log-likelihoods.

    Args:
        total_nlls: list of total NLL values, one per chunk/batch.
            Each entry is the *sum* of per-token NLLs for that chunk.
        token_counts: list of token counts, parallel to total_nlls.

    Returns:
        Perplexity = exp(mean NLL per token).

    Raises:
        ValueError: if no tokens provided.
    """
    total_tokens = sum(token_counts)
    if total_tokens == 0:
        raise ValueError("Cannot compute perplexity with zero tokens")
    total_nll = sum(total_nlls)
    mean_nll = total_nll / total_tokens
    return math.exp(mean_nll)


def format_bytes(n: int) -> str:
    """Human-readable byte string: B, KB, MB, GB."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.2f} KB"
    elif n < 1024 ** 3:
        return f"{n / 1024**2:.2f} MB"
    else:
        return f"{n / 1024**3:.2f} GB"


def timed(fn: Callable[[], T]) -> tuple[T, float]:
    """Run fn() and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest language-model-improvements/tests/test_eval_utils.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add language-model-improvements/eval_utils.py \
        language-model-improvements/tests/test_eval_utils.py
git commit -m "feat(eval_utils): pure helpers for perplexity, formatting, timing"
git push
```

---

## Task 3: Write the baseline eval script — perplexity on WikiText-2

This is the core of phase 2. Not TDD'd (requires GPU + real model), built incrementally like the phase 1 inspection script.

**Files:**
- Create: `~/Desktop/Experiments/turboquant-experiments/language-model-improvements/scripts/02_baseline_eval.py`

- [ ] **Step 1: Write the complete eval script**

Create `language-model-improvements/scripts/02_baseline_eval.py`:

```python
"""Phase 2 baseline eval: measure perplexity, memory, throughput for Llama-3.1-8B.

This script produces the fp16 baseline numbers against which all TurboQuant
experiments will be compared. The eval harness must be FROZEN after this
script produces its first results — no tweaks once phase 3 starts.

Run on a GPU-equipped machine:
    uv run python language-model-improvements/scripts/02_baseline_eval.py \
        --model NousResearch/Meta-Llama-3.1-8B-Instruct \
        --output language-model-improvements/results/baseline.json

Determinism: fixed seed, fixed dataset slice, fixed stride. Re-running
with the same args must produce the same numbers (modulo float precision).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from language_model_improvements.eval_utils import perplexity_from_nlls, format_bytes, timed
from language_model_improvements.kv_utils import kv_cache_bytes


def load_wikitext2_test(tokenizer, max_tokens: int = 32768, seed: int = 42):
    """Load WikiText-2 test split, tokenize, and return a single long tensor.

    We concatenate the entire test split into one sequence (standard for
    perplexity evaluation), then truncate to max_tokens for reproducibility.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Concatenate all text with newlines (standard approach)
    full_text = "\n\n".join([t for t in ds["text"] if t.strip()])
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids[0]  # shape: (total_tokens,)
    if input_ids.shape[0] > max_tokens:
        input_ids = input_ids[:max_tokens]
    return input_ids


def eval_perplexity(model, input_ids, stride: int = 512, max_length: int = 2048):
    """Sliding-window perplexity evaluation.

    Uses a sliding window of size `max_length` with step `stride`.
    This is the standard approach from the HuggingFace perplexity docs.

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
        trg_len = end_loc - prev_end_loc  # number of tokens we score this window
        input_chunk = input_ids[begin_loc:end_loc].unsqueeze(0).to(device)

        # Target labels: -100 for context tokens (already scored), real ids for new tokens
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids)
            # outputs.loss is the mean NLL over non-ignored tokens
            nll = outputs.loss.float().item()

        total_nlls.append(nll * trg_len)
        token_counts.append(trg_len)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    peak_mem = torch.cuda.max_memory_allocated()
    ppl = perplexity_from_nlls(total_nlls, token_counts)
    return ppl, total_nlls, token_counts, peak_mem


def eval_generation_throughput(model, tokenizer, prompt: str, max_new_tokens: int = 128, num_runs: int = 3):
    """Measure generation speed in tokens/sec.

    Runs generation `num_runs` times and returns the median tokens/sec.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    speeds = []

    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated_tokens = out.shape[1] - inputs.input_ids.shape[1]
        speeds.append(generated_tokens / elapsed)

    return float(np.median(speeds))


def extended_outlier_analysis(model, tokenizer, context_lengths, thresholds):
    """Run outlier analysis at multiple context lengths and thresholds.

    Addresses the phase 1 finding that outliers weren't detected at 20 tokens / 10x.
    """
    from language_model_improvements.kv_utils import find_outlier_channels

    # Generate a longer sequence for analysis
    prompt = "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings. "
    results = {}

    for ctx_len in context_lengths:
        # Repeat the prompt to fill context
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        # Repeat to reach desired length
        repeats = max(1, ctx_len // input_ids.shape[1])
        input_ids = input_ids.repeat(1, repeats)[:, :ctx_len]

        with torch.no_grad():
            out = model(input_ids, use_cache=True)

        pkv = out.past_key_values
        # Extract KV using iteration (same approach as phase 1 fix)
        kv_layers = []
        for item in pkv:
            if isinstance(item, tuple):
                kv_layers.append((item[0], item[1]))

        n_layers = len(kv_layers)
        ctx_results = {}
        for layer_idx in [0, n_layers // 2, n_layers - 1]:
            k, v = kv_layers[layer_idx]
            batch, n_heads, seq, head_dim = k.shape
            k_np = k[0].permute(1, 0, 2).reshape(seq, n_heads * head_dim).float().cpu().numpy()
            v_np = v[0].permute(1, 0, 2).reshape(seq, n_heads * head_dim).float().cpu().numpy()

            layer_results = {}
            for thresh in thresholds:
                k_outliers = find_outlier_channels(k_np, threshold_factor=thresh)
                v_outliers = find_outlier_channels(v_np, threshold_factor=thresh)
                layer_results[f"thresh_{thresh}x"] = {
                    "K_outlier_count": len(k_outliers),
                    "V_outlier_count": len(v_outliers),
                    "K_outlier_channels": k_outliers[:10],  # cap at 10 for readability
                    "V_outlier_channels": v_outliers[:10],
                }
            # Also store distribution stats
            layer_results["K_stats"] = {
                "mean": float(k_np.mean()),
                "std": float(k_np.std()),
                "min": float(k_np.min()),
                "max": float(k_np.max()),
                "p99.9": float(np.quantile(np.abs(k_np), 0.999)),
            }
            layer_results["V_stats"] = {
                "mean": float(v_np.mean()),
                "std": float(v_np.std()),
                "min": float(v_np.min()),
                "max": float(v_np.max()),
                "p99.9": float(np.quantile(np.abs(v_np), 0.999)),
            }
            ctx_results[f"layer_{layer_idx}"] = layer_results

        results[f"ctx_{ctx_len}"] = ctx_results
        print(f"  outlier analysis done for ctx_len={ctx_len}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Baseline eval harness for TurboQuant experiments")
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output", default="language-model-improvements/results/baseline.json")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Max tokens from WikiText-2 to evaluate on")
    parser.add_argument("--stride", type=int, default=512,
                        help="Sliding window stride for perplexity")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Sliding window size for perplexity")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Model: {args.model}")
    print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    print(f"Seed: {args.seed}")
    print(f"WikiText-2 max tokens: {args.max_tokens}")
    print(f"Stride: {args.stride}, Window: {args.max_length}")
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
    print()

    # Model info
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    # 1. Perplexity
    print("Evaluating perplexity on WikiText-2...")
    input_ids = load_wikitext2_test(tokenizer, max_tokens=args.max_tokens, seed=args.seed)
    print(f"  dataset tokens: {input_ids.shape[0]:,}")

    ppl, nlls, tcounts, peak_mem = eval_perplexity(
        model, input_ids, stride=args.stride, max_length=args.max_length
    )
    total_scored = sum(tcounts)
    print(f"  perplexity: {ppl:.2f}")
    print(f"  tokens scored: {total_scored:,}")
    print(f"  peak GPU memory: {format_bytes(peak_mem)}")
    print()

    # 2. Generation throughput
    print("Measuring generation throughput...")
    gen_prompt = "Explain the concept of quantization in neural networks in detail."
    tokens_per_sec = eval_generation_throughput(model, tokenizer, gen_prompt, max_new_tokens=128, num_runs=3)
    print(f"  tokens/sec (median of 3): {tokens_per_sec:.1f}")
    print()

    # 3. KV cache memory (analytical)
    per_token_kv = kv_cache_bytes(
        seq_len=1, num_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads, head_dim=head_dim, p_bytes=2,
    )
    kv_at_max_length = kv_cache_bytes(
        seq_len=args.max_length, num_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads, head_dim=head_dim, p_bytes=2,
    )
    print(f"KV cache per token: {format_bytes(per_token_kv)}")
    print(f"KV cache at window={args.max_length}: {format_bytes(kv_at_max_length)}")
    print()

    # 4. Extended outlier analysis (addresses phase 1 finding)
    print("Running extended outlier analysis...")
    outlier_results = extended_outlier_analysis(
        model, tokenizer,
        context_lengths=[512, 2048, 8192],
        thresholds=[3.0, 5.0, 10.0],
    )
    print()

    # Build results JSON
    results = {
        "meta": {
            "model": args.model,
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "stride": args.stride,
            "max_length": args.max_length,
            "dtype": "float16",
        },
        "model_info": {
            "total_params": total_params,
            "weight_bytes": weight_bytes,
            "num_layers": cfg.num_hidden_layers,
            "num_q_heads": cfg.num_attention_heads,
            "num_kv_heads": cfg.num_key_value_heads,
            "head_dim": head_dim,
            "hidden_size": cfg.hidden_size,
        },
        "perplexity": {
            "value": round(ppl, 4),
            "tokens_scored": total_scored,
            "dataset": "wikitext-2-raw-v1 (test split)",
        },
        "memory": {
            "peak_gpu_bytes": peak_mem,
            "peak_gpu_human": format_bytes(peak_mem),
            "weight_bytes": weight_bytes,
            "weight_human": format_bytes(weight_bytes),
            "kv_per_token_bytes": per_token_kv,
            "kv_at_window_bytes": kv_at_max_length,
            "kv_at_window_human": format_bytes(kv_at_max_length),
        },
        "throughput": {
            "tokens_per_sec": round(tokens_per_sec, 2),
            "gen_prompt": gen_prompt,
            "max_new_tokens": 128,
            "num_runs": 3,
        },
        "outlier_analysis": outlier_results,
    }

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {args.output}")

    # Print summary
    print()
    print("=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(f"  Model           : {args.model}")
    print(f"  Dtype           : fp16")
    print(f"  Perplexity      : {ppl:.2f}")
    print(f"  Tokens/sec      : {tokens_per_sec:.1f}")
    print(f"  Peak GPU mem    : {format_bytes(peak_mem)}")
    print(f"  Weight mem      : {format_bytes(weight_bytes)}")
    print(f"  KV/token        : {format_bytes(per_token_kv)}")
    print(f"  KV at {args.max_length} ctx  : {format_bytes(kv_at_max_length)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit locally (before running on GPU)**

```bash
cd ~/Desktop/Experiments/turboquant-experiments
git add language-model-improvements/scripts/02_baseline_eval.py
git commit -m "feat(phase2): baseline eval harness — perplexity, memory, throughput, outlier sweep"
git push
```

---

## Task 4: Run the baseline eval on RunPod

All steps in this task run **inside the pod**.

- [ ] **Step 1: Launch pod and bootstrap**

Same as phase 1 — A100 80GB, PyTorch 2.4 template. Then:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/MaximusStupidus/turboquant-experiments.git
cd turboquant-experiments
uv sync --extra dev
export HF_HOME=/workspace/hf_cache
```

- [ ] **Step 2: Run unit tests to verify environment**

```bash
uv run pytest language-model-improvements/tests/ -v
```
Expected: all tests pass (7 from phase 1 + 7 new = 14 total).

- [ ] **Step 3: Run the baseline eval**

```bash
uv run python language-model-improvements/scripts/02_baseline_eval.py --model NousResearch/Meta-Llama-3.1-8B-Instruct --output language-model-improvements/results/baseline.json
```

Expected runtime: ~10-20 minutes (perplexity eval over 32k tokens with sliding window, plus generation throughput, plus outlier analysis at 3 context lengths).

Read the output carefully. Key numbers to look for:
- **Perplexity** should be in the range 6–12 for Llama-3.1-8B on WikiText-2 (if it's >50 or <3, something is wrong).
- **Tokens/sec** should be in the range 30–100 for an A100.
- **Outlier analysis** — check if outliers appear at 2048 or 8192 tokens where they didn't at 20 tokens. Check if 3x and 5x thresholds reveal what 10x didn't.

- [ ] **Step 4: Commit and push results from the pod**

```bash
git config user.name "Ojas Jain"
git config user.email "96643674+MaximusStupidus@users.noreply.github.com"
git add language-model-improvements/results/baseline.json
git commit -m "data(phase2): fp16 baseline results — perplexity, memory, throughput, outlier sweep"
```

For push, set up the token-based remote (replace `TOKEN` with your GitHub PAT):
```bash
git remote set-url origin https://TOKEN@github.com/MaximusStupidus/turboquant-experiments.git
git push
```

- [ ] **Step 5: Shut down the pod**

---

## Task 5: Pull results and review on laptop

- [ ] **Step 1: Pull**

```bash
cd ~/Desktop/Experiments/turboquant-experiments
git pull
```

- [ ] **Step 2: Inspect the JSON**

```bash
uv run python -c "
import json
with open('language-model-improvements/results/baseline.json') as f:
    d = json.load(f)
print(json.dumps(d, indent=2))
"
```

Read the full output. Check:
- Do the model_info numbers match phase 1?
- Is perplexity reasonable?
- Are the outlier results different from phase 1 (longer context, lower thresholds)?

---

## Task 6: Clean up debug file from phase 1

- [ ] **Step 1: Remove the debug script that's no longer needed**

```bash
cd ~/Desktop/Experiments/turboquant-experiments
git rm debug_cache.py
git commit -m "chore: remove phase 1 debug script"
git push
```

---

## Task 7: Writing checkpoint — `notes/02-what-im-measuring.md`

**Files:**
- Create: `~/Desktop/Experiments/turboquant-experiments/notes/02-what-im-measuring.md`

- [ ] **Step 1: Owner writes the note**

The note should explain, in the owner's own words:

1. **Why perplexity?** What does it measure, intuitively? What would it mean for perplexity to go up by 0.5? By 5? At what point does a perplexity increase become "the model got noticeably worse"?

2. **Why peak GPU memory?** What component dominates it — weights or KV cache? How will this metric change when we apply TurboQuant (hint: the weights stay fp16, only the KV cache shrinks)?

3. **Why tokens/sec?** What are we timing — just generation, or also prompt processing? Why might TurboQuant change throughput (hint: smaller KV cache = less memory traffic = potentially faster attention)?

4. **Why the extended outlier analysis?** What did the phase 1 result (no outliers at 20 tokens / 10x) tell us, and what do the new numbers (512 / 2048 / 8192 tokens at 3x / 5x / 10x) add?

5. **What would "TurboQuant is working well" look like in these numbers?** What would "TurboQuant is broken" look like?

- [ ] **Step 2: Submit for supervisor review**

- [ ] **Step 3: Commit**

```bash
cd ~/Desktop/Experiments/turboquant-experiments
git add notes/02-what-im-measuring.md
git commit -m "docs(phase2): metrics writeup — what we measure and why"
git push
```

---

## Phase 2 done

At this point:
- The eval harness exists, is deterministic, and produces a structured JSON.
- The fp16 baseline numbers are committed.
- The harness is now **FROZEN** — no changes until all phase 3 experiments are done.
- The owner has written a note explaining what each metric measures and what changes would mean.
- The extended outlier analysis has been run at proper context lengths and thresholds.

Phase 3 begins with a separate planning session. We do not pre-plan phase 3 here.

---

## Self-review checklist

- [x] Spec coverage: perplexity ✅, peak GPU memory ✅, tokens/sec ✅, KV cache memory ✅, deterministic ✅, re-runnable ✅, baseline JSON ✅, writing checkpoint ✅, extended outlier analysis (addresses phase 1 finding) ✅.
- [x] Placeholder scan: no TBDs, no "implement later", all code blocks complete.
- [x] Type consistency: `perplexity_from_nlls` signature matches between tests (Task 2) and call site (Task 3). `kv_cache_bytes` and `find_outlier_channels` signatures match existing code in `kv_utils.py`. `format_bytes` and `timed` consistent between test and implementation.
- [x] Harness freeze rule from design doc is explicitly mentioned in the "Phase 2 done" section.
- [x] No phase 3 work has leaked in (no quantization, no TurboQuant).
