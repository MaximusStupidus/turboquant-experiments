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

# Make the repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from language_model_improvements.eval_utils import perplexity_from_nlls, format_bytes
from language_model_improvements.kv_utils import kv_cache_bytes, find_outlier_channels


# ─── Section 1: Load and tokenize WikiText-2 ────────────────────────────────

def load_wikitext2_test(tokenizer, max_tokens: int = 32768):
    """Load WikiText-2 test split, concatenate, tokenize, truncate.

    WikiText-2 is a standard benchmark for LM perplexity. We concatenate
    the entire test split into one long token sequence (standard practice),
    then truncate to max_tokens for reproducibility and speed.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Join all non-empty lines with double newlines (standard)
    full_text = "\n\n".join([t for t in ds["text"] if t.strip()])
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids[0]  # shape: (total_tokens,)
    if input_ids.shape[0] > max_tokens:
        input_ids = input_ids[:max_tokens]
    return input_ids


# ─── Section 2: Sliding-window perplexity ────────────────────────────────────

def eval_perplexity(model, input_ids, stride: int = 512, max_length: int = 2048):
    """Sliding-window perplexity evaluation.

    How it works:
    - We slide a window of size `max_length` across the token sequence,
      advancing by `stride` tokens each step.
    - In each window, only the rightmost `stride` tokens are scored (the rest
      is context). This ensures every scored token has ample left-context.
    - The model returns loss = mean NLL over scored tokens. We multiply by
      the count to get total NLL per window, then combine all windows at the end
      using perplexity_from_nlls() (which does the correct exp-of-mean, not
      mean-of-exp).

    Returns:
        (perplexity, total_nlls, token_counts, peak_mem_bytes)
    """
    device = model.device
    seq_len = input_ids.shape[0]
    total_nlls = []
    token_counts = []

    # Reset memory tracking so peak_mem only reflects this evaluation
    torch.cuda.reset_peak_memory_stats()

    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # how many NEW tokens to score

        input_chunk = input_ids[begin_loc:end_loc].unsqueeze(0).to(device)

        # Labels: -100 for context tokens (already scored), real ids for new ones.
        # PyTorch cross-entropy ignores positions with label=-100.
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


# ─── Section 3: Generation throughput ────────────────────────────────────────

def eval_generation_throughput(
    model, tokenizer, prompt: str,
    max_new_tokens: int = 128, num_runs: int = 3,
):
    """Measure generation speed in tokens/sec.

    We time model.generate() with torch.cuda.synchronize() to ensure we're
    measuring actual GPU work, not just kernel-launch overhead. We run
    num_runs times and take the median to smooth out warmup effects.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    speeds = []

    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy — deterministic
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated_tokens = out.shape[1] - inputs.input_ids.shape[1]
        speed = generated_tokens / elapsed
        speeds.append(speed)
        print(f"  run {i+1}/{num_runs}: {generated_tokens} tokens in {elapsed:.2f}s = {speed:.1f} tok/s")

    return float(np.median(speeds))


# ─── Section 4: Extended outlier analysis ────────────────────────────────────

def extended_outlier_analysis(model, tokenizer, context_lengths, thresholds):
    """Run outlier detection at multiple context lengths and thresholds.

    Addresses the phase 1 finding: no outliers at 20 tokens / 10x threshold.
    Here we test longer contexts (512, 2048, 8192) and lower thresholds
    (3x, 5x, 10x) to see if outliers emerge.
    """
    # Use a repeating prompt to fill long contexts
    prompt = "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings. "
    results = {}

    for ctx_len in context_lengths:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        # Repeat prompt tokens to fill the desired context length
        repeats = max(1, ctx_len // input_ids.shape[1])
        input_ids = input_ids.repeat(1, repeats)[:, :ctx_len]

        with torch.no_grad():
            out = model(input_ids, use_cache=True)

        # Extract KV layers (same iteration approach from phase 1)
        kv_layers = []
        for item in out.past_key_values:
            if isinstance(item, tuple):
                kv_layers.append((item[0], item[1]))

        n_layers = len(kv_layers)
        ctx_results = {}

        for layer_idx in [0, n_layers // 2, n_layers - 1]:
            k, v = kv_layers[layer_idx]
            batch, n_heads, seq, head_dim = k.shape
            # Reshape to (seq_len, n_kv_heads * head_dim) for analysis
            k_np = k[0].permute(1, 0, 2).reshape(seq, n_heads * head_dim).float().cpu().numpy()
            v_np = v[0].permute(1, 0, 2).reshape(seq, n_heads * head_dim).float().cpu().numpy()

            layer_results = {}
            for thresh in thresholds:
                k_outliers = find_outlier_channels(k_np, threshold_factor=thresh)
                v_outliers = find_outlier_channels(v_np, threshold_factor=thresh)
                layer_results[f"thresh_{thresh}x"] = {
                    "K_outlier_count": len(k_outliers),
                    "V_outlier_count": len(v_outliers),
                    "K_outlier_channels": k_outliers[:10],
                    "V_outlier_channels": v_outliers[:10],
                }

            # Distribution stats (for comparison across context lengths)
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
        print(f"  outlier analysis done: ctx_len={ctx_len}")

    return results


# ─── Section 5: Main — orchestrate and save to JSON ─────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline eval harness")
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output", default="language-model-improvements/results/baseline.json")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Max tokens from WikiText-2 to use")
    parser.add_argument("--stride", type=int, default=512,
                        help="Sliding window stride for perplexity")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Sliding window size for perplexity")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Model: {args.model}")
    print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    print(f"Seed: {args.seed}")
    print(f"WikiText-2 max tokens: {args.max_tokens}")
    print(f"Stride: {args.stride}, Window: {args.max_length}")
    print()

    # ── Load model ──
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

    # Model metadata
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    # ── 1. Perplexity ──
    print("=" * 50)
    print("PERPLEXITY (WikiText-2)")
    print("=" * 50)
    input_ids = load_wikitext2_test(tokenizer, max_tokens=args.max_tokens)
    print(f"  dataset tokens: {input_ids.shape[0]:,}")
    print(f"  evaluating...")

    ppl, nlls, tcounts, peak_mem = eval_perplexity(
        model, input_ids, stride=args.stride, max_length=args.max_length
    )
    total_scored = sum(tcounts)
    print(f"  perplexity: {ppl:.4f}")
    print(f"  tokens scored: {total_scored:,}")
    print(f"  peak GPU memory: {format_bytes(peak_mem)}")
    print()

    # ── 2. Generation throughput ──
    print("=" * 50)
    print("GENERATION THROUGHPUT")
    print("=" * 50)
    gen_prompt = "Explain the concept of quantization in neural networks in detail."
    tokens_per_sec = eval_generation_throughput(
        model, tokenizer, gen_prompt, max_new_tokens=128, num_runs=3
    )
    print(f"  median tokens/sec: {tokens_per_sec:.1f}")
    print()

    # ── 3. KV cache memory (analytical) ──
    per_token_kv = kv_cache_bytes(
        seq_len=1, num_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads, head_dim=head_dim, p_bytes=2,
    )
    kv_at_window = kv_cache_bytes(
        seq_len=args.max_length, num_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads, head_dim=head_dim, p_bytes=2,
    )

    # ── 4. Extended outlier analysis ──
    print("=" * 50)
    print("EXTENDED OUTLIER ANALYSIS")
    print("=" * 50)
    outlier_results = extended_outlier_analysis(
        model, tokenizer,
        context_lengths=[512, 2048, 8192],
        thresholds=[3.0, 5.0, 10.0],
    )
    print()

    # ── 5. Build and save results JSON ──
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
            "kv_at_window_bytes": kv_at_window,
            "kv_at_window_human": format_bytes(kv_at_window),
        },
        "throughput": {
            "tokens_per_sec": round(tokens_per_sec, 2),
            "gen_prompt": gen_prompt,
            "max_new_tokens": 128,
            "num_runs": 3,
        },
        "outlier_analysis": outlier_results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ──
    print("=" * 50)
    print("BASELINE SUMMARY")
    print("=" * 50)
    print(f"  Model           : {args.model}")
    print(f"  Dtype           : fp16")
    print(f"  Perplexity      : {ppl:.4f}")
    print(f"  Tokens/sec      : {tokens_per_sec:.1f}")
    print(f"  Peak GPU mem    : {format_bytes(peak_mem)}")
    print(f"  Weight mem      : {format_bytes(weight_bytes)}")
    print(f"  KV/token        : {format_bytes(per_token_kv)}")
    print(f"  KV at {args.max_length} ctx  : {format_bytes(kv_at_window)}")
    print("=" * 50)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
