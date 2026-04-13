"""Phase 3: evaluate TurboQuant and KIVI KV-cache quantization on Llama-3.1-8B.

Runs the FROZEN eval harness (same perplexity/throughput functions from
eval_core.py, identical to what produced baseline.json) with different cache
objects:

1. KIVI 4-bit  — HF QuantizedCache, naive per-channel quantization
2. KIVI 2-bit  — same, more aggressive
3. TurboQuant key=4, value=4 — random projection + quantization
4. TurboQuant key=4, value=2 — asymmetric (V compressed more)
5. TurboQuant key=2, value=2 — most aggressive

Each config produces a JSON with the same structure as baseline.json.
A summary JSON collects all configs + the baseline for easy comparison.

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
# Monkey-patch: turboquant uses np.trapz which was removed in NumPy 2.0+
# It was renamed to np.trapezoid. Patch it so turboquant works.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from language_model_improvements.eval_core import (
    load_wikitext2_test,
    eval_perplexity_autoregressive,
    eval_generation_throughput,
)
from language_model_improvements.eval_utils import format_bytes
from language_model_improvements.kv_utils import kv_cache_bytes


# ─── Cache factory functions ─────────────────────────────────────────────────
#
# A "cache factory" is a function that returns a fresh cache object.
# We pass it to eval_perplexity() and eval_generation_throughput(),
# which call it at the start of each evaluation window to get a new cache.
# This is the ONLY thing that changes between configs — the measurement
# code is identical.

def make_kivi_factory(model_config, nbits):
    """Factory for HF's built-in QuantizedCache (KIVI-style).

    KIVI does naive per-channel linear quantization — no rotation or
    projection. This is our "naive baseline" that TurboQuant should beat.
    """
    def factory():
        from transformers import QuantizedCache
        return QuantizedCache(
            backend="quanto",
            nbits=nbits,
            config=model_config,
        )
    return factory


def make_turboquant_factory(bits):
    """Factory for community TurboQuant cache."""
    def factory():
        from turboquant import TurboQuantCache
        return TurboQuantCache(bits=bits)
    return factory


def make_handrolled_factory(model_config, bits):
    """Factory for our from-scratch TurboQuant implementation.

    Uses random orthogonal projection + uniform scalar quantization.
    Should produce similar results to the community package — if it does,
    our implementation is validated. If it doesn't, we have a bug to find.
    """
    def factory():
        from language_model_improvements.handrolled_turboquant import HandrolledTurboQuantCache
        return HandrolledTurboQuantCache(
            num_layers=model_config.num_hidden_layers,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.hidden_size // model_config.num_attention_heads,
            bits=bits,
            device="cuda",
            dtype=torch.float16,
        )
    return factory


# ─── Config definitions ──────────────────────────────────────────────────────

def get_configs(model_config):
    """The configs we sweep over.

    - fp16 baseline (no compression)
    - KIVI 4-bit and 2-bit (naive per-channel quantization)
    - Community TurboQuant at 4/3/2 bit
    - Our handrolled TurboQuant at 4/2 bit (for validation)
    """
    return [
        ("fp16 (baseline)",  "baseline_autoreg.json", None),
        ("KIVI 4-bit",       "kivi_4bit.json",        make_kivi_factory(model_config, nbits=4)),
        ("KIVI 2-bit",       "kivi_2bit.json",        make_kivi_factory(model_config, nbits=2)),
        ("TurboQuant 4-bit", "turboquant_4bit.json",  make_turboquant_factory(bits=4)),
        ("TurboQuant 3-bit", "turboquant_3bit.json",  make_turboquant_factory(bits=3)),
        ("TurboQuant 2-bit", "turboquant_2bit.json",  make_turboquant_factory(bits=2)),
        ("Handrolled 4-bit", "handrolled_4bit.json",  make_handrolled_factory(model_config, bits=4)),
        ("Handrolled 2-bit", "handrolled_2bit.json",  make_handrolled_factory(model_config, bits=2)),
    ]


# ─── Run one config ──────────────────────────────────────────────────────────

def run_single_config(model, tokenizer, input_ids, config_name, cache_factory, args):
    """Run the full eval (perplexity + throughput) for one cache config."""
    print(f"\n{'='*60}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*60}")

    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # ── Perplexity (autoregressive — correct for cache quantization) ──
    print("  Evaluating perplexity (autoregressive)...")
    try:
        ppl, total_nll, total_scored, peak_mem = eval_perplexity_autoregressive(
            model, input_ids,
            cache_factory=cache_factory,
            prefill_len=args.prefill_len,
            max_eval_tokens=args.max_eval_tokens,
        )
        print(f"  perplexity: {ppl:.4f}")
        print(f"  tokens scored: {total_scored:,}")
        print(f"  peak GPU memory: {format_bytes(peak_mem)}")
    except Exception as e:
        print(f"  PERPLEXITY FAILED: {e}")
        traceback.print_exc()
        ppl, total_scored, peak_mem = None, 0, 0

    # ── Throughput ──
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

    # ── KV memory reference (fp16 analytical, for comparison) ──
    per_token_fp16 = kv_cache_bytes(
        seq_len=1, num_layers=cfg.num_hidden_layers,
        n_kv_heads=cfg.num_key_value_heads, head_dim=head_dim, p_bytes=2,
    )

    return {
        "config_name": config_name,
        "meta": {
            "model": args.model,
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "seed": args.seed,
            "max_eval_tokens": args.max_eval_tokens,
            "prefill_len": args.prefill_len,
            "eval_mode": "autoregressive",
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
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Max tokens to load from WikiText-2")
    parser.add_argument("--max-eval-tokens", type=int, default=2048,
                        help="Total tokens for autoregressive eval (prefill + scored)")
    parser.add_argument("--prefill-len", type=int, default=128,
                        help="Tokens to prefill before autoregressive scoring")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-handrolled", action="store_true",
                        help="Only run handrolled configs (skip fp16/KIVI/community TQ)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Model: {args.model}")
    print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    print(f"Seed: {args.seed}")
    print()

    # Load model once — shared across all configs
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

    # Load dataset once — shared across all configs
    print("Loading WikiText-2...")
    input_ids = load_wikitext2_test(tokenizer, max_tokens=args.max_tokens)
    print(f"  dataset tokens: {input_ids.shape[0]:,}")

    # ── Run the sweep ──
    configs = get_configs(model.config)
    if args.only_handrolled:
        configs = [(n, f, cf) for n, f, cf in configs if "Handrolled" in n]
        print(f"  --only-handrolled: running {len(configs)} configs")
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

        # Free GPU cache between configs to avoid memory buildup
        torch.cuda.empty_cache()

    # ── Build summary (baseline is now part of the sweep as the first config) ──
    summary = {"all_results": all_results}
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # ── Print comparison table ──
    print(f"\n{'='*70}")
    print("COMPARISON TABLE (autoregressive eval)")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'PPL':>8} {'tok/s':>8} {'Peak Mem':>12}")
    print("-" * 55)
    for r in all_results:
        ppl_s = f"{r['perplexity']['value']:.2f}" if r['perplexity']['value'] else "FAIL"
        tps_s = f"{r['throughput']['tokens_per_sec']:.1f}" if r['throughput']['tokens_per_sec'] else "FAIL"
        mem_s = r['memory']['peak_gpu_human']
        print(f"{r['config_name']:<25} {ppl_s:>8} {tps_s:>8} {mem_s:>12}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
