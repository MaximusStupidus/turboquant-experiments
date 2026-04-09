"""Phase 1 inspection script: load Llama-3.1-8B-Instruct and report on its KV cache.

Run on a GPU-equipped machine:
    uv run python language-model-improvements/scripts/01_inspect_kv_cache.py \
        --output language-model-improvements/results/01_kv_cache_inspection.txt
"""
import argparse
import sys
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make the repo root importable so we can use kv_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from language_model_improvements.kv_utils import kv_cache_bytes, find_outlier_channels


class _Tee:
    """Write to multiple streams simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()


def _get_kv_layer(pkv, layer_idx):
    """Extract (K, V) tensors from layer_idx, handling all cache formats."""
    if hasattr(pkv, 'key_cache'):
        return pkv.key_cache[layer_idx], pkv.value_cache[layer_idx]
    if hasattr(pkv, 'to_legacy_cache'):
        legacy = pkv.to_legacy_cache()
        return legacy[layer_idx]
    return pkv[layer_idx]


def report_config(model):
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print("=== Model config ===")
    print(f"  model_type        : {cfg.model_type}")
    print(f"  hidden_size       : {cfg.hidden_size}")
    print(f"  num_hidden_layers : {cfg.num_hidden_layers}")
    print(f"  num_attention_heads (Q heads): {cfg.num_attention_heads}")
    print(f"  num_key_value_heads (KV heads): {cfg.num_key_value_heads}")
    print(f"  head_dim          : {head_dim}")
    print(f"  vocab_size        : {cfg.vocab_size}")
    print(f"  max_position_embeddings: {cfg.max_position_embeddings}")
    print(f"  torch_dtype       : {model.dtype}")
    return cfg, head_dim


def report_weight_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print()
    print("=== Model weights ===")
    print(f"  total parameters  : {total_params:,}")
    print(f"  total bytes       : {total_bytes:,} ({total_bytes / 1e9:.2f} GB)")
    return total_bytes


def inspect_kv_cache_structure(model, tokenizer):
    prompt = (
        "Explain in one sentence what the KV cache is and why it exists "
        "in transformer language models."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print()
    print("=== Forward pass ===")
    print(f"  prompt: {prompt!r}")
    print(f"  input_ids shape: {tuple(inputs.input_ids.shape)}")
    print(f"  num tokens: {inputs.input_ids.shape[1]}")

    with torch.no_grad():
        out = model(**inputs, use_cache=True)

    pkv = out.past_key_values
    print()
    print("=== past_key_values structure ===")
    print(f"  type              : {type(pkv).__name__}")
    print(f"  num layers        : {len(pkv)}")

    # Handle DynamicCache, legacy tuple, or other formats
    k0, v0 = _get_kv_layer(pkv, 0)
    fmt = type(pkv).__name__

    print(f"  K[layer=0] shape  : {tuple(k0.shape)}")
    print(f"  K[layer=0] dtype  : {k0.dtype}")
    print(f"  K[layer=0] device : {k0.device}")
    print(f"  V[layer=0] shape  : {tuple(v0.shape)}")
    print(f"  V[layer=0] dtype  : {v0.dtype}")
    print()
    print(f"  Expected shape: (batch=1, n_kv_heads, seq_len, head_dim)")

    return pkv


def report_kv_memory_sweep(cfg, head_dim, weight_bytes):
    n_layers = cfg.num_hidden_layers
    n_kv_heads = cfg.num_key_value_heads
    p_bytes = 2  # fp16

    per_token = kv_cache_bytes(
        seq_len=1, num_layers=n_layers, n_kv_heads=n_kv_heads,
        head_dim=head_dim, p_bytes=p_bytes,
    )

    print()
    print("=== KV cache memory (analytical, fp16) ===")
    print(f"  per token: {per_token:,} bytes ({per_token / 1024:.1f} KB)")
    print()
    print(f"  {'seq_len':>8}  {'KV bytes':>16}  {'KV (GB)':>10}  {'vs weights':>12}")
    print("  " + "-" * 54)
    for seq_len in [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        b = kv_cache_bytes(
            seq_len=seq_len, num_layers=n_layers, n_kv_heads=n_kv_heads,
            head_dim=head_dim, p_bytes=p_bytes,
        )
        ratio = b / weight_bytes
        print(f"  {seq_len:>8}  {b:>16,}  {b / 1e9:>9.2f}  {ratio:>11.1%}")

    print()
    print(f"  Model weight memory: {weight_bytes / 1e9:.2f} GB")
    crossover = weight_bytes / per_token
    print(f"  KV = weights crossover at: ~{int(crossover):,} tokens")


def report_value_distributions(pkv, layer_idx: int, num_layers: int):
    """Print K and V per-channel value statistics for one chosen layer."""
    k, v = _get_kv_layer(pkv, layer_idx)

    # Shape is (batch, n_kv_heads, seq_len, head_dim).
    # Flatten to (n_tokens, n_channels) where n_channels = n_kv_heads * head_dim.
    batch, n_heads, seq_len, head_dim = k.shape
    k_np = k[0].permute(1, 0, 2).reshape(seq_len, n_heads * head_dim).float().cpu().numpy()
    v_np = v[0].permute(1, 0, 2).reshape(seq_len, n_heads * head_dim).float().cpu().numpy()

    def stats(name, arr):
        print(f"  {name}: shape={arr.shape}  "
              f"mean={arr.mean():+.4f}  std={arr.std():.4f}  "
              f"min={arr.min():+.3f}  max={arr.max():+.3f}  "
              f"p99.9={np.quantile(np.abs(arr), 0.999):.3f}")

    label = f"layer {layer_idx}"
    if layer_idx == 0:
        label += " (first)"
    elif layer_idx == num_layers - 1:
        label += " (last)"
    else:
        label += " (middle)"

    print()
    print(f"=== K/V value distributions — {label} ===")
    stats("K", k_np)
    stats("V", v_np)

    k_outliers = find_outlier_channels(k_np, threshold_factor=10.0)
    v_outliers = find_outlier_channels(v_np, threshold_factor=10.0)
    print(f"  K outlier channels (>10x median max-abs): {k_outliers if k_outliers else 'none'}")
    print(f"  V outlier channels (>10x median max-abs): {v_outliers if v_outliers else 'none'}")

    # If outliers exist, show their magnitudes vs typical
    if k_outliers or v_outliers:
        print()
        if k_outliers:
            k_maxabs = np.abs(k_np).max(axis=0)
            k_median = float(np.median(k_maxabs))
            for ch in k_outliers:
                print(f"    K channel {ch}: max-abs={k_maxabs[ch]:.3f}  "
                      f"({k_maxabs[ch]/k_median:.1f}x typical)")
        if v_outliers:
            v_maxabs = np.abs(v_np).max(axis=0)
            v_median = float(np.median(v_maxabs))
            for ch in v_outliers:
                print(f"    V channel {ch}: max-abs={v_maxabs[ch]:.3f}  "
                      f"({v_maxabs[ch]/v_median:.1f}x typical)")


def main():
    parser = argparse.ArgumentParser(description="Inspect KV cache of a HuggingFace model")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output", default=None,
                        help="If set, also write the report to this path.")
    args = parser.parse_args()

    # Tee stdout to file if --output is set
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        f = open(args.output, "w")
        sys.stdout = _Tee(sys.__stdout__, f)

    print(f"Model: {args.model}")
    print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    print(f"Torch: {torch.__version__}")
    print()

    # Load model
    print(f"Loading {args.model} in fp16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Loaded.")
    print()

    # 1. Config
    cfg, head_dim = report_config(model)

    # 2. Weight memory
    weight_bytes = report_weight_memory(model)

    # 3. Forward pass + KV cache structure
    pkv = inspect_kv_cache_structure(model, tokenizer)

    # 4. KV memory sweep
    report_kv_memory_sweep(cfg, head_dim, weight_bytes)

    # 5. Value distributions at three layers: first, middle, last
    n_layers = cfg.num_hidden_layers
    for layer_idx in [0, n_layers // 2, n_layers - 1]:
        report_value_distributions(pkv, layer_idx, n_layers)

    print()
    print("=== Done ===")

    if args.output:
        print(f"\nReport saved to: {args.output}", file=sys.__stdout__)


if __name__ == "__main__":
    main()
