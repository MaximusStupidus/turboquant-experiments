"""Hand-rolled TurboQuant KV cache — from-scratch implementation.

This is a minimal, commented implementation of the TurboQuant algorithm
for educational purposes. It implements the same two-step idea as the
community `turboquant` package:

    1. Random projection (flatten outlier distributions)
    2. Uniform scalar quantization (compress to N bits)

But written from scratch so every line is understandable.

The key insight: the model doesn't need the K/V vectors preserved exactly.
It only needs the DOT PRODUCTS (Q·K) preserved approximately. The JL lemma
guarantees that multiplying by a random matrix preserves dot products.
As a side effect, the random rotation also flattens the value distribution,
which makes the subsequent bit-quantization much more effective (no outlier
channels eating the bucket budget).

Algorithm:
    QUANTIZE (on write):
        K_proj = K @ R                      # rotate into "flat" space
        indices = uniform_quantize(K_proj)   # quantize the now-flat values
        store indices + scale + zero_point   # compact storage

    DEQUANTIZE (on read):
        K_proj_approx = dequantize(indices, scale, zero_point)
        K_approx = K_proj_approx @ R^T      # rotate back
        return K_approx                      # used by attention

Usage:
    from handrolled_turboquant import HandrolledTurboQuantCache

    cache = HandrolledTurboQuantCache(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        bits=4,
        device="cuda",
        dtype=torch.float16,
    )
    outputs = model(**inputs, past_key_values=cache, use_cache=True)
"""
import math
import torch
from transformers import DynamicCache


class HandrolledTurboQuantCache(DynamicCache):
    """KV cache that applies random projection + uniform quantization.

    Inherits from DynamicCache so the HuggingFace model can use it as a
    drop-in replacement. We override the `update()` method to:
    1. Project the incoming K/V with random matrices (flatten distributions)
    2. Quantize the projected values to N bits (compress)
    3. Immediately dequantize (decompress) and unproject (rotate back)
    4. Return the approximate K/V values (lossy) to the attention layer
    5. Also store the quantized representation internally (for memory savings)

    The attention layer sees the lossy values, so the quality impact of
    quantization is reflected in the model's outputs.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        bits: int = 4,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bits = bits
        self.num_levels = 2 ** bits  # e.g., 16 for 4-bit, 4 for 2-bit
        self.num_layers = num_layers
        self.head_dim = head_dim

        # ── Generate random projection matrices ──
        # One matrix per layer, shape (head_dim, head_dim).
        # Each entry is drawn from N(0, 1/head_dim), which gives the
        # matrix the right scaling for the JL lemma.
        #
        # We use the SAME matrix for K and V within a layer (simpler),
        # and different matrices across layers (each layer's K/V have
        # different distributions, so independent rotations are better).
        #
        # Fixed seed for reproducibility — same projection every run.
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.projections = []
        for _ in range(num_layers):
            # Generate random Gaussian, then orthogonalize with QR decomposition.
            # Q is a random orthogonal matrix: it rotates without distorting norms.
            # This is critical — a non-orthogonal matrix would stretch/squish vectors,
            # adding distortion ON TOP of the quantization error. An orthogonal matrix
            # preserves norms and dot products exactly, so the ONLY error comes from
            # the quantization step. This is what TurboQuant (and the Hadamard variant)
            # does in practice.
            raw = torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float32)
            Q, _ = torch.linalg.qr(raw)
            self.projections.append(Q.to(dtype).to(device))

        # Storage for quantized representations
        # Each layer stores: list of (k_indices, k_scales, k_zeros,
        #                              v_indices, v_scales, v_zeros)
        # one entry per token that has been cached.
        self._quantized_layers = [[] for _ in range(num_layers)]

    def _quantize_tensor(self, x: torch.Tensor, R: torch.Tensor):
        """Project and quantize a tensor.

        Args:
            x: shape (batch, n_heads, seq_len, head_dim) — the K or V tensor
            R: shape (head_dim, head_dim) — the projection matrix for this layer

        Returns:
            (indices, scale, zero_point, x_approx)
            - indices: uint8 tensor, same shape as x, values in [0, num_levels-1]
            - scale: per-head-per-token scaling factor
            - zero_point: per-head-per-token offset
            - x_approx: the dequantized + unprojected approximate tensor (fp16)
        """
        # Step 1: Project into "flat" space
        # x @ R: (batch, n_heads, seq_len, head_dim) @ (head_dim, head_dim)
        x_proj = x @ R

        # Step 2: Find the range per head per token for uniform quantization
        # We quantize each (head, token) independently so each gets its own scale.
        # x_proj shape: (batch, n_heads, seq_len, head_dim)
        min_val = x_proj.amin(dim=-1, keepdim=True)   # (batch, n_heads, seq_len, 1)
        max_val = x_proj.amax(dim=-1, keepdim=True)   # (batch, n_heads, seq_len, 1)

        # Avoid division by zero if all values in a vector are identical
        range_val = max_val - min_val
        range_val = torch.clamp(range_val, min=1e-8)

        # Step 3: Quantize — map [min_val, max_val] to [0, num_levels - 1]
        # normalized: values in [0, 1]
        normalized = (x_proj - min_val) / range_val
        # indices: integers in [0, num_levels - 1]
        indices = torch.round(normalized * (self.num_levels - 1)).to(torch.uint8)

        # Step 4: Dequantize — map back from indices to approximate projected values
        x_proj_approx = indices.to(x.dtype) / (self.num_levels - 1) * range_val + min_val

        # Step 5: Unproject — rotate back to original coordinate system
        # x_proj_approx @ R^T: back to original space
        x_approx = x_proj_approx @ R.T

        return indices, min_val.squeeze(-1), max_val.squeeze(-1), x_approx

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Override DynamicCache.update() to apply TurboQuant.

        This is called by the model at each layer during the forward pass.
        The model passes fresh K/V states and expects to get back the full
        accumulated K/V (including all previous tokens).

        We:
        1. Quantize the NEW key/value states (project + quantize)
        2. Store the compressed representation
        3. Return the APPROXIMATE (lossy) key/value states

        The lossy values are what the attention layer uses, so the quality
        impact of quantization is reflected in the model's outputs.
        """
        R = self.projections[layer_idx]

        # Quantize the new K and V
        k_idx, k_min, k_max, k_approx = self._quantize_tensor(key_states, R)
        v_idx, v_min, v_max, v_approx = self._quantize_tensor(value_states, R)

        # Store the compressed representation
        self._quantized_layers[layer_idx].append({
            "k_idx": k_idx, "k_min": k_min, "k_max": k_max,
            "v_idx": v_idx, "v_min": v_min, "v_max": v_max,
        })

        # Use the parent class to accumulate the APPROXIMATE values
        # This is what the model reads during attention.
        # We pass k_approx/v_approx (lossy) instead of key_states/value_states (exact).
        return super().update(k_approx, v_approx, layer_idx, cache_kwargs)
