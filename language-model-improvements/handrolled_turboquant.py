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
import numpy as np
from scipy.stats import norm
from transformers import DynamicCache


def compute_gaussian_codebook(bits: int, n_points: int = 10000) -> torch.Tensor:
    """Compute MSE-optimal quantization centroids for a standard Gaussian.

    For a Gaussian distribution N(0,1), the optimal way to place 2^bits
    bucket centers is NOT uniform spacing. Instead, we:

    1. Divide the Gaussian into 2^bits equal-probability regions
       (each region contains 1/2^bits of the total probability mass).
    2. The centroid of each region is the conditional expectation
       E[X | X is in region_i] — i.e., the average value of X given
       that X falls in that region.

    This is called Lloyd-Max quantization. It minimizes the mean squared
    error between the original values and their quantized representations.

    At 4-bit (16 buckets), uniform and optimal are nearly identical.
    At 2-bit (4 buckets), optimal placement is significantly better
    because it concentrates buckets where the data density is highest.

    Returns:
        codebook: tensor of shape (2^bits,) containing the optimal centroids,
                  sorted in ascending order.
    """
    num_levels = 2 ** bits

    # Divide the Gaussian into equal-probability regions.
    # The boundaries between regions are the quantiles at 1/N, 2/N, ..., (N-1)/N.
    boundaries = [norm.ppf(i / num_levels) for i in range(1, num_levels)]
    boundaries = [-float('inf')] + boundaries + [float('inf')]

    # For each region, compute the conditional expectation E[X | lower < X < upper].
    # For a standard Gaussian:
    #   E[X | a < X < b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a))
    # where phi = PDF, Phi = CDF.
    centroids = []
    for i in range(num_levels):
        a, b = boundaries[i], boundaries[i + 1]
        # phi(a) - phi(b) gives the expected value integral
        pdf_a = norm.pdf(a) if a != -float('inf') else 0.0
        pdf_b = norm.pdf(b) if b != float('inf') else 0.0
        cdf_diff = norm.cdf(b) - norm.cdf(a)
        if cdf_diff > 0:
            centroid = (pdf_a - pdf_b) / cdf_diff
        else:
            centroid = (a + b) / 2
        centroids.append(centroid)

    return torch.tensor(centroids, dtype=torch.float32)


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

        # Precompute the MSE-optimal codebook for a standard Gaussian.
        # After orthogonal projection, the values are approximately Gaussian,
        # so this codebook is near-optimal for the projected data.
        # Shape: (num_levels,) — e.g., [-1.51, -0.45, +0.45, +1.51] for 2-bit
        self.codebook = compute_gaussian_codebook(bits).to(dtype).to(device)

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
        """Project and quantize a tensor using MSE-optimal codebook.

        Args:
            x: shape (batch, n_heads, seq_len, head_dim) — the K or V tensor
            R: shape (head_dim, head_dim) — the projection matrix for this layer

        Returns:
            (indices, scale, mean, x_approx)
            - indices: uint8 tensor, same shape as x, values in [0, num_levels-1]
            - scale: per-head-per-token standard deviation (for rescaling)
            - mean: per-head-per-token mean (for centering)
            - x_approx: the dequantized + unprojected approximate tensor
        """
        # Step 1: Project into "flat" space
        # After orthogonal projection, the values are approximately Gaussian.
        x_proj = x @ R

        # Step 2: Normalize to standard Gaussian (mean=0, std=1)
        # The codebook was computed for N(0,1), so we need to standardize
        # the projected values before quantizing, then rescale after.
        # We compute mean and std per (head, token) vector.
        mean = x_proj.mean(dim=-1, keepdim=True)    # (batch, n_heads, seq_len, 1)
        std = x_proj.std(dim=-1, keepdim=True)       # (batch, n_heads, seq_len, 1)
        std = torch.clamp(std, min=1e-8)             # avoid division by zero

        x_normalized = (x_proj - mean) / std          # approximately N(0,1)

        # Step 3: Quantize — find nearest codebook centroid for each value
        # codebook shape: (num_levels,)
        # x_normalized shape: (batch, n_heads, seq_len, head_dim)
        # We compute distance from each value to each centroid, pick the closest.
        #
        # Expand dimensions for broadcasting:
        # x_normalized: (..., head_dim, 1) vs codebook: (num_levels,)
        diffs = (x_normalized.unsqueeze(-1) - self.codebook)  # (..., head_dim, num_levels)
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)  # (..., head_dim)

        # Step 4: Dequantize — look up centroid values and rescale
        # Map indices back to codebook values (still in normalized space)
        x_normalized_approx = self.codebook[indices.long()]

        # Rescale back to original projected space
        x_proj_approx = x_normalized_approx * std + mean

        # Step 5: Unproject — rotate back to original coordinate system
        x_approx = x_proj_approx @ R.T

        return indices, std.squeeze(-1), mean.squeeze(-1), x_approx

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
        k_idx, k_std, k_mean, k_approx = self._quantize_tensor(key_states, R)
        v_idx, v_std, v_mean, v_approx = self._quantize_tensor(value_states, R)

        # Store the compressed representation
        self._quantized_layers[layer_idx].append({
            "k_idx": k_idx, "k_std": k_std, "k_mean": k_mean,
            "v_idx": v_idx, "v_std": v_std, "v_mean": v_mean,
        })

        # Use the parent class to accumulate the APPROXIMATE values
        # This is what the model reads during attention.
        # We pass k_approx/v_approx (lossy) instead of key_states/value_states (exact).
        return super().update(k_approx, v_approx, layer_idx, cache_kwargs)
