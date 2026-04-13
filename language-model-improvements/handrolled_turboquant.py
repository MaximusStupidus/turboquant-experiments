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
from transformers import DynamicCache


def compute_beta_codebook(dim: int, bits: int) -> torch.Tensor:
    """Compute MSE-optimal quantization centroids for the unit-sphere distribution.

    When you take a vector, normalize it to unit norm, then multiply by a
    random orthogonal matrix, each coordinate of the result follows a
    Beta((d-1)/2, (d-1)/2) distribution mapped to [-1, 1].

    This is NOT a Gaussian. For d=128, alpha=63.5, the Beta distribution is
    much more concentrated around 0 than a Gaussian — lighter tails, sharper
    peak. Using a Gaussian codebook wastes centroids on tail regions where
    no data lives. Using the correct Beta codebook places centroids where
    the data actually is.

    The procedure (Lloyd-Max quantization for Beta):
    1. Divide the Beta distribution into 2^bits equal-probability regions
       using Beta quantiles.
    2. Place each centroid at the conditional expectation within its region,
       computed via numerical integration.

    Args:
        dim: the vector dimension (e.g., 128 for head_dim). Determines the
             Beta shape parameter alpha = (dim-1)/2.
        bits: number of quantization bits.

    Returns:
        codebook: tensor of shape (2^bits,) containing optimal centroids
                  in [-1, 1], sorted ascending.
    """
    from scipy.special import betaincinv
    from scipy.stats import beta as beta_dist

    num_levels = 2 ** bits
    alpha = (dim - 1) / 2.0

    # Special case: 1-bit (2 centroids) has a closed-form solution
    if bits == 1:
        c = math.sqrt(2.0 / (math.pi * dim))
        return torch.tensor([-c, c], dtype=torch.float32)

    # Beta distribution on [0, 1] with shape (alpha, alpha)
    rv = beta_dist(alpha, alpha)

    # Find boundaries that divide the Beta into equal-probability regions.
    # betaincinv gives the quantile function of the regularized incomplete beta.
    # Map from [0,1] to [-1,1] via: x_mapped = 2*x - 1
    boundaries_01 = []
    for i in range(1, num_levels):
        q = float(betaincinv(alpha, alpha, i / num_levels))
        boundaries_01.append(q)

    # Compute centroid for each region: E[X | lower < X < upper]
    # Using numerical integration on a fine grid
    centroids = []
    lowers = [0.0] + boundaries_01
    uppers = boundaries_01 + [1.0]

    for lo, hi in zip(lowers, uppers):
        # Fine grid within [lo, hi]
        x = np.linspace(lo, hi, 2000)
        pdf_vals = rv.pdf(x)
        # E[X | lo < X < hi] = integral(x * pdf) / integral(pdf)
        numerator = np.trapezoid(x * pdf_vals, x)
        denominator = np.trapezoid(pdf_vals, x)
        if denominator > 0:
            centroid_01 = numerator / denominator
        else:
            centroid_01 = (lo + hi) / 2.0
        # Map from [0,1] to [-1,1]
        centroids.append(2.0 * centroid_01 - 1.0)

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

        # Precompute the MSE-optimal codebook for the unit-sphere distribution.
        # After normalizing to unit norm and rotating by an orthogonal matrix,
        # each coordinate follows Beta((d-1)/2, (d-1)/2) mapped to [-1,1].
        # The codebook is optimized for this EXACT distribution.
        self.codebook = compute_beta_codebook(head_dim, bits).to(dtype).to(device)

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
        """Project and quantize a tensor using unit-sphere normalization + Beta codebook.

        The correct TurboQuant procedure:
        1. Normalize each vector to unit norm (store the norm as a scalar)
        2. Rotate by orthogonal matrix R
        3. Each coordinate now follows Beta((d-1)/2, (d-1)/2) on [-1,1]
        4. Quantize using the precomputed Beta-optimal codebook
        5. Dequantize, unrotate, rescale by stored norm

        Args:
            x: shape (batch, n_heads, seq_len, head_dim) — the K or V tensor
            R: shape (head_dim, head_dim) — the projection matrix for this layer

        Returns:
            (indices, norms, x_approx)
            - indices: uint8 tensor, values in [0, num_levels-1]
            - norms: L2 norms per vector, shape (batch, n_heads, seq_len)
            - x_approx: the dequantized + unprojected approximate tensor
        """
        # Work in float32 for numerical precision during quantization
        x_f32 = x.float()

        # Step 1: Normalize to unit sphere
        # Store the L2 norm — this is the ONLY scalar we need per vector
        # (vs. mean+std = 2 scalars in the wrong Gaussian approach)
        norms = x_f32.norm(dim=-1, keepdim=True)      # (batch, n_heads, seq_len, 1)
        x_unit = x_f32 / (norms + 1e-10)               # unit norm vectors

        # Step 2: Rotate by orthogonal matrix
        # Each coordinate of x_rotated now follows Beta((d-1)/2, (d-1)/2) on [-1,1]
        x_rotated = x_unit @ R.float()

        # Step 3: Quantize — find nearest Beta-optimal centroid for each value
        # codebook shape: (num_levels,) — centroids in [-1, 1]
        # x_rotated values are in [-1, 1]
        diffs = (x_rotated.unsqueeze(-1) - self.codebook.float())  # (..., head_dim, num_levels)
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

        # Step 4: Dequantize — look up centroid values
        x_rotated_approx = self.codebook[indices.long()].float()

        # Step 5: Unrotate — back to original coordinate system
        x_unit_approx = x_rotated_approx @ R.float().T

        # Step 6: Rescale by stored norm
        x_approx = x_unit_approx * norms

        return indices, norms.squeeze(-1), x_approx.to(x.dtype)

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
        k_idx, k_norms, k_approx = self._quantize_tensor(key_states, R)
        v_idx, v_norms, v_approx = self._quantize_tensor(value_states, R)

        # Store the compressed representation
        self._quantized_layers[layer_idx].append({
            "k_idx": k_idx, "k_norms": k_norms,
            "v_idx": v_idx, "v_norms": v_norms,
        })

        # Use the parent class to accumulate the APPROXIMATE values
        # This is what the model reads during attention.
        # We pass k_approx/v_approx (lossy) instead of key_states/value_states (exact).
        return super().update(k_approx, v_approx, layer_idx, cache_kwargs)
