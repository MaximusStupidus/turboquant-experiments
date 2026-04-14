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
        residual_length: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bits = bits
        self.num_levels = 2 ** bits  # e.g., 16 for 4-bit, 4 for 2-bit
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.residual_length = residual_length

        # Precompute the MSE-optimal codebook for the unit-sphere distribution.
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

        # Per-layer state for the sliding residual buffer.
        # _quantized_k/v: accumulated dequantized older tokens (or None)
        # _residual_k/v: list of recent exact token tensors
        # _residual_count: how many tokens are in the residual right now
        #
        # Strategy: new tokens always enter the residual (exact). When the
        # residual exceeds residual_length, the OLDEST tokens in the residual
        # get quantized and merged into _quantized. The MOST RECENT tokens
        # stay exact. This means the model's immediate context is always
        # high-quality, and only distant past context is lossy.
        self._quantized_k = [None] * num_layers
        self._quantized_v = [None] * num_layers
        self._residual_k = [[] for _ in range(num_layers)]
        self._residual_v = [[] for _ in range(num_layers)]

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
        """Override DynamicCache.update() with sliding residual buffer.

        How it works:
        1. New tokens ALWAYS enter the residual buffer (exact, no quantization).
        2. If the residual exceeds residual_length, the OLDEST tokens in the
           residual get quantized and pushed into the quantized portion.
        3. The MOST RECENT residual_length tokens stay exact.
        4. We return [quantized (lossy) | residual (exact)] concatenated.

        This means the tokens the model is actively attending to most strongly
        (the most recent ones) are always at full precision. Only distant past
        tokens — whose attention weights are typically small — are lossy.

        We do NOT use super().update() because DynamicCache's internal storage
        format varies across transformers versions. Instead we manage our own
        state and just return the (key, value) tensors that the model expects.
        """
        R = self.projections[layer_idx]

        # Step 1: Add new tokens to the residual (exact)
        self._residual_k[layer_idx].append(key_states)
        self._residual_v[layer_idx].append(value_states)

        # Concatenate all residual tokens into one tensor
        res_k = torch.cat(self._residual_k[layer_idx], dim=2)
        res_v = torch.cat(self._residual_v[layer_idx], dim=2)
        res_len = res_k.shape[2]

        # Step 2: If residual overflows, quantize the oldest tokens
        if res_len > self.residual_length:
            overflow = res_len - self.residual_length

            # Split: [oldest (to quantize) | newest (keep exact)]
            old_k = res_k[:, :, :overflow, :]
            old_v = res_v[:, :, :overflow, :]
            keep_k = res_k[:, :, overflow:, :]
            keep_v = res_v[:, :, overflow:, :]

            # Quantize the oldest tokens
            _, _, old_k_approx = self._quantize_tensor(old_k, R)
            _, _, old_v_approx = self._quantize_tensor(old_v, R)

            # Merge into quantized portion
            if self._quantized_k[layer_idx] is not None:
                self._quantized_k[layer_idx] = torch.cat(
                    [self._quantized_k[layer_idx], old_k_approx], dim=2)
                self._quantized_v[layer_idx] = torch.cat(
                    [self._quantized_v[layer_idx], old_v_approx], dim=2)
            else:
                self._quantized_k[layer_idx] = old_k_approx
                self._quantized_v[layer_idx] = old_v_approx

            # Update residual to only the newest tokens
            self._residual_k[layer_idx] = [keep_k]
            self._residual_v[layer_idx] = [keep_v]
            res_k = keep_k
            res_v = keep_v

        # Step 3: Build full view [quantized | residual] and return
        if self._quantized_k[layer_idx] is not None:
            full_k = torch.cat([self._quantized_k[layer_idx], res_k], dim=2)
            full_v = torch.cat([self._quantized_v[layer_idx], res_v], dim=2)
        else:
            full_k = res_k
            full_v = res_v

        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the total cached sequence length for a layer."""
        total = 0
        if self._quantized_k[layer_idx] is not None:
            total += self._quantized_k[layer_idx].shape[2]
        for t in self._residual_k[layer_idx]:
            total += t.shape[2]
        return total

    def __len__(self):
        """Number of layers in the cache."""
        return self.num_layers

    def __iter__(self):
        """Iterate over layers, yielding (key, value) tuples."""
        for i in range(self.num_layers):
            res_k = torch.cat(self._residual_k[i], dim=2) if self._residual_k[i] else None
            res_v = torch.cat(self._residual_v[i], dim=2) if self._residual_v[i] else None
            if self._quantized_k[i] is not None and res_k is not None:
                k = torch.cat([self._quantized_k[i], res_k], dim=2)
                v = torch.cat([self._quantized_v[i], res_v], dim=2)
            elif res_k is not None:
                k, v = res_k, res_v
            else:
                continue
            yield (k, v)
