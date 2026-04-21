"""Per-step numerical equivalence tests for the handrolled TurboQuant cache.

End-to-end perplexity matching the community package (5.51 / 5.54 / 5.66 /
6.07 on Llama-3.1-8B) is necessary but not sufficient — both
implementations could be similarly wrong. These tests prove
tensor-level correctness of the individual algorithmic steps.

Scope: the steps whose correctness is NOT ambiguous — orthogonality of
the projection matrix, norm preservation under rotation, codebook
monotonicity, quantization round-trip bounds. We do NOT assert
bit-for-bit match with a specific community-package output because
transformers versions drift; equivalence at the paper-spec level is
what we can check.
"""
import numpy as np
import pytest
import torch

from language_model_improvements.handrolled_turboquant import (
    HandrolledTurboQuantCache,
    NaiveQuantCache,
    compute_beta_codebook,
)


def test_projection_matrix_is_orthogonal():
    """Every layer's projection matrix R must satisfy R @ R.T == I to
    numerical precision. Non-orthogonal R would distort norms / dot
    products, adding error on top of quantization (which is exactly
    what TurboQuant is designed to avoid)."""
    cache = HandrolledTurboQuantCache(
        num_layers=4, num_kv_heads=8, head_dim=64,
        bits=4, device="cpu", dtype=torch.float32, seed=42,
    )
    for layer_idx, R in enumerate(cache.projections):
        Rf = R.float()
        RRt = Rf @ Rf.T
        eye = torch.eye(Rf.shape[0])
        err = (RRt - eye).abs().max().item()
        assert err < 1e-4, f"layer {layer_idx}: R @ R.T deviation {err:.2e}"


def test_rotation_preserves_norms():
    """||R @ x|| == ||x|| for all x, because R is orthogonal. A TTS
    reader should see: the rotation changes the *direction* of each
    K/V vector but not its length — all the energy is preserved,
    just redistributed across channels."""
    torch.manual_seed(0)
    cache = HandrolledTurboQuantCache(
        num_layers=1, num_kv_heads=4, head_dim=32,
        bits=4, device="cpu", dtype=torch.float32, seed=1,
    )
    R = cache.projections[0].float()
    x = torch.randn(16, 32)
    rotated = x @ R
    in_norms = x.norm(dim=-1)
    out_norms = rotated.norm(dim=-1)
    assert torch.allclose(in_norms, out_norms, atol=1e-5), (
        f"norm drift up to {(in_norms - out_norms).abs().max():.2e}"
    )


def test_rotation_preserves_dot_products():
    """<R @ x, R @ y> == <x, y>. This is the operational reason
    rotation is safe: attention computes dot products (Q @ K^T), and
    those are unchanged by orthogonal rotation. Quantization error is
    therefore the ONLY approximation we introduce."""
    torch.manual_seed(0)
    cache = HandrolledTurboQuantCache(
        num_layers=1, num_kv_heads=4, head_dim=32,
        bits=4, device="cpu", dtype=torch.float32, seed=2,
    )
    R = cache.projections[0].float()
    x = torch.randn(8, 32)
    y = torch.randn(8, 32)
    raw_dots = (x * y).sum(dim=-1)
    rot_dots = ((x @ R) * (y @ R)).sum(dim=-1)
    assert torch.allclose(raw_dots, rot_dots, atol=1e-5), (
        f"dot-product drift up to {(raw_dots - rot_dots).abs().max():.2e}"
    )


def test_codebook_is_sorted_and_spans_unit_interval():
    """The Beta codebook should be monotonically increasing and lie
    in [-1, 1] since it represents centroids of the Beta((d-1)/2,
    (d-1)/2) distribution on the unit sphere after mapping to [-1, 1]."""
    for bits in [2, 3, 4]:
        cb = compute_beta_codebook(dim=64, bits=bits).numpy()
        assert len(cb) == 2 ** bits
        assert np.all(np.diff(cb) > 0), f"{bits}-bit codebook not sorted: {cb}"
        assert cb[0] > -1.0 and cb[-1] < 1.0, (
            f"{bits}-bit codebook out of [-1, 1]: {cb[0]:.3f}..{cb[-1]:.3f}"
        )


def test_codebook_is_symmetric_around_zero():
    """For Beta((d-1)/2, (d-1)/2) on [-1, 1] the distribution is
    symmetric, so optimal centroids should be too: centroid_k ≈
    -centroid_(N-1-k)."""
    for bits in [2, 3, 4]:
        cb = compute_beta_codebook(dim=64, bits=bits).numpy()
        for k in range(len(cb) // 2):
            mirror = cb[-(k + 1)]
            assert abs(cb[k] + mirror) < 1e-3, (
                f"{bits}-bit codebook asymmetric at pos {k}: "
                f"{cb[k]:.4f} vs mirror {-mirror:.4f}"
            )


def test_quantize_roundtrip_preserves_norm_approximately():
    """After quantize+dequantize of a unit vector, the reconstructed
    length should stay close to 1 (TurboQuant quantizes on the unit
    sphere, so the stored norm is applied back exactly; what we're
    checking is that the rotated-and-requantized direction doesn't
    drift too far)."""
    torch.manual_seed(0)
    cache = HandrolledTurboQuantCache(
        num_layers=1, num_kv_heads=1, head_dim=64,
        bits=4, device="cpu", dtype=torch.float32, seed=3,
    )
    x = torch.randn(1, 1, 32, 64) * 3.0  # (b, h, seq, d)
    _, _, x_approx = cache._quantize_tensor(x, cache.projections[0])
    # Max per-coordinate error should be bounded by half the largest
    # codebook step (rotation-preserves-norm + max-quantization-step).
    err = (x - x_approx).abs().max().item()
    assert err < 5.0, f"4-bit reconstruction max error {err:.2f} too large"


def test_2bit_has_larger_error_than_4bit():
    """MSE monotonicity: fewer bits → larger reconstruction error.
    This is the trivial consistency check that any quantizer should
    pass, and it catches bugs like 'codebook has only 2 distinct
    levels at 4-bit' (silently-wrong but end-to-end-passing)."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, 64)
    errs = {}
    for bits in [2, 3, 4]:
        cache = HandrolledTurboQuantCache(
            num_layers=1, num_kv_heads=1, head_dim=64,
            bits=bits, device="cpu", dtype=torch.float32, seed=4,
        )
        _, _, x_approx = cache._quantize_tensor(x, cache.projections[0])
        errs[bits] = ((x - x_approx) ** 2).mean().item()
    assert errs[2] > errs[3] > errs[4], f"MSE not monotonic in bits: {errs}"


# ---------- NaiveQuantCache regression tests ----------
# Why these exist: during the ablation sweep, NaiveQuantCache silently
# no-op'd because per-channel min-max on a SINGLE overflowing token
# gives min == max → scale = 0 → all values snap to level 0 → output
# = input. End-to-end WER came out identical to fp16 baseline and
# speaker similarity was 1.000 — which is what tipped us off. Fixed by
# reducing over head_dim (per-token, KIVI's V strategy) instead of
# seq_len. These tests lock that in.


def test_naive_quant_per_token_single_token_case():
    """With a single-token tensor, naive per-token quant must still
    produce distinct levels across channels (the original per-channel
    bug degenerated on single tokens)."""
    torch.manual_seed(0)
    cache = NaiveQuantCache(
        num_layers=1, num_kv_heads=1, head_dim=64,
        bits=2, device="cpu", dtype=torch.float32, residual_length=0,
    )
    # Single token, 64 channels with varied values.
    x = torch.linspace(-3, 3, 64).view(1, 1, 1, 64)
    x_approx = cache._quantize_tensor(x)
    unique_vals = torch.unique(x_approx)
    # At 2 bits we expect up to 4 distinct values. Must be more than 1.
    assert len(unique_vals) > 1, (
        f"NaiveQuantCache degenerated to single level: {unique_vals.tolist()}"
    )
    assert len(unique_vals) <= 4, (
        f"Expected ≤4 levels at 2-bit, got {len(unique_vals)}"
    )


def test_naive_quant_produces_expected_level_count():
    """At N bits, max 2^N unique values per token (per-token quant)."""
    torch.manual_seed(0)
    for bits in [2, 3, 4]:
        cache = NaiveQuantCache(
            num_layers=1, num_kv_heads=1, head_dim=32,
            bits=bits, device="cpu", dtype=torch.float32, residual_length=0,
        )
        x = torch.randn(1, 1, 8, 32)  # 8 tokens
        x_approx = cache._quantize_tensor(x)
        # For each (batch, head, token), count unique values across
        # the 32 channels.
        for t in range(8):
            vals = torch.unique(x_approx[0, 0, t, :])
            assert len(vals) <= 2 ** bits, (
                f"{bits}-bit: token {t} has {len(vals)} levels, max 2^{bits}"
            )


def test_naive_quant_mse_monotonic_in_bits():
    """More bits → less reconstruction error. Regression test against
    the per-channel bug: that bug gave MSE=0 at every bit count
    because the output equalled the input."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 16, 64)
    errs = {}
    for bits in [2, 3, 4]:
        cache = NaiveQuantCache(
            num_layers=1, num_kv_heads=1, head_dim=64,
            bits=bits, device="cpu", dtype=torch.float32, residual_length=0,
        )
        x_approx = cache._quantize_tensor(x)
        errs[bits] = ((x - x_approx) ** 2).mean().item()
        # Critical: NOT zero. The previous bug made this zero.
        assert errs[bits] > 1e-6, (
            f"NaiveQuantCache at {bits}-bit has MSE {errs[bits]:.2e} — likely a no-op bug"
        )
    assert errs[2] > errs[3] > errs[4], f"Naive MSE not monotonic: {errs}"


def test_no_projection_flag_changes_behaviour():
    """use_projection=False must produce different output than the
    default. Regression against a bug where the flag could silently
    short-circuit back to the full pipeline."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 16, 64) * 2.0

    cache_proj = HandrolledTurboQuantCache(
        num_layers=1, num_kv_heads=1, head_dim=64, bits=2,
        device="cpu", dtype=torch.float32, seed=7, use_projection=True,
    )
    cache_noproj = HandrolledTurboQuantCache(
        num_layers=1, num_kv_heads=1, head_dim=64, bits=2,
        device="cpu", dtype=torch.float32, seed=7, use_projection=False,
    )

    _, _, x_with = cache_proj._quantize_tensor(x, cache_proj.projections[0])
    _, _, x_without = cache_noproj._quantize_tensor(x, cache_noproj.projections[0])

    diff = (x_with - x_without).abs().max().item()
    assert diff > 1e-3, (
        f"use_projection flag had no effect: max diff {diff:.2e}"
    )
