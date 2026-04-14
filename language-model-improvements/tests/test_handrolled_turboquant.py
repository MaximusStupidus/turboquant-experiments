"""Tests for the hand-rolled TurboQuant implementation.

These run on CPU — no GPU needed. They verify the mathematical properties
we learned in Phase 0:
1. The projection preserves dot products (JL property)
2. The quantize→dequantize round-trip introduces bounded error
3. Lower bits = more error (sanity check)
"""
import math
import torch
import pytest

from language_model_improvements.handrolled_turboquant import HandrolledTurboQuantCache


@pytest.fixture
def cache():
    """A small cache for testing (CPU, fp32 for numerical clarity).
    residual_length=0 so all tokens get quantized — tests the quantization logic.
    """
    return HandrolledTurboQuantCache(
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        bits=4,
        device="cpu",
        dtype=torch.float32,
        seed=42,
        residual_length=0,
    )


def test_update_returns_tensors_of_correct_shape(cache):
    """update() should return (keys, values) with accumulated sequence length."""
    k = torch.randn(1, 4, 1, 64)  # batch=1, heads=4, seq=1, dim=64
    v = torch.randn(1, 4, 1, 64)

    # First token
    ret_k, ret_v = cache.update(k, v, layer_idx=0)
    assert ret_k.shape == (1, 4, 1, 64)
    assert ret_v.shape == (1, 4, 1, 64)

    # Second token — cache should accumulate
    k2 = torch.randn(1, 4, 1, 64)
    v2 = torch.randn(1, 4, 1, 64)
    ret_k2, ret_v2 = cache.update(k2, v2, layer_idx=0)
    assert ret_k2.shape == (1, 4, 2, 64)  # seq_len grew from 1 to 2
    assert ret_v2.shape == (1, 4, 2, 64)


def test_update_returns_approximate_not_exact(cache):
    """The returned values should be DIFFERENT from the input (lossy)."""
    k = torch.randn(1, 4, 1, 64)
    v = torch.randn(1, 4, 1, 64)

    ret_k, ret_v = cache.update(k, v, layer_idx=0)

    # Should NOT be identical (quantization introduces error)
    assert not torch.equal(ret_k, k), "Expected lossy values, got exact copy"
    assert not torch.equal(ret_v, v), "Expected lossy values, got exact copy"

    # But should be CLOSE (quantization error is bounded)
    k_err = (ret_k - k).abs().mean().item()
    v_err = (ret_v - v).abs().mean().item()
    assert k_err < 0.5, f"Key error too large: {k_err}"
    assert v_err < 0.5, f"Value error too large: {v_err}"


def test_dot_product_approximately_preserved(cache):
    """Core JL property: Q·K_approx ≈ Q·K.

    We use normalized error (absolute error / product of norms) instead of
    relative error on the dot product value, because random vectors can have
    near-zero dot products where relative error blows up meaninglessly.
    """
    # Use multiple vectors to get a stable measurement
    q = torch.randn(1, 4, 32, 64)   # 32 query vectors
    k = torch.randn(1, 4, 32, 64)   # 32 key vectors
    v = torch.randn(1, 4, 32, 64)

    # True dot products: Q @ K^T gives all pairwise dot products
    true_dots = torch.matmul(q, k.transpose(-1, -2))  # (1, 4, 32, 32)

    # Approximate K from cache
    ret_k, ret_v = cache.update(k, v, layer_idx=0)
    approx_dots = torch.matmul(q, ret_k.transpose(-1, -2))

    # Normalized error: |true - approx| / (||q|| * ||k||)
    # This measures how much the dot product error is relative to the
    # magnitude of the vectors, not relative to the dot product value itself.
    q_norms = q.norm(dim=-1, keepdim=True)  # (1, 4, 32, 1)
    k_norms = k.norm(dim=-1, keepdim=True).transpose(-1, -2)  # (1, 4, 1, 32)
    scale = q_norms * k_norms  # (1, 4, 32, 32)

    normalized_error = ((approx_dots - true_dots).abs() / (scale + 1e-6)).mean().item()
    assert normalized_error < 0.15, f"Normalized dot product error too large: {normalized_error:.4f}"


def test_more_bits_means_less_error():
    """Fundamental sanity check: 4-bit should be more accurate than 2-bit."""
    k = torch.randn(1, 4, 10, 64)
    v = torch.randn(1, 4, 10, 64)

    errors = {}
    for bits in [2, 3, 4]:
        c = HandrolledTurboQuantCache(
            num_layers=1, num_kv_heads=4, head_dim=64,
            bits=bits, device="cpu", dtype=torch.float32, seed=42,
            residual_length=0,
        )
        ret_k, _ = c.update(k, v, layer_idx=0)
        err = (ret_k - k).abs().mean().item()
        errors[bits] = err

    # 4-bit error < 3-bit error < 2-bit error
    assert errors[4] < errors[3], f"4-bit ({errors[4]}) should be < 3-bit ({errors[3]})"
    assert errors[3] < errors[2], f"3-bit ({errors[3]}) should be < 2-bit ({errors[2]})"


def test_different_layers_use_different_projections():
    """Each layer should have its own random matrix → different approximations."""
    cache = HandrolledTurboQuantCache(
        num_layers=2, num_kv_heads=4, head_dim=64,
        bits=4, device="cpu", dtype=torch.float32, seed=42,
        residual_length=0,
    )

    k = torch.randn(1, 4, 1, 64)
    v = torch.randn(1, 4, 1, 64)

    ret_k_layer0, _ = cache.update(k, v, layer_idx=0)
    ret_k_layer1, _ = cache.update(k, v, layer_idx=1)

    # Same input, different layers → different approximations
    assert not torch.equal(ret_k_layer0, ret_k_layer1), \
        "Expected different projections per layer"


def test_residual_buffer_keeps_most_recent_exact():
    """The most recent residual_length tokens should be EXACT.
    Older tokens that overflow the residual should be quantized (lossy)."""
    cache = HandrolledTurboQuantCache(
        num_layers=1, num_kv_heads=4, head_dim=64,
        bits=2, device="cpu", dtype=torch.float32, seed=42,
        residual_length=5,
    )

    # Insert 5 tokens (fills residual exactly) — all exact
    k1 = torch.randn(1, 4, 5, 64)
    v1 = torch.randn(1, 4, 5, 64)
    ret_k, _ = cache.update(k1, v1, layer_idx=0)
    assert torch.equal(ret_k, k1), "Initial tokens within residual should be exact"

    # Insert 3 more tokens — residual overflows (8 > 5)
    # Oldest 3 tokens get quantized, most recent 5 stay exact
    k2 = torch.randn(1, 4, 3, 64)
    v2 = torch.randn(1, 4, 3, 64)
    ret_k2, _ = cache.update(k2, v2, layer_idx=0)
    assert ret_k2.shape == (1, 4, 8, 64)

    # The LAST 3 tokens (most recent) should be exact
    assert torch.equal(ret_k2[:, :, -3:, :], k2), \
        "Most recent tokens should be exact (in residual)"

    # The FIRST 3 tokens (oldest, overflowed) should be lossy
    assert not torch.equal(ret_k2[:, :, :3, :], k1[:, :, :3, :]), \
        "Oldest tokens should be quantized (lossy)"
