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
    """A small cache for testing (CPU, fp32 for numerical clarity)."""
    return HandrolledTurboQuantCache(
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        bits=4,
        device="cpu",
        dtype=torch.float32,
        seed=42,
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
    """Core JL property: Q·K_approx ≈ Q·K."""
    q = torch.randn(1, 4, 1, 64)
    k = torch.randn(1, 4, 1, 64)
    v = torch.randn(1, 4, 1, 64)

    # True dot product
    true_dot = (q * k).sum(dim=-1)  # (1, 4, 1)

    # Approximate K from cache
    ret_k, ret_v = cache.update(k, v, layer_idx=0)
    approx_dot = (q * ret_k).sum(dim=-1)  # (1, 4, 1)

    # Should be close (within ~20% relative error for 4-bit at dim=64)
    rel_error = ((approx_dot - true_dot).abs() / (true_dot.abs() + 1e-6)).mean().item()
    assert rel_error < 0.3, f"Dot product relative error too large: {rel_error:.4f}"


def test_more_bits_means_less_error():
    """Fundamental sanity check: 4-bit should be more accurate than 2-bit."""
    k = torch.randn(1, 4, 10, 64)
    v = torch.randn(1, 4, 10, 64)

    errors = {}
    for bits in [2, 3, 4]:
        c = HandrolledTurboQuantCache(
            num_layers=1, num_kv_heads=4, head_dim=64,
            bits=bits, device="cpu", dtype=torch.float32, seed=42,
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
    )

    k = torch.randn(1, 4, 1, 64)
    v = torch.randn(1, 4, 1, 64)

    ret_k_layer0, _ = cache.update(k, v, layer_idx=0)
    ret_k_layer1, _ = cache.update(k, v, layer_idx=1)

    # Same input, different layers → different approximations
    # (because each layer uses a different random projection matrix)
    assert not torch.equal(ret_k_layer0, ret_k_layer1), \
        "Expected different projections per layer"
