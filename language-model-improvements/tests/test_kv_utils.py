"""Unit tests for pure KV-cache helpers."""
import numpy as np
import pytest
from language_model_improvements.kv_utils import kv_cache_bytes, find_outlier_channels


# --- kv_cache_bytes ---

def test_kv_cache_bytes_llama_3_1_8b_one_token_fp16():
    """Llama-3.1-8B at 1 token in fp16 should be 131,072 bytes (~128 KB)."""
    assert kv_cache_bytes(
        seq_len=1, num_layers=32, n_kv_heads=8,
        head_dim=128, p_bytes=2, batch_size=1,
    ) == 131_072


def test_kv_cache_bytes_scales_linearly_with_seq_len():
    base = kv_cache_bytes(
        seq_len=1000, num_layers=32, n_kv_heads=8,
        head_dim=128, p_bytes=2, batch_size=1,
    )
    doubled = kv_cache_bytes(
        seq_len=2000, num_layers=32, n_kv_heads=8,
        head_dim=128, p_bytes=2, batch_size=1,
    )
    assert doubled == 2 * base


def test_kv_cache_bytes_scales_linearly_with_batch():
    base = kv_cache_bytes(
        seq_len=512, num_layers=16, n_kv_heads=4,
        head_dim=64, p_bytes=2, batch_size=1,
    )
    doubled = kv_cache_bytes(
        seq_len=512, num_layers=16, n_kv_heads=4,
        head_dim=64, p_bytes=2, batch_size=2,
    )
    assert doubled == 2 * base


def test_kv_cache_bytes_int8_is_half_of_fp16():
    fp16 = kv_cache_bytes(
        seq_len=4096, num_layers=32, n_kv_heads=8,
        head_dim=128, p_bytes=2, batch_size=1,
    )
    int8 = kv_cache_bytes(
        seq_len=4096, num_layers=32, n_kv_heads=8,
        head_dim=128, p_bytes=1, batch_size=1,
    )
    assert int8 * 2 == fp16


# --- find_outlier_channels ---

def test_find_outlier_channels_picks_the_obvious_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 10))
    x[:, 3] *= 50.0
    outliers = find_outlier_channels(x, threshold_factor=10.0)
    assert outliers == [3]


def test_find_outlier_channels_returns_empty_when_uniform():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((100, 8))
    outliers = find_outlier_channels(x, threshold_factor=10.0)
    assert outliers == []


def test_find_outlier_channels_picks_multiple():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((200, 12))
    x[:, 1] *= 30.0
    x[:, 7] *= 40.0
    outliers = find_outlier_channels(x, threshold_factor=10.0)
    assert outliers == [1, 7]
