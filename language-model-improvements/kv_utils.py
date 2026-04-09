"""Pure helpers for reasoning about KV cache memory.

Kept dependency-free (no torch, no transformers) so they can be unit-tested
fast on CPU and reused from any script.
"""
import numpy as np
from numpy.typing import NDArray


def kv_cache_bytes(
    *,
    seq_len: int,
    num_layers: int,
    n_kv_heads: int,
    head_dim: int,
    p_bytes: int,
    batch_size: int = 1,
) -> int:
    """Total KV-cache memory for one forward pass, in bytes.

    Formula:
        2 * batch * seq * n_layers * n_kv_heads * head_dim * p_bytes
    The leading 2 accounts for storing both K and V.
    """
    return 2 * batch_size * seq_len * num_layers * n_kv_heads * head_dim * p_bytes


def find_outlier_channels(
    x: NDArray,
    threshold_factor: float = 10.0,
) -> list[int]:
    """Return indices of channels whose max-abs value exceeds threshold_factor * median.

    A channel is flagged if its per-channel max-abs value exceeds
    threshold_factor * median(per-channel max-abs values across all channels).

    Args:
        x: 2-D array, shape (num_tokens, num_channels).
        threshold_factor: how many times the typical channel magnitude a channel
            must reach to count as an outlier.
    """
    per_channel_max = np.abs(x).max(axis=0)
    typical = float(np.median(per_channel_max))
    threshold = threshold_factor * typical
    return [int(i) for i in np.where(per_channel_max > threshold)[0]]
