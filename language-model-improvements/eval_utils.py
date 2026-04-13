"""Pure helpers for the eval harness.

No torch, no transformers — just math and formatting so these can be
unit-tested fast on CPU.
"""
import math
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def perplexity_from_nlls(
    total_nlls: list[float],
    token_counts: list[int],
) -> float:
    """Compute perplexity from accumulated negative log-likelihoods.

    We evaluate perplexity in chunks (sliding windows). Each chunk gives us:
    - total_nll: the SUM of per-token NLLs for that chunk
    - token_count: how many tokens were scored in that chunk

    Perplexity = exp(total NLL across all chunks / total tokens across all chunks)

    This is the correct way to combine chunks. The WRONG way would be to compute
    perplexity per chunk and then average — that gives a different (incorrect) number
    because exp(mean(x)) != mean(exp(x)).
    """
    total_tokens = sum(token_counts)
    if total_tokens == 0:
        raise ValueError("Cannot compute perplexity with zero tokens")
    total_nll = sum(total_nlls)
    mean_nll = total_nll / total_tokens
    return math.exp(mean_nll)


def format_bytes(n: int) -> str:
    """Human-readable byte string: B, KB, MB, GB."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.2f} KB"
    elif n < 1024 ** 3:
        return f"{n / 1024**2:.2f} MB"
    else:
        return f"{n / 1024**3:.2f} GB"


def timed(fn: Callable[[], T]) -> tuple[T, float]:
    """Run fn() and return (result, elapsed_seconds).

    Uses perf_counter for high-resolution timing. We use this to measure
    generation throughput — wrap the model.generate() call in timed() and
    divide tokens by elapsed to get tokens/sec.
    """
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed
