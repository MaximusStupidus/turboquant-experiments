"""Unit tests for eval helpers — pure math, no GPU."""
import math
import pytest
from language_model_improvements.eval_utils import (
    perplexity_from_nlls,
    format_bytes,
    timed,
)


# --- perplexity_from_nlls ---

def test_perplexity_perfect_model():
    """A model that assigns probability 1 to every token has perplexity 1.
    NLL = -log(1) = 0 for every token, so exp(0) = 1."""
    nlls = [0.0, 0.0, 0.0]
    token_counts = [10, 10, 10]
    assert perplexity_from_nlls(nlls, token_counts) == pytest.approx(1.0)


def test_perplexity_known_value():
    """If avg NLL = ln(100) ≈ 4.605, perplexity should be 100.
    50 tokens, total NLL = ln(100)*50, so mean NLL = ln(100), ppl = 100."""
    nll_per_token = math.log(100)
    nlls = [nll_per_token * 50]
    token_counts = [50]
    assert perplexity_from_nlls(nlls, token_counts) == pytest.approx(100.0, rel=1e-4)


def test_perplexity_multiple_chunks_same_as_one():
    """Perplexity must be the same whether computed in one chunk or many.
    This catches the common mistake of averaging per-chunk perplexities."""
    nll_per_token = 2.5
    ppl_one = perplexity_from_nlls([nll_per_token * 100], [100])
    ppl_two = perplexity_from_nlls(
        [nll_per_token * 60, nll_per_token * 40], [60, 40]
    )
    assert ppl_one == pytest.approx(ppl_two, rel=1e-6)


def test_perplexity_empty_raises():
    """No data should raise ValueError — can't compute ppl from nothing."""
    with pytest.raises(ValueError):
        perplexity_from_nlls([], [])


# --- format_bytes ---

def test_format_bytes():
    assert format_bytes(0) == "0 B"
    assert format_bytes(1023) == "1023 B"
    assert format_bytes(1024) == "1.00 KB"
    assert format_bytes(1_048_576) == "1.00 MB"
    assert format_bytes(1_073_741_824) == "1.00 GB"
    assert format_bytes(16_060_522_496) == "14.96 GB"


# --- timed ---

def test_timed_returns_result_and_elapsed():
    """timed() should return (result, elapsed_seconds)."""
    result, elapsed = timed(lambda: sum(range(1000)))
    assert result == 499500
    assert elapsed > 0
    assert elapsed < 5  # should be near-instant
