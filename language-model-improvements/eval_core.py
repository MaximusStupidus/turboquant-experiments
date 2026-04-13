"""Shared evaluation functions for the TurboQuant experiment.

These are the measurement functions used by both the baseline eval (phase 2)
and the TurboQuant/KIVI eval (phase 3). They are extracted here so that both
scripts use IDENTICAL measurement code — the only thing that changes between
experiments is the cache object passed to the model.

The cache object is injected via a `cache_factory` parameter: a callable that
returns a fresh cache object. For fp16, pass None (default cache). For KIVI,
pass a factory that returns QuantizedCache. For TurboQuant, pass a factory
that returns TurboQuantCache. The eval code doesn't know or care which one
it's using — it just measures what comes out.

DO NOT MODIFY these functions after baseline.json has been produced.
If a bug is found, fix it here, re-run ALL experiments (baseline included),
and re-commit all results.
"""
import time
import numpy as np
import torch
from datasets import load_dataset

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from language_model_improvements.eval_utils import perplexity_from_nlls


def load_wikitext2_test(tokenizer, max_tokens: int = 32768):
    """Load WikiText-2 test split, concatenate, tokenize, truncate.

    WikiText-2 is a standard benchmark for LM perplexity. We concatenate
    the entire test split into one long token sequence (standard practice),
    then truncate to max_tokens for reproducibility and speed.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join([t for t in ds["text"] if t.strip()])
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids[0]
    if input_ids.shape[0] > max_tokens:
        input_ids = input_ids[:max_tokens]
    return input_ids


def eval_perplexity(model, input_ids, stride: int = 512, max_length: int = 2048,
                    cache_factory=None):
    """Sliding-window perplexity evaluation.

    Args:
        model: the HF causal LM
        input_ids: 1-D tensor of token ids
        stride: sliding window step size
        max_length: sliding window size
        cache_factory: callable returning a fresh cache object, or None for
            default fp16 DynamicCache. This is how TurboQuant and KIVI caches
            get injected — the measurement code stays identical.

    Returns:
        (perplexity, total_nlls, token_counts, peak_mem_bytes)
    """
    device = model.device
    seq_len = input_ids.shape[0]
    total_nlls = []
    token_counts = []

    torch.cuda.reset_peak_memory_stats()

    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_chunk = input_ids[begin_loc:end_loc].unsqueeze(0).to(device)

        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100

        kwargs = {}
        if cache_factory is not None:
            kwargs["past_key_values"] = cache_factory()

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids, **kwargs)
            nll = outputs.loss.float().item()

        total_nlls.append(nll * trg_len)
        token_counts.append(trg_len)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    peak_mem = torch.cuda.max_memory_allocated()
    ppl = perplexity_from_nlls(total_nlls, token_counts)
    return ppl, total_nlls, token_counts, peak_mem


def eval_perplexity_autoregressive(model, input_ids, cache_factory=None,
                                   prefill_len: int = 128, max_eval_tokens: int = 2048):
    """Autoregressive perplexity evaluation — token by token.

    This is the CORRECT way to measure KV cache quantization effects.
    The key difference from eval_perplexity (sliding-window):

    Sliding-window: feeds 2048 tokens in one shot. The cache is written
    but never read back within the same pass. Quantization methods like
    KIVI that only affect read-back show ZERO quality impact.

    Autoregressive: feeds tokens one at a time after an initial prefill.
    Each step reads the cached K/V from ALL previous steps. If the cache
    stores quantized values, the model reads lossy values, and perplexity
    reflects the real quality impact.

    How it works:
    1. Prefill: feed first `prefill_len` tokens in one shot → fills the cache.
       We don't score these (the cache isn't being read back yet during prefill).
    2. Autoregressive loop: feed one token at a time.
       - Model reads cached K/V from all previous tokens (quantized if applicable)
       - Model outputs logits predicting the next token
       - We score the prediction against the actual next token (NLL)
       - New token's K/V gets added to the cache

    Args:
        model: the HF causal LM
        input_ids: 1-D tensor of token ids
        cache_factory: callable returning a fresh cache, or None for fp16
        prefill_len: tokens to feed in one shot to initialize cache
        max_eval_tokens: total tokens to use (prefill + autoregressive)

    Returns:
        (perplexity, total_nll, tokens_scored, peak_mem_bytes)
    """
    import math

    device = model.device
    seq_len = min(input_ids.shape[0], max_eval_tokens)
    input_ids = input_ids[:seq_len]

    torch.cuda.reset_peak_memory_stats()

    # Create cache
    cache = cache_factory() if cache_factory is not None else None

    # Step 1: Prefill — feed first prefill_len tokens in one shot
    prefill_ids = input_ids[:prefill_len].unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(prefill_ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values

    # We CAN score tokens 1..prefill_len from the prefill output, but
    # those scores are unaffected by cache quantization (cache is being
    # written for the first time, not read back). For a fair comparison
    # we only score the autoregressive tokens.

    total_nll = 0.0
    tokens_scored = 0

    # Step 2: Autoregressive — feed one token at a time
    # At step i, we feed token[i] and the model predicts token[i+1].
    # We score: was token[i+1] predicted correctly?
    for i in range(prefill_len, seq_len - 1):
        input_token = input_ids[i].unsqueeze(0).unsqueeze(0).to(device)  # shape (1, 1)

        with torch.no_grad():
            out = model(input_token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values

        # out.logits shape: (1, 1, vocab_size)
        # This predicts the token at position i+1
        logits = out.logits[0, 0].float()  # (vocab_size,)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # The actual next token
        target = input_ids[i + 1].item()
        nll = -log_probs[target].item()

        total_nll += nll
        tokens_scored += 1

        # Progress print every 256 tokens
        if tokens_scored % 256 == 0:
            running_ppl = math.exp(total_nll / tokens_scored)
            print(f"    scored {tokens_scored} tokens, running PPL: {running_ppl:.2f}")

    peak_mem = torch.cuda.max_memory_allocated()
    ppl = math.exp(total_nll / tokens_scored) if tokens_scored > 0 else float('inf')
    return ppl, total_nll, tokens_scored, peak_mem


def eval_generation_throughput(
    model, tokenizer, prompt: str,
    max_new_tokens: int = 128, num_runs: int = 3,
    cache_factory=None,
):
    """Measure generation speed in tokens/sec.

    Uses torch.cuda.synchronize() to ensure we time actual GPU work.
    Runs num_runs times and returns median to smooth warmup effects.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    speeds = []

    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        if cache_factory is not None:
            gen_kwargs["past_key_values"] = cache_factory()

        with torch.no_grad():
            out = model.generate(**gen_kwargs)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated_tokens = out.shape[1] - inputs.input_ids.shape[1]
        speed = generated_tokens / elapsed
        speeds.append(speed)
        print(f"    run {i+1}/{num_runs}: {generated_tokens} tokens in {elapsed:.2f}s = {speed:.1f} tok/s")

    return float(np.median(speeds))
