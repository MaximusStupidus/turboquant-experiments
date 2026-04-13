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
