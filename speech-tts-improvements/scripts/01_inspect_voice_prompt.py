"""Inspect the actual structure of a voice prompt .pt file.

The pre-built voice prompts use an older DynamicCache format that doesn't
work with current transformers. We need to understand exactly what's inside
so we can write a shim that converts to the new format.
"""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import sys
import torch

VIBEVOICE_REPO = os.environ.get("VIBEVOICE_REPO", "/home/ubuntu/VibeVoice")
sys.path.insert(0, VIBEVOICE_REPO)

voice_path = os.path.join(VIBEVOICE_REPO, "demo/voices/streaming_model/en-Carter_man.pt")
print(f"Loading: {voice_path}")
prompt = torch.load(voice_path, map_location="cuda", weights_only=False)

print(f"\nTop-level type: {type(prompt).__name__}")
print(f"Keys: {list(prompt.keys())}")

for key in prompt:
    val = prompt[key]
    print(f"\n=== {key} ===")
    print(f"  type: {type(val).__name__}")
    print(f"  attrs: {[x for x in dir(val) if not x.startswith('_')][:20]}")

    # Check past_key_values specifically
    if hasattr(val, "past_key_values"):
        pkv = val.past_key_values
        print(f"  past_key_values type: {type(pkv).__name__}")
        print(f"  past_key_values attrs: {[x for x in dir(pkv) if not x.startswith('_')][:30]}")

        # Check if it's a tuple of tuples (legacy) or DynamicCache
        if isinstance(pkv, tuple):
            print(f"  LEGACY FORMAT: tuple of length {len(pkv)}")
            if len(pkv) > 0 and isinstance(pkv[0], tuple):
                print(f"  inner[0] len: {len(pkv[0])}")
                if len(pkv[0]) > 0 and hasattr(pkv[0][0], "shape"):
                    print(f"  pkv[0][0] shape: {pkv[0][0].shape}")
        else:
            if hasattr(pkv, "layers"):
                print(f"  DynamicCache.layers len: {len(pkv.layers)}")
            if hasattr(pkv, "key_cache"):
                print(f"  DynamicCache.key_cache len: {len(pkv.key_cache)}")
                if len(pkv.key_cache) > 0:
                    print(f"  key_cache[0] shape: {pkv.key_cache[0].shape}")
            # Try to iterate
            try:
                items = list(iter(pkv))
                print(f"  iter() gave {len(items)} items")
                if len(items) > 0:
                    print(f"  item[0] type: {type(items[0]).__name__}")
                    if isinstance(items[0], tuple):
                        print(f"  item[0] len: {len(items[0])}")
                        if hasattr(items[0][0], "shape"):
                            print(f"  item[0][0] shape: {items[0][0].shape}")
            except Exception as e:
                print(f"  iter() failed: {e}")

    if hasattr(val, "last_hidden_state"):
        lhs = val.last_hidden_state
        if hasattr(lhs, "shape"):
            print(f"  last_hidden_state shape: {lhs.shape}")
