"""Dump the raw __dict__ of the loaded cache to see what fields actually exist."""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import sys, torch

VIBEVOICE_REPO = os.environ.get("VIBEVOICE_REPO", "/home/ubuntu/VibeVoice")
sys.path.insert(0, VIBEVOICE_REPO)

voice_path = os.path.join(VIBEVOICE_REPO, "demo/voices/streaming_model/en-Carter_man.pt")
prompt = torch.load(voice_path, map_location="cuda", weights_only=False)

for key in prompt:
    val = prompt[key]
    print(f"\n=== {key} ({type(val).__name__}) ===")
    pkv = val.past_key_values if hasattr(val, "past_key_values") else None
    if pkv is None:
        print("  no past_key_values")
        continue
    print(f"  pkv type: {type(pkv).__name__}")
    print(f"  pkv __dict__ keys: {list(pkv.__dict__.keys())}")
    for k, v in pkv.__dict__.items():
        if isinstance(v, list):
            print(f"  {k}: list of len {len(v)}")
            if len(v) > 0 and hasattr(v[0], "shape"):
                print(f"    [0]: shape={v[0].shape} dtype={v[0].dtype}")
        elif hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}" if not callable(v) else f"  {k}: <callable>")
