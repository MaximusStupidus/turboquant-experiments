"""Inspect VibeVoice's actual model structure to understand the 4/20 layer split.

The cache error says `layer_idx` out of range in self.layers. The cache we
loaded has 4 layers (base LM) or 20 layers (TTS LM). But the error happens
inside the Qwen2 model's forward, which iterates over its OWN num_hidden_layers.
We need to check if language_model has 24 layers or 4 layers internally.
"""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import sys
import torch

VIBEVOICE_REPO = os.environ.get("VIBEVOICE_REPO", "/home/ubuntu/VibeVoice")
sys.path.insert(0, VIBEVOICE_REPO)

# Monkey-patch register
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
for _cls in (AutoModel, AutoModelForCausalLM, AutoConfig):
    _orig = _cls.register
    def _mk(orig):
        def patched(*a, **kw):
            kw.setdefault("exist_ok", True)
            return orig(*a, **kw)
        return patched
    _cls.register = staticmethod(_mk(_orig))

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)

print("Loading model...")
model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
    "microsoft/VibeVoice-Realtime-0.5B",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()

print("\n=== model.model top-level attributes ===")
inner = model.model
for name in dir(inner):
    if not name.startswith("_"):
        attr = getattr(inner, name)
        if hasattr(attr, "layers") or hasattr(attr, "num_hidden_layers"):
            print(f"  {name}: {type(attr).__name__}")

print("\n=== Sub-model layer counts ===")
for attr_name in ["language_model", "tts_language_model", "tts_lm", "lm"]:
    if hasattr(inner, attr_name):
        sub = getattr(inner, attr_name)
        print(f"  model.model.{attr_name}: {type(sub).__name__}")
        if hasattr(sub, "layers"):
            print(f"    .layers: {len(sub.layers)}")
        if hasattr(sub, "config"):
            if hasattr(sub.config, "num_hidden_layers"):
                print(f"    .config.num_hidden_layers: {sub.config.num_hidden_layers}")
        # Check nested
        if hasattr(sub, "model") and hasattr(sub.model, "layers"):
            print(f"    .model.layers: {len(sub.model.layers)}")

print("\n=== Config values ===")
print(f"  decoder.num_hidden_layers: {model.config.decoder_config.num_hidden_layers}")
print(f"  tts_backbone_num_hidden_layers: {model.config.tts_backbone_num_hidden_layers}")
