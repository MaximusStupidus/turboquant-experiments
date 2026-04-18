"""Verify the AWS environment is correctly set up before generation."""
import torch
import transformers
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
try:
    import parler_tts
    print(f"parler_tts: {parler_tts.__version__}")
except Exception as e:
    print(f"parler_tts import FAILED: {e}")

print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"vram (GB): {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")

import numpy as np
print(f"numpy: {np.__version__}")

import soundfile as sf
print(f"soundfile: {sf.__version__}")

print("\nAll imports OK.")
