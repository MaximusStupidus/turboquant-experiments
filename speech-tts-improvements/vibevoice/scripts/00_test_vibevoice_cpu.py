"""CPU-only wrapper for VibeVoice test — forces CPU to avoid MPS segfaults on Mac."""
import torch
# Disable MPS detection BEFORE importing the test script
torch.backends.mps.is_available = lambda: False

# Now run the actual test
exec(open("speech-tts-improvements/scripts/00_test_vibevoice.py").read())
