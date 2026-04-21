"""Shared test data for Part 2 Parler experiments.

Fixed voices and texts, so every config generates the exact same audio
modulo the cache quantization. This is what makes the comparison fair.
"""

VOICES = {
    "jon":   "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.",
    "laura": "Laura's voice is clear and calm, delivered at a moderate speed with a neutral pitch. The recording is very clear with no background noise.",
    "gary":  "Gary speaks with a slightly expressive tone at a moderate speed. The recording is very clear and close up, with no background noise.",
}

TEXTS = {
    "short":  "Hello, this is a test of voice quality after KV cache compression.",
    "medium": "Hello, this message is from Dr. [redacted], [redacted], [redacted], and [redacted]. We are testing voice quality after KV cache compression on open source TurboQuant.",
    "long":   "Large language models store a KV cache during text generation — cached Key and Value vectors from all previous tokens that the model reads back at each step to compute attention. At long context lengths, this cache can exceed the model weights in memory. TurboQuant compresses it.",
}

# Parler-TTS emits ~86 audio frames/sec; generation_config.max_length=2580
# corresponds to ~30 s of audio. Parler's custom generate() IGNORES
# `max_new_tokens` and uses `max_length` for the delay pattern mask, so
# we pass max_length directly per prompt sized to ~1.3x natural speech
# duration (3 words/sec). Word counts: short=12, medium=34, long=56.
MAX_LENGTH = {
    "short":  450,   # ~5 s of audio for ~12-word prompt
    "medium": 1200,  # ~14 s of audio for ~34-word prompt
    "long":   2400,  # ~28 s of audio for ~56-word prompt (close to model cap 2580)
}
