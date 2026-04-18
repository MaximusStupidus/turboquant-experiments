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
