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
    # Harvard Sentences (IEEE Recommended Practices for Speech Quality
    # Measurements, 1969) — phonetically balanced, no proper nouns or
    # acronyms. Standard TTS/speech-quality benchmark.
    "short":  "The birch canoe slid on the smooth planks.",
    "medium": "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. It's easy to tell the depth of a well.",
    "long":   "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. It's easy to tell the depth of a well. These days a chicken leg is a rare dish. Rice is often served in round bowls. The juice of lemons makes fine punch.",
}

# Parler-TTS emits ~86 audio frames/sec; generation_config.max_length=2580
# ≈ 30 s of audio. Parler's custom generate() uses `max_length` (NOT
# `max_new_tokens`) for the delay pattern mask. Values below give ~1.5x
# headroom over natural speech duration so the model can emit EOS
# cleanly instead of running out of budget mid-sentence.
# Word counts: short=9, medium=23, long=51.
MAX_LENGTH = {
    "short":  500,   # ~5.8 s of audio for ~9-word prompt
    "medium": 1400,  # ~16 s of audio for ~23-word prompt
    "long":   2580,  # full Parler default — for ~51-word prompt (~20 s)
}
