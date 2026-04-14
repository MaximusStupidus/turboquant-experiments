"""Phase 2.0 — Test that VibeVoice-Realtime-0.5B inference works.

This script verifies we can:
1. Load the model
2. Load a voice prompt
3. Generate speech from text
4. Save it as a WAV file

Run on GPU:
    uv run python speech-tts-improvements/scripts/00_test_vibevoice.py

Requires: VibeVoice repo cloned alongside our repo (see setup instructions).
"""
import sys
import os
import time
import torch
import copy

# Add VibeVoice repo to path
VIBEVOICE_REPO = os.environ.get("VIBEVOICE_REPO", "/workspace/VibeVoice")
sys.path.insert(0, VIBEVOICE_REPO)

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor


def main():
    print("=== VibeVoice-Realtime-0.5B Test ===\n")

    # Load model
    print("Loading model...")
    t0 = time.time()
    processor = VibeVoiceStreamingProcessor.from_pretrained(
        "microsoft/VibeVoice-Realtime-0.5B"
    )
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        "microsoft/VibeVoice-Realtime-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Print model config
    cfg = model.config.decoder_config
    print("=== Model Config ===")
    print(f"  num_hidden_layers    : {cfg.num_hidden_layers}")
    print(f"  num_attention_heads  : {cfg.num_attention_heads}")
    print(f"  num_key_value_heads  : {cfg.num_key_value_heads}")
    print(f"  hidden_size          : {cfg.hidden_size}")
    print(f"  head_dim             : {cfg.hidden_size // cfg.num_attention_heads}")
    print(f"  tts_backbone_layers  : {model.config.tts_backbone_num_hidden_layers}")
    print()

    # Load voice prompt
    voice_path = os.path.join(VIBEVOICE_REPO, "demo/voices/streaming_model/en-Carter_man.pt")
    if not os.path.exists(voice_path):
        print(f"ERROR: Voice file not found at {voice_path}")
        print(f"Make sure the VibeVoice repo is cloned at {VIBEVOICE_REPO}")
        sys.exit(1)

    print(f"Loading voice prompt: {voice_path}")
    all_prefilled_outputs = torch.load(voice_path, map_location="cuda", weights_only=False)
    print(f"Voice prompt loaded.\n")

    # Inspect the cache structure
    print("=== Voice Prompt Cache Structure ===")
    for key in all_prefilled_outputs:
        val = all_prefilled_outputs[key]
        if hasattr(val, 'shape'):
            print(f"  {key}: shape={val.shape} dtype={val.dtype}")
        elif isinstance(val, (list, tuple)):
            print(f"  {key}: type={type(val).__name__} len={len(val)}")
            if len(val) > 0 and hasattr(val[0], 'shape'):
                print(f"    [0]: shape={val[0].shape}")
        elif hasattr(val, 'key_cache'):
            print(f"  {key}: type={type(val).__name__} layers={len(val.key_cache)}")
            if len(val.key_cache) > 0:
                print(f"    key_cache[0]: shape={val.key_cache[0].shape}")
        else:
            print(f"  {key}: type={type(val).__name__}")
    print()

    # Generate speech
    test_text = "Hello, this is a test of voice quality after KV cache compression."
    print(f"Generating speech for: {test_text!r}")

    inputs = processor.process_input_with_cached_prompt(
        text=test_text,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to("cuda")

    t0 = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=1.5,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs),
    )
    gen_time = time.time() - t0

    # Save audio
    output_dir = "speech-tts-improvements/results/baseline"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_output.wav")
    processor.save_audio(outputs.speech_outputs[0], output_path=output_path)

    # Calculate RTF
    import torchaudio
    waveform, sr = torchaudio.load(output_path)
    audio_duration = waveform.shape[1] / sr

    print(f"\n=== Results ===")
    print(f"  Output: {output_path}")
    print(f"  Audio duration: {audio_duration:.2f}s")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  RTF: {gen_time / audio_duration:.3f} (< 1.0 = real-time)")
    print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("\n=== Test PASSED ===")


if __name__ == "__main__":
    main()
