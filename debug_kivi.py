"""Debug script: figure out why KIVI QuantizedCache shows no perplexity change.

Checks:
1. Is the QuantizedCache actually being used by the model?
2. Does _quantized_keys get populated (i.e., does quantization actually happen)?
3. What is the residual_length?
4. Are the dequantized values different from the original fp16 values?
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantizedCache

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")
print("Model loaded.\n")

# Create the cache the same way our sweep script does
cache = QuantizedCache(backend="quanto", nbits=4, config=model.config)

print("=== Cache object info ===")
print(f"  type: {type(cache).__name__}")
print(f"  residual_length: {cache._quantized_key_value_cache[0].residual_length if hasattr(cache, '_quantized_key_value_cache') else 'N/A'}")

# Check internal structure
if hasattr(cache, '_quantized_key_value_cache'):
    layer0 = cache._quantized_key_value_cache[0]
    print(f"  layer type: {type(layer0).__name__}")
    print(f"  layer residual_length: {layer0.residual_length}")
    print(f"  layer nbits: {layer0.nbits}")
    print(f"  dir: {[x for x in dir(layer0) if not x.startswith('__')]}")
else:
    print(f"  dir: {[x for x in dir(cache) if not x.startswith('__')]}")

# Run a forward pass with 512 tokens
print("\n=== Running forward pass with 512 tokens ===")
prompt = "The history of artificial intelligence " * 50
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
print(f"  input tokens: {inputs.input_ids.shape[1]}")

with torch.no_grad():
    outputs = model(**inputs, past_key_values=cache, use_cache=True)

print("\n=== Cache state after forward pass ===")
returned_cache = outputs.past_key_values
print(f"  returned cache type: {type(returned_cache).__name__}")
print(f"  same object as input? {returned_cache is cache}")

# Check if quantization actually happened
if hasattr(returned_cache, '_quantized_key_value_cache'):
    layer0 = returned_cache._quantized_key_value_cache[0]
    print(f"\n  Layer 0 after forward pass:")
    print(f"    type: {type(layer0).__name__}")

    # Check for quantized data
    if hasattr(layer0, '_quantized_keys'):
        qk = layer0._quantized_keys
        print(f"    _quantized_keys type: {type(qk)}")
        print(f"    _quantized_keys empty: {qk.numel() == 0 if hasattr(qk, 'numel') else 'N/A'}")
        if hasattr(qk, 'shape'):
            print(f"    _quantized_keys shape: {qk.shape}")
    else:
        print(f"    NO _quantized_keys attribute")
        print(f"    available attrs: {[x for x in dir(layer0) if 'quant' in x.lower() or 'key' in x.lower()]}")

    # Check residual keys
    if hasattr(layer0, 'keys'):
        print(f"    residual keys shape: {layer0.keys.shape if hasattr(layer0.keys, 'shape') else 'empty'}")
    if hasattr(layer0, '_keys'):
        print(f"    _keys shape: {layer0._keys.shape if hasattr(layer0._keys, 'shape') else 'empty'}")

else:
    print(f"  NO _quantized_key_value_cache attribute")
    print(f"  returned cache attrs: {[x for x in dir(returned_cache) if not x.startswith('__')]}")

# Now compare: run the SAME input WITHOUT the quantized cache
print("\n=== Comparison: fp16 vs KIVI output ===")
with torch.no_grad():
    out_fp16 = model(**inputs, use_cache=False)
    out_kivi = model(**inputs, past_key_values=QuantizedCache(backend="quanto", nbits=2, config=model.config), use_cache=True)

loss_fp16 = out_fp16.loss.item()
loss_kivi = out_kivi.loss.item()
print(f"  fp16 loss:  {loss_fp16:.6f}")
print(f"  KIVI 2-bit loss: {loss_kivi:.6f}")
print(f"  difference: {abs(loss_fp16 - loss_kivi):.6f}")
print(f"  identical: {loss_fp16 == loss_kivi}")

# Also test with labels to make sure loss is computed
print("\n=== With explicit labels ===")
labels = inputs.input_ids.clone()
with torch.no_grad():
    out_fp16_l = model(**inputs, labels=labels, use_cache=False)
    kivi_cache = QuantizedCache(backend="quanto", nbits=2, config=model.config)
    out_kivi_l = model(**inputs, labels=labels, past_key_values=kivi_cache, use_cache=True)

print(f"  fp16 loss:  {out_fp16_l.loss.item():.6f}")
print(f"  KIVI 2-bit loss: {out_kivi_l.loss.item():.6f}")
print(f"  difference: {abs(out_fp16_l.loss.item() - out_kivi_l.loss.item()):.6f}")
