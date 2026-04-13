"""Debug script v2: dig into KIVI QuantizedCache.layers structure."""
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

# Create cache
cache = QuantizedCache(backend="quanto", nbits=2, config=model.config)

print("=== Cache structure ===")
print(f"  type: {type(cache).__name__}")
print(f"  has .layers: {hasattr(cache, 'layers')}")
if hasattr(cache, 'layers'):
    print(f"  len(layers): {len(cache.layers)}")
    if len(cache.layers) > 0:
        layer0 = cache.layers[0]
        print(f"  layer[0] type: {type(layer0).__name__}")
        print(f"  layer[0] attrs: {[x for x in dir(layer0) if not x.startswith('__')]}")
        if hasattr(layer0, 'residual_length'):
            print(f"  layer[0].residual_length: {layer0.residual_length}")
        if hasattr(layer0, 'nbits'):
            print(f"  layer[0].nbits: {layer0.nbits}")

# Forward pass with 512 tokens
print("\n=== Forward pass ===")
prompt = "The history of artificial intelligence " * 50
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
labels = inputs.input_ids.clone()
print(f"  tokens: {inputs.input_ids.shape[1]}")

with torch.no_grad():
    out = model(**inputs, labels=labels, past_key_values=cache, use_cache=True)

print(f"  loss with KIVI 2-bit: {out.loss.item():.6f}")

# Check cache state after forward pass
print("\n=== Cache after forward pass ===")
if hasattr(cache, 'layers') and len(cache.layers) > 0:
    layer0 = cache.layers[0]
    print(f"  layer[0] type: {type(layer0).__name__}")
    all_attrs = [x for x in dir(layer0) if not x.startswith('_')]
    print(f"  public attrs: {all_attrs}")
    # Check every attribute that might hold quantized data
    for attr in dir(layer0):
        if 'quant' in attr.lower() or 'key' in attr.lower() or 'value' in attr.lower():
            val = getattr(layer0, attr)
            if hasattr(val, 'shape'):
                print(f"  .{attr}: shape={val.shape} dtype={val.dtype}")
            elif hasattr(val, 'numel'):
                print(f"  .{attr}: numel={val.numel()}")
            elif not callable(val):
                print(f"  .{attr}: {val}")

# Compare fp16 vs KIVI
print("\n=== fp16 vs KIVI 2-bit comparison ===")
with torch.no_grad():
    out_fp16 = model(**inputs, labels=labels, use_cache=True)
    fresh_cache = QuantizedCache(backend="quanto", nbits=2, config=model.config)
    out_kivi = model(**inputs, labels=labels, past_key_values=fresh_cache, use_cache=True)

print(f"  fp16 loss:      {out_fp16.loss.item():.8f}")
print(f"  KIVI 2-bit loss: {out_kivi.loss.item():.8f}")
print(f"  difference:     {abs(out_fp16.loss.item() - out_kivi.loss.item()):.8f}")
print(f"  bit-identical:  {out_fp16.loss.item() == out_kivi.loss.item()}")
