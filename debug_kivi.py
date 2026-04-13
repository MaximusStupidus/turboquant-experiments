"""Debug v4: definitive test — does cache.update() return original or dequantized values?"""
import torch
from transformers import QuantizedCache, AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")
print("Model loaded.\n")

# First check: does dequantize produce different values?
print("=== Check 1: does quantize->dequantize change values? ===")
cache = QuantizedCache(backend="quanto", nbits=2, config=model.config)
prompt = "The history of artificial intelligence " * 50
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")

with torch.no_grad():
    out = model(**inputs, past_key_values=cache, use_cache=True)

layer0 = cache.layers[0]
qk = layer0._quantized_keys
deq = qk.dequantize()

print(f"  quantized type: {type(qk).__name__}")
print(f"  dequantized shape: {deq.shape}, dtype: {deq.dtype}")
print(f"  dequantized min: {deq.min():.4f}, max: {deq.max():.4f}")
print(f"  num unique values in dequantized: {deq.unique().numel()}")
print(f"  (if truly 2-bit quantized, should have very few unique values)")

# Second check: what does update() return?
print("\n=== Check 2: does update() return original or dequantized? ===")
fresh_cache = QuantizedCache(backend="quanto", nbits=2, config=model.config)
fake_key = torch.randn(1, 8, 300, 128, dtype=torch.float16, device="cuda")
fake_value = torch.randn(1, 8, 300, 128, dtype=torch.float16, device="cuda")

returned_k, returned_v = fresh_cache.update(fake_key, fake_value, layer_idx=0)

diff_mean = (fake_key - returned_k).abs().mean().item()
diff_max = (fake_key - returned_k).abs().max().item()
print(f"  input vs returned key mean diff: {diff_mean:.6f}")
print(f"  input vs returned key max diff:  {diff_max:.6f}")
print(f"  bit-identical: {torch.equal(fake_key, returned_k)}")
print()
if diff_mean == 0:
    print("  DIAGNOSIS: update() returns ORIGINAL values, not dequantized.")
    print("  This means KIVI quantization has ZERO effect on the forward pass.")
    print("  The cache stores quantized data but the model never reads it back —")
    print("  it uses the original values that update() returned.")
else:
    print(f"  DIAGNOSIS: update() returns DEQUANTIZED values (lossy).")
    print(f"  Quantization should affect the forward pass.")
