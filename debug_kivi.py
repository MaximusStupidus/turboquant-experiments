"""Debug script v3: check quanto quantization internals."""
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

# Create and run
cache = QuantizedCache(backend="quanto", nbits=2, config=model.config)
prompt = "The history of artificial intelligence " * 50
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")

with torch.no_grad():
    out = model(**inputs, labels=inputs.input_ids.clone(), past_key_values=cache, use_cache=True)

layer0 = cache.layers[0]

print("=== Quantization details ===")
print(f"  nbits: {layer0.nbits}")
print(f"  qtype: {layer0.qtype}")
print(f"  q_group_size: {layer0.q_group_size}")
print(f"  residual_length: {layer0.residual_length}")
print()

# Check the actual _quantized_keys object
qk = layer0._quantized_keys
print(f"=== _quantized_keys ===")
print(f"  type: {type(qk).__name__}")
print(f"  dtype: {qk.dtype}")
print(f"  shape: {qk.shape}")

# Is it a quanto QTensor or a regular tensor?
import optimum.quanto as quanto
print(f"  is QTensor: {isinstance(qk, quanto.QTensor)}")
print(f"  quanto version: {quanto.__version__}")

# Try manual quantization to see if quanto works at all
print("\n=== Manual quanto test ===")
test_tensor = torch.randn(1, 8, 64, 128, dtype=torch.float16, device="cuda")
try:
    from optimum.quanto import quantize_activation, qint2, qint4
    q_test = quantize_activation(test_tensor, qtype=qint2)
    deq_test = q_test.dequantize()
    diff = (test_tensor - deq_test).abs().mean().item()
    print(f"  manual qint2 quantize->dequantize diff: {diff:.6f}")
    print(f"  (should be > 0 if quantization is real)")
except Exception as e:
    print(f"  manual quantization failed: {e}")

# Check what _quantize actually does
print("\n=== layer._quantize source ===")
import inspect
print(inspect.getsource(layer0._quantize))

print("\n=== layer._dequantize source ===")
print(inspect.getsource(layer0._dequantize))
