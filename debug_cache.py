"""Debug script: inspect DynamicCache attributes."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
)
tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")
out = model(**tok("hello", return_tensors="pt").to("cuda"), use_cache=True)
pkv = out.past_key_values

print("type:", type(pkv))
print()
print("dir:", [x for x in dir(pkv) if not x.startswith('_')])
print()
print("has key_cache:", hasattr(pkv, 'key_cache'))
print("has to_legacy_cache:", hasattr(pkv, 'to_legacy_cache'))
print()

# Try various access patterns
for method in ['key_cache', 'value_cache']:
    attr = getattr(pkv, method, 'MISSING')
    if attr != 'MISSING':
        print(f"pkv.{method} type: {type(attr)}, len: {len(attr) if hasattr(attr, '__len__') else 'N/A'}")
        if hasattr(attr, '__len__') and len(attr) > 0:
            print(f"  [0] shape: {attr[0].shape}, dtype: {attr[0].dtype}")

# Try getitem
try:
    item = pkv[0]
    print(f"\npkv[0] works! type: {type(item)}")
except Exception as e:
    print(f"\npkv[0] failed: {e}")

# Try iteration
try:
    first = next(iter(pkv))
    print(f"iter(pkv) works! first type: {type(first)}, len: {len(first) if hasattr(first, '__len__') else 'N/A'}")
except Exception as e:
    print(f"iter(pkv) failed: {e}")
