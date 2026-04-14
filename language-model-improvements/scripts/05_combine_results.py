"""Combine all individual result JSONs into one summary for final plotting.

The autoregressive eval results are in these files:
- baseline_autoreg.json (fp16, autoregressive eval)
- kivi_4bit.json, kivi_2bit.json
- turboquant_4bit.json, turboquant_3bit.json, turboquant_2bit.json
- handrolled_4bit.json, handrolled_2bit.json

This script reads all of them and writes a combined summary.json.
"""
import json
import os

RESULTS_DIR = "language-model-improvements/results"

# The configs we want in the final comparison, in display order
CONFIGS = [
    ("fp16 (baseline)",  "baseline_autoreg.json"),
    ("KIVI 4-bit",       "kivi_4bit.json"),
    ("KIVI 2-bit",       "kivi_2bit.json"),
    ("TurboQuant 4-bit", "turboquant_4bit.json"),
    ("TurboQuant 3-bit", "turboquant_3bit.json"),
    ("TurboQuant 2-bit", "turboquant_2bit.json"),
    ("Handrolled 4-bit", "handrolled_4bit.json"),
    ("Handrolled 2-bit", "handrolled_2bit.json"),
]

all_results = []
for display_name, filename in CONFIGS:
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        # Ensure the display name is consistent
        data["config_name"] = display_name
        all_results.append(data)
        ppl = data["perplexity"]["value"]
        tps = data["throughput"]["tokens_per_sec"]
        mem = data["memory"]["peak_gpu_human"]
        print(f"  {display_name:<25} PPL={ppl:>7}  tok/s={tps:>6}  mem={mem}")
    else:
        print(f"  {display_name:<25} MISSING: {filename}")

summary = {"all_results": all_results}
out_path = os.path.join(RESULTS_DIR, "summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved combined summary: {out_path} ({len(all_results)} configs)")
