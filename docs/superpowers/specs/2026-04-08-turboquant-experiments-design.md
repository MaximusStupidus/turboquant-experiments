# TurboQuant Experiments — Project Design

**Status:** Draft, awaiting user review.
**Date:** 2026-04-08
**Owner:** Ojas (with supervisor pairing)

---

## 1. Purpose

Run a pair of comparative benchmarks that apply **TurboQuant** (a KV-cache quantization method based on random projection followed by bit-quantization) to two different autoregressive transformer models:

- **Part 1:** `meta-llama/Llama-3.1-8B-Instruct` (text generation).
- **Part 2:** `microsoft/VibeVoice-1.5B` (autoregressive TTS — a transformer language model whose vocabulary is audio tokens).

Both parts are open-sourced as a single public repository so that anyone can vet, reproduce, and extend the results.

## 2. Priority order (the most important section in this doc)

The owner has explicitly ranked the success criteria, in this order:

1. **Learning-first** — by the end of part 1, the owner must be able to explain TurboQuant, KV caches, and the experimental results in their own words to a smart friend. The artifact is the *explanation*; code is there to back it up.
2. **Benchmark-first** — multiple quantization configurations measured rigorously on standard evals (perplexity on WikiText, latency, memory, KV cache memory specifically), with clean plots.
3. **Reproducibility-first** — the repo must be runnable end-to-end on a fresh RunPod box with one command, producing the same numbers.

These are pursued in this order, not in parallel. When any of the three are in tension, **learning wins, then benchmark, then reproducibility.** Specifically: shipping a flashier benchmark at the cost of the owner's understanding is *not* acceptable.

### The discipline that protects priority 1: writing-checkpoint gates

Every phase ends with a markdown writeup in the owner's own words, committed to the repo, before the next phase begins. If the writeup can't be written, the phase isn't done. This rule is non-negotiable for the duration of the project. The supervisor will enforce it even when it feels inconvenient — that is the supervisor's primary job.

### Compute philosophy: don't optimize for frugality, optimize for learning

Compute cost is **explicitly not a primary constraint** on this project. If renting a GPU accelerates learning — by letting us inspect the *real* Llama-3.1-8B instead of a smaller stand-in, by letting us iterate faster, by letting us actually *see* the numbers we're trying to understand — then we rent the GPU. Spending \$5, \$50, or \$200 on GPU time to remove a friction that would otherwise compromise understanding is a good trade. The owner has explicitly said: *"Learning can have good cost."*

What we still keep, because it's hygiene rather than frugality:
- **Don't leave pods idle.** Shut them down when not actively in use. A pod that's running but unattended is wasted money for zero learning.
- **Know what you're about to run before you start the meter.** Scripts are tested on the smallest viable setup before they're launched on a big GPU. Not because the big GPU is expensive, but because debugging on the big GPU is *slower* (longer iteration loops, more state to manage) and the friction itself slows learning.
- **Pick the right GPU for the task, not the biggest available.** If the workload fits on an A100 40GB, an H100 is wasted overhead.

The owner's MacBook is *not expected* to run the real models. It will be used for editing code, reading, writing notes, and small CPU-only sanity checks. Anything involving Llama-3.1-8B inference happens on a rented GPU.

## 3. Background context the owner brought in

Recorded so future contributors know the starting point this plan was designed around:

- **TurboQuant familiarity:** "Heard the name, knows it's something that makes models smaller/faster, hasn't read the paper yet." → phase 0 must teach the technique from first principles, not assume prior reading.
- **Transformer internals familiarity:** Knows transformers use attention and there's a thing called KV cache, but couldn't derive *why* the cache exists from first principles. → phase 0 derives this explicitly.
- **PyTorch / HF / remote-GPU familiarity:** Has used HuggingFace at the pipeline/API level, hasn't really written PyTorch or touched model internals, has not used a rented GPU before. → first PyTorch code is small (~50 lines) and runs locally before any GPU is rented.
- **GPU access:** RunPod account with billing already set up.

## 4. Repository structure

One repo, both parts. The intentional shape is **flat and minimal** right now — only the things we already know we need. The internal structure inside each part-subfolder is deliberately *not* fully specified, because we don't yet know what shape it will want until we've worked inside a RunPod pod for a few hours. We will add directories when the work needs them, not before.

```
turboquant-experiments/
├── README.md                            # Top-level explainer, written last (phase 4)
├── pyproject.toml                       # uv-managed, see §5
│
├── notes/                               # Owner's own-words writeups (gated checkpoints)
│   ├── 00-foundations.md                # Phase 0 deliverable (already drafted)
│   ├── 01-kv-cache-reality.md           # Phase 1 deliverable
│   ├── 02-what-im-measuring.md          # Phase 2 deliverable
│   ├── 03-results.md                    # Phase 3 deliverable
│   └── scratch/                         # Pedagogical scripts/plots, not project code
│       ├── jl_demo.py
│       ├── jl_demo.png
│       ├── outlier_demo.py
│       └── outlier_demo.png
│
├── language-model-improvements/         # Part 1: TurboQuant on Llama-3.1-8B
│   ├── scripts/                         # Will be populated phase-by-phase
│   ├── results/                         # CSV/JSON outputs synced back from pods
│   └── README.md                        # Part 1 reproduction instructions
│
├── speech-tts-improvements/             # Part 2: TurboQuant on VibeVoice
│   └── (empty until part 1 ships)
│
└── docs/
    └── superpowers/specs/
        └── 2026-04-08-turboquant-experiments-design.md   # this file
```

**Naming:** Both part-folders are kebab-case and end in `-improvements`. Parallel structure on purpose, so they read as a pair.

**Why one repo:** the conceptual work in part 1 (KV cache, TurboQuant) is the foundation for part 2, and a single artifact tells the whole story. `notes/` is shared across both parts.

**Why the internal structure is barely specified:** I had originally listed three pre-named scripts inside `language-model-improvements/scripts/`. I removed that. Listing files we haven't written yet creates fake structure that the real work will then have to fight against. We will add files when phase 1 actually demands them. If phase 1 turns out to want notebooks instead of scripts, or a `kernels/` subfolder, or anything else — we add it then, not now.

## 4a. RunPod workflow — where the code lives, where the work happens

This is the section that actually needs thinking-through, since the directory tree above is a *local-machine* view but most of the real work runs inside a rented pod. Here's how the two layers fit together.

**Source of truth: GitHub.** The repo lives on GitHub (created at the start of phase 1, kept private until phase 4 if you prefer). The owner's MacBook has a clone for editing, reading, writing notes, and committing. RunPod pods get the code by cloning the same repo. Nothing important lives only on the pod.

**Pod lifecycle (the loop we'll actually run during phases 1–3):**

1. Spin up a fresh pod with the right GPU and a PyTorch image.
2. `git clone <the-repo>` inside the pod.
3. `uv sync` to install the locked Python environment.
4. Run the phase's script. Output files (CSV, JSON, plots) are written into `language-model-improvements/results/` *inside the pod*.
5. Either `git add` + `git commit` + `git push` from inside the pod (cleanest, treats the pod as a normal dev box), **or** scp the results files back to the laptop and commit from there. We'll pick whichever feels less clunky after the first phase 1 session.
6. **Shut down the pod immediately when done.** No "I'll come back to it later" — that's how idle-pod bills happen.

**What the owner edits where:**
- **Notes / writeups (`notes/*.md`):** edit on the laptop. They're reading and reflection, not GPU work.
- **Scripts (`language-model-improvements/scripts/*.py`):** can be edited *either* on the laptop (then `git pull` on the pod) or directly on the pod via VS Code Remote / Cursor Remote / SSH. We'll see which feels better in phase 1. The "edit on laptop, run on pod" loop has higher latency but cleaner state; the "edit directly on pod" loop is faster but the pod is ephemeral and you have to remember to commit before shutdown.
- **Results (`language-model-improvements/results/`):** generated on the pod, synced back to git so the laptop sees them.

**One concrete decision for now, others deferred until phase 1 reveals them:**
- **Decision:** the repo is the source of truth for *all* code and *all* results. Pods are ephemeral compute environments, not storage. If a pod dies and a result wasn't committed, the result is lost — and that's an acceptable risk because we can re-run.
- **Deferred:** which exact RunPod GPU to use, which exact base image, whether to use VS Code Remote vs. raw SSH vs. JupyterLab inside the pod, whether large weight files should be cached on a RunPod network volume vs. re-downloaded each session. All of these are "we'll find out the right answer by trying it" questions that don't deserve to be settled in this design doc.

What the owner needs to know going into phase 1: **the laptop is for thinking, the pod is for running, the GitHub repo is the bridge.** That's it.

## 5. Tooling decisions

- **Python environment:** `uv`. Reason: fast, reproducible, identical behavior on the owner's laptop and on the rented RunPod box, one less environment to debug at 11pm. No conda, no plain pip.
- **Source control:** git. Public GitHub repo (created at the start of phase 1, not now, to avoid premature publication of incomplete work).
- **Plotting:** matplotlib. Reason: lowest dependency footprint, already installed, plots reproduce identically across machines.
- **No experiment-tracking SaaS** (W&B, MLflow, etc.) for v1. CSV/JSON in `results/` is enough and keeps the repo self-contained.

## 6. Phase plan

Phases are sequential. Each phase has a **goal**, an **output** (code + a writing checkpoint), and a **gate** that must be passed before the next phase begins.

### Phase 0 — Foundations (no GPU, no code-as-project)

**Goal:** Owner can explain on a whiteboard (a) the attention equation, (b) why autoregressive decoding caches K and V, (c) what quantization is and why models tolerate it, and (d) what TurboQuant does and why it's a particularly good fit for KV caches.

**Output:** `notes/00-foundations.md` — already drafted in this conversation, includes inline `[Note : ...]` sharpenings from supervisor review.

**Status:** ✅ Passing.

### Phase 1 — First contact with the real model (RunPod from the start)

**Goal:** Owner writes ~50 lines of PyTorch that load **the actual Llama-3.1-8B-Instruct** on a rented GPU, run a forward pass, and **print the KV cache tensor shapes and memory footprint with their own eyes**, on the real model — not a smaller stand-in.

We use a real GPU from day one of phase 1 because (a) the laptop almost certainly cannot load Llama-3.1-8B comfortably, (b) inspecting a stand-in model would teach the wrong shapes and the wrong intuitions, and (c) the friction of "make it work locally first" buys us nothing — the whole point is to see the *actual numbers* on the *actual model* we'll use throughout the project.

**Outputs:**
- `language-model-improvements/scripts/01_inspect_kv_cache.py` — loads `meta-llama/Llama-3.1-8B-Instruct`, runs one forward pass with `use_cache=True`, prints the structure of `past_key_values`, computes the per-token memory and total memory, compares to the model weight memory, prints typical value distributions for K and V (mean, std, min, max, p99.9 — to make the outlier-channel story concrete on real data).
- `notes/01-kv-cache-reality.md` — owner's writeup of what the script printed and what (if anything) surprised them. The "surprise" prompt is intentional: it forces a comparison of mental model to reality.

**Why this phase exists:** This is the "aha" moment. Seeing on the terminal that a 2k-context KV cache rivals the model weight memory — *on the real model* — makes the entire TurboQuant motivation visceral in a way no paper can.

**RunPod hygiene for phase 1:** A small GPU (A100 40GB or even a 24GB-class card) is plenty for inspecting Llama-3.1-8B at modest context lengths. Pod is launched when we're ready to run, shut down when we're done editing notes. Total wall time on the meter for phase 1 is expected to be small — maybe an hour or two of active GPU time across the phase — but we are *not* optimizing this number. If it takes longer, it takes longer.

**Gate:** `notes/01-kv-cache-reality.md` written and reviewed.

### Phase 2 — Eval harness (rented GPU)

**Goal:** Build the **baseline measurement infrastructure before introducing any quantization**. A script that takes a model, runs it on a fixed slice of WikiText-2, and outputs:

1. Perplexity
2. Peak GPU memory
3. Tokens per second (generation throughput)
4. **KV cache memory specifically** (separated from model-weight memory)

The script must be deterministic (fixed seed, fixed dataset slice) and re-runnable.

**Outputs:**
- `language-model-improvements/scripts/02_baseline_eval.py`
- `language-model-improvements/results/baseline.json` — fp16 Llama-3.1-8B numbers
- `notes/02-what-im-measuring.md` — owner explains in their own words *why each metric matters* and *what it would mean for it to go up or down*. This is a writing checkpoint specifically because misunderstanding what a metric measures is the most common failure mode in benchmarking.

**Discipline rule (load-bearing):** **The eval harness must be frozen before phase 3 begins.** No tweaks, no metric additions, no dataset changes once we've seen the first set of TurboQuant numbers — otherwise our before/after numbers are not comparable. If we discover the harness is wrong mid-experiment, we re-run *all* baselines with the corrected harness, not just the new variant.

**Gate:** `notes/02-what-im-measuring.md` written and reviewed, baseline JSON checked in.

### Phase 3 — Apply TurboQuant (rented GPU)

**Goal:** Run the same harness with TurboQuant applied to the KV cache, at multiple quantization configurations. Compare against fp16 baseline and at least one naive baseline (e.g. naive int8 KV cache) so TurboQuant has something to look better than (or worse than — we don't pre-decide which).

**Outputs:**
- `language-model-improvements/scripts/03_turboquant_eval.py`
- `language-model-improvements/results/turboquant_sweep.json`
- Plots of perplexity vs. KV memory tradeoff for each config
- `notes/03-results.md` — owner's interpretation of the numbers in their own words. **Numbers without interpretation don't count.** If a result is surprising, the writeup must engage with the surprise rather than wave it away.

**RunPod hygiene (specific to this phase):**
- Before launching a long sweep, dry-run the script on a single config to make sure it produces the expected output format. This is about iteration speed, not cost.
- Walk in with a written list of runs to perform — not because we're rationing GPU time, but because "what was I about to test?" is a real failure mode after a context-switch.
- Pod is shut down when the sweep finishes and the results are downloaded. No leaving pods running idle overnight.
- **No dollar cap.** If a config is interesting and worth re-running with different parameters, we re-run it. If the sweep takes 8 hours of H100 time instead of 2, that's fine.

**Mindset rule for this phase (recorded because of an exchange in phase 0):** Trust no claim — supervisor's, paper's, or library's — about what numbers to expect, until it's been measured on this exact setup. If the numbers look weird, the response is "huh, why?" not "we must be running it wrong, the paper says it works." Sometimes the paper is wrong on a particular setup. Pre-committing to this mindset before we see results.

**Gate:** `notes/03-results.md` written and reviewed.

### Phase 4 — Write-up & repo polish

**Goal:** Public-facing README that someone unfamiliar with TurboQuant can read and understand. Reproducible from a fresh clone.

**Outputs:**
- `README.md` (top-level)
- `language-model-improvements/README.md` (reproduction instructions)
- Polished plots
- Optional short blog-post-style explainer

**Gate:** Repo public; the owner can verbally walk someone through the project end-to-end without notes.

### Hard rule between part 1 and part 2

**We do not start part 2 until part 1 is fully shipped — repo public, README written, owner can explain the results.** Resist the temptation to start part 2 "in parallel because it's exciting." Part 2 will go faster *because* part 1 is finished, not despite it.

## 7. Decisions deliberately deferred

These are decisions we are choosing not to make right now because they require information we don't have yet. They are recorded so future-us doesn't accidentally settle them by drift.

- **Which TurboQuant implementation/library to use** — to be decided at the start of phase 3 after surveying the landscape (HF integration? a research repo? hand-roll a minimal version for the learning?).
- **Which exact quantization configurations to sweep** — depends on what the chosen library exposes.
- **Choice of GPU on RunPod** (A100 40GB vs H100 vs 4090) — depends on what fits Llama-3.1-8B + a few-thousand-token KV cache comfortably; will be decided at the start of phase 3 with a one-line memory calculation.
- **Whether to also include MMLU or another downstream eval** in addition to perplexity — to be decided at the start of phase 2; default is "perplexity only for v1, add MMLU only if time and budget allow."
- **Part 2 details (VibeVoice setup, audio quality metrics, voice-similarity measurement)** — fully deferred until part 1 ships.

## 8. Out of scope for v1

To prevent scope creep, these are explicitly *not* part of this project:

- Training or fine-tuning any model. We only measure inference.
- Comparing TurboQuant against every other KV-cache quantization method. We compare against fp16 baseline and one naive baseline. That's enough for the first version.
- Multi-GPU experiments. Single-GPU only.
- Custom CUDA kernels. We use whatever the chosen library provides.
- Throughput optimization beyond what falls out of measurement. We measure tokens/sec but we do not chase it.

## 9. Risk register

Things most likely to go wrong, and the pre-committed response.

| Risk | Likelihood | Pre-committed response |
|---|---|---|
| TurboQuant library doesn't exist in usable form for Llama-3.1-8B | Medium | Hand-roll a minimal "project + bit-quantize" wrapper around HF's KV cache. The math is simple; the engineering is the hard part. Treat the hand-roll as a *learning win* if it happens. |
| Numbers don't show TurboQuant winning | Medium | Report honestly. A negative result is still a publishable result and is more useful to the community than a fake positive. The priority is *understanding*, not *advocacy*. |
| Owner's laptop can't load Llama-3.1-8B | Certain (assumed up front) | Not a risk, just a fact. All real model work happens on RunPod from phase 1 onwards. The laptop is for editing, reading, writing notes, and CPU-only sanity checks. |
| RunPod costs higher than expected | Possible | Not treated as a problem in itself. We accept the cost when it buys learning. The only thing we *do* avoid is leaving pods idle, because idle pods buy zero learning. |
| Owner's understanding outpaces the writing checkpoints | Medium | Skipping a checkpoint is allowed *only* via an explicit "I am consciously overriding this gate" statement, recorded in this design doc as an amendment. No drift-based skipping. |

## 10. Open items for owner review

Before this design is locked, please react to the following — push back on anything that feels wrong:

1. Repo structure (§4) — does the `language-model-improvements/` + `speech-tts-improvements/` + shared `notes/` layout make sense? And does the §4a RunPod workflow section answer your "I'm not sure how the internals will look" concern, or does something specific still feel undefined?
2. Tooling (§5) — `uv` for Python, matplotlib for plots, no experiment tracker. OK?
3. Phase gates (§6) — every phase has a markdown writeup gate. You already approved this in conversation but I'm re-confirming it's recorded here.
4. Compute philosophy (§2, "Compute philosophy") — confirm I captured "don't optimize for frugality, optimize for learning" the way you meant it.
5. Out-of-scope list (§8) — anything you'd add or remove?
6. Risk register (§9) — anything obvious I missed?

Once you've reviewed and approved (or amended) this doc, the next step is to invoke the writing-plans skill and produce a detailed implementation plan for **phase 1 only** (we'll write subsequent phase plans as we get to them — don't pre-plan phases we haven't started).
