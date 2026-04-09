# Phase 1 — KV Cache Inspection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the project repo, then load real Llama-3.1-8B-Instruct on a rented RunPod GPU, inspect its KV cache structure on real data, measure per-token and per-context KV memory, and characterize the K/V value distributions (including outlier channels). Owner writes a "what surprised me" note. This phase produces no quantization — only first-hand evidence of why the KV cache matters.

**Architecture:** A single inspection script (`01_inspect_kv_cache.py`) that takes the model name as a CLI arg, loads it, runs one forward pass, and prints/saves a structured report covering: model config, weight memory, KV cache tensor shapes, per-token KV bytes, KV memory at a sweep of context lengths (1k → 128k), and per-channel value statistics for K and V at one chosen layer. Two pure helper functions (`kv_cache_bytes` and `find_outlier_channels`) are TDD'd because they have well-defined inputs/outputs; the rest of the script is exploratory and developed by run-and-observe.

**Tech Stack:** Python 3.11+, `uv` for environment management, `torch`, `transformers`, `pytest` for the two unit tests, `huggingface_hub` for gated-model auth, RunPod for the GPU (single A100 40GB or A100 80GB; H100 is overkill for inspection).

---

## File structure

The current working directory `/Users/ojasjain/Desktop/Experiments/TurboQuant Experimentation/` sits inside an unrelated parent git repo (`dental-demo` branch) and is messy. **We start fresh** in a sibling directory `~/Desktop/Experiments/turboquant-experiments/` and move the existing artifacts (foundations doc, design doc, pedagogical scripts) into it. This avoids nested-repo problems and gives the project a clean history from commit 1.

After Phase 1, the repo will look like:

```
turboquant-experiments/
├── .gitignore
├── README.md                              # one-paragraph placeholder; real version in phase 4
├── pyproject.toml                         # uv-managed
├── uv.lock                                # generated
├── notes/
│   ├── 00-foundations.md                  # moved from old location
│   ├── 01-kv-cache-reality.md             # NEW — owner's writeup
│   └── scratch/
│       ├── jl_demo.py                     # moved
│       ├── jl_demo.png                    # moved
│       ├── outlier_demo.py                # moved
│       └── outlier_demo.png               # moved
├── language-model-improvements/
│   ├── README.md                          # placeholder
│   ├── kv_utils.py                        # NEW — pure helpers (TDD'd)
│   ├── scripts/
│   │   └── 01_inspect_kv_cache.py         # NEW — main inspection script
│   ├── results/
│   │   └── 01_kv_cache_inspection.txt     # NEW — captured output
│   └── tests/
│       └── test_kv_utils.py               # NEW — unit tests for helpers
├── speech-tts-improvements/               # empty for now
│   └── .gitkeep
└── docs/
    └── superpowers/
        ├── specs/
        │   └── 2026-04-08-turboquant-experiments-design.md   # moved
        └── plans/
            └── 2026-04-08-phase1-kv-cache-inspection.md      # this file (moved)
```

**Where TDD applies and where it doesn't.** Two helper functions (`kv_cache_bytes`, `find_outlier_channels`) are pure: same inputs always give same outputs, no model needed, and the math is checkable by hand. We TDD those (Tasks 8–9). The main inspection script (`01_inspect_kv_cache.py`) is exploratory: it loads a real LLM, runs a forward pass, and prints whatever structure that LLM happens to expose. Writing tests for it would mean either mocking out HuggingFace (which defeats the point — we want to see the *real* shape) or running the real model in CI (slow, GPU-required, fragile). For the script itself, we use a "build-incrementally and run-after-each-step" loop instead of TDD. Tasks 10–17 reflect this. This is a deliberate choice and the trade-off is recorded in §10 of the design doc.

---

## Section A — Repository setup (laptop, no GPU yet)

### Task 1: Create fresh project directory

**Files:**
- Create directory: `~/Desktop/Experiments/turboquant-experiments/`

- [ ] **Step 1: Verify the parent directory exists and the target name is free**

Run: `ls ~/Desktop/Experiments/ | grep -i turboquant`
Expected: only `TurboQuant Experimentation` (the messy old folder), nothing named `turboquant-experiments`.

- [ ] **Step 2: Create the new directory**

Run: `mkdir -p ~/Desktop/Experiments/turboquant-experiments`
Expected: no output, exit code 0.

- [ ] **Step 3: Initialize git inside it**

Run: `cd ~/Desktop/Experiments/turboquant-experiments && git init`
Expected: `Initialized empty Git repository in ...`

---

### Task 2: Migrate existing artifacts from the old folder

**Files moved (not copied — we want the old folder gone after this task):**
- From `~/Desktop/Experiments/TurboQuant Experimentation/notes/00-foundations.md` → `~/Desktop/Experiments/turboquant-experiments/notes/00-foundations.md`
- From `~/Desktop/Experiments/TurboQuant Experimentation/jl_demo.py` → `~/Desktop/Experiments/turboquant-experiments/notes/scratch/jl_demo.py`
- From `~/Desktop/Experiments/TurboQuant Experimentation/jl_demo.png` → `~/Desktop/Experiments/turboquant-experiments/notes/scratch/jl_demo.png`
- From `~/Desktop/Experiments/TurboQuant Experimentation/outlier_demo.py` → `~/Desktop/Experiments/turboquant-experiments/notes/scratch/outlier_demo.py`
- From `~/Desktop/Experiments/TurboQuant Experimentation/outlier_demo.png` → `~/Desktop/Experiments/turboquant-experiments/notes/scratch/outlier_demo.png`
- From `~/Desktop/Experiments/TurboQuant Experimentation/docs/superpowers/specs/2026-04-08-turboquant-experiments-design.md` → `~/Desktop/Experiments/turboquant-experiments/docs/superpowers/specs/2026-04-08-turboquant-experiments-design.md`
- From `~/Desktop/Experiments/TurboQuant Experimentation/docs/superpowers/specs/_style.css` → `~/Desktop/Experiments/turboquant-experiments/docs/superpowers/specs/_style.css`
- From `~/Desktop/Experiments/TurboQuant Experimentation/docs/superpowers/specs/2026-04-08-turboquant-experiments-design.html` → `~/Desktop/Experiments/turboquant-experiments/docs/superpowers/specs/2026-04-08-turboquant-experiments-design.html`
- From `~/Desktop/Experiments/TurboQuant Experimentation/docs/superpowers/plans/2026-04-08-phase1-kv-cache-inspection.md` → `~/Desktop/Experiments/turboquant-experiments/docs/superpowers/plans/2026-04-08-phase1-kv-cache-inspection.md`

- [ ] **Step 1: Create the destination directories**

Run:
```bash
cd ~/Desktop/Experiments/turboquant-experiments
mkdir -p notes/scratch
mkdir -p docs/superpowers/specs
mkdir -p docs/superpowers/plans
mkdir -p language-model-improvements/scripts
mkdir -p language-model-improvements/results
mkdir -p language-model-improvements/tests
mkdir -p speech-tts-improvements
```
Expected: no output.

- [ ] **Step 2: Move foundations doc**

Run:
```bash
mv "/Users/ojasjain/Desktop/Experiments/TurboQuant Experimentation/notes/00-foundations.md" \
   ~/Desktop/Experiments/turboquant-experiments/notes/00-foundations.md
```
Expected: no output.

- [ ] **Step 3: Move pedagogical scripts and plots**

Run:
```bash
SRC="/Users/ojasjain/Desktop/Experiments/TurboQuant Experimentation"
DST=~/Desktop/Experiments/turboquant-experiments/notes/scratch
mv "$SRC/jl_demo.py"      "$DST/jl_demo.py"
mv "$SRC/jl_demo.png"     "$DST/jl_demo.png"
mv "$SRC/outlier_demo.py" "$DST/outlier_demo.py"
mv "$SRC/outlier_demo.png" "$DST/outlier_demo.png"
```
Expected: no output.

- [ ] **Step 4: Move design doc, style, rendered HTML, and this plan**

Run:
```bash
SRC="/Users/ojasjain/Desktop/Experiments/TurboQuant Experimentation/docs/superpowers"
DST=~/Desktop/Experiments/turboquant-experiments/docs/superpowers
mv "$SRC/specs/2026-04-08-turboquant-experiments-design.md"   "$DST/specs/"
mv "$SRC/specs/_style.css"                                     "$DST/specs/"
mv "$SRC/specs/2026-04-08-turboquant-experiments-design.html" "$DST/specs/"
mv "$SRC/plans/2026-04-08-phase1-kv-cache-inspection.md"      "$DST/plans/"
```
Expected: no output.

- [ ] **Step 5: Add a placeholder for empty `speech-tts-improvements/`**

Run: `touch ~/Desktop/Experiments/turboquant-experiments/speech-tts-improvements/.gitkeep`
Expected: no output.

- [ ] **Step 6: Sanity check file layout**

Run: `cd ~/Desktop/Experiments/turboquant-experiments && find . -type f | sort`
Expected: see the foundations doc, the four scratch files, the three doc files, and the .gitkeep — nothing else yet.

- [ ] **Step 7: Confirm the old folder still exists but is now mostly empty**

Run: `find "/Users/ojasjain/Desktop/Experiments/TurboQuant Experimentation" -type f`
Expected: empty or only system files. If anything important remains, stop and ask the owner before deleting the old folder.

---

### Task 3: Write a minimal `.gitignore`

**Files:**
- Create: `~/Desktop/Experiments/turboquant-experiments/.gitignore`

- [ ] **Step 1: Write the .gitignore**

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.venv/
venv/

# uv
.python-version

# Editor
.vscode/
.idea/
*.swp

# OS
.DS_Store

# Model weights — never commit, always re-download or use HF cache
*.bin
*.safetensors
*.gguf

# Local secrets
.env
.env.*
hf_token.txt

# Pytest
.pytest_cache/
```

- [ ] **Step 2: Verify it was written**

Run: `cat ~/Desktop/Experiments/turboquant-experiments/.gitignore | wc -l`
Expected: ~25 lines.

---

### Task 4: Create `pyproject.toml` with `uv`

**Files:**
- Create: `~/Desktop/Experiments/turboquant-experiments/pyproject.toml`

- [ ] **Step 1: Verify uv is installed**

Run: `uv --version`
Expected: a version string like `uv 0.4.x` or newer. If not installed, run `curl -LsSf https://astral.sh/uv/install.sh | sh` first.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "turboquant-experiments"
version = "0.1.0"
description = "Comparative benchmarks of TurboQuant on Llama-3.1-8B and VibeVoice"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4",
    "transformers>=4.45",
    "accelerate>=0.34",
    "numpy>=2.0",
    "matplotlib>=3.9",
    "huggingface-hub>=0.25",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.uv]
# Pin nothing extra here; uv.lock is the source of truth.
```

- [ ] **Step 3: Resolve and lock dependencies**

Run: `cd ~/Desktop/Experiments/turboquant-experiments && uv sync --extra dev`
Expected: uv creates `.venv/`, downloads packages, writes `uv.lock`. Output ends with something like `Resolved N packages in Xs`. **First-time sync may take a few minutes** because torch is large.

- [ ] **Step 4: Sanity-check the env can import torch and transformers**

Run:
```bash
cd ~/Desktop/Experiments/turboquant-experiments
uv run python -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"
```
Expected: prints two version strings. No errors.

---

### Task 5: Write a placeholder top-level `README.md`

**Files:**
- Create: `~/Desktop/Experiments/turboquant-experiments/README.md`

- [ ] **Step 1: Write the placeholder**

```markdown
# turboquant-experiments

Open comparative benchmarks of [TurboQuant](https://arxiv.org/abs/2509.04349) applied to two autoregressive transformer models:

1. **Part 1 — `language-model-improvements/`** — TurboQuant on Llama-3.1-8B-Instruct (text generation).
2. **Part 2 — `speech-tts-improvements/`** — TurboQuant on Microsoft VibeVoice-1.5B (autoregressive TTS).

The full design is in [`docs/superpowers/specs/2026-04-08-turboquant-experiments-design.md`](docs/superpowers/specs/2026-04-08-turboquant-experiments-design.md). Conceptual notes are in [`notes/`](notes/).

This README will be replaced with a real explainer in phase 4 once the experiments have actually run.
```

- [ ] **Step 2: Verify it renders cleanly in plain text**

Run: `cat ~/Desktop/Experiments/turboquant-experiments/README.md`
Expected: see the markdown above.

---

### Task 6: Write placeholder `language-model-improvements/README.md`

**Files:**
- Create: `~/Desktop/Experiments/turboquant-experiments/language-model-improvements/README.md`

- [ ] **Step 1: Write the placeholder**

```markdown
# language-model-improvements

Part 1 of the turboquant-experiments project: applying TurboQuant to the KV cache of Llama-3.1-8B-Instruct.

## Phase 1 — KV cache inspection

Run on a GPU-equipped machine (RunPod):

```bash
uv run python language-model-improvements/scripts/01_inspect_kv_cache.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output language-model-improvements/results/01_kv_cache_inspection.txt
```

Reproduction instructions for later phases will be added as those phases land.
```

---

### Task 7: First commit

- [ ] **Step 1: Stage everything**

Run:
```bash
cd ~/Desktop/Experiments/turboquant-experiments
git add .gitignore pyproject.toml uv.lock README.md \
        notes/ \
        language-model-improvements/README.md \
        speech-tts-improvements/.gitkeep \
        docs/
```
Expected: no output.

- [ ] **Step 2: Inspect what's about to be committed**

Run: `git status`
Expected: all the files listed above shown as new, no model weights, no `.venv/`.

- [ ] **Step 3: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
chore: bootstrap turboquant-experiments repo

Brings over the phase 0 foundations doc, design spec, pedagogical
JL/outlier scripts, and a minimal uv-managed Python environment.
No model code yet; phase 1 inspection script lands next.
EOF
)"
```
Expected: a commit summary listing all the files.

---

## Section B — Pure helper functions (TDD'd, runs locally on CPU)

The two helpers below have hand-checkable inputs and outputs, so we test them before we write them. They run on CPU; no GPU needed. Get these solid on the laptop, then push to the pod via git.

### Task 8: TDD `kv_cache_bytes`

**Files:**
- Create: `~/Desktop/Experiments/turboquant-experiments/language-model-improvements/kv_utils.py`
- Create: `~/Desktop/Experiments/turboquant-experiments/language-model-improvements/tests/test_kv_utils.py`
- Create: `~/Desktop/Experiments/turboquant-experiments/language-model-improvements/tests/__init__.py` (empty)

- [ ] **Step 1: Create empty test package marker**

Run: `touch ~/Desktop/Experiments/turboquant-experiments/language-model-improvements/tests/__init__.py`

- [ ] **Step 2: Write the failing test**

Create `language-model-improvements/tests/test_kv_utils.py`:

```python
"""Unit tests for pure KV-cache helpers."""
import pytest
from language_model_improvements.kv_utils import kv_cache_bytes


def test_kv_cache_bytes_llama_3_1_8b_one_token_fp16():
    """Llama-3.1-8B at 1 token in fp16 should be ~128 KB.

    n_layers = 32, n_kv_heads = 8, head_dim = 128, p_bytes = 2 (fp16),
    seq_len = 1, batch = 1.
    Expected: 2 * 1 * 1 * 32 * 8 * 128 * 2 = 131_072 bytes.
    """
    assert kv_cache_bytes(
        seq_len=1,
        num_layers=32,
        n_kv_heads=8,
        head_dim=128,
        p_bytes=2,
        batch_size=1,
    ) == 131_072


def test_kv_cache_bytes_scales_linearly_with_seq_len():
    """Doubling seq_len should exactly double the result."""
    base = kv_cache_bytes(
        seq_len=1000, num_layers=32, n_kv_heads=8,
        head_dim=128, p_bytes=2, batch_size=1,
    )
    doubled = kv_cache_bytes(
        seq_len=2000, num_layers=32, n_kv_heads=8,
        head_dim=128, p_bytes=2, batch_size=1,
    )
    assert doubled == 2 * base


def test_kv_cache_bytes_scales_linearly_with_batch():
    """Doubling batch size should exactly double the result."""
    base = kv_cache_bytes(
        seq_len=512, num_layers=16, n_kv_heads=4,
        head_dim=64, p_bytes=2, batch_size=1,
    )
    doubled = kv_cache_bytes(
        seq_len=512, num_layers=16, n_kv_heads=4,
        head_dim=64, p_bytes=2, batch_size=2,
    )
    assert doubled == 2 * base


def test_kv_cache_bytes_int8_is_half_of_fp16():
    """Switching p_bytes from 2 to 1 should halve the result."""
    fp16 = kv_cache_bytes(
        seq_len=4096, num_layers=32, n_kv_heads=8,
        head_dim=128, p_bytes=2, batch_size=1,
    )
    int8 = kv_cache_bytes(
        seq_len=4096, num_layers=32, n_kv_heads=8,
        head_dim=128, p_bytes=1, batch_size=1,
    )
    assert int8 * 2 == fp16
```

(`language_model_improvements` here is a Python module path; we need to make it importable. The package layout is set up in step 4.)

- [ ] **Step 3: Make `language-model-improvements` importable as a package**

The folder name has a hyphen, which Python doesn't allow in module names, so we expose it via a `[tool.setuptools]` entry. Edit `~/Desktop/Experiments/turboquant-experiments/pyproject.toml` and add this to the bottom:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["language_model_improvements*"]
```

Then create a symlink so the importable name has an underscore:

Run:
```bash
cd ~/Desktop/Experiments/turboquant-experiments
ln -s language-model-improvements language_model_improvements
```

Expected: no output. `ls -la | grep language` should now show both the directory and the symlink.

Also create an `__init__.py` to make it a proper package:

Run: `touch ~/Desktop/Experiments/turboquant-experiments/language-model-improvements/__init__.py`

- [ ] **Step 4: Run the tests to confirm they fail**

Run: `cd ~/Desktop/Experiments/turboquant-experiments && uv run pytest language-model-improvements/tests/test_kv_utils.py -v`
Expected: All four tests fail with `ModuleNotFoundError: No module named 'language_model_improvements.kv_utils'` (because we haven't written `kv_utils.py` yet).

- [ ] **Step 5: Write the minimal implementation**

Create `~/Desktop/Experiments/turboquant-experiments/language-model-improvements/kv_utils.py`:

```python
"""Pure helpers for reasoning about KV cache memory.

Kept dependency-free (no torch, no transformers) so they can be unit-tested
fast on CPU and reused from any script.
"""


def kv_cache_bytes(
    *,
    seq_len: int,
    num_layers: int,
    n_kv_heads: int,
    head_dim: int,
    p_bytes: int,
    batch_size: int = 1,
) -> int:
    """Total KV-cache memory for one forward pass, in bytes.

    Formula:
        2 * batch * seq * n_layers * n_kv_heads * head_dim * p_bytes
    The leading 2 accounts for storing both K and V.

    Args:
        seq_len: number of tokens in the cache (prompt + generated).
        num_layers: transformer decoder layers (e.g. 32 for Llama-3.1-8B).
        n_kv_heads: KV attention heads (note: GQA models have n_kv_heads < n_q_heads).
        head_dim: per-head dimension.
        p_bytes: bytes per stored number (fp16 = 2, fp32 = 4, int8 = 1).
        batch_size: number of sequences cached in parallel.
    """
    return 2 * batch_size * seq_len * num_layers * n_kv_heads * head_dim * p_bytes
```

- [ ] **Step 6: Run the tests to confirm they pass**

Run: `cd ~/Desktop/Experiments/turboquant-experiments && uv run pytest language-model-improvements/tests/test_kv_utils.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

Run:
```bash
cd ~/Desktop/Experiments/turboquant-experiments
git add language-model-improvements/__init__.py \
        language-model-improvements/kv_utils.py \
        language-model-improvements/tests/__init__.py \
        language-model-improvements/tests/test_kv_utils.py \
        language_model_improvements \
        pyproject.toml uv.lock
git commit -m "feat(kv_utils): pure helper for KV cache memory math"
```

---

### Task 9: TDD `find_outlier_channels`

**Files:**
- Modify: `~/Desktop/Experiments/turboquant-experiments/language-model-improvements/kv_utils.py` (append a new function)
- Modify: `~/Desktop/Experiments/turboquant-experiments/language-model-improvements/tests/test_kv_utils.py` (append new tests)

- [ ] **Step 1: Write the failing tests**

Append to `language-model-improvements/tests/test_kv_utils.py`:

```python
import numpy as np
from language_model_improvements.kv_utils import find_outlier_channels


def test_find_outlier_channels_picks_the_obvious_one():
    """A 1-of-10 outlier channel should be flagged."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 10))   # 100 tokens, 10 channels
    x[:, 3] *= 50.0                       # channel 3 is the outlier
    outliers = find_outlier_channels(x, threshold_factor=10.0)
    assert outliers == [3]


def test_find_outlier_channels_returns_empty_when_uniform():
    """If no channel exceeds threshold * median, return []."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((100, 8))
    outliers = find_outlier_channels(x, threshold_factor=10.0)
    assert outliers == []


def test_find_outlier_channels_picks_multiple():
    """Multiple outlier channels should all be flagged, in order."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((200, 12))
    x[:, 1] *= 30.0
    x[:, 7] *= 40.0
    outliers = find_outlier_channels(x, threshold_factor=10.0)
    assert outliers == [1, 7]
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `cd ~/Desktop/Experiments/turboquant-experiments && uv run pytest language-model-improvements/tests/test_kv_utils.py -v`
Expected: 3 new failures with `ImportError: cannot import name 'find_outlier_channels'`. The previous 4 tests still pass.

- [ ] **Step 3: Write the minimal implementation**

Append to `~/Desktop/Experiments/turboquant-experiments/language-model-improvements/kv_utils.py`:

```python
import numpy as np
from numpy.typing import NDArray


def find_outlier_channels(
    x: NDArray,
    threshold_factor: float = 10.0,
) -> list[int]:
    """Return indices of channels (last-axis dimensions) that look like outliers.

    A channel is flagged if its per-channel max-abs value exceeds
    `threshold_factor * median(per-channel max-abs values across all channels)`.

    Args:
        x: 2-D array, shape (num_tokens, num_channels). Real K/V tensors are
            higher-dimensional; flatten/reshape to 2-D before calling.
        threshold_factor: how many times the typical channel magnitude a channel
            must reach to count as an outlier. 10.0 is a sensible default;
            paper-grade outliers are usually 30–100x.
    """
    per_channel_max = np.abs(x).max(axis=0)
    typical = float(np.median(per_channel_max))
    threshold = threshold_factor * typical
    return [int(i) for i in np.where(per_channel_max > threshold)[0]]
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `cd ~/Desktop/Experiments/turboquant-experiments && uv run pytest language-model-improvements/tests/test_kv_utils.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

Run:
```bash
cd ~/Desktop/Experiments/turboquant-experiments
git add language-model-improvements/kv_utils.py \
        language-model-improvements/tests/test_kv_utils.py
git commit -m "feat(kv_utils): add find_outlier_channels for K/V analysis"
```

---

## Section C — Push to GitHub and prepare RunPod

### Task 10: Create the GitHub repo and push

This task is partly manual because GitHub auth lives outside the agent.

- [ ] **Step 1: Owner creates the empty GitHub repo**

In the browser (or via `gh repo create turboquant-experiments --private --confirm`), create a new private repository named `turboquant-experiments`. Do NOT initialize with a README — we already have one.

- [ ] **Step 2: Add the remote and push**

Run (replace `<OWNER>` with the GitHub username):
```bash
cd ~/Desktop/Experiments/turboquant-experiments
git remote add origin git@github.com:<OWNER>/turboquant-experiments.git
git branch -M main
git push -u origin main
```
Expected: push succeeds, `main` is set to track `origin/main`.

- [ ] **Step 3: Confirm in browser**

Open the repo URL. Verify the directory layout matches §"File structure" above.

---

### Task 11: Pick and launch a RunPod pod

Manual but worth scripting in the plan so the choices are recorded.

- [ ] **Step 1: Choose GPU**

Pick **A100 80GB** (or A100 40GB if 80GB is unavailable). Reasoning:
- Llama-3.1-8B in fp16 needs ~16 GB for weights alone.
- A few-thousand-token KV cache adds ~1–2 GB.
- 40 GB has comfortable headroom; 80 GB has *more* and will let us push to 32k context easily.
- H100 is overkill for inspection — save it for phase 3 when we sweep configs.

- [ ] **Step 2: Choose template**

Pick the official **PyTorch 2.4 (CUDA 12.x) Ubuntu 22.04** template. We will install our exact deps via `uv sync`, but starting from a torch image saves the long torch download.

- [ ] **Step 3: Launch the pod**

In the RunPod UI: select the GPU + template, set disk to 50 GB (model weights + cache), launch. Wait for it to come up.

- [ ] **Step 4: Connect via Web Terminal or SSH**

The Web Terminal in the RunPod dashboard is fine for phase 1 (exploration). For longer phases we'll consider VS Code Remote.

---

### Task 12: Bootstrap the pod environment

All commands in this task run **inside the pod**.

- [ ] **Step 1: Install uv**

Run inside pod: `curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env`
Expected: `uv` is on PATH.

- [ ] **Step 2: Clone the repo**

Run:
```bash
cd ~
git clone https://github.com/<OWNER>/turboquant-experiments.git
cd turboquant-experiments
```

- [ ] **Step 3: Sync the environment**

Run: `uv sync --extra dev`
Expected: uv resolves and installs all deps. On a torch base image, torch is already cached so this should be fast.

- [ ] **Step 4: Verify GPU is visible to torch**

Run:
```bash
uv run python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```
Expected: `cuda available: True` and the GPU model name. **If this prints False, stop and debug — none of the rest of the plan will work.**

- [ ] **Step 5: Verify the unit tests still pass on the pod**

Run: `uv run pytest language-model-improvements/tests/test_kv_utils.py -v`
Expected: 7 passed. (This is a sanity check that the pod environment matches the laptop environment.)

- [ ] **Step 6: Authenticate with HuggingFace (Llama-3.1 is gated)**

Run: `uv run huggingface-cli login`
Paste the HF access token when prompted. (The token must have been granted access to `meta-llama/Llama-3.1-8B-Instruct` in advance via the HF model page.)
Expected: `Login successful`.

---

## Section D — The inspection script (incremental, run-and-observe)

This is **not** TDD'd. We build the script in small slices, run it after each slice, look at the output, and react to what we see. Each step ends in a commit so the history is a clean log of what we learned in what order.

All steps in this section run **inside the pod** (or can be developed on the laptop and pushed, then `git pull`'d on the pod — the loop is owner's choice).

### Task 13: Slice 1 — load the model and print its config

**Files:**
- Create: `~/turboquant-experiments/language-model-improvements/scripts/01_inspect_kv_cache.py`

- [ ] **Step 1: Write the slice 1 script**

Create the file with:

```python
"""Phase 1 inspection script: load Llama-3.1-8B-Instruct and report on its KV cache.

This script is built up across multiple commits during phase 1. Each slice adds
one piece of information; the final version produces a structured report saved
to `language-model-improvements/results/01_kv_cache_inspection.txt`.

Slice 1: load + print model config.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output", default=None,
                        help="If set, also write the report to this path.")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    cfg = model.config
    print()
    print("=== Model config ===")
    print(f"  hidden_size       : {cfg.hidden_size}")
    print(f"  num_hidden_layers : {cfg.num_hidden_layers}")
    print(f"  num_attention_heads (Q heads): {cfg.num_attention_heads}")
    print(f"  num_key_value_heads (KV heads): {cfg.num_key_value_heads}")
    print(f"  head_dim          : {cfg.hidden_size // cfg.num_attention_heads}")
    print(f"  vocab_size        : {cfg.vocab_size}")
    print(f"  max_position_embeddings: {cfg.max_position_embeddings}")
    print(f"  torch_dtype       : {model.dtype}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run inside pod:
```bash
cd ~/turboquant-experiments
uv run python language-model-improvements/scripts/01_inspect_kv_cache.py
```
Expected: model downloads from HF (first run only, ~16 GB, takes a few minutes), then prints the config block. **Read every line aloud to yourself before moving on** — these are the numbers your KV cache math will use.

- [ ] **Step 3: Commit**

Run:
```bash
git add language-model-improvements/scripts/01_inspect_kv_cache.py
git commit -m "feat(phase1): slice 1 — load model and print config"
git push
```

---

### Task 14: Slice 2 — print model weight memory

- [ ] **Step 1: Add a weight-memory section to the script**

Insert this block in `01_inspect_kv_cache.py` right after the config print, before `if __name__ == "__main__"`:

```python
def report_weight_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print()
    print("=== Model weights ===")
    print(f"  total parameters: {total_params:,}")
    print(f"  total bytes     : {total_bytes:,} ({total_bytes / 1e9:.2f} GB)")
```

And call it from `main()` after the config print:

```python
    report_weight_memory(model)
```

- [ ] **Step 2: Run it**

Run: `uv run python language-model-improvements/scripts/01_inspect_kv_cache.py`
Expected: same config output, plus a weight-memory block. For Llama-3.1-8B in fp16 expect roughly 8.0 B parameters and ~16 GB.

- [ ] **Step 3: Commit**

```bash
git add language-model-improvements/scripts/01_inspect_kv_cache.py
git commit -m "feat(phase1): slice 2 — print model weight memory"
git push
```

---

### Task 15: Slice 3 — run a forward pass and inspect `past_key_values`

- [ ] **Step 1: Add forward-pass + cache inspection**

Append a new function and call it from `main()`:

```python
def inspect_kv_cache_structure(model, tokenizer):
    """Run a single forward pass with cache enabled and inspect the result."""
    prompt = (
        "Explain in one sentence what the KV cache is and why it exists "
        "in transformer language models."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print()
    print("=== Forward pass ===")
    print(f"  prompt: {prompt!r}")
    print(f"  input_ids shape: {tuple(inputs.input_ids.shape)}")

    with torch.no_grad():
        out = model(**inputs, use_cache=True)

    pkv = out.past_key_values
    print()
    print("=== past_key_values structure ===")
    print(f"  type: {type(pkv).__name__}")
    print(f"  num layers: {len(pkv)}")
    layer0 = pkv[0]
    print(f"  per-layer type: {type(layer0).__name__}")
    print(f"  per-layer length (K, V): {len(layer0)}")
    k0, v0 = layer0
    print(f"  K[0] shape: {tuple(k0.shape)}  dtype: {k0.dtype}  device: {k0.device}")
    print(f"  V[0] shape: {tuple(v0.shape)}  dtype: {v0.dtype}  device: {v0.device}")
    return pkv, inputs
```

Then in `main()`, after `report_weight_memory(model)`:

```python
    pkv, inputs = inspect_kv_cache_structure(model, tokenizer)
```

- [ ] **Step 2: Run it**

Run: `uv run python language-model-improvements/scripts/01_inspect_kv_cache.py`
Expected output includes a `past_key_values structure` block. K/V shapes should be `(1, 8, seq_len, 128)` for Llama-3.1-8B (batch=1, n_kv_heads=8, seq_len=tokens in prompt, head_dim=128). **If the shape doesn't match this, stop and figure out why before continuing.**

- [ ] **Step 3: Commit**

```bash
git add language-model-improvements/scripts/01_inspect_kv_cache.py
git commit -m "feat(phase1): slice 3 — run forward pass and inspect past_key_values"
git push
```

---

### Task 16: Slice 4 — KV memory math at a sweep of context lengths

- [ ] **Step 1: Use `kv_cache_bytes` to print a sweep**

Append a new function:

```python
from language_model_improvements.kv_utils import kv_cache_bytes


def report_kv_memory_sweep(model):
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    p_bytes = 2  # fp16

    per_token = kv_cache_bytes(
        seq_len=1, num_layers=n_layers, n_kv_heads=n_kv_heads,
        head_dim=head_dim, p_bytes=p_bytes,
    )

    print()
    print("=== KV cache memory (analytical) ===")
    print(f"  per token: {per_token:,} bytes ({per_token / 1024:.1f} KB)")
    print()
    print(f"  {'seq_len':>8}  {'KV bytes':>16}  {'KV (GB)':>10}")
    print("  " + "-" * 40)
    for seq_len in [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        b = kv_cache_bytes(
            seq_len=seq_len, num_layers=n_layers, n_kv_heads=n_kv_heads,
            head_dim=head_dim, p_bytes=p_bytes,
        )
        print(f"  {seq_len:>8}  {b:>16,}  {b / 1e9:>9.2f}")

    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print()
    print(f"  Model weight memory : {weight_bytes / 1e9:.2f} GB")
    print(f"  KV at 8k  / weights : {kv_cache_bytes(seq_len=8192, num_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim, p_bytes=p_bytes) / weight_bytes:.2%}")
    print(f"  KV at 128k/ weights : {kv_cache_bytes(seq_len=131072, num_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim, p_bytes=p_bytes) / weight_bytes:.2%}")
```

Call it from `main()` after `inspect_kv_cache_structure(...)`:

```python
    report_kv_memory_sweep(model)
```

- [ ] **Step 2: Run it**

Run: `uv run python language-model-improvements/scripts/01_inspect_kv_cache.py`
Expected: a memory sweep table. **Look for the "KV at 128k vs weights" ratio — it should be around 100% (i.e. the KV cache at full context is ~the same size as the model itself). This is the moment you should feel the entire motivation for TurboQuant.**

- [ ] **Step 3: Commit**

```bash
git add language-model-improvements/scripts/01_inspect_kv_cache.py
git commit -m "feat(phase1): slice 4 — KV memory sweep across context lengths"
git push
```

---

### Task 17: Slice 5 — K/V value distribution and outlier channels

- [ ] **Step 1: Add per-channel value statistics**

Append:

```python
import numpy as np
from language_model_improvements.kv_utils import find_outlier_channels


def report_value_distributions(pkv, layer_idx: int = 0):
    """Print K and V per-channel value statistics for one chosen layer."""
    k, v = pkv[layer_idx]
    # Shape is (batch, n_kv_heads, seq_len, head_dim).
    # Flatten to (n_tokens, n_channels) where n_channels = n_kv_heads * head_dim.
    k_np = k[0].permute(1, 0, 2).reshape(k.shape[2], -1).float().cpu().numpy()
    v_np = v[0].permute(1, 0, 2).reshape(v.shape[2], -1).float().cpu().numpy()

    def stats(name, arr):
        print(f"  {name}: shape={arr.shape}  "
              f"mean={arr.mean():+.3f}  std={arr.std():.3f}  "
              f"min={arr.min():+.2f}  max={arr.max():+.2f}  "
              f"p99.9={np.quantile(np.abs(arr), 0.999):.2f}")

    print()
    print(f"=== K/V value distributions (layer {layer_idx}) ===")
    stats("K", k_np)
    stats("V", v_np)

    k_outliers = find_outlier_channels(k_np, threshold_factor=10.0)
    v_outliers = find_outlier_channels(v_np, threshold_factor=10.0)
    print(f"  K outlier channels (>10x median): {k_outliers}")
    print(f"  V outlier channels (>10x median): {v_outliers}")
```

Call it from `main()` after `report_kv_memory_sweep(model)`:

```python
    report_value_distributions(pkv, layer_idx=0)
    report_value_distributions(pkv, layer_idx=cfg.num_hidden_layers // 2)
    report_value_distributions(pkv, layer_idx=cfg.num_hidden_layers - 1)
```

(Inspect three layers — early, middle, late — because outlier behavior often changes with depth.)

- [ ] **Step 2: Run it**

Run: `uv run python language-model-improvements/scripts/01_inspect_kv_cache.py`
Expected: three K/V value-distribution blocks, one per chosen layer. **Look at the outlier channel lists — if they're non-empty on real Llama K/V data, you've just empirically confirmed the central premise of TurboQuant on the actual model. If they're empty, that's also interesting and worth investigating.**

- [ ] **Step 3: Commit**

```bash
git add language-model-improvements/scripts/01_inspect_kv_cache.py
git commit -m "feat(phase1): slice 5 — K/V distributions and outlier channels"
git push
```

---

### Task 18: Slice 6 — tee the report to a results file

- [ ] **Step 1: Wrap stdout when `--output` is provided**

Modify `main()` so that if `args.output` is set, all `print` calls also go to the file. Simplest approach — wrap stdout with a `Tee`:

```python
import sys


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()
```

In `main()`, after parsing args, before any other code:

```python
    if args.output:
        import os
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        f = open(args.output, "w")
        sys.stdout = _Tee(sys.__stdout__, f)
```

- [ ] **Step 2: Run with `--output`**

Run:
```bash
cd ~/turboquant-experiments
uv run python language-model-improvements/scripts/01_inspect_kv_cache.py \
    --output language-model-improvements/results/01_kv_cache_inspection.txt
```
Expected: same console output as before, AND a file at the specified path containing the same content.

- [ ] **Step 3: Verify the file is well-formed**

Run: `wc -l language-model-improvements/results/01_kv_cache_inspection.txt`
Expected: dozens of lines.

Run: `head -30 language-model-improvements/results/01_kv_cache_inspection.txt`
Expected: the model config block at the top.

- [ ] **Step 4: Commit**

```bash
git add language-model-improvements/scripts/01_inspect_kv_cache.py \
        language-model-improvements/results/01_kv_cache_inspection.txt
git commit -m "feat(phase1): slice 6 — save inspection report to results/"
git push
```

---

## Section E — Writing checkpoint and shutdown

### Task 19: Owner writes `notes/01-kv-cache-reality.md`

This is the **gating writeup for phase 1**. The supervisor will not let phase 2 begin without it.

**Files:**
- Create: `~/turboquant-experiments/notes/01-kv-cache-reality.md`

- [ ] **Step 1: Re-read the captured report**

Run: `cat language-model-improvements/results/01_kv_cache_inspection.txt`

- [ ] **Step 2: Write the note**

The note should answer, in the owner's own words, the following prompts. ~600–1000 words. Length is not the goal; honest reflection is.

1. **What did the model config look like?** Did the numbers match what you expected from the foundations doc? Anything different?
2. **How big are the model weights?** Did the per-token KV cache size you computed earlier in `notes/00-foundations.md` match what the script printed? If not, where was the discrepancy?
3. **The KV-vs-weights crossover.** At what context length does KV cache memory equal the model weight memory? Was this earlier or later than you expected? What does that imply for inference serving economics?
4. **K/V distributions.** What did you see for mean / std / min / max / p99.9? Were the outlier channels detected, or was the distribution clean? If outliers were detected, which channels and at which layers? Did early/middle/late layers behave differently?
5. **What surprised you?** This is the most important question. Anything that didn't match your prior mental model. If nothing surprised you, that itself is information — write that down too, and the supervisor will probe to find out whether the foundations were unusually well-calibrated or whether the inspection didn't go deep enough.
6. **What questions does this raise that you'd like answered in phase 2 or phase 3?**

- [ ] **Step 3: Submit the note for supervisor review**

Show the file to the supervisor. Iterate until the supervisor signs off. **Do not move to phase 2 without this sign-off.**

- [ ] **Step 4: Commit**

```bash
cd ~/turboquant-experiments
git add notes/01-kv-cache-reality.md
git commit -m "docs(phase1): own-words writeup of KV cache inspection findings"
git push
```

---

### Task 20: Shut down the pod

- [ ] **Step 1: Verify everything is committed and pushed**

Run inside pod:
```bash
git status
git log --oneline -10
```
Expected: working tree clean, recent commits visible. If anything is uncommitted, commit and push it now — the pod is about to die.

- [ ] **Step 2: Stop the pod from the RunPod dashboard**

Click "Stop" (or "Terminate" if you don't plan to resume). **Confirm the meter has stopped before closing the tab.**

- [ ] **Step 3: From the laptop, pull the latest**

Run on laptop:
```bash
cd ~/Desktop/Experiments/turboquant-experiments
git pull
```
Expected: all phase 1 commits are now visible locally.

---

## Phase 1 done

At this point:
- The repo exists, is on GitHub, has a clean phase 0 + phase 1 history.
- The unit tests pass.
- The inspection script produces a structured report on real Llama-3.1-8B.
- The owner has written a personal-voice note about what they observed and what surprised them, and the supervisor has signed off.
- The pod is shut down.

Phase 2 begins with a separate writing-plans invocation. We deliberately do not pre-plan phase 2 here.

---

## Self-review checklist (run after writing this plan)

- [x] Spec coverage: every Phase 1 deliverable from `docs/superpowers/specs/2026-04-08-turboquant-experiments-design.md` §6 has a task. The script (`01_inspect_kv_cache.py`) and the writeup (`notes/01-kv-cache-reality.md`) are both covered.
- [x] Placeholder scan: no TBDs, no "implement later", no "similar to Task N" without showing the code, no "add error handling" hand-waves.
- [x] Type consistency: `kv_cache_bytes` and `find_outlier_channels` have consistent signatures across the test file (Task 8/9), the implementation (Task 8/9), and the call sites in the inspection script (Task 16/17).
- [x] TDD vs. exploratory split is explicit and justified (see "Where TDD applies and where it doesn't" under File structure).
- [x] No phase 2 work has leaked in (no eval harness, no perplexity, no quantization).
