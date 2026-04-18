# Part 2 Pivot — Parler-TTS Implementation Plan

> **For the fresh session executing this:** every script below already exists in the repo. Your job is to SSH into the AWS instance, set up the environment, and run them. You do NOT need to write new code — everything is pre-authored. You just run, observe, debug specific errors if they appear, and commit results.
>
> **Hard timebox: 2 hours of GPU wall-clock time.** If we don't have working baseline audio by 45 min in, stop and reassess. If baseline works but quantization doesn't by 90 min in, commit baseline only and stop.

---

## Goal

Generate 36 WAV files (4 cache configs × 3 voices × 3 text prompts) using Parler-TTS Mini v1, where 3 of the 4 configs apply `HandrolledTurboQuantCache` at 4-bit, 3-bit, 2-bit. Then pull the WAVs to the laptop for CPU-side metrics (UTMOS, speaker similarity, WER).

## Pre-flight

Before starting the fresh session:

- [ ] User shares AWS SSH details: public IP, key path on Mac (e.g. `~/.ssh/turboquant-key.pem`), username (`ubuntu`)
- [ ] Confirm instance is running and has ≥ 50 GB free disk
- [ ] Confirm GitHub PAT is saved somewhere (needed for committing results from the pod)

## Execution order (all happens on the AWS instance unless noted)

### Phase 1: Environment setup (~10 min)

**Task 1.1.** SSH in:

```bash
ssh -i ~/.ssh/turboquant-key.pem -o StrictHostKeyChecking=accept-new ubuntu@<PUBLIC_IP>
```

**Task 1.2.** Install system deps (this AMI is bare — no pip):

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv ffmpeg git build-essential
```

**Task 1.3.** Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

**Task 1.4.** Clone repo (or pull if already cloned):

```bash
cd ~
if [ -d turboquant-experiments ]; then
    cd turboquant-experiments && git pull
else
    git clone https://github.com/MaximusStupidus/turboquant-experiments.git
    cd turboquant-experiments
fi
```

**Task 1.5.** Sync the env:

```bash
uv sync --extra dev
```

**Task 1.6.** Install Parler-TTS with strict deps (remember: it pins `transformers==4.46.1` — we accept the pin to sidestep API drift):

```bash
uv pip install "numpy<2"
uv pip install git+https://github.com/huggingface/parler-tts.git
uv pip install soundfile
```

**Task 1.7.** Set env:

```bash
export HF_HOME=~/hf_cache
```

**Task 1.8.** Verify:

```bash
uv run python speech-tts-improvements/parler/scripts/00_check_setup.py
```

**Expected output:** prints CUDA available, GPU name, torch/transformers/parler_tts versions. If transformers is NOT 4.46.1, that's a problem — the parler install should have downgraded it.

### Phase 2: Baseline generation (~20 min)

**Task 2.1.** Generate baseline audio for all (voice × text) combos:

```bash
uv run python speech-tts-improvements/parler/scripts/01_generate_baseline.py
```

Creates `speech-tts-improvements/parler/results/baseline/*.wav` (9 files).

**Task 2.2.** Listen test (on laptop after syncing): baseline WAVs should sound like clear human speech. If they sound like static or silence, **stop** — the model/codec isn't working and we need to debug before touching quantization.

### Phase 3: TurboQuant sweep (~60 min)

**Task 3.1.** Run the sweep for 4-bit, 3-bit, 2-bit:

```bash
uv run python speech-tts-improvements/parler/scripts/02_generate_turboquant.py
```

Creates `speech-tts-improvements/parler/results/tq_4bit/*.wav`, `tq_3bit/*.wav`, `tq_2bit/*.wav` (9 files each).

Also writes `speech-tts-improvements/parler/results/timings.json` with RTF + peak memory per config.

### Phase 4: Commit and shut down (~5 min)

**Task 4.1.** Commit results from pod:

```bash
git config user.name "Ojas Jain"
git config user.email "96643674+MaximusStupidus@users.noreply.github.com"
git remote set-url origin https://<PAT>@github.com/MaximusStupidus/turboquant-experiments.git
git add speech-tts-improvements/parler/results/
git commit -m "data(part2-parler): baseline + TurboQuant sweep audio + timings"
git push
```

**Task 4.2.** Exit SSH. **Shut down the instance from AWS console** (not `sudo shutdown` — the instance stays allocated and continues billing on stop vs terminate; pick based on whether we want to resume tomorrow).

### Phase 5: CPU-side metrics (laptop, ~15 min)

Back on laptop:

```bash
cd ~/Desktop/Experiments/turboquant-experiments
git pull
# Audio quality metrics (UTMOS, speaker sim, WER)
python3 speech-tts-improvements/parler/scripts/03_compute_metrics.py
```

Writes `speech-tts-improvements/parler/results/metrics.json` with all scores.

### Phase 6: Plots and write-up (laptop, ~15 min)

```bash
python3 speech-tts-improvements/parler/scripts/04_plot_results.py
```

Writes `speech-tts-improvements/parler/results/plots/*.png`.

Owner writes `notes/04-parler-results.md`. Final commit + push.

---

## Troubleshooting cheat sheet (for the fresh session)

### If `parler-tts` install fails

- Ensure `build-essential git` are installed (descript-audiotools needs to compile)
- Try `pip install --upgrade pip` first
- If a specific dep fails, `uv pip install --no-deps git+https://github.com/huggingface/parler-tts.git` then install deps one by one

### If transformers version mismatches

- Parler pins `transformers==4.46.1` exactly. If uv sync installs a newer version first, the parler install will downgrade it. That's fine.
- If after install `transformers` is >4.46, explicitly: `uv pip install "transformers==4.46.1"`

### If model download fails (HF token)

- Set `export HF_TOKEN=<your_token>` (public model but token raises rate limits)

### If "DAC model_type already registered"

- Known issue on newer transformers. If we hit this on 4.46.1 it's unexpected; if it happens, add `exist_ok=True` patch like we did for VibeVoice

### If audio is garbage

- Check `model.config.sampling_rate` == 44100
- Check that `audio_arr` is a 1-D numpy array in [-1, 1]
- Try `do_sample=True` with `set_seed(42)` — greedy can collapse sometimes

### If TurboQuant cache errors out with "layer_idx out of range"

- Our `HandrolledTurboQuantCache` creates empty layers and fills them via `update()`. That's correct. If Parler expects pre-sized layers, we may need to override `__init__` to pre-create `num_layers` empty `DynamicLayer` objects.
- See `language-model-improvements/handrolled_turboquant.py` — pattern matches `DynamicCache` which is what Parler expects.

### If `EncoderDecoderCache` wrapping fails

- Print `type(past_key_values)` inside `model.generate()` to verify auto-wrap happened
- If needed, manually wrap: `from transformers import EncoderDecoderCache; EncoderDecoderCache(our_cache, DynamicCache())`
