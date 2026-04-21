"""Build a self-contained HTML frontend that lets you A/B the generated
audio across fp16 baseline and TurboQuant 4/3/2-bit.

Reads metrics.json + the WAVs in results/ and writes
results/audio_comparison.html. Load the HTML locally or via GitHub
Pages — each audio element streams the corresponding WAV via a
relative path.

Usage:
    python3 speech-tts-improvements/parler/scripts/05_build_frontend.py
"""
import os
import sys
import json

import numpy as np
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS = os.path.join(REPO_ROOT, "speech-tts-improvements/parler/results")
sys.path.insert(0, os.path.join(REPO_ROOT, "speech-tts-improvements", "parler"))

from voices_and_texts import VOICES, TEXTS  # noqa: E402

CONFIGS = [
    ("baseline", "fp16 baseline", "reference · what clean speech sounds like"),
    ("tq_4bit", "TurboQuant 4-bit", "should sound ≈ identical to baseline"),
    ("tq_3bit", "TurboQuant 3-bit", "subtle artifacts on long clips"),
    ("tq_2bit", "TurboQuant 2-bit", "audible degradation on long clips"),
    ("noproj_2bit", "No-projection 2-bit", "ablation · rotation removed → severely broken"),
    ("naive_2bit", "Naive min-max 2-bit", "ablation · no rotation, no Beta codebook · ~same WER as TQ 2-bit, slightly different voice"),
]
VOICES_ORDER = ["jon", "laura", "gary"]
TEXTS_ORDER = ["short", "medium", "long"]


def wav_duration(path: str):
    try:
        a, sr = sf.read(path)
        return len(a) / sr
    except Exception:
        return None


def chip_class(value: float, good_thr: float, mid_thr: float, invert: bool = False) -> str:
    if value is None:
        return ""
    if invert:
        return "good" if value < good_thr else ("mid" if value < mid_thr else "bad")
    return "good" if value > good_thr else ("mid" if value > mid_thr else "bad")


def cell_html(metrics, config: str, voice: str, text_name: str) -> str:
    key = f"{voice}__{text_name}"
    rel_path = f"{config}/{key}.wav"
    full_path = os.path.join(RESULTS, rel_path)
    dur = wav_duration(full_path)
    entry = metrics.get(config, {}).get(key, {})
    wer = entry.get("wer", {}).get("wer")
    transcript = entry.get("wer", {}).get("transcript", "") or ""
    sim = entry.get("speaker_similarity_vs_baseline")

    chips = []
    if dur is not None:
        chips.append(f'<span class="meta-chip dur">{dur:.1f}s</span>')
    if wer is not None:
        chips.append(f'<span class="meta-chip wer {chip_class(wer, 0.30, 0.60, invert=True)}">WER {wer:.2f}</span>')
    if isinstance(sim, (int, float)):
        chips.append(f'<span class="meta-chip sim {chip_class(float(sim), 0.75, 0.50)}">SpkSim {float(sim):.2f}</span>')

    transcript_html = f'<div class="transcript">&ldquo;{transcript.strip()}&rdquo;</div>' if transcript else ""
    return (
        '    <div class="cell">\n'
        f'      <audio controls preload="none" src="{rel_path}"></audio>\n'
        f'      <div class="meta">{"".join(chips)}</div>\n'
        f'      {transcript_html}\n'
        '    </div>'
    )


def main() -> None:
    with open(os.path.join(RESULTS, "metrics.json")) as f:
        metrics = json.load(f)

    # Per-config means for the summary table
    summary_rows = []
    for cfg, label, _ in CONFIGS:
        wers = [
            v["wer"]["wer"]
            for v in metrics.get(cfg, {}).values()
            if isinstance(v, dict) and "wer" in v.get("wer", {})
        ]
        mean_wer = f"{np.mean(wers):.2f}" if wers else "—"
        sims = [
            v["speaker_similarity_vs_baseline"]
            for v in metrics.get(cfg, {}).values()
            if isinstance(v, dict)
            and isinstance(v.get("speaker_similarity_vs_baseline"), (int, float))
        ]
        mean_sim = f"{np.mean(sims):.2f}" if sims else "—"
        summary_rows.append(
            f"<tr><td>{label}</td><td>{mean_wer}</td><td>{mean_sim}</td></tr>"
        )

    rows_html = []
    for voice in VOICES_ORDER:
        for text_name in TEXTS_ORDER:
            ref_text = TEXTS[text_name]
            row_label = (
                '<div class="row-label">'
                f'<div class="voice">{voice}</div>'
                f'<div class="text-kind">{text_name}</div>'
                f'<div class="reference">&ldquo;{ref_text}&rdquo;</div>'
                "</div>"
            )
            cells = "".join(cell_html(metrics, cfg, voice, text_name) for cfg, _, _ in CONFIGS)
            rows_html.append(f'<div class="row">\n  {row_label}\n{cells}\n</div>')

    header_html = "\n".join(
        f'  <div class="header-cell"><div class="hc-title">{label}</div>'
        f'<div class="hc-expected">{expected}</div></div>'
        for _, label, expected in CONFIGS
    )

    html = _HTML_TEMPLATE.format(
        summary_rows="".join(summary_rows),
        header_html=header_html,
        rows_html="\n".join(rows_html),
    )

    out = os.path.join(RESULTS, "audio_comparison.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({len(html):,} chars)")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>TurboQuant on Parler-TTS — audio comparison</title>
<style>
:root {{
  --bg: #0b0d10;
  --panel: #13161b;
  --border: #1f232b;
  --fg: #e8ecf2;
  --fg-dim: #8a93a3;
  --accent: #7aa2ff;
  --good: #5ac58a;
  --mid: #e6b44c;
  --bad: #ff7a7a;
  --baseline: #6ec4a7;
  --q4: #a7c3ff;
  --q3: #ffb86c;
  --q2: #ff7a9c;
}}
* {{ box-sizing: border-box; }}
html, body {{
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, \"SF Pro Text\", Segoe UI, Inter, system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  margin: 0;
  padding: 0;
  font-feature-settings: \"ss01\", \"ss02\", \"cv11\";
}}
.wrapper {{
  max-width: 1400px;
  margin: 0 auto;
  padding: 48px 32px 96px;
}}
header {{
  margin-bottom: 40px;
  padding-bottom: 28px;
  border-bottom: 1px solid var(--border);
}}
h1 {{
  margin: 0 0 8px;
  font-size: 28px;
  font-weight: 600;
  letter-spacing: -0.01em;
}}
header p {{
  margin: 6px 0 0;
  color: var(--fg-dim);
  max-width: 700px;
}}
.summary {{
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 32px;
  align-items: start;
  margin-top: 28px;
}}
.summary table {{
  border-collapse: collapse;
  font-size: 13px;
}}
.summary th, .summary td {{
  padding: 8px 16px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}}
.summary th {{
  color: var(--fg-dim);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 11px;
}}
.summary td:nth-child(2), .summary td:nth-child(3) {{ font-variant-numeric: tabular-nums; }}
.legend {{
  color: var(--fg-dim);
  font-size: 12px;
  line-height: 1.7;
}}
.legend strong {{ color: var(--fg); font-weight: 600; }}
.grid {{
  display: grid;
  grid-template-columns: 220px repeat(6, 1fr);
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  font-size: 12px;
}}
.grid-header {{
  display: contents;
}}
.header-cell, .row-label, .cell {{
  background: var(--panel);
  padding: 16px 18px;
}}
.header-cell {{
  font-weight: 600;
  color: var(--fg);
  background: #181c22;
  font-size: 13px;
}}
.header-cell .hc-title {{ font-size: 13px; font-weight: 600; }}
.header-cell .hc-expected {{
  font-size: 11px;
  font-weight: 400;
  font-style: italic;
  color: var(--fg-dim);
  margin-top: 4px;
  line-height: 1.4;
}}
.grid > .header-cell:nth-child(2) {{ color: var(--baseline); }}
.grid > .header-cell:nth-child(3) {{ color: var(--q4); }}
.grid > .header-cell:nth-child(4) {{ color: var(--q3); }}
.grid > .header-cell:nth-child(5) {{ color: var(--q2); }}
.grid > .header-cell:nth-child(6) {{ color: #ff7a7a; }}
.grid > .header-cell:nth-child(7) {{ color: #c4b5fd; }}
.row {{
  display: contents;
}}
.row-label {{
  background: #181c22;
}}
.row-label .voice {{
  font-weight: 600;
  font-size: 14px;
  color: var(--fg);
  text-transform: capitalize;
}}
.row-label .text-kind {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--fg-dim);
  margin-top: 2px;
}}
.row-label .reference {{
  font-size: 12px;
  color: var(--fg-dim);
  margin-top: 10px;
  line-height: 1.5;
  max-height: 120px;
  overflow: auto;
}}
.cell audio {{
  width: 100%;
  height: 36px;
  border-radius: 8px;
  filter: invert(0.88) hue-rotate(180deg);
}}
.cell .meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
}}
.meta-chip {{
  display: inline-block;
  padding: 3px 9px;
  border-radius: 6px;
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  font-weight: 500;
  background: rgba(255, 255, 255, 0.05);
  color: var(--fg-dim);
  border: 1px solid var(--border);
}}
.meta-chip.dur {{ color: var(--fg); }}
.meta-chip.good {{ color: var(--good); border-color: rgba(90, 197, 138, 0.35); background: rgba(90, 197, 138, 0.08); }}
.meta-chip.mid {{ color: var(--mid); border-color: rgba(230, 180, 76, 0.35); background: rgba(230, 180, 76, 0.08); }}
.meta-chip.bad {{ color: var(--bad); border-color: rgba(255, 122, 122, 0.35); background: rgba(255, 122, 122, 0.08); }}
.cell .transcript {{
  margin-top: 10px;
  font-size: 11px;
  color: var(--fg-dim);
  line-height: 1.5;
  font-style: italic;
  opacity: 0.8;
  max-height: 80px;
  overflow: auto;
}}
@media (max-width: 1100px) {{
  .grid {{ grid-template-columns: 1fr; }}
  .header-cell {{ display: none; }}
  .row {{
    display: block;
    padding: 16px 0;
    border-bottom: 1px solid var(--border);
  }}
  .row-label {{
    display: block;
    margin-bottom: 12px;
  }}
  .cell {{
    display: block;
    margin-bottom: 10px;
  }}
}}
footer {{
  margin-top: 48px;
  color: var(--fg-dim);
  font-size: 12px;
  line-height: 1.6;
}}
footer a {{ color: var(--accent); text-decoration: none; }}
footer a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class=\"wrapper\">

<header>
  <h1>TurboQuant on Parler-TTS Mini v1 &mdash; audio comparison</h1>
  <p style="margin-bottom:18px;">
    <a href="algorithm_explained.html" style="color:var(--accent); text-decoration:none; border-bottom:1px solid var(--accent);">&larr; How the algorithm works</a>
  </p>
  <p>
    9 prompts (3 voices &times; 3 text lengths) generated at 6 cache
    settings: fp16 baseline, TurboQuant 4-bit / 3-bit / 2-bit, and two
    2-bit ablations (no-projection + naive min-max). Generation used
    <code>do_sample=True, temperature=1.0, seed=42</code>,
    <code>max_length</code> per text length. Metrics: WER via Whisper
    small.en, speaker similarity vs fp16 baseline via ECAPA-TDNN.
  </p>

  <div class=\"summary\">
    <table>
      <thead>
        <tr><th>Config</th><th>Mean WER</th><th>Mean spk-sim</th></tr>
      </thead>
      <tbody>
        {summary_rows}
      </tbody>
    </table>
    <div class=\"legend\">
      <strong>Reading the cells.</strong>
      Chips use red/yellow/green by threshold &mdash; WER &lt; 0.30 good,
      0.30&ndash;0.60 mid, &gt; 0.60 bad. Speaker similarity &gt; 0.75
      good, 0.50&ndash;0.75 mid, &lt; 0.50 bad. The reference text under
      each row's voice label is the exact prompt fed to Parler; the
      italicised line under each audio cell is Whisper's transcript of
      the generated audio.
      <br><br>
      <strong>What to listen for.</strong> Differences are subtle on
      short prompts because the 128-token residual buffer keeps the
      first ~1.5&nbsp;s of every clip at fp16 regardless of bit level.
      The clearest A/B is on <em>long</em> prompts:
      <code>gary__long</code> at 2-bit is visibly broken
      (&ldquo;ghoul the sheet...rude juice of terror&rdquo;);
      <code>laura__long</code> at 3-bit shows gradual drift.
      <code>noproj_2bit</code> is the ablation that proves the rotation
      matters &mdash; compare any long clip there to TurboQuant 2-bit.
    </div>
  </div>
</header>

<div class=\"grid\">
  <div class=\"grid-header\">
    <div class=\"header-cell\">prompt</div>
{header_html}
  </div>

{rows_html}
</div>

<footer>
  Part of the
  <a href=\"https://github.com/MaximusStupidus/turboquant-experiments\">turboquant-experiments</a>
  project. Generated by
  <code>speech-tts-improvements/parler/scripts/05_build_frontend.py</code>.
  Load this file locally or via GitHub Pages &mdash; each audio tag
  streams the corresponding WAV relative to this file's location.
</footer>

</div>
</body>
</html>
"""


if __name__ == "__main__":
    main()
