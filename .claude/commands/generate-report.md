# generate-report

Generate all HTML output files and standalone model.py for a paper.

Output: `outputs/{name}/summary.html`, `outputs/{name}/model.py`, `outputs/{name}/train.html`, `outputs/{name}/evaluate.html`

## Arguments

$ARGUMENTS

First argument: paper name slug (e.g. `attention-is-all-you-need`). Must have `analyses/{name}/innovations.md`.

## Steps

**Step 1 — Load inputs.**

Read these reference files before generating anything:
1. `analyses/{name}/innovations.md` — primary source. If missing: tell user to run `/analyze-innovations {name}` first.
2. `analyses/{name}/raw.md` — supplemental detail.
3. `prompts/html_report_system.md` — HTML design rules (dark theme, self-contained, no CDN).
4. `outputs/_template/summary.html` — **reference template**: follow its CSS variables, class names, and section structure exactly.
5. `outputs/_template/evaluate.html` — shows the metric card + ROC/PR chart layout.
6. `outputs/_template/train.html` — shows the loss curve + metric chart layout.

**Step 2 — Create output directory.**

Run: `mkdir -p outputs/{name}`

**Step 3 — Extract architecture figure.**

Before generating the HTML, check for an extracted figure:

```bash
python3 scripts/extract_figures.py papers/{name}.pdf analyses/{name}/
```

Parse the JSON output and get `arch_figure`. If `arch_figure` has a `b64` field (non-null), you have a base64-encoded PNG to embed. Store it as `ARCH_B64` and `ARCH_CAPTION`.

If `extract_figures.py` fails (PyMuPDF not installed), set `ARCH_B64 = null`. The architecture section will fall back to text-only.

**Step 4 — Generate `outputs/{name}/summary.html`.**

Write a complete, self-contained HTML file. All CSS must be inlined in `<style>` — no external URLs, no CDN links. The file must render correctly when opened directly in a browser without internet access.

Use this exact structure and CSS framework:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{paper title} — Research Summary</title>
<style>
/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0f1117;
  --surface: #1a1d27;
  --surface2: #222535;
  --border: #2e3148;
  --text: #e2e4f0;
  --text-muted: #8b8fa8;
  --accent: #7c6af5;
  --accent2: #4fc3f7;
  --accent3: #69f0ae;
  --warn: #ffb74d;
  --red: #ef5350;
  --font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
  --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --radius: 8px;
}
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 15px;
  line-height: 1.7;
  max-width: 960px;
  margin: 0 auto;
  padding: 40px 24px 80px;
}

/* ── Header ── */
.paper-header {
  border-left: 4px solid var(--accent);
  padding: 24px 28px;
  background: var(--surface);
  border-radius: 0 var(--radius) var(--radius) 0;
  margin-bottom: 32px;
}
.paper-header h1 { font-size: 1.6rem; font-weight: 700; color: #fff; margin-bottom: 8px; }
.paper-meta { color: var(--text-muted); font-size: 0.875rem; margin-bottom: 12px; }
.paper-meta span { margin-right: 16px; }
.tldr {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 18px;
  font-size: 0.95rem;
  color: var(--accent2);
  font-style: italic;
  margin-top: 14px;
}
.tldr::before { content: "💡 TL;DR: "; font-style: normal; font-weight: 600; }

/* ── TOC ── */
.toc {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 24px;
  margin-bottom: 40px;
}
.toc h3 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 12px; }
.toc ol { list-style: none; counter-reset: toc; }
.toc ol li { counter-increment: toc; padding: 4px 0; }
.toc ol li::before { content: counter(toc) ". "; color: var(--accent); font-weight: 600; }
.toc a { color: var(--text); text-decoration: none; }
.toc a:hover { color: var(--accent); }

/* ── Sections ── */
section { margin-bottom: 44px; }
.section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}
.section-icon {
  width: 36px; height: 36px;
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem;
  flex-shrink: 0;
}
.icon-problem  { background: #3d1a1a; }
.icon-contrib  { background: #1a2d1a; }
.icon-arch     { background: #1a1a3d; }
.icon-loss     { background: #2d2d1a; }
.icon-train    { background: #1a2a2d; }
.icon-results  { background: #2d1a2d; }
.icon-impl     { background: #2a1a10; }
.section-header h2 { font-size: 1.15rem; font-weight: 700; }

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 22px;
  margin-bottom: 14px;
}
.card h3 { font-size: 0.95rem; font-weight: 600; color: var(--accent2); margin-bottom: 10px; }
.card p, .card li { color: var(--text); font-size: 0.925rem; }

/* ── Contributions ── */
.contribution {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent3);
  border-radius: var(--radius);
  padding: 16px 20px;
  margin-bottom: 12px;
}
.contribution .contrib-title { font-weight: 700; color: var(--accent3); margin-bottom: 6px; }
.contribution .contrib-why { font-size: 0.875rem; color: var(--text-muted); margin-top: 6px; }

/* ── Architecture figure from paper ── */
.figure-block {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 16px;
  text-align: center;
}
.arch-figure {
  max-width: 100%;
  height: auto;
  border-radius: 6px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.5);
}
.figure-caption {
  margin-top: 12px;
  color: var(--text-muted);
  font-size: 0.875rem;
  font-style: italic;
}

/* ── Architecture components ── */
.arch-component {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 20px;
  margin-bottom: 12px;
}
.arch-component h4 { color: var(--accent); font-size: 0.925rem; margin-bottom: 8px; }
.arch-component .tag {
  display: inline-block;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-right: 6px;
  margin-bottom: 4px;
  font-family: var(--font-mono);
}

/* ── Code / equations ── */
.equation {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--warn);
  border-radius: var(--radius);
  padding: 16px 20px;
  font-family: var(--font-mono);
  font-size: 0.9rem;
  overflow-x: auto;
  margin: 12px 0;
  white-space: pre;
}
code {
  background: var(--surface2);
  border-radius: 4px;
  padding: 2px 6px;
  font-family: var(--font-mono);
  font-size: 0.875em;
  color: var(--accent2);
}

/* ── Table ── */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
  margin: 14px 0;
}
th {
  background: var(--surface2);
  border: 1px solid var(--border);
  padding: 10px 14px;
  text-align: left;
  color: var(--text-muted);
  font-weight: 600;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
td {
  border: 1px solid var(--border);
  padding: 10px 14px;
  vertical-align: top;
}
tr:nth-child(even) td { background: var(--surface); }

/* ── Checklist ── */
.checklist { list-style: none; }
.checklist li { padding: 6px 0; font-size: 0.9rem; }
.checklist li::before { content: "☐ "; color: var(--accent); font-size: 1.1em; }

/* ── Badges ── */
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 600;
  margin-right: 6px;
}
.badge-year  { background: #1e2d4a; color: var(--accent2); }
.badge-venue { background: #2a1e3d; color: var(--accent); }
.badge-arxiv { background: #1e3a2d; color: var(--accent3); }

/* ── Footer ── */
footer {
  margin-top: 60px;
  padding-top: 20px;
  border-top: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 0.8rem;
  text-align: center;
}

/* ── Utility ── */
ul, ol { padding-left: 22px; }
li { margin-bottom: 4px; }
strong { color: #fff; }
.muted { color: var(--text-muted); }
.highlight { color: var(--accent2); }
.divider { border: none; border-top: 1px solid var(--border); margin: 24px 0; }
</style>
</head>
<body>

<!-- ══════════════════════════════════════ -->
<!--  HEADER                               -->
<!-- ══════════════════════════════════════ -->
<div class="paper-header">
  <h1>{PAPER TITLE}</h1>
  <div class="paper-meta">
    <span>👤 {AUTHORS}</span>
    <span class="badge badge-year">{YEAR}</span>
    <span class="badge badge-arxiv">arXiv:{ARXIV_ID}</span>
  </div>
  <div class="tldr">{ONE-LINE SUMMARY — "This paper proposes X that achieves Y by Z."}</div>
</div>

<!-- ══════════════════════════════════════ -->
<!--  TABLE OF CONTENTS                    -->
<!-- ══════════════════════════════════════ -->
<nav class="toc">
  <h3>Contents</h3>
  <ol>
    <li><a href="#problem">Problem Statement</a></li>
    <li><a href="#contributions">Core Contributions</a></li>
    <li><a href="#architecture">Model Architecture</a></li>
    <li><a href="#loss">Loss Design</a></li>
    <li><a href="#training">Training Strategy</a></li>
    <li><a href="#results">Key Results</a></li>
    <li><a href="#implementation">Implementation Notes</a></li>
  </ol>
</nav>

<!-- ══════════════════════════════════════ -->
<!--  1. PROBLEM STATEMENT                 -->
<!-- ══════════════════════════════════════ -->
<section id="problem">
  <div class="section-header">
    <div class="section-icon icon-problem">🎯</div>
    <h2>1. Problem Statement</h2>
  </div>

  <div class="card">
    <h3>Core Problem</h3>
    <p>{CORE PROBLEM DESCRIPTION}</p>
  </div>

  <div class="card">
    <h3>Prior Art Limitations</h3>
    <ul>
      <li>{LIMITATION 1}</li>
      <li>{LIMITATION 2}</li>
    </ul>
  </div>
</section>

<!-- ══════════════════════════════════════ -->
<!--  2. CONTRIBUTIONS                     -->
<!-- ══════════════════════════════════════ -->
<section id="contributions">
  <div class="section-header">
    <div class="section-icon icon-contrib">✨</div>
    <h2>2. Core Contributions</h2>
  </div>

  <div class="contribution">
    <div class="contrib-title">{CONTRIBUTION 1 NAME}</div>
    <div>{DESCRIPTION}</div>
    <div class="contrib-why">Why it matters: {IMPACT}</div>
  </div>

  <!-- repeat .contribution block for each contribution -->
</section>

<!-- ══════════════════════════════════════ -->
<!--  3. MODEL ARCHITECTURE                -->
<!-- ══════════════════════════════════════ -->
<section id="architecture">
  <div class="section-header">
    <div class="section-icon icon-arch">🏗️</div>
    <h2>3. Model Architecture</h2>
  </div>

  <!-- ① Paper figure (if extracted) — INSERT WHEN ARCH_B64 IS NOT NULL -->
  <!-- Replace the entire block below with actual base64 data when available -->
  <!-- IF ARCH_B64 IS NOT NULL:
  <div class="figure-block">
    <img src="data:image/png;base64,{ARCH_B64}"
         alt="Model Architecture — extracted from paper"
         class="arch-figure">
    <div class="figure-caption">{ARCH_CAPTION}</div>
  </div>
  -->
  <!-- IF ARCH_B64 IS NULL: omit the figure-block entirely -->

  <div class="card">
    <h3>High-Level Design</h3>
    <p>{OVERALL ARCHITECTURE DESCRIPTION}</p>
  </div>

  <!-- ASCII diagram as fallback / supplement -->
  <div class="equation">{ASCII ARCHITECTURE DIAGRAM}</div>

  <!-- One .arch-component per key component -->
  <div class="arch-component">
    <h4>{COMPONENT NAME}</h4>
    <p>{DESCRIPTION}</p>
    <div>
      <span class="tag">input: {SHAPE}</span>
      <span class="tag">output: {SHAPE}</span>
    </div>
  </div>
</section>

<!-- ══════════════════════════════════════ -->
<!--  4. LOSS DESIGN                       -->
<!-- ══════════════════════════════════════ -->
<section id="loss">
  <div class="section-header">
    <div class="section-icon icon-loss">📐</div>
    <h2>4. Loss Design</h2>
  </div>

  <div class="card">
    <h3>Primary Loss</h3>
    <div class="equation">{LOSS EQUATION}</div>
    <p>{EXPLANATION OF EACH TERM}</p>
  </div>

  <!-- repeat card for each auxiliary loss -->

  <div class="card">
    <h3>Loss Weighting</h3>
    <p>{WEIGHTING STRATEGY}</p>
  </div>
</section>

<!-- ══════════════════════════════════════ -->
<!--  5. TRAINING STRATEGY                 -->
<!-- ══════════════════════════════════════ -->
<section id="training">
  <div class="section-header">
    <div class="section-icon icon-train">⚙️</div>
    <h2>5. Training Strategy</h2>
  </div>

  <table>
    <tr><th>Aspect</th><th>Detail</th></tr>
    <tr><td>Dataset</td><td>{DATASET}</td></tr>
    <tr><td>Optimizer</td><td>{OPTIMIZER}</td></tr>
    <tr><td>Learning Rate</td><td>{LR + SCHEDULE}</td></tr>
    <tr><td>Batch Size</td><td>{BATCH SIZE}</td></tr>
    <tr><td>Epochs / Steps</td><td>{VALUE}</td></tr>
    <tr><td>Hardware</td><td>{GPUs, time}</td></tr>
    <tr><td>Regularization</td><td>{DROPOUT, WEIGHT DECAY}</td></tr>
    <tr><td>Key Tricks</td><td>{WARMUP, CLIPPING, EMA, etc.}</td></tr>
  </table>
</section>

<!-- ══════════════════════════════════════ -->
<!--  6. KEY RESULTS                       -->
<!-- ══════════════════════════════════════ -->
<section id="results">
  <div class="section-header">
    <div class="section-icon icon-results">📊</div>
    <h2>6. Key Results</h2>
  </div>

  <table>
    <tr><th>Benchmark</th><th>Metric</th><th>This Paper</th><th>Prior SOTA</th><th>Delta</th></tr>
    <!-- fill rows from innovations.md -->
  </table>

  <div class="card" style="margin-top:16px;">
    <h3>Most Important Findings</h3>
    <ul>
      <li>{FINDING 1}</li>
      <li>{FINDING 2}</li>
    </ul>
  </div>

  <div class="card">
    <h3>Ablation Insights</h3>
    <p>{WHAT ABLATIONS REVEAL}</p>
  </div>
</section>

<!-- ══════════════════════════════════════ -->
<!--  7. IMPLEMENTATION NOTES              -->
<!-- ══════════════════════════════════════ -->
<section id="implementation">
  <div class="section-header">
    <div class="section-icon icon-impl">🔧</div>
    <h2>7. Implementation Notes</h2>
  </div>

  <div class="card">
    <h3>Critical Details</h3>
    <p>{EASY-TO-MISS DETAILS}</p>
  </div>

  <div class="card">
    <h3>Reproduction Checklist</h3>
    <ul class="checklist">
      <li>{ITEM}</li>
      <li>{ITEM}</li>
    </ul>
  </div>
</section>

<footer>
  Generated by auto-research · {DATE} · <a href="../../analyses/{NAME}/innovations.md" style="color:var(--text-muted)">raw analysis</a>
</footer>

</body>
</html>
```

Populate **every** placeholder with actual content from `analyses/{name}/innovations.md`. Do not leave any `{PLACEHOLDER}` text in the final output. Write the complete file to `outputs/{name}/summary.html`.

**Step 4 — Generate `outputs/{name}/model.py`.**

Check if `reproductions/{name}/model.py` already exists:
- **If yes**: copy its content to `outputs/{name}/model.py` (read and write)
- **If no**: generate a standalone PyTorch `model.py` from scratch based solely on Section 3 (Model Architecture) of `analyses/{name}/innovations.md`. This file must be complete and runnable — do not depend on any other files in `reproductions/`.

The `outputs/{name}/model.py` must be self-contained: include all imports, the `ModelConfig` dataclass, all sub-modules, and the top-level model class. Add a `if __name__ == "__main__":` block that instantiates the model and prints parameter count + a sample forward pass shape.

**Step 5 — Generate training and evaluation visualisations.**

Check whether training logs exist for this paper:

```bash
find logs/{name} -name "metrics.jsonl" | sort | tail -1
find logs/{name} -name "test_results.json" | sort | tail -1
```

- **If `metrics.jsonl` found**: run `python3 scripts/generate_viz.py --log-dir <that_dir> --output-dir outputs/{name}` to generate `train.html` and `evaluate.html`.
- **If `test_results.json` not found**: note that `evaluate.html` will be skipped; the user should run `python test.py` in `reproductions/{name}/` first, then re-run `/generate-report {name}`.
- **If no logs at all**: skip this step and note it in the confirmation output.

**Step 6 — Confirm.**

Run `find outputs/{name} -type f | sort` to list output files.

Print:
```
✓ Report generated
  outputs/{name}/
  ├── summary.html    innovation analysis (self-contained dark-theme HTML)
  ├── model.py        standalone PyTorch model
  ├── train.html      training curves (loss, AUC, LR)         [if logs exist]
  └── evaluate.html   ROC/PR curves, confusion matrix, metrics [if test_results.json exists]

  Open in browser:
    open outputs/{name}/summary.html
    open outputs/{name}/train.html
    open outputs/{name}/evaluate.html
```
