# auto-research

Full pipeline: parse paper → analyze innovations → generate PyTorch reproduction code → generate HTML report.
Runs all four stages sequentially with progress tracking.

## Arguments

$ARGUMENTS

First argument: PDF file path OR arxiv URL
Second argument (optional): paper name slug. If omitted, derived from arxiv ID or filename.

## Steps

**Step 1 — Initialize session.**

Print the pipeline header:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AUTO RESEARCH PIPELINE
 Input: {arg1}
 Name:  {resolved_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/4] Parsing paper...
```

---

**Step 2 — Parse paper.**

Execute the full logic of the `/parse-paper` skill inline:

- Run `python3 scripts/fetch_paper.py "{arg1}" "{arg2}"` to fetch/copy the paper
- Run `python3 scripts/parse_pdf.py "papers/{name}.pdf"` to extract text
- Run `python3 scripts/extract_figures.py "papers/{name}.pdf" "analyses/{name}/"` to extract architecture figures (sets `arch_figure` in raw.md; skip gracefully if PyMuPDF not installed)
- Write `analyses/{name}/raw.md` following the raw.md schema

Print on completion:
```
✓ [1/4] Parse complete → analyses/{name}/raw.md

[2/4] Analyzing innovations...
```

---

**Step 3 — Analyze innovations.**

Execute the full logic of the `/analyze-innovations` skill inline:

- Read `analyses/{name}/raw.md`, `prompts/innovations_system.md`, and `prompts/math_specialist_system.md`
- Search raw.md for any `github.com` URL — record as the official repo URL (or "not found")
- Run detection scan to determine which sections are present (`sections_detected` MANIFEST)
- **Round 1 (Math Specialist)**: if theoretical content detected, extract T.1–T.4 as `{THEORY_DRAFT}` using `math_specialist_system.md` — notation fidelity pass, no architectural interpretation
- **Round 2 (Systems Analyst)**: full analysis pass — architecture, loss, training, efficiency; write T.5 (Theory→Design Connection) by synthesizing THEORY_DRAFT with architecture findings
- Write `analyses/{name}/innovations.md`

Print on completion:
```
✓ [2/4] Analysis complete → analyses/{name}/innovations.md

[3/4] Generating PyTorch reproduction...
```

---

**Step 4 — Generate PyTorch code.**

Execute the full logic of the `/reproduce-code` skill inline. Framework is always PyTorch — do not use JAX or any other framework.

- Check innovations.md Section 0 for GitHub URL; if present, run `python3 scripts/fetch_repo.py {url} analyses/{name}/` and use official model/loss as the base
- Read `analyses/{name}/innovations.md` and `prompts/reproduce_system.md`
- Create `outputs/{name}/reproduction/` directory
- Generate all 7 files: config.py, model.py, loss.py, dataset.py, train.py, test.py, README.md

Print on completion:
```
✓ [3/4] Code generated → outputs/{name}/reproduction/

[4/4] Generating HTML report...
```

---

**Step 5 — Generate HTML report.**

Execute the full logic of the `/generate-report` skill inline:

- Read `analyses/{name}/innovations.md` and `prompts/html_report_system.md`
- Run `mkdir -p outputs/{name}/html outputs/{name}/reproduction`
- Write `outputs/{name}/html/summary.html` — complete self-contained dark-theme HTML
- The full reproduction code (config.py, model.py, loss.py, dataset.py, train.py, test.py, README.md) is already in `outputs/{name}/reproduction/` from Stage 3

Print on completion:
```
✓ [4/4] Report generated → outputs/{name}/
```

---

**Step 6 — Final summary.**

Print the complete summary:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PIPELINE COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Paper:   {full title}
Authors: {authors}
Year:    {year}

OUTPUTS
  papers/{name}.pdf
  analyses/{name}/raw.md
  analyses/{name}/innovations.md
  outputs/{name}/
    ├── html/
    │   ├── summary.html     ← open in browser
    │   ├── train.html       ← training curves [if logs exist]
    │   └── evaluate.html    ← ROC/PR/confusion [if test_results.json exists]
    └── reproduction/
        ├── config.py
        ├── model.py
        ├── loss.py
        ├── dataset.py
        ├── train.py
        ├── test.py
        └── README.md

KEY INNOVATIONS
{3-5 bullet points summarizing the paper's contributions}

ARCHITECTURE
{1-2 sentence description of model structure}

TO BROWSE THE REPORT
  open outputs/{name}/html/summary.html

TO RUN THE REPRODUCTION
  cd outputs/{name}/reproduction
  pip install torch  # + any extra deps in README.md
  python train.py

IF TRAINING FAILS (auto-fix runtime errors)
  /fix-reproduction {name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
