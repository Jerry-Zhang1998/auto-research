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
- Write `analyses/{name}/raw.md` following the raw.md schema

Print on completion:
```
✓ [1/4] Parse complete → analyses/{name}/raw.md

[2/4] Analyzing innovations...
```

---

**Step 3 — Analyze innovations.**

Execute the full logic of the `/analyze-innovations` skill inline:

- Read `analyses/{name}/raw.md` and `prompts/innovations_system.md`
- Search raw.md for any `github.com` URL — record as the official repo URL (or "not found")
- Produce the complete structured innovation analysis including Section 0 (Repository)
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
- Create `reproductions/{name}/` directory
- Generate all 7 files: config.py, model.py, loss.py, dataset.py, train.py, test.py, README.md

Print on completion:
```
✓ [3/4] Code generated → reproductions/{name}/

[4/4] Generating HTML report...
```

---

**Step 5 — Generate HTML report.**

Execute the full logic of the `/generate-report` skill inline:

- Read `analyses/{name}/innovations.md` and `prompts/html_report_system.md`
- Run `mkdir -p outputs/{name}`
- Write `outputs/{name}/summary.html` — complete self-contained dark-theme HTML
- Copy `reproductions/{name}/model.py` → `outputs/{name}/model.py`

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
  reproductions/{name}/
    ├── config.py
    ├── model.py
    ├── loss.py
    ├── dataset.py
    ├── train.py
    └── README.md
  outputs/{name}/
    ├── summary.html     ← open in browser
    └── model.py         ← standalone PyTorch model

KEY INNOVATIONS
{3-5 bullet points summarizing the paper's contributions}

ARCHITECTURE
{1-2 sentence description of model structure}

TO BROWSE THE REPORT
  open outputs/{name}/summary.html

TO RUN THE REPRODUCTION
  cd reproductions/{name}
  pip install torch  # + any extra deps in README.md
  python train.py

IF TRAINING FAILS (auto-fix runtime errors)
  /fix-reproduction {name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
