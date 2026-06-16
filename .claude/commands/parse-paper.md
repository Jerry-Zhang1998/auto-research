# parse-paper

Parse an academic paper from a local PDF or arxiv URL into a structured markdown document.

## Arguments

$ARGUMENTS

First argument: PDF file path OR arxiv URL (e.g. `https://arxiv.org/abs/1706.03762` or `./papers/foo.pdf`)
Second argument (optional): short paper name slug (e.g. `attention-is-all-you-need`). If omitted, derive from filename or arxiv ID.

## Steps

**Step 1 — Resolve the paper source.**

Run `python3 scripts/fetch_paper.py "$ARG1" "$ARG2"`. This script will:
- If the input is an arxiv URL or ID: download the PDF to `papers/{name}.pdf` and print metadata (title, authors, year, abstract)
- If the input is a local PDF path: copy it to `papers/{name}.pdf` and print its path

Read the script output to get the resolved `{name}`, `{pdf_path}`, and the
authoritative `{title}` / `{authors}` / `{year}` (for arxiv inputs these come from
arxiv metadata and are more reliable than text-derived guesses).

**Step 2 — Extract paper text.**

Run `python3 scripts/parse_pdf.py papers/{name}.pdf` to extract the full text with section detection. The script prints JSON with keys: `title`, `abstract`, `sections` (list of `{heading, text}`), `figures`, `tables`, `references_count`.

**Step 3 — Structure the raw document.**

Create `analyses/{name}/` directory if it doesn't exist.

Read the prompt template from `prompts/parse_system.md` for guidance on section extraction.

Write `analyses/{name}/raw.md` following this exact schema.

**Title/authors/year source rule**: for arxiv inputs use the values from
`fetch_paper.py` (Step 1) — they are authoritative. Only fall back to
`parse_pdf.py`'s text-derived `title` for local PDFs where no metadata exists.

```
---
title: {full paper title}
authors: {author list}
year: {year}
arxiv: {arxiv ID or "local"}
pdf: papers/{name}.pdf
parsed: {today's date}
---

# {title}

## Abstract
{abstract text}

## 1. Introduction / Problem Statement
{extracted text}

## 2. Related Work
{extracted text — summarize if very long}

## 3. Method / Proposed Approach
{extracted text — preserve all equations, algorithm descriptions, and architectural details verbatim}

## 4. Model Architecture
{extracted architectural details — if merged with Method, extract and copy the architecture-specific content here}

## 5. Loss Functions & Training Objectives
{all loss equations and descriptions — preserve LaTeX notation}

## 6. Experiments & Results
{datasets, baselines, metrics, key numbers}

## 7. Ablation Studies
{if present}

## 8. Conclusion
{extracted text}

## Raw Notes
{any figures described, table captions, key equations not captured above}
```

If a section is absent in the paper, write `{not present}`.

**Step 4 — Extract figures.**

Run `python3 scripts/extract_figures.py papers/{name}.pdf analyses/{name}/`.

The script outputs JSON with:
- `total` — number of figures found
- `figures` — list of `{index, file, page, width, height, caption}`
- `arch_figure` — the likely architecture diagram: `{file, caption, b64_file, has_b64}`. The base64 PNG is written to the `b64_file` sidecar, NOT included in stdout.

Add to the frontmatter of `analyses/{name}/raw.md` (update the file):
```yaml
figures_dir: analyses/{name}/figures/
arch_figure: analyses/{name}/figures/{arch_figure_filename}
arch_caption: {arch_figure_caption}
figure_count: {total}
```

**Critical — never embed the base64 blob in raw.md.** Store only the figure
*path* and *caption*. The base64 string (often 150K+ tokens) makes raw.md
unreadable and is re-ingested by every downstream skill. The PNG is read directly
from disk by `/generate-report` at HTML-build time — it does not need to live in
raw.md. The `extract_figures.py` output writes b64 to a sidecar file, not stdout.

If PyMuPDF is not installed, skip this step and note it in the confirm output.

**Step 5 — Confirm.**

Print a summary:
```
✓ Parsed: {title}
  PDF      → papers/{name}.pdf
  Raw      → analyses/{name}/raw.md
  Figures  → analyses/{name}/figures/ ({N} extracted, arch: fig_XXX.png)
  Sections found: {list section headings}
  Next: /analyze-innovations {name}
```
