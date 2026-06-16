# HTML Report Generation — Design Rules

You are generating a self-contained HTML research summary report. Follow these rules strictly.

## Visual Design

- **Dark theme only**: background `#0f1117`, surface `#1a1d27`, text `#e2e4f0`
- **Accent colors**: purple `#7c6af5` (primary), cyan `#4fc3f7` (highlight), green `#69f0ae` (success/contributions), amber `#ffb74d` (equations)
- **No external dependencies**: all CSS inline in `<style>`, no CDN, no Google Fonts fallback is fine (monospace stack covers equations)
- **Single scrollable page**: no iframes, no JavaScript required for reading

## Content Rules

### Dynamic Section Rendering
- Read the `sections_detected` dict from `innovations.md` YAML frontmatter
- **Only render HTML sections whose key is `true`** — absent sections are completely omitted
- Never render an empty section with placeholder text or "N/A" — omit the entire `<section>` block
- The TOC `<ol>` must list only the sections that were actually rendered
- The `<div class="profile-bar">` must show section chips: `chip-on` for present, `chip-off` for absent

### Completeness (for present sections)
- Every present section from `innovations.md` must appear in the HTML — no truncation
- Tables in innovations.md must become HTML `<table>` elements
- Loss equations must go in `<div class="equation">` blocks (monospace, amber border)

### Theoretical Analysis Section (if `theoretical_analysis = true`)
- Use `<div class="theory-claim">` for the core claim/theorem
- Use `<div class="derivation">` (cyan left border, monospace, pre) for the derivation steps
- Use `<ul class="assumption-list">` for assumptions (⟹ prefix)
- The "Theory → Design Connection" card is the most important — don't omit or shorten it
- If the theory is informal (no formal proof): label clearly with "Key Insight (informal):"

### Efficiency Analysis Section (if `efficiency_analysis = true`)
- Use `<div class="efficiency-grid">` with four `<div class="efficiency-card">` cards
- Cards: Complexity, Parameters, Throughput, GPU Memory
- If any metric is not reported in the paper: omit that specific card (not the whole section)
- Use `var(--accent3)` (green) for the numeric values to signal positive/favorable numbers

### Accuracy
- Copy numbers and equations exactly — do not paraphrase metrics or loss terms
- Paper title, authors, year, arxiv ID must match the raw.md frontmatter exactly
- The TL;DR sentence must match Section 1.3 "Proposed Solution" from innovations.md

### Architecture Figure (from paper PDF)

The `arch_figure` field in `analyses/{name}/raw.md` frontmatter (or from `extract_figures.py` output) may provide a `b64` base64-encoded PNG of the actual architecture diagram from the paper.

**If `b64` is available** — embed it as the first element of the architecture section:
```html
<div class="figure-block">
  <img src="data:image/png;base64,{ARCH_B64}"
       alt="Model Architecture — extracted from paper"
       class="arch-figure">
  <div class="figure-caption">{ARCH_CAPTION}</div>
</div>
```

**If `b64` is null** — omit the `figure-block` entirely; fall back to ASCII diagram only.

The `.figure-block` CSS is already in the template (`outputs/_template/summary.html`). Use it as-is — do not inline different styles.

### Architecture Diagram (ASCII supplement)
- After the figure (or in its absence), convert the text diagram from innovations.md Section 3.3 into ASCII art inside a `<div class="equation">` block
- If no diagram exists in innovations.md, construct one based on the component descriptions

### Model Components
- Each component from innovations.md Section 3.2 gets one `.arch-component` div
- Include input/output tensor shapes as `.tag` spans where stated
- Mark components from prior work: `<span class="badge badge-venue">from Vaswani 2017</span>`

### Results Table
- Fill every row from innovations.md Section 6.1
- If the paper improves on SOTA, add `color: var(--accent3)` style to the Delta cell
- If it does not beat SOTA (shows baselines only), note that in a `<p class="muted">` below the table

### Implementation Checklist
- Use `<ul class="checklist">` — each `<li>` is one verifiable reproduction step
- Items must be specific: "Initialize projection matrices with kaiming_uniform" not "check initialization"

## Self-Containment Check
Before writing the file, verify mentally:
1. No `<link>` to external CSS ✓
2. No `<script src>` ✓  
3. No image `src` pointing to external URLs — base64 `data:` URIs are allowed ✓
4. All `{PLACEHOLDER}` text replaced with real content ✓
5. File opens and renders in an offline browser ✓
6. If `ARCH_B64` is null, no broken `<img>` tag in the architecture section ✓
7. TOC only lists sections that exist in the body ✓
8. No empty `<section>` blocks for absent sections ✓
9. `sections_detected = false` sections have no HTML at all — not even an empty header ✓
