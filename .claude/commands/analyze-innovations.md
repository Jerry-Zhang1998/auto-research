# analyze-innovations

Extract and structure the core innovations from a parsed paper into a detailed analysis document.

## Arguments

$ARGUMENTS

First argument: paper name slug (e.g. `attention-is-all-you-need`). Must match a folder in `analyses/`.

## Steps

**Step 1 — Load the parsed paper.**

Read `analyses/{name}/raw.md`. If the file does not exist, tell the user to run `/parse-paper` first.

Also read the analysis prompt schema from `prompts/innovations_system.md`.

**Step 2 — Deep analysis.**

Carefully read the entire raw.md. Then produce `analyses/{name}/innovations.md` with the following structure (do not skip any section — write "N/A" only if truly absent):

```markdown
---
paper: {title}
name: {name}
analyzed: {today's date}
---

# Innovation Analysis: {title}

## 1. Problem Statement

### 1.1 Core Problem
{What specific problem or limitation does this paper address? Be precise — quote the paper's own framing.}

### 1.2 Prior Art Limitations
{What are the key limitations of existing approaches that motivated this work?}

### 1.3 Proposed Solution (one-line)
{Single sentence: "This paper proposes X that achieves Y by doing Z."}

---

## 2. Core Contributions

List each contribution as a numbered item. For each:
- **Contribution**: {name}
- **What it is**: {description}
- **Why it matters**: {novelty and impact}

---

## 3. Model Architecture

### 3.1 High-Level Design
{Overall architecture description — what goes in, what comes out, how data flows}

### 3.2 Key Components
For each major component:

#### {Component Name}
- **Role**: {what it does in the model}
- **Design**: {how it works — equations if applicable}
- **Input/Output shape**: {tensor dimensions if specified}
- **Innovation**: {what's novel about this component vs prior work}

### 3.3 Architecture Diagram (text)
{ASCII or text description of the data flow and component connections}

---

## 4. Loss Design

### 4.1 Primary Loss
```
L_total = {equation}
```
{Explanation of each term}

### 4.2 Auxiliary Losses
{Each auxiliary loss with equation and purpose}

### 4.3 Loss Weighting Strategy
{How losses are balanced — fixed weights, scheduled, learned}

### 4.4 Training Objectives Summary
{Why this loss design was chosen — what behavior it incentivizes}

---

## 5. Training Strategy

| Aspect | Detail |
|--------|--------|
| Dataset | {name, size, preprocessing} |
| Optimizer | {Adam/SGD/etc with β values} |
| Learning Rate | {value, schedule} |
| Batch Size | {value} |
| Epochs/Iterations | {value} |
| Hardware | {GPUs, training time if stated} |
| Regularization | {dropout, weight decay, etc.} |
| Data Augmentation | {if applicable} |
| Key Training Tricks | {warmup, gradient clipping, EMA, etc.} |

---

## 6. Key Results

### 6.1 Main Benchmarks
| Benchmark | Metric | This Paper | Prior SOTA | Improvement |
|-----------|--------|------------|------------|-------------|
{fill from experiments section}

### 6.2 Most Important Findings
{3-5 bullet points of the most significant experimental findings}

### 6.3 Ablation Insights
{What the ablations reveal about which components matter most}

---

## 7. Implementation Notes for Reproduction

### 7.1 Critical Details
{Implementation details that are easy to miss but crucial for reproduction — initialization schemes, specific activation functions, normalization placement, etc.}

### 7.2 Potential Pitfalls
{Common failure modes or tricky parts based on the paper's own discussion}

### 7.3 Hyperparameter Sensitivity
{Which hyperparameters the paper identifies as most important}

### 7.4 Reproduction Checklist
- [ ] {specific thing to verify}
- [ ] {specific thing to verify}
...

---

## 8. Paper Significance

### 8.1 Why This Paper Matters
{1 paragraph — the broader impact on the field}

### 8.2 Limitations Acknowledged
{What the authors themselves acknowledge as limitations}

### 8.3 Follow-up Directions
{Research directions the paper suggests or that naturally follow}
```

**Step 3 — Confirm.**

Print:
```
✓ Analyzed: {title}
  Output  → analyses/{name}/innovations.md
  Contributions: {count}
  Components: {list key model components}
  Next: /reproduce-code {name}
```
