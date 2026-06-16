# analyze-innovations

Extract and structure the core innovations from a parsed paper into a detailed analysis document.
Sections are generated conditionally — only those present in the paper are written.

## Arguments

$ARGUMENTS

First argument: paper name slug (e.g. `attention-is-all-you-need`). Must match a folder in `analyses/`.

## Steps

---

**Step 1 — Load the parsed paper.**

Read `analyses/{name}/raw.md`. If the file does not exist, tell the user to run `/parse-paper` first.

Also read:
- `prompts/innovations_system.md` — analysis principles and calibration guidance
- `analyses/_template/innovations.md` — output schema and section structure

---

**Step 1b — Search for official GitHub repository.**

Scan the full text of `analyses/{name}/raw.md` for any line containing `github.com`.

Extract all GitHub URLs (pattern: `https?://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+`).

Prefer URLs near phrases like "code available", "implementation", "official", "our code", "released at".

Record as `{GITHUB_URL}` (or "not found" if none).

---

**Step 1c — Detection scan (Paper Profile).**

Before writing anything, quickly scan `raw.md` and decide which analytical dimensions are present.
Answer YES / NO for each based on the detection indicators in `prompts/innovations_system.md`:

| Dimension | Indicator to look for |
|-----------|----------------------|
| `theoretical_analysis` | theorem/proof/lemma/convergence/bound/derivation/"we prove"/"we show" |
| `model_architecture` | named components, architecture diagrams, tensor shapes, "we propose a [module/block/layer]" |
| `loss_design` | new loss equation (L=...), multi-term objective, novel training objective |
| `training_strategy` | non-standard optimizer choices, curriculum, multi-stage, training tricks as contribution |
| `efficiency_analysis` | FLOPs count, "X× faster", O(n) complexity claim, latency/throughput numbers |
| `ablation_study` | "ablation", table of model variants, "effect of", "w/ or w/o" component |
| `implementation_notes` | "implementation details" section, specific initialization, engineering choices |

Record as `{MANIFEST}`:
```json
{
  "theoretical_analysis": true/false,
  "model_architecture": true/false,
  "loss_design": true/false,
  "training_strategy": true/false,
  "efficiency_analysis": true/false,
  "ablation_study": true/false,
  "implementation_notes": true/false
}
```

Also determine:
- `{PAPER_TYPE}`: Architecture | Theory | Empirical | System | Benchmark | Survey
- `{PRIMARY_DOMAIN}`: Vision | NLP | RL | Optimization | Multimodal | Graph | Audio | Other

---

**Step 1d — Round 1: Math Specialist Pass.**

**Run only if `theoretical_analysis = true` in `{MANIFEST}`. Otherwise set `{THEORY_DRAFT}` = null and skip to Step 2.**

Read `prompts/math_specialist_system.md` now.

Then locate the math-dense portions of `raw.md`:
- Sections titled "Analysis", "Theory", "Theoretical Analysis", "Method" that contain multi-line equations
- All paragraphs containing theorem/proposition/lemma/corollary labels
- Any appendix sections with derivations or proofs

Read those portions with the focus and rules in `math_specialist_system.md`:
- Your only goal in this pass is faithful mathematical extraction — not understanding what it means for architecture
- Produce `{THEORY_DRAFT}` in the T.1–T.5 structure defined in `math_specialist_system.md`
- Leave T.5 as the placeholder `[TO BE COMPLETED IN ROUND 2 — requires architecture analysis context]`
- Do not analyze architecture, loss, or training during this pass

**If `theoretical_analysis = true` but the paper's mathematical content is entirely informal** (no equations, no derivation steps): still produce a THEORY_DRAFT marking `Nature: informal argument` and use T.3 to capture the verbal logical chain instead of equations. Set T.4 to "None stated."

---

**Step 2 — Round 2: Systems Analyst Deep Analysis.**

Carefully read all of `raw.md`. Then produce `analyses/{name}/innovations.md`.

**Critical rule**: only write sections whose key in `{MANIFEST}` is `true`. Do NOT write the header for absent sections. Do NOT write "N/A" placeholders — absent sections are simply not present in the file.

**Round 2 role**: this is the Systems Analyst pass. Your focus is architecture, loss, training, and implementation. For the Theoretical Analysis section (when `{THEORY_DRAFT}` is not null):
- **T.1–T.4**: start from `{THEORY_DRAFT}`. Verify each field against `raw.md`; correct notation errors or missed steps, but do not simplify. Copy the content in.
- **T.5 (Theory → Design Connection)**: write this now with full cross-pass awareness. You have just analyzed the architecture (Section 3) and loss (Section 4) — explain specifically how the theoretical result from T.2/T.3 directly justifies those design choices. This is the synthesis only Round 2 can produce. Be concrete: name the design decision and cite the theorem/equation that motivates it.

The always-present sections are: Repository (0), Problem Statement (1), Core Contributions (2), Key Results (6), Paper Significance (8), and the Paper Profile header.

Write the file following this schema:

```markdown
---
paper: {title}
name: {name}
analyzed: {today's date}
github: {GITHUB_URL}
paper_type: {PAPER_TYPE}
sections_detected:
  theoretical_analysis: {true/false}
  model_architecture: {true/false}
  loss_design: {true/false}
  training_strategy: {true/false}
  efficiency_analysis: {true/false}
  ablation_study: {true/false}
  implementation_notes: {true/false}
---

# Innovation Analysis: {title}

## Paper Profile

**Type**: {PAPER_TYPE}
**Domain**: {PRIMARY_DOMAIN}
**Primary Contribution**: {one-phrase summary}

Sections present in this analysis:
- [x] Problem Statement
- [{x or space}] Theoretical Analysis{if false: — not present (empirical paper / no formal derivations)}
- [x] Core Contributions
- [{x or space}] Model Architecture{if false: — not present}
- [{x or space}] Loss Design{if false: — not present (uses standard objective)}
- [{x or space}] Training Strategy{if false: — not present (standard training)}
- [{x or space}] Efficiency Analysis{if false: — not present}
- [x] Key Results
- [{x or space}] Ablation Study{if false: — not present}
- [{x or space}] Implementation Notes{if false: — not present}
- [x] Paper Significance

---

## 0. Repository

- **GitHub**: {GITHUB_URL}
- **Status**: official | not-found
- **Notes**: {any relevant info, or "N/A"}

---

## 1. Problem Statement

### 1.1 Core Problem
{Quote the paper's exact framing where possible. What specific limitation or gap is addressed?}

### 1.2 Prior Art Limitations
{What are the key limitations of existing approaches? Be specific — not "slow" but "O(n²) attention prevents context windows beyond 4K tokens"}

### 1.3 Proposed Solution (one-line)
{Single sentence: "This paper proposes X that achieves Y by doing Z."}

---

[IF theoretical_analysis = true]
## Theoretical Analysis

### T.1 Theoretical Framework
{What theoretical basis: variational inference, information theory, PAC-learning, optimal transport,
spectral analysis, Bayesian framework, etc. What does this framework provide?}

### T.2 Core Claim / Main Result
{The central theorem, proposition, or theoretical claim. Quote from paper directly if possible.
Label clearly: "Theorem 1 (informal):", "Proposition 2:", "Key insight (no formal proof):"}

### T.3 Key Derivation Steps
{The essential mathematical steps — preserve the paper's notation exactly.
2-4 most critical steps that form the logical chain.}
```
Starting point: {equation or assumption}
   ↓  {technique applied: Jensen's inequality / substitution / integration / ...}
{intermediate form}
   ↓  {next step}
Core result: {final equation}
```

### T.4 Assumptions & Conditions
{What must hold for the theoretical result to apply?
e.g., i.i.d. samples, Lipschitz-continuous gradients, specific distribution family.
Note which assumptions are standard vs which are restrictive.}

### T.5 Theory → Design Connection
{The "so what": how does this theoretical result directly justify the model design, loss choice, or training procedure?
This is what a practitioner needs to know to understand WHY the paper's choices are principled.}

---
[END IF]

## 2. Core Contributions

List each contribution. For each:
- **Contribution**: {name}
- **What it is**: {description}
- **Why it matters**: {novelty and impact vs prior work}

---

[IF model_architecture = true]
## 3. Model Architecture

### 3.1 High-Level Design
{Overall architecture — inputs, outputs, data flow}

### 3.2 Key Components

#### {Component Name}
- **Role**: {what it does}
- **Design**: {how it works, equations if applicable}
- **Input/Output shape**: {tensor dimensions}
- **Innovation**: {what's novel vs prior work}
- **Prior work**: {if from another paper: "standard X from Author et al. YEAR"}

### 3.3 Architecture Diagram (text)
{ASCII diagram of data flow and component connections}

---
[END IF]

[IF loss_design = true]
## 4. Loss Design

### 4.1 Primary Loss
```
L_total = {equation — preserve paper notation}
```
{Explanation of each term: what it penalizes, why}

### 4.2 Auxiliary Losses
{Each loss: equation, purpose, when it's applied}

### 4.3 Loss Weighting Strategy
{Fixed weights / scheduled / learned. Values from paper.}

### 4.4 Training Objectives Summary
{Why this loss design incentivizes the desired behavior}

---
[END IF]

[IF training_strategy = true]
## 5. Training Strategy

| Aspect | Detail |
|--------|--------|
| Dataset | {name, size, preprocessing} |
| Optimizer | {type, β₁, β₂, ε} |
| Learning Rate | {value, schedule — warmup steps, decay} |
| Batch Size | {value, accumulation steps if applicable} |
| Epochs / Iterations | {value} |
| Hardware | {GPU type, count, training time} |
| Regularization | {dropout rate, weight decay λ} |
| Data Augmentation | {if applicable} |
| Key Tricks | {warmup, gradient clipping, EMA factor, stop-gradient, etc.} |

---
[END IF]

[IF efficiency_analysis = true]
## Efficiency Analysis

### E.1 Computational Complexity
{Time complexity of key operations with big-O notation.
What is the bottleneck? How does the paper improve it?}

### E.2 Parameter Count
{Total parameters. Direct comparison to baselines from the paper.}

### E.3 Inference Speed / Throughput
{Latency, FPS, tokens/sec, or explicit speedup factor. Hardware context required.}

### E.4 Memory Footprint
{Peak GPU memory. Impact on max feasible batch size.}

### E.5 Scalability
{How performance scales with data size, model size, or compute. Scaling law results if reported.}

---
[END IF]

## 6. Key Results

### 6.1 Main Benchmarks
| Benchmark | Metric | This Paper | Prior SOTA | Improvement |
|-----------|--------|------------|------------|-------------|
{fill every row from the paper's main results table}

### 6.2 Most Important Findings
{3-5 bullet points of the most significant findings}

### 6.3 Ablation Insights
{If ablation_study = true: what each ablation reveals about which components matter.}
{If ablation_study = false: write "No ablation study present in this paper."}

---

[IF implementation_notes = true]
## 7. Implementation Notes

### 7.1 Critical Details
{Details easy to miss but crucial for reproduction: initialization, norm placement, activation choice, etc.}

### 7.2 Potential Pitfalls
{Common failure modes based on the paper's discussion}

### 7.3 Hyperparameter Sensitivity
{Which hyperparameters the paper identifies as most important / most sensitive}

### 7.4 Reproduction Checklist
- [ ] {specific verifiable item}
- [ ] {specific verifiable item}

---
[END IF]

## 8. Paper Significance

### 8.1 Why This Paper Matters
{1 paragraph — broader impact on the field}

### 8.2 Limitations Acknowledged
{What the authors themselves acknowledge}

### 8.3 Follow-up Directions
{Research directions the paper suggests or that naturally follow}
```

---

**Step 3 — Confirm.**

Print:
```
✓ Analyzed: {title}
  Output      → analyses/{name}/innovations.md
  GitHub      → {GITHUB_URL or "not found"}
  Paper type  → {PAPER_TYPE}
  Sections written:
    [x] Problem Statement
    [{x/ }] Theoretical Analysis
    [x] Core Contributions
    [{x/ }] Model Architecture
    [{x/ }] Loss Design
    [{x/ }] Training Strategy
    [{x/ }] Efficiency Analysis
    [x] Key Results  (ablation: {yes/no})
    [{x/ }] Implementation Notes
    [x] Paper Significance
  Next: /reproduce-code {name}
```
