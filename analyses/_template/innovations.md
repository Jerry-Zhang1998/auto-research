---
paper: {title}
name: {name}
analyzed: {YYYY-MM-DD}
github: {https://github.com/... or "not found"}
---

# Innovation Analysis: {title}

## 0. Repository

- **GitHub**: {https://github.com/... or "not found"}
- **Status**: official | not-found
- **Notes**: {branch, key files, extra dependencies, or "N/A"}

---

## 1. Problem Statement

### 1.1 Core Problem
{What specific problem does this paper address?}

### 1.2 Prior Art Limitations
{Key limitations of existing approaches}

### 1.3 Proposed Solution (one-line)
{Single sentence: "This paper proposes X that achieves Y by doing Z."}

---

## 2. Core Contributions

1. **{Contribution}**: {description} — {why it matters}
2. **{Contribution}**: {description} — {why it matters}

---

## 3. Model Architecture

### 3.1 High-Level Design
{Overall architecture description}

### 3.2 Key Components

#### {Component Name}
- **Role**: {what it does}
- **Design**: {how it works — equations if applicable}
- **Input/Output shape**: {tensor dimensions}
- **Innovation**: {what's novel}

### 3.3 Architecture Diagram (text)
```
Input → [Component A] → [Component B] → Output
                ↑
          [Side branch]
```

---

## 4. Loss Design

### 4.1 Primary Loss
```
L_total = ...
```
{Explanation}

### 4.2 Auxiliary Losses
{Each auxiliary loss with equation and purpose}

### 4.3 Loss Weighting Strategy
{Fixed weights / scheduled / learned}

---

## 5. Training Strategy

| Aspect | Detail |
|--------|--------|
| Dataset | {name, size} |
| Optimizer | {type, β values} |
| Learning Rate | {value, schedule} |
| Batch Size | {value} |
| Epochs/Steps | {value} |
| Hardware | {GPUs, time} |
| Regularization | {dropout, weight decay} |
| Key Tricks | {warmup, clipping, EMA, etc.} |

---

## 6. Key Results

| Benchmark | Metric | This Paper | Prior SOTA | Delta |
|-----------|--------|------------|------------|-------|
| | | | | |

### Most Important Findings
- 
- 

### Ablation Insights
{What the ablations reveal}

---

## 7. Implementation Notes

### Critical Details
{Easy-to-miss but crucial implementation details}

### Potential Pitfalls
{Common failure modes}

### Reproduction Checklist
- [ ] {thing to verify}
- [ ] {thing to verify}

---

## 8. Paper Significance

### Why This Paper Matters
{1 paragraph}

### Limitations
{What the authors acknowledge}

### Follow-up Directions
{Natural next research steps}
