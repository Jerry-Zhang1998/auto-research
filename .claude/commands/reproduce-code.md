# reproduce-code

Generate a complete PyTorch code reproduction from a paper's innovation analysis.
Framework is always PyTorch (≥ 2.0). All generated code uses `torch`, `torch.nn`, and `torch.optim`.

## Arguments

$ARGUMENTS

First argument: paper name slug (e.g. `attention-is-all-you-need`). Must match a folder in `analyses/`.

## Steps

**Step 0 — Check for official GitHub repository.**

Read `analyses/{name}/innovations.md`. Look at the frontmatter `github:` field **and** the `## 0. Repository` section.

If a GitHub URL is present (not "not found"):

1. Run:
   ```bash
   python3 scripts/fetch_repo.py {GITHUB_URL} analyses/{name}/
   ```

2. Parse the JSON output.
   - **If success**: read the official source files identified under `file_summaries.model` and `file_summaries.loss` (the first result in each, up to 80 preview lines). These are the ground-truth implementation.
     - Record `{HAS_OFFICIAL_REPO}` = true
     - Use the official `model.py` and `loss.py` as the **primary source** when generating those files below. Adapt them to use our config dataclass and add type hints, but do NOT rewrite the mathematical logic.
     - For `train.py` and `test.py`: still follow the BaseTrainer/BaseEvaluator pattern — wrap the official model/loss.
     - Note any extra dependencies from `requirements` in the generated `README.md`.
   - **If clone failed** (private repo, network error, etc.): record `{HAS_OFFICIAL_REPO}` = false. Print a warning and continue with generating from scratch.

If no GitHub URL: set `{HAS_OFFICIAL_REPO}` = false.

---

**Step 1 — Load all reference material.**

Read in this order:
1. `analyses/{name}/innovations.md` — source of truth for the paper. If missing, tell the user to run `/analyze-innovations {name}` first.
2. `analyses/{name}/raw.md` — for implementation-level details missed in the analysis.
3. `prompts/reproduce_system.md` — code generation standards to follow exactly.
4. `outputs/_template/reproduction/train.py` — **reference template**: shows the exact BaseTrainer subclass pattern to follow for `train.py`.
5. `outputs/_template/reproduction/test.py` — **reference template**: shows the exact BaseEvaluator pattern to follow for `test.py`.
6. `src/base/base_trainer.py` — understand the interface: which methods are abstract, what keys `train_step` must return.
7. `src/base/base_evaluator.py` — understand `load_checkpoint` and `evaluate` signatures.

**Step 2 — Create the reproduction directory.**

Run: `mkdir -p outputs/{name}/reproduction`

**Step 3 — Generate each file.**

Generate all 7 files below. Rules:
- No placeholder TODOs — implement every component described in the analysis.
- Use type hints throughout.
- `config.py`, `model.py`, `loss.py`, `dataset.py` are fully paper-specific.
  - If `{HAS_OFFICIAL_REPO}` = true: adapt `model.py` and `loss.py` from the official code (keep mathematical logic, port to our config dataclass). Generate `dataset.py` from the paper.
  - If `{HAS_OFFICIAL_REPO}` = false: write all four from scratch based on `innovations.md`.
- `train.py` and `test.py` must follow the template pattern exactly (subclass BaseTrainer / use BaseEvaluator) — do NOT rewrite the training loop; that logic lives in `src/base/`.
- First line of `train.py` and `test.py`: `sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))`

---

### File: `outputs/{name}/reproduction/config.py`

All hyperparameters in a single dataclass. Every value must be taken from the paper; add a comment for values not explicitly stated.

```python
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

@dataclass
class ModelConfig:
    # Architecture
    ...

@dataclass  
class TrainConfig:
    # Training
    ...

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "outputs"
```

---

### File: `outputs/{name}/reproduction/model.py`

Full model architecture. Implement every component identified in the innovations analysis. Include:
- All sub-modules as separate `nn.Module` classes
- Forward method with clear tensor shape comments
- `__repr__` or docstring describing input/output shapes
- Parameter count utility at the bottom

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from config import ModelConfig

# Implement each component from Section 3 of innovations.md
```

---

### File: `outputs/{name}/reproduction/loss.py`

All loss functions from Section 4 of the innovations analysis.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from config import TrainConfig

class {PaperName}Loss(nn.Module):
    """
    Implements: {loss equation from paper}
    """
    def forward(self, ...) -> Dict[str, torch.Tensor]:
        # Returns dict with keys: 'total', 'primary', 'aux_*'
        ...
```

---

### File: `outputs/{name}/reproduction/dataset.py`

Data loading and preprocessing pipeline matching the paper's experimental setup.

```python
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Tuple
from config import TrainConfig

class {PaperName}Dataset(Dataset):
    """Dataset as described in Section 5 (Experiments)."""
    ...

def get_dataloader(config: TrainConfig, split: str = "train") -> DataLoader:
    ...
```

---

### File: `outputs/{name}/reproduction/train.py`

Thin `PaperTrainer(BaseTrainer)` subclass — only override `train_step` and `eval_step`.
All loop logic (logging, checkpointing, LR stepping, gradient clipping, metric computation)
is handled by `BaseTrainer` in `src/base/base_trainer.py`. Follow the pattern in
`outputs/_template/reproduction/train.py` exactly.

Must include:
- `sys.path.insert(0, ...)` pointing to project root so `src/` is importable
- `from src.utils.seed import set_seed`
- `from src.utils.logger import MetricLogger` — writes to `logs/{name}/{run_name}/`
- `from src.utils.checkpoint import CheckpointManager`
- `from src.metrics.classification import ClassificationMetrics` (or RegressionMetrics)
- AdamW optimizer with paper's hyperparameters
- LR scheduler matching the paper (warmup + cosine / step decay / etc.)
- Gradient clipping if the paper specifies it
- Per-step metric logging via `logger.log_metrics(step, epoch, train_loss=..., val_auc=...)`
- Checkpoint saving: `ckpt_best.pt` + `ckpt_latest.pt`
- `--lr`, `--epochs`, `--batch-size`, `--run-name` CLI arguments

The `run_name` defaults to `run_YYYYMMDD_HHMMSS` so each run has its own log directory.

---

### File: `outputs/{name}/reproduction/test.py`

Uses `BaseEvaluator` from `src/base/base_evaluator.py`. Follow `outputs/_template/reproduction/test.py` exactly.
Must include:
- `sys.path.insert(0, ...)` pointing to project root
- CLI args: `--run-name`, `--checkpoint` (direct path override), `--split` (test/val), `--task` (classification/regression)
- Auto-resolves checkpoint: if no `--checkpoint` given, looks for `logs/{name}/{run_name}/ckpt_best.pt`; if no `--run-name`, picks the most recent run directory
- `evaluator.load_checkpoint(ckpt_path)` to restore model weights
- `evaluator.evaluate(loader)` — returns full results: metrics + ROC/PR curves + confusion matrix
- Saves results to `logs/{name}/{run_name}/test_results.json` via `evaluator.save_results()`

---

### File: `outputs/{name}/reproduction/README.md`

```markdown
# {Paper Title} — Reproduction

Paper: {arxiv URL or citation}
Analysis: analyses/{name}/innovations.md
Official code: {GITHUB_URL or "N/A — generated from paper"}
  (cloned to analyses/{name}/_official_repo/ — model.py and loss.py adapted from official implementation)

## Quick Start
pip install torch torchvision  # + any paper-specific deps
# Place dataset in datasets/{dataset_name}/processed/

python train.py                       # train with default config
python train.py --run-name exp_01     # named run
python test.py  --run-name exp_01     # evaluate best checkpoint

## Files
- config.py    — all hyperparameters (values from paper)
- model.py     — {list major components}
- loss.py      — {loss function names}
- dataset.py   — {dataset name and preprocessing}
- train.py     — training loop (logs to logs/{name}/{run_name}/)
- test.py      — evaluation on test split

## Logs
Metrics are written to logs/{name}/{run_name}/:
  config.json        — config snapshot
  metrics.jsonl      — per-step metrics (train_loss, val_auc, …)
  train.log          — human-readable text
  metrics.csv        — exported at end of training
  ckpt_best.pt       — best validation checkpoint
  test_results.json  — populated after running test.py

## Key Implementation Decisions
{List any ambiguities in the paper and the choices made}

## Expected Results
{Target metrics from the paper}

## Known Gaps
{Any parts not reproduced, and why}
```

---

**Step 4 — Verify structure.**

Run:
```bash
find outputs/{name}/reproduction -type f | sort
wc -l outputs/{name}/reproduction/*.py
```

**Step 5 — Confirm.**

Print:
```
✓ Code generated: {title}
  Directory → outputs/{name}/reproduction/
  Files:
    config.py    ({N} lines)  — {N} hyperparameters
    model.py     ({N} lines)  — {list components}
    loss.py      ({N} lines)  — {list losses}
    dataset.py   ({N} lines)  — {dataset name}
    train.py     ({N} lines)  — logs to logs/{name}/{run_name}/
    test.py      ({N} lines)  — evaluates ckpt_best.pt
    README.md

  To train:
    cd outputs/{name}/reproduction
    python train.py --run-name exp_01

  To test:
    python test.py --run-name exp_01
```
