# reproduce-code

Generate a complete PyTorch code reproduction from a paper's innovation analysis.
Framework is always PyTorch (≥ 2.0). All generated code uses `torch`, `torch.nn`, and `torch.optim`.

## Arguments

$ARGUMENTS

First argument: paper name slug (e.g. `attention-is-all-you-need`). Must match a folder in `analyses/`.

## Steps

**Step 1 — Load the analysis.**

Read `analyses/{name}/innovations.md`. If missing, tell the user to run `/analyze-innovations {name}` first.
Also read `analyses/{name}/raw.md` for implementation-level details.
Read `prompts/reproduce_system.md` for code generation guidelines.

**Step 2 — Create the reproduction directory.**

Run: `mkdir -p reproductions/{name}`

**Step 3 — Generate each file.**

Generate all files below. Each file must be complete, runnable, and well-structured. No placeholder TODOs — implement every component described in the analysis. Use type hints throughout.

---

### File: `reproductions/{name}/config.py`

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

### File: `reproductions/{name}/model.py`

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

### File: `reproductions/{name}/loss.py`

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

### File: `reproductions/{name}/dataset.py`

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

### File: `reproductions/{name}/train.py`

Complete training loop with:
- Optimizer and LR scheduler matching the paper
- Gradient clipping if mentioned
- Logging (loss, metrics every N steps)
- Checkpointing (save best + latest)
- Evaluation loop
- Reproducible seeding

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os, json, time
from typing import Optional
from config import Config
from model import {ModelClass}
from loss import {LossClass}
from dataset import get_dataloader

def train(config: Config):
    ...

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    config = Config()
    train(config)
```

---

### File: `reproductions/{name}/train.py`

Full training loop that integrates with `src/`. Must include:
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

### File: `reproductions/{name}/test.py`

Standalone test / evaluation script. Must include:
- `sys.path.insert(0, ...)` pointing to project root
- CLI args: `--run-name`, `--checkpoint` (direct path override), `--split` (test/val), `--task` (classification/regression)
- Auto-resolves checkpoint: if no `--checkpoint` given, looks for `logs/{name}/{run_name}/ckpt_best.pt`; if no `--run-name`, picks the most recent run directory
- Loads model weights via `CheckpointManager.load()`
- Runs full pass over the test split with `torch.no_grad()`
- Computes all metrics via `ClassificationMetrics.compute_all()` or `RegressionMetrics.compute_all()`
- Prints a formatted results table
- Saves results to `logs/{name}/{run_name}/test_results.json`

---

### File: `reproductions/{name}/README.md`

```markdown
# {Paper Title} — Reproduction

Paper: {arxiv URL or citation}
Analysis: analyses/{name}/innovations.md

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
find reproductions/{name} -type f | sort
wc -l reproductions/{name}/*.py
```

**Step 5 — Confirm.**

Print:
```
✓ Code generated: {title}
  Directory → reproductions/{name}/
  Files:
    config.py    ({N} lines)  — {N} hyperparameters
    model.py     ({N} lines)  — {list components}
    loss.py      ({N} lines)  — {list losses}
    dataset.py   ({N} lines)  — {dataset name}
    train.py     ({N} lines)  — logs to logs/{name}/{run_name}/
    test.py      ({N} lines)  — evaluates ckpt_best.pt
    README.md

  To train:
    cd reproductions/{name}
    python train.py --run-name exp_01

  To test:
    python test.py --run-name exp_01
```
