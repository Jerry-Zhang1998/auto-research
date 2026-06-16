# {Paper Title} — Reproduction

Paper: {arxiv URL or citation}
Analysis: [analyses/{name}/innovations.md](../../analyses/{name}/innovations.md)

## Quick Start

```bash
pip install torch torchvision  # + paper-specific deps

python train.py
# optional overrides:
python train.py --lr 1e-3 --epochs 50 --batch-size 128
```

## Files

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters (values taken from paper) |
| `model.py` | {List major architectural components} |
| `loss.py` | {Loss function names} |
| `dataset.py` | {Dataset name and preprocessing steps} |
| `train.py` | Full training loop with checkpointing |

## Implementation Decisions

{List any ambiguities in the paper and the choices made, with reasoning}

## Expected Results

| Benchmark | Metric | Target |
|-----------|--------|--------|
| | | |

## Known Gaps

{Any parts not fully reproduced and why (e.g., proprietary dataset, missing details)}

## Citation

```bibtex
@article{...}
```
