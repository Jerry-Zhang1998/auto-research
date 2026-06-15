"""
Test / evaluation entry point. Loads best checkpoint and reports final metrics.

Usage:
    # Use best checkpoint from a specific run:
    python test.py --run-name run_20260615_143022

    # Or point directly to a checkpoint file:
    python test.py --checkpoint ../../logs/my-paper/run_0/ckpt_best.pt

Results are saved to:
    logs/{PAPER_NAME}/{run_name}/test_results.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import Config
from model import ExampleModel
from dataset import get_dataloader

from src.utils.seed import set_seed
from src.utils.checkpoint import CheckpointManager
from src.metrics.classification import ClassificationMetrics   # or RegressionMetrics


# ── Evaluation loop ──────────────────────────────────────────────────────────

def evaluate(model, test_loader, device, task="classification"):
    model.eval()
    all_logits, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs  = inputs.to(device,  non_blocking=True)
            outputs = model(inputs)

            all_logits.append(outputs.cpu())
            all_targets.append(targets.cpu())

    logits_cat  = torch.cat(all_logits,  dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    if task == "classification":
        return ClassificationMetrics.compute_all(targets_cat, logits_cat)
    else:
        from src.metrics.regression import RegressionMetrics
        return RegressionMetrics.compute_all(targets_cat, logits_cat)


def print_results(metrics: dict, run_name: str, ckpt_path: str) -> None:
    print("\n" + "═" * 56)
    print(f"  TEST RESULTS")
    print(f"  Run:  {run_name}")
    print(f"  Ckpt: {ckpt_path}")
    print("─" * 56)
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.6f}")
    print("═" * 56 + "\n")


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name",   type=str, default=None,
                   help="Run subdirectory under logs/{paper_name}/. "
                        "Uses latest run if not specified.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Direct path to .pt file. Overrides --run-name.")
    p.add_argument("--split",      type=str, default="test",
                   choices=["test", "val"],
                   help="Dataset split to evaluate on.")
    p.add_argument("--task",       type=str, default="classification",
                   choices=["classification", "regression"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Config()
    set_seed(config.seed)

    device     = torch.device(config.device if torch.cuda.is_available() else "cpu")
    paper_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    logs_root  = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "logs", paper_name)
    )

    # ── Resolve checkpoint path ──────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = args.checkpoint
        run_name  = os.path.basename(os.path.dirname(ckpt_path))
    else:
        if args.run_name:
            run_dir = os.path.join(logs_root, args.run_name)
        else:
            # Pick the most recently modified run directory
            runs = sorted(
                (d for d in [os.path.join(logs_root, x) for x in os.listdir(logs_root)]
                 if os.path.isdir(d)),
                key=os.path.getmtime,
                reverse=True,
            )
            if not runs:
                raise FileNotFoundError(f"No run directories found in {logs_root}")
            run_dir = runs[0]

        run_name  = os.path.basename(run_dir)
        ckpt_path = os.path.join(run_dir, "ckpt_best.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run training first: python train.py"
        )

    # ── Load model ───────────────────────────────────────────────────────────
    model = ExampleModel(config.model).to(device)
    ckpt  = CheckpointManager(os.path.dirname(ckpt_path))
    ckpt.load(ckpt_path, model, device=str(device))

    # ── Load data ────────────────────────────────────────────────────────────
    test_loader = get_dataloader(config.train, split=args.split)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    metrics = evaluate(model, test_loader, device, task=args.task)
    print_results(metrics, run_name, ckpt_path)

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(ckpt_path), "test_results.json")
    with open(out_path, "w") as f:
        json.dump({"run": run_name, "checkpoint": ckpt_path,
                   "split": args.split, "metrics": metrics}, f, indent=2)
    print(f"Results saved → {out_path}")
