"""
Test / evaluation entry point — uses BaseEvaluator from src/.

Resolves the best checkpoint automatically from the logs directory,
runs inference on the test split, computes full metrics + ROC/PR curves,
and saves test_results.json (used by generate_viz.py to build evaluate.html).

Usage:
    python test.py                             # auto-picks latest run's ckpt_best.pt
    python test.py --run-name exp_01           # pick specific run
    python test.py --checkpoint path/to.pt    # explicit checkpoint path
    python test.py --split val                 # evaluate on val split instead
    python test.py --task regression           # switch metric mode
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import Config
from model import ExampleModel
from dataset import get_dataloader
from src.base.base_evaluator import BaseEvaluator
from src.utils.seed import set_seed


def resolve_checkpoint(logs_root: str, run_name: str | None) -> tuple[str, str]:
    """Returns (ckpt_path, run_name)."""
    if run_name:
        run_dir = os.path.join(logs_root, run_name)
    else:
        runs = sorted(
            [os.path.join(logs_root, d) for d in os.listdir(logs_root)
             if os.path.isdir(os.path.join(logs_root, d))],
            key=os.path.getmtime, reverse=True,
        )
        if not runs:
            raise FileNotFoundError(f"No run directories in {logs_root}. Train first.")
        run_dir = runs[0]

    run  = os.path.basename(run_dir)
    ckpt = os.path.join(run_dir, "ckpt_best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No ckpt_best.pt in {run_dir}. Train first.")
    return ckpt, run


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name",   type=str)
    p.add_argument("--checkpoint", type=str, help="Direct path; overrides --run-name")
    p.add_argument("--split",      type=str, default="test", choices=["test", "val"])
    p.add_argument("--task",       type=str, default="classification",
                   choices=["classification", "regression"])
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    config = Config()
    set_seed(config.seed)

    paper     = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    logs_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "logs", paper)
    )

    if args.checkpoint:
        ckpt_path = args.checkpoint
        run_name  = os.path.basename(os.path.dirname(ckpt_path))
    else:
        ckpt_path, run_name = resolve_checkpoint(logs_root, args.run_name)

    out_path = os.path.join(os.path.dirname(ckpt_path), "test_results.json")

    model     = ExampleModel(config.model)
    evaluator = BaseEvaluator(model, task=args.task)
    evaluator.load_checkpoint(ckpt_path)

    results = evaluator.evaluate(get_dataloader(config.train, split=args.split))
    results["run"]        = run_name
    results["checkpoint"] = ckpt_path
    results["split"]      = args.split

    evaluator.save_results(results, out_path)
    print(f"\nTo visualise:\n  python3 scripts/generate_viz.py "
          f"--log-dir {os.path.dirname(ckpt_path)} --output-dir outputs/{paper}")
