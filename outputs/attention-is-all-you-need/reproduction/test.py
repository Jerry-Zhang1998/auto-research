"""
Transformer evaluation entry point.

Computes perplexity (from label-smoothed CE loss) on the test/val split.
For BLEU, run beam-search decoding separately (requires sacrebleu).

Usage:
    python test.py                          # auto-picks latest run's ckpt_best.pt
    python test.py --run-name exp_base      # pick specific run
    python test.py --checkpoint path/to.pt  # explicit checkpoint
    python test.py --split val

Results → logs/attention-is-all-you-need/{run_name}/test_results.json
Then visualise:
    python3 scripts/generate_viz.py --log-dir <run_dir> --output-dir outputs/attention-is-all-you-need/html
"""
import os
import sys
import json
import math
import argparse
import torch
from typing import Optional, Tuple

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from config import Config
from model import Transformer
from loss import LabelSmoothingCrossEntropy
from dataset import get_dataloader
from src.utils.seed import set_seed


def resolve_checkpoint(logs_root: str, run_name: Optional[str]) -> Tuple[str, str]:
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


def evaluate(
    model: Transformer,
    criterion: LabelSmoothingCrossEntropy,
    loader,
    device: torch.device,
) -> dict:
    """Run inference and return perplexity + token-level accuracy."""
    model.eval()
    total_loss  = 0.0
    total_steps = 0
    correct     = 0
    total_toks  = 0

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device, non_blocking=True)    # [B, S]
            tgt = tgt.to(device, non_blocking=True)    # [B, T]

            dec_input  = tgt[:, :-1]   # [B, T-1]
            dec_target = tgt[:, 1:]    # [B, T-1]

            logits    = model(src, dec_input)           # [B, T-1, V]
            loss_dict = criterion(logits, dec_target)

            total_loss  += loss_dict["total"].item()
            total_steps += 1

            # Token-level accuracy (ignore padding)
            preds   = logits.argmax(dim=-1)             # [B, T-1]
            non_pad = dec_target.ne(model.pad_id)
            correct    += (preds.eq(dec_target) & non_pad).sum().item()
            total_toks += non_pad.sum().item()

    avg_loss   = total_loss / max(total_steps, 1)
    perplexity = math.exp(min(avg_loss, 20))   # clip before exp to avoid overflow
    accuracy   = correct / max(total_toks, 1)

    metrics = {
        "loss":       avg_loss,
        "perplexity": perplexity,
        "accuracy":   accuracy,
    }
    return {"metrics": metrics, "curves": {}, "confusion": []}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name",   type=str)
    p.add_argument("--checkpoint", type=str, help="Direct path; overrides --run-name")
    p.add_argument("--split",      type=str, default="test", choices=["test", "val"])
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    config = Config()
    set_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    paper     = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs", paper)
    )

    if args.checkpoint:
        ckpt_path = args.checkpoint
        run_name  = os.path.basename(os.path.dirname(ckpt_path))
    else:
        ckpt_path, run_name = resolve_checkpoint(logs_root, args.run_name)

    out_path = os.path.join(os.path.dirname(ckpt_path), "test_results.json")

    # Load model
    vocab_size = config.model.src_vocab_size
    model      = Transformer(config.model).to(device)
    ckpt       = torch.load(ckpt_path, map_location=str(device))
    model.load_state_dict(ckpt["model"])
    print(f"Loaded epoch={ckpt.get('epoch','?')} from {ckpt_path}")

    criterion = LabelSmoothingCrossEntropy(config.train, vocab_size, config.model.pad_token_id)
    criterion.to(device)

    loader  = get_dataloader(config.train, split=args.split)
    results = evaluate(model, criterion, loader, device)
    results["run"]        = run_name
    results["checkpoint"] = ckpt_path
    results["split"]      = args.split

    # Print
    print("\n" + "=" * 50)
    print("  TEST RESULTS")
    print("-" * 50)
    for k, v in results["metrics"].items():
        print(f"  {k:<20}  {v:.6f}")
    print("=" * 50)

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out_path}")
    print(f"\nTo visualise:\n  python3 scripts/generate_viz.py "
          f"--log-dir {os.path.dirname(ckpt_path)} --output-dir outputs/{paper}/html")
