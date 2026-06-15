"""
Training entry point — subclasses BaseTrainer with paper-specific forward pass.

All training loop logic (logging, checkpointing, LR scheduling, gradient clipping,
metric computation) is handled by BaseTrainer in src/base/base_trainer.py.
Only override train_step() and eval_step() here.

Usage:
    python train.py
    python train.py --lr 3e-4 --epochs 100 --run-name exp_01
    python train.py --task regression   # switch metric mode

Logs → logs/{paper_name}/{run_name}/
    config.json   metrics.jsonl   train.log   metrics.csv
    ckpt_best.pt  ckpt_latest.pt
"""
import os
import sys
import argparse
import torch
from datetime import datetime

# ── Project root on path so src/ is importable ──────────────────────────────
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import Config
from model import ExampleModel
from loss import PaperLoss
from dataset import get_dataloader
from src.base.base_trainer import BaseTrainer
from src.utils.seed import set_seed


# ── Paper-specific trainer ───────────────────────────────────────────────────

class PaperTrainer(BaseTrainer):
    """
    Override train_step and eval_step with paper-specific forward pass.
    Both must return a dict with at least "loss" (scalar tensor).
    Include "logits" [N,C] and "targets" [N] so BaseTrainer can compute AUC / F1.
    """

    def train_step(self, batch):
        inputs, targets = batch
        outputs   = self.model(inputs)          # [N, C] or [N]
        loss_dict = self.criterion(outputs, targets)   # must contain "total"
        return {
            **loss_dict,
            "loss":    loss_dict["total"],
            "logits":  outputs.detach(),
            "targets": targets,
        }

    def eval_step(self, batch):
        inputs, targets = batch
        outputs   = self.model(inputs)
        loss_dict = self.criterion(outputs, targets)
        return {
            **loss_dict,
            "loss":    loss_dict["total"],
            "logits":  outputs,
            "targets": targets,
        }


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr",         type=float)
    p.add_argument("--epochs",     type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--run-name",   type=str,
                   help="Run subdirectory name (default: run_YYYYMMDD_HHMMSS)")
    p.add_argument("--task",       type=str, default="classification",
                   choices=["classification", "regression"])
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    config = Config()
    if args.lr:         config.train.lr         = args.lr
    if args.epochs:     config.train.max_epochs = args.epochs
    if args.batch_size: config.train.batch_size = args.batch_size

    set_seed(config.seed)

    paper   = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    run     = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "logs", paper, run)
    )

    model     = ExampleModel(config.model)
    criterion = PaperLoss(config.train)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.max_epochs
    )

    trainer = PaperTrainer(
        model, criterion, optimizer, config,
        log_dir=log_dir,
        scheduler=scheduler,
        task=args.task,
    )
    trainer.fit(
        get_dataloader(config.train, "train"),
        get_dataloader(config.train, "val"),
    )
