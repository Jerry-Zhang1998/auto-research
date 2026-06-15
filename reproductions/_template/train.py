"""
Training entry point.

Usage:
    python train.py
    python train.py --lr 1e-3 --epochs 100 --batch-size 64 --run-name exp_01

Logs are written to:
    logs/{PAPER_NAME}/{run_name}/
        config.json
        metrics.jsonl
        train.log
        metrics.csv
        ckpt_best.pt
        ckpt_latest.pt
"""
import argparse
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn

# Allow importing from project root (src/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import Config
from model import ExampleModel
from loss import PaperLoss
from dataset import get_dataloader

from src.utils.seed import set_seed
from src.utils.logger import MetricLogger
from src.utils.checkpoint import CheckpointManager
from src.metrics.classification import ClassificationMetrics   # or RegressionMetrics


# ── Paper-specific trainer ───────────────────────────────────────────────────

def train_step(model, criterion, optimizer, batch, device, grad_clip):
    model.train()
    inputs, targets = batch
    inputs  = inputs.to(device,  non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss_dict = criterion(outputs, targets)
    loss = loss_dict["total"]
    loss.backward()

    if grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()
    return loss_dict, outputs.detach(), targets


def eval_step(model, criterion, batch, device):
    model.eval()
    inputs, targets = batch
    inputs  = inputs.to(device,  non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    with torch.no_grad():
        outputs   = model(inputs)
        loss_dict = criterion(outputs, targets)

    return loss_dict, outputs, targets


# ── Main training loop ───────────────────────────────────────────────────────

def train(config: Config, run_name: str) -> None:
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Determine paper name from directory name (one level up from reproductions/)
    paper_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    log_dir    = os.path.join(
        os.path.dirname(__file__), "..", "..", "logs", paper_name, run_name
    )
    log_dir = os.path.normpath(log_dir)

    logger = MetricLogger(log_dir)
    ckpt   = CheckpointManager(log_dir, monitor="val_loss", mode="min")
    logger.log_config(config)

    # Model + loss + optimizer
    model     = ExampleModel(config.model).to(device)
    criterion = PaperLoss(config.train).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )

    # LR scheduler (cosine with warmup — adjust per paper)
    total_steps = config.train.max_epochs * 100   # rough estimate; update with real dataset size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    train_loader = get_dataloader(config.train, "train")
    val_loader   = get_dataloader(config.train, "val")

    global_step = 0
    log_every   = getattr(config.train, "log_every", 50)

    for epoch in range(config.train.max_epochs):
        t0 = time.time()

        # ── Train ──
        train_losses: dict[str, float] = {}
        train_count = 0
        for batch in train_loader:
            loss_dict, logits, targets = train_step(
                model, criterion, optimizer, batch, device, config.train.grad_clip
            )
            scheduler.step()
            global_step += 1
            train_count += 1

            for k, v in loss_dict.items():
                train_losses[k] = train_losses.get(k, 0.0) + v.item()

            if global_step % log_every == 0:
                avg = {k: v / train_count for k, v in train_losses.items()}
                logger.log_metrics(global_step, epoch=epoch,
                                   lr=scheduler.get_last_lr()[0],
                                   **{f"train_{k}": v for k, v in avg.items()})

        # ── Validate ──
        val_losses: dict[str, float] = {}
        all_logits, all_targets = [], []
        val_count = 0
        for batch in val_loader:
            loss_dict, logits, targets = eval_step(model, criterion, batch, device)
            val_count += 1
            for k, v in loss_dict.items():
                val_losses[k] = val_losses.get(k, 0.0) + v.item()
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

        val_avg = {k: v / max(val_count, 1) for k, v in val_losses.items()}

        # Compute metrics on validation set
        logits_cat  = torch.cat(all_logits,  dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        val_metrics = ClassificationMetrics.compute_all(targets_cat, logits_cat)

        # Log everything
        logger.log_metrics(
            global_step, epoch=epoch,
            **{f"val_{k}": v for k, v in val_avg.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            epoch_time_s=time.time() - t0,
        )

        # Checkpoint
        ckpt.save(
            step=global_step, epoch=epoch,
            model=model, optimizer=optimizer, scheduler=scheduler,
            metrics={"val_loss": val_avg.get("total", float("inf")),
                     "val_auc":  val_metrics.get("auc", 0.0)},
        )

        print(
            f"Epoch {epoch:03d}/{config.train.max_epochs} | "
            f"train_loss={train_losses.get('total', 0) / max(train_count, 1):.4f} | "
            f"val_loss={val_avg.get('total', 0):.4f} | "
            f"val_auc={val_metrics.get('auc', 0):.4f} | "
            f"time={time.time()-t0:.1f}s"
        )

    logger.export_csv()
    logger.close()
    print(f"\nBest checkpoint → {ckpt.best_path()}")


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--batch-size",  type=int,   default=None)
    p.add_argument("--run-name",    type=str,   default=None,
                   help="subdirectory name under logs/{paper_name}/. "
                        "Defaults to run_YYYYMMDD_HHMMSS")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Config()

    if args.lr:         config.train.lr          = args.lr
    if args.epochs:     config.train.max_epochs  = args.epochs
    if args.batch_size: config.train.batch_size  = args.batch_size

    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train(config, run_name)
