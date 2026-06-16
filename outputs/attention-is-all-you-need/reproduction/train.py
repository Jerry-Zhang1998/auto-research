"""
Transformer training entry point.

Subclasses BaseTrainer with the Transformer-specific forward pass.
Implements the Noam LR schedule (Eq. 6) per step rather than per epoch.

Usage:
    python train.py
    python train.py --epochs 50 --run-name exp_base
    python train.py --warmup-steps 4000 --d-model 512

Logs → logs/attention-is-all-you-need/{run_name}/
    config.json   metrics.jsonl   train.log   metrics.csv
    ckpt_best.pt  ckpt_latest.pt
"""
import os
import sys
import math
import argparse
import torch
from datetime import datetime

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from config import Config
from model import Transformer
from loss import LabelSmoothingCrossEntropy
from dataset import get_dataloader
from src.base.base_trainer import BaseTrainer
from src.utils.seed import set_seed


# ── Noam LR Schedule (Eq. 6) ─────────────────────────────────────────────────

def noam_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """lrate = d_model^{-0.5} · min(step^{-0.5}, step · warmup^{-1.5})"""
    if step == 0:
        step = 1
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))


# ── Paper-specific Trainer ────────────────────────────────────────────────────

class TransformerTrainer(BaseTrainer):
    """
    train_step: teacher-forced forward pass + label-smoothing CE loss.
    eval_step:  same without Noam LR update.

    The Noam schedule (Eq. 6) is per-step, not per-epoch, so we update
    optimizer.param_groups[0]['lr'] directly inside train_step.
    Pass scheduler=None to BaseTrainer to disable per-epoch stepping.
    """

    def train_step(self, batch):
        src, tgt = batch                 # [B, S], [B, T]

        # Noam LR update — happens before optimizer.step() inside BaseTrainer
        step = self.global_step + 1      # global_step is pre-increment in BaseTrainer
        lr   = noam_lr(step, self.config.model.d_model, self.config.train.warmup_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        # Teacher-forcing: feed tgt[:, :-1] → predict tgt[:, 1:]
        dec_input  = tgt[:, :-1]         # [B, T-1]  (BOS ... last-but-one token)
        dec_target = tgt[:, 1:]          # [B, T-1]  (first-after-BOS ... EOS)

        logits     = self.model(src, dec_input)   # [B, T-1, V]
        loss_dict  = self.criterion(logits, dec_target)

        return {
            **loss_dict,
            "loss": loss_dict["total"],
        }

    def eval_step(self, batch):
        src, tgt   = batch
        dec_input  = tgt[:, :-1]
        dec_target = tgt[:, 1:]

        logits    = self.model(src, dec_input)   # [B, T-1, V]
        loss_dict = self.criterion(logits, dec_target)

        return {
            **loss_dict,
            "loss": loss_dict["total"],
        }


# ── Entry Point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Transformer (Attention Is All You Need)")
    p.add_argument("--epochs",        type=int)
    p.add_argument("--batch-size",    type=int)
    p.add_argument("--warmup-steps",  type=int)
    p.add_argument("--d-model",       type=int)
    p.add_argument("--n-heads",       type=int)
    p.add_argument("--n-layers",      type=int)
    p.add_argument("--dropout",       type=float)
    p.add_argument("--run-name",      type=str,
                   help="Run subdirectory (default: run_YYYYMMDD_HHMMSS)")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    config = Config()

    # CLI overrides
    if args.epochs:       config.train.max_epochs    = args.epochs
    if args.batch_size:   config.train.batch_size    = args.batch_size
    if args.warmup_steps: config.train.warmup_steps  = args.warmup_steps
    if args.d_model:      config.model.d_model       = args.d_model
    if args.n_heads:      config.model.n_heads       = args.n_heads
    if args.n_layers:
        config.model.n_encoder_layers = args.n_layers
        config.model.n_decoder_layers = args.n_layers
    if args.dropout:      config.model.dropout       = args.dropout

    set_seed(config.seed)

    paper   = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run     = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs", paper, run)
    )

    train_loader = get_dataloader(config.train, "train")
    val_loader   = get_dataloader(config.train, "val")

    vocab_size = config.model.src_vocab_size
    model      = Transformer(config.model)
    criterion  = LabelSmoothingCrossEntropy(config.train, vocab_size, config.model.pad_token_id)

    # Adam with paper's non-standard β₂=0.98, ε=1e-9 (Section 5.3)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0,   # set by Noam schedule in train_step
        betas=(config.train.beta1, config.train.beta2),
        eps=config.train.eps,
        weight_decay=config.train.weight_decay,
    )

    print(f"Model:  {model}")
    print(f"Params: {model.count_parameters():,}")
    print(f"Logs →  {log_dir}")

    trainer = TransformerTrainer(
        model, criterion, optimizer, config,
        log_dir=log_dir,
        scheduler=None,      # Noam schedule is per-step; updated manually in train_step
        task="classification",
    )
    trainer.fit(train_loader, val_loader)
