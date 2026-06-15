"""Checkpoint manager: saves best + latest, keeps history clean."""
import glob
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """
    Saves two checkpoints per run:
        {log_dir}/ckpt_latest.pt   — overwritten every save_every epochs
        {log_dir}/ckpt_best.pt     — overwritten when the monitored metric improves

    Optional: keep the top-k best checkpoints by name.
    """

    def __init__(
        self,
        log_dir: str,
        monitor: str = "val_loss",
        mode: str = "min",           # "min" | "max"
        save_every: int = 1,
    ):
        self.log_dir    = log_dir
        self.monitor    = monitor
        self.mode       = mode
        self.save_every = save_every
        self._best      = float("inf") if mode == "min" else float("-inf")

        os.makedirs(log_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def save(
        self,
        step:      int,
        epoch:     int,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics:   Dict[str, float],
        scheduler: Optional[Any] = None,
    ) -> None:
        payload = {
            "step":      step,
            "epoch":     epoch,
            "metrics":   metrics,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()

        # Always save latest
        if epoch % self.save_every == 0:
            torch.save(payload, os.path.join(self.log_dir, "ckpt_latest.pt"))

        # Save best if improved
        val = metrics.get(self.monitor, None)
        if val is not None and self._is_better(val):
            self._best = val
            torch.save(payload, os.path.join(self.log_dir, "ckpt_best.pt"))
            print(f"  [ckpt] Best {self.monitor}={val:.6f} → ckpt_best.pt")

    def load(
        self,
        path:      str,
        model:     nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device:    str = "cpu",
    ) -> Dict:
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"  [ckpt] Loaded epoch={ckpt.get('epoch')}, step={ckpt.get('step')} from {path}")
        return ckpt

    def best_path(self) -> str:
        return os.path.join(self.log_dir, "ckpt_best.pt")

    def latest_path(self) -> str:
        return os.path.join(self.log_dir, "ckpt_latest.pt")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _is_better(self, val: float) -> bool:
        if self.mode == "min":
            return val < self._best
        return val > self._best
