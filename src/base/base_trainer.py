"""Base training loop. Subclass and implement train_step / eval_step."""
import os
import time
import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import DataLoader

from src.utils.logger import MetricLogger
from src.utils.checkpoint import CheckpointManager


class BaseTrainer:
    """
    Usage:

        class MyTrainer(BaseTrainer):
            def train_step(self, batch):
                x, y = batch
                logits = self.model(x)
                loss = self.criterion(logits, y)
                return {"loss": loss, "logits": logits.detach(), "targets": y}

            def eval_step(self, batch):
                x, y = batch
                with torch.no_grad():
                    logits = self.model(x)
                return {"logits": logits, "targets": y}

        trainer = MyTrainer(model, criterion, optimizer, config, log_dir="logs/my-paper/run_0")
        trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config,
        log_dir: str,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.criterion.to(self.device)

        self.logger = MetricLogger(log_dir)
        self.ckpt = CheckpointManager(log_dir)
        self.global_step = 0

    # ── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Run one forward + loss computation.
        Must return a dict containing at least {"loss": scalar_tensor}.
        May also return "logits" and "targets" for metric computation.
        """
        ...

    @abstractmethod
    def eval_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Run one forward pass for evaluation (no grad).
        Must return {"logits": ..., "targets": ...}.
        """
        ...

    # ── Training loop ────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> None:
        epochs = epochs or self.config.train.max_epochs
        log_every = getattr(self.config.train, "log_every", 50)
        self.logger.log_config(self.config)

        for epoch in range(epochs):
            train_metrics = self._run_epoch(train_loader, phase="train", log_every=log_every)
            val_metrics   = self._run_epoch(val_loader,   phase="val",   log_every=None)

            self.logger.log_metrics(
                step=self.global_step, epoch=epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}":   v for k, v in val_metrics.items()},
            )

            monitor_metric = val_metrics.get("loss", float("inf"))
            self.ckpt.save(
                step=self.global_step,
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                metrics={"val_loss": monitor_metric},
            )

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_metrics.get('loss', 0):.4f} | "
                f"val_loss={val_metrics.get('loss', 0):.4f} | "
                f"step={self.global_step}"
            )

            if self.scheduler is not None:
                self.scheduler.step()

        self.logger.export_csv()
        print(f"\nTraining complete. Logs → {self.logger.log_dir}")

    def _run_epoch(
        self,
        loader: DataLoader,
        phase: str,
        log_every: Optional[int],
    ) -> Dict[str, float]:
        is_train = phase == "train"
        self.model.train(is_train)

        accum: Dict[str, float] = {}
        count = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(loader):
            batch = self._to_device(batch)

            if is_train:
                self.optimizer.zero_grad()
                out = self.train_step(batch)
                loss = out["loss"]
                loss.backward()
                if hasattr(self.config.train, "grad_clip") and self.config.train.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                self.optimizer.step()
                self.global_step += 1
            else:
                with torch.no_grad():
                    out = self.eval_step(batch)

            for k, v in out.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    accum[k] = accum.get(k, 0.0) + v.item()
            count += 1

            if is_train and log_every and (self.global_step % log_every == 0):
                step_metrics = {k: v / count for k, v in accum.items()}
                self.logger.log_metrics(step=self.global_step, phase=phase, **step_metrics)

        elapsed = time.time() - t0
        avg = {k: v / max(count, 1) for k, v in accum.items()}
        avg["epoch_time_s"] = elapsed
        return avg

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _to_device(self, batch: Any) -> Any:
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(x) for x in batch)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        return batch

    def load_checkpoint(self, path: str) -> Dict:
        return self.ckpt.load(path, self.model, self.optimizer)
