"""Base training loop. Subclass and implement train_step / eval_step."""
import time
import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader

from src.utils.logger import MetricLogger
from src.utils.checkpoint import CheckpointManager


class BaseTrainer:
    """
    Subclass this and implement train_step() and eval_step().

    train_step(batch) → dict  must contain "loss" (scalar tensor).
                              optionally "logits" [N,C] and "targets" [N] for metric logging.
    eval_step(batch)  → dict  must contain "logits" [N,C] and "targets" [N].
                              optionally scalar tensors (e.g. "loss") which are averaged.

    Example:

        class MyTrainer(BaseTrainer):
            def train_step(self, batch):
                x, y = batch
                logits = self.model(x)
                losses = self.criterion(logits, y)   # returns {"total":..., "primary":...}
                return {**losses, "loss": losses["total"],
                        "logits": logits.detach(), "targets": y}

            def eval_step(self, batch):
                x, y = batch
                logits = self.model(x)
                losses = self.criterion(logits, y)
                return {**losses, "loss": losses["total"],
                        "logits": logits, "targets": y}

        trainer = MyTrainer(model, criterion, optimizer, config,
                            log_dir="logs/my-paper/run_0", task="classification")
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
        task: str = "classification",   # "classification" | "regression"
    ):
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.config     = config
        self.task       = task
        self.device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir    = log_dir

        self.model.to(self.device)
        self.criterion.to(self.device)

        self.logger      = MetricLogger(log_dir)
        self.ckpt        = CheckpointManager(log_dir, monitor="val_loss", mode="min")
        self.global_step = 0

    # ── Abstract ─────────────────────────────────────────────────────────────

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def eval_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        ...

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> None:
        epochs    = epochs or self.config.train.max_epochs
        log_every = getattr(self.config.train, "log_every", 50)
        self.logger.log_config(self.config)

        for epoch in range(epochs):
            t0 = time.time()

            train_scalars = self._train_epoch(train_loader, log_every)
            val_scalars, val_task_metrics = self._val_epoch(val_loader)

            all_metrics = {
                **{f"train_{k}": v for k, v in train_scalars.items()},
                **{f"val_{k}":   v for k, v in val_scalars.items()},
                **{f"val_{k}":   v for k, v in val_task_metrics.items()},
                "epoch_time_s": time.time() - t0,
            }
            self.logger.log_metrics(step=self.global_step, epoch=epoch, **all_metrics)

            monitor_val = val_scalars.get("loss", float("inf"))
            self.ckpt.save(
                step=self.global_step, epoch=epoch,
                model=self.model, optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics={"val_loss": monitor_val,
                         **{f"val_{k}": v for k, v in val_task_metrics.items()}},
            )

            auc_str = f"  val_auc={val_task_metrics.get('auc', 0):.4f}" if "auc" in val_task_metrics else ""
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_scalars.get('loss', 0):.4f} | "
                f"val_loss={val_scalars.get('loss', 0):.4f}"
                f"{auc_str} | step={self.global_step}"
            )

            if self.scheduler is not None:
                self.scheduler.step()

        self.logger.export_csv()
        self.logger.close()
        print(f"\nTraining complete. Logs → {self.logger.log_dir}")
        print(f"Best checkpoint  → {self.ckpt.best_path()}")

    def load_checkpoint(self, path: str) -> Dict:
        return self.ckpt.load(path, self.model, self.optimizer, self.scheduler)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, log_every: int) -> Dict[str, float]:
        self.model.train()
        accum: Dict[str, float] = {}
        count = 0

        for batch in loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            out  = self.train_step(batch)
            loss = out["loss"]
            loss.backward()

            if hasattr(self.config.train, "grad_clip") and self.config.train.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)

            self.optimizer.step()
            self.global_step += 1
            count += 1

            for k, v in out.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    accum[k] = accum.get(k, 0.0) + v.item()

            if log_every and self.global_step % log_every == 0:
                step_avg = {k: v / count for k, v in accum.items()}
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_metrics(step=self.global_step, lr=lr,
                                        **{f"train_{k}": v for k, v in step_avg.items()})

        return {k: v / max(count, 1) for k, v in accum.items()}

    def _val_epoch(self, loader: DataLoader):
        self.model.eval()
        accum: Dict[str, float] = {}
        count = 0
        all_logits:  List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                out   = self.eval_step(batch)
                count += 1

                for k, v in out.items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        accum[k] = accum.get(k, 0.0) + v.item()

                if "logits" in out and "targets" in out:
                    all_logits.append(out["logits"].cpu())
                    all_targets.append(out["targets"].cpu())

        scalars = {k: v / max(count, 1) for k, v in accum.items()}

        # Compute task-level metrics from accumulated logits/targets
        task_metrics: Dict[str, float] = {}
        if all_logits and all_targets:
            logits  = torch.cat(all_logits,  dim=0)
            targets = torch.cat(all_targets, dim=0)
            try:
                if self.task == "classification":
                    from src.metrics.classification import ClassificationMetrics
                    task_metrics = ClassificationMetrics.compute_all(targets, logits)
                else:
                    from src.metrics.regression import RegressionMetrics
                    task_metrics = RegressionMetrics.compute_all(targets, logits)
            except Exception:
                pass

        return scalars, task_metrics

    def _to_device(self, batch: Any) -> Any:
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(x) for x in batch)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        return batch
