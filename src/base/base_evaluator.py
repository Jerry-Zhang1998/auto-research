"""Standalone evaluator for test-time inference."""
import json
import os
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader

from src.metrics.classification import ClassificationMetrics
from src.metrics.regression import RegressionMetrics


class BaseEvaluator:
    """
    Usage:

        evaluator = BaseEvaluator(model, device="cuda", task="classification")
        results = evaluator.evaluate(test_loader)
        evaluator.save_results(results, "logs/my-paper/run_0/test_results.json")
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        task: str = "classification",   # "classification" | "regression"
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        all_logits: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                inputs, targets = self._unpack(batch)
                inputs = self._to_device(inputs)
                logits = self.model(inputs)
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())

        logits  = torch.cat(all_logits,  dim=0)
        targets = torch.cat(all_targets, dim=0)

        if self.task == "classification":
            metrics = ClassificationMetrics.compute_all(targets, logits)
        else:
            metrics = RegressionMetrics.compute_all(targets, logits)

        self._print_results(metrics)
        return metrics

    def save_results(self, results: Dict[str, float], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Test results saved → {path}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _unpack(self, batch: Any):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        raise ValueError(f"Expected (inputs, targets) batch, got {type(batch)}")

    def _to_device(self, x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, non_blocking=True)
        if isinstance(x, (list, tuple)):
            return type(x)(self._to_device(i) for i in x)
        return x

    @staticmethod
    def _print_results(metrics: Dict[str, float]) -> None:
        print("\n── Test Results " + "─" * 40)
        for k, v in metrics.items():
            print(f"  {k:<25} {v:.6f}")
        print("─" * 56)
