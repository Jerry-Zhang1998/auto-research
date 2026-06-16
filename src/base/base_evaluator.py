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

        evaluator = BaseEvaluator(model, task="classification")
        evaluator.load_checkpoint("logs/my-paper/run_0/ckpt_best.pt")
        results = evaluator.evaluate(test_loader)
        evaluator.save_results(results, "logs/my-paper/run_0/test_results.json")
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        task: str = "classification",
    ):
        self.model  = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task   = task
        self.model.to(self.device)
        self.model.eval()

    def load_checkpoint(self, path: str, map_location: Optional[str] = None) -> Dict:
        loc  = map_location or str(self.device)
        ckpt = torch.load(path, map_location=loc)
        key  = "model" if "model" in ckpt else "model_state"
        self.model.load_state_dict(ckpt[key])
        print(f"  Loaded epoch={ckpt.get('epoch','?')} step={ckpt.get('step','?')} from {path}")
        return ckpt

    def evaluate(self, loader: DataLoader) -> Dict:
        """
        Returns a dict suitable for save_results():
            {
              "metrics":  {auc, accuracy, f1, precision, recall, auc_pr},
              "curves":   {roc: {fpr, tpr}, pr: {precision, recall}},   # binary only
              "confusion": [[...]],   # int matrix, classification only
            }
        """
        all_logits:  List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                inputs, targets = self._unpack(batch)
                inputs  = self._to_device(inputs)
                outputs = self.model(inputs)
                all_logits.append(outputs.cpu())
                all_targets.append(targets.cpu())

        logits  = torch.cat(all_logits,  dim=0)
        targets = torch.cat(all_targets, dim=0)

        results: Dict[str, Any] = {}

        if self.task == "classification":
            results["metrics"]   = ClassificationMetrics.compute_all(targets, logits)
            results["curves"]    = ClassificationMetrics.compute_curves(targets, logits)
            results["confusion"] = ClassificationMetrics.compute_confusion_matrix(targets, logits)
        else:
            results["metrics"]   = RegressionMetrics.compute_all(targets, logits)
            results["curves"]    = {}
            results["confusion"] = []

        self._print_results(results["metrics"])
        return results

    def save_results(self, results: Dict, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Test results → {path}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _unpack(self, batch: Any):
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            input_keys  = [k for k in batch if k in ("input", "inputs", "x", "image", "data")]
            target_keys = [k for k in batch if k in ("label", "labels", "target", "targets", "y")]
            if input_keys and target_keys:
                return batch[input_keys[0]], batch[target_keys[0]]
        raise ValueError(
            f"Cannot unpack batch of type {type(batch)}. "
            "Override _unpack() in your evaluator subclass to handle this batch format."
        )

    def _to_device(self, x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, non_blocking=True)
        if isinstance(x, (list, tuple)):
            return type(x)(self._to_device(i) for i in x)
        return x

    @staticmethod
    def _print_results(metrics: Dict[str, float]) -> None:
        print("\n" + "═" * 50)
        print("  TEST RESULTS")
        print("─" * 50)
        for k, v in metrics.items():
            bar = "█" * int(v * 20) if 0 <= v <= 1 else ""
            print(f"  {k:<20}  {v:.6f}  {bar}")
        print("═" * 50)
