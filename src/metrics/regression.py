"""Regression metrics: MSE, RMSE, MAE, R², MAPE."""
import numpy as np
import torch
from typing import Dict


class RegressionMetrics:

    @staticmethod
    def compute_all(
        targets: torch.Tensor,
        predictions: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Args:
            targets:     [N] or [N, 1] ground-truth values
            predictions: [N] or [N, 1] model outputs
        Returns:
            dict with keys: mse, rmse, mae, r2, mape
        """
        y = _to_np(targets).ravel()
        p = _to_np(predictions).ravel()

        ss_res = ((y - p) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()

        mse  = float(ss_res / len(y))
        mae  = float(np.abs(y - p).mean())
        r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        mape = float((np.abs((y - p) / np.where(y != 0, y, 1e-8))).mean() * 100)

        return {
            "mse":  mse,
            "rmse": float(np.sqrt(mse)),
            "mae":  mae,
            "r2":   r2,
            "mape": mape,
        }

    @staticmethod
    def compute_mse(targets, predictions) -> float:
        return RegressionMetrics.compute_all(targets, predictions)["mse"]

    @staticmethod
    def compute_rmse(targets, predictions) -> float:
        return RegressionMetrics.compute_all(targets, predictions)["rmse"]

    @staticmethod
    def compute_mae(targets, predictions) -> float:
        return RegressionMetrics.compute_all(targets, predictions)["mae"]

    @staticmethod
    def compute_r2(targets, predictions) -> float:
        return RegressionMetrics.compute_all(targets, predictions)["r2"]


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
