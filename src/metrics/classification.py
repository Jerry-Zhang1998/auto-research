"""Classification metrics: AUC, Accuracy, F1, Precision, Recall, AP."""
import numpy as np
import torch
from typing import Dict, Optional


class ClassificationMetrics:

    @staticmethod
    def compute_all(
        targets: torch.Tensor,
        logits: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Args:
            targets: [N] integer class labels  OR  [N, C] one-hot
            logits:  [N, C] raw scores (pre-softmax)  OR  [N] for binary
        Returns:
            dict with keys: auc, auc_pr, accuracy, f1, precision, recall
        """
        y_true = targets.numpy() if isinstance(targets, torch.Tensor) else targets
        y_score = logits.numpy() if isinstance(logits, torch.Tensor) else logits

        # Flatten one-hot targets
        if y_true.ndim == 2:
            y_true = y_true.argmax(axis=1)

        metrics: Dict[str, float] = {}

        # Probabilities
        if y_score.ndim == 1:
            y_prob = _sigmoid(y_score)
            y_pred = (y_prob >= threshold).astype(int)
            n_classes = 2
        else:
            y_prob = _softmax(y_score)
            y_pred = y_prob.argmax(axis=1)
            n_classes = y_score.shape[1]

        # AUC (ROC)
        metrics["auc"] = _roc_auc(y_true, y_prob, n_classes)

        # AUC-PR (average precision)
        metrics["auc_pr"] = _average_precision(y_true, y_prob, n_classes)

        # Accuracy
        metrics["accuracy"] = float((y_pred == y_true).mean())

        # Precision / Recall / F1 (macro)
        p, r, f1 = _prf_macro(y_true, y_pred, n_classes)
        metrics["precision"] = p
        metrics["recall"]    = r
        metrics["f1"]        = f1

        return metrics

    @staticmethod
    def compute_auc(targets, logits) -> float:
        return ClassificationMetrics.compute_all(targets, logits)["auc"]

    @staticmethod
    def compute_accuracy(targets, logits) -> float:
        return ClassificationMetrics.compute_all(targets, logits)["accuracy"]

    @staticmethod
    def compute_f1(targets, logits) -> float:
        return ClassificationMetrics.compute_all(targets, logits)["f1"]


# ── Internal helpers (no sklearn required) ────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _roc_auc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if n_classes == 2:
            prob = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
            return float(roc_auc_score(y_true, prob))
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ImportError:
        pass

    # Fallback: binary only
    if n_classes > 2:
        return float("nan")
    prob = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
    return _binary_roc_auc(y_true, prob)


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = tp / max(y_true.sum(), 1)
    fpr = fp / max((1 - y_true).sum(), 1)
    # trapezoid rule
    return float(np.trapz(tpr, fpr))


def _average_precision(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    try:
        from sklearn.metrics import average_precision_score
        if n_classes == 2:
            prob = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
            return float(average_precision_score(y_true, prob))
        # one-vs-rest macro
        scores = []
        for c in range(n_classes):
            binary = (y_true == c).astype(int)
            scores.append(float(average_precision_score(binary, y_prob[:, c])))
        return float(np.mean(scores))
    except ImportError:
        return float("nan")


def _prf_macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    ps, rs, f1s = [], [], []
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        ps.append(p); rs.append(r); f1s.append(f)
    return float(np.mean(ps)), float(np.mean(rs)), float(np.mean(f1s))
