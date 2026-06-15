"""Classification metrics: AUC, Accuracy, F1, Precision, Recall, AP, ROC/PR curves."""
import numpy as np
import torch
from typing import Dict, List, Tuple


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
            dict: auc, auc_pr, accuracy, f1, precision, recall
        """
        y_true, y_prob, y_pred, n_classes = _prepare(targets, logits, threshold)
        return {
            "auc":       _roc_auc(y_true, y_prob, n_classes),
            "auc_pr":    _average_precision(y_true, y_prob, n_classes),
            "accuracy":  float((y_pred == y_true).mean()),
            **dict(zip(["precision", "recall", "f1"], _prf_macro(y_true, y_pred, n_classes))),
        }

    @staticmethod
    def compute_curves(
        targets: torch.Tensor,
        logits: torch.Tensor,
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Returns ROC and PR curve data for binary classification.

        Returns:
            {
              "roc": {"fpr": [...], "tpr": [...], "thresholds": [...]},
              "pr":  {"precision": [...], "recall": [...], "thresholds": [...]},
            }
        """
        y_true, y_prob, _, n_classes = _prepare(targets, logits)
        if n_classes != 2:
            return {"roc": {}, "pr": {}}

        prob = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
        return {
            "roc": _roc_curve(y_true, prob),
            "pr":  _pr_curve(y_true, prob),
        }

    @staticmethod
    def compute_confusion_matrix(
        targets: torch.Tensor,
        logits: torch.Tensor,
        threshold: float = 0.5,
    ) -> List[List[int]]:
        """Returns confusion matrix as list-of-lists (for JSON serialisation)."""
        y_true, _, y_pred, n_classes = _prepare(targets, logits, threshold)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        return cm.tolist()

    @staticmethod
    def compute_auc(targets, logits) -> float:
        return ClassificationMetrics.compute_all(targets, logits)["auc"]

    @staticmethod
    def compute_accuracy(targets, logits) -> float:
        return ClassificationMetrics.compute_all(targets, logits)["accuracy"]

    @staticmethod
    def compute_f1(targets, logits) -> float:
        return ClassificationMetrics.compute_all(targets, logits)["f1"]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _prepare(targets, logits, threshold=0.5):
    y_true  = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    y_score = logits.detach().cpu().numpy()  if isinstance(logits,  torch.Tensor) else np.asarray(logits)

    if y_true.ndim == 2:
        y_true = y_true.argmax(axis=1)

    if y_score.ndim == 1:
        y_prob   = _sigmoid(y_score)
        y_pred   = (y_prob >= threshold).astype(int)
        n_classes = 2
    else:
        y_prob   = _softmax(y_score)
        y_pred   = y_prob.argmax(axis=1)
        n_classes = y_score.shape[1]

    return y_true, y_prob, y_pred, n_classes


def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _roc_auc(y_true, y_prob, n_classes):
    try:
        from sklearn.metrics import roc_auc_score
        prob = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
        if n_classes == 2:
            return float(roc_auc_score(y_true, prob))
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ImportError:
        pass
    if n_classes > 2:
        return float("nan")
    prob = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
    return float(_binary_roc_auc(y_true, prob))


def _binary_roc_auc(y_true, y_score):
    order    = np.argsort(-y_score)
    y_sorted = y_true[order]
    tpr = np.cumsum(y_sorted)      / max(y_true.sum(), 1)
    fpr = np.cumsum(1 - y_sorted)  / max((1 - y_true).sum(), 1)
    return float(np.trapz(tpr, fpr))


def _roc_curve(y_true, y_score) -> Dict[str, List[float]]:
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y_true, y_score)
        # Downsample to ≤200 points for JSON size
        idx = np.linspace(0, len(fpr)-1, min(200, len(fpr)), dtype=int)
        return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist(), "thresholds": thr[idx].tolist()}
    except ImportError:
        pass
    # Fallback: compute manually
    order    = np.argsort(-y_score)
    y_sorted = y_true[order]
    P = y_true.sum(); N = len(y_true) - P
    tpr = np.concatenate([[0], np.cumsum(y_sorted)   / max(P, 1), [1]])
    fpr = np.concatenate([[0], np.cumsum(1-y_sorted) / max(N, 1), [1]])
    idx = np.linspace(0, len(fpr)-1, min(200, len(fpr)), dtype=int)
    return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist(), "thresholds": []}


def _pr_curve(y_true, y_score) -> Dict[str, List[float]]:
    try:
        from sklearn.metrics import precision_recall_curve
        p, r, thr = precision_recall_curve(y_true, y_score)
        idx = np.linspace(0, len(p)-1, min(200, len(p)), dtype=int)
        return {"precision": p[idx].tolist(), "recall": r[idx].tolist(), "thresholds": thr[min(idx, len(thr)-1)].tolist() if len(thr) else []}
    except ImportError:
        return {"precision": [], "recall": [], "thresholds": []}


def _average_precision(y_true, y_prob, n_classes):
    try:
        from sklearn.metrics import average_precision_score
        if n_classes == 2:
            prob = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
            return float(average_precision_score(y_true, prob))
        scores = [float(average_precision_score((y_true==c).astype(int), y_prob[:,c])) for c in range(n_classes)]
        return float(np.mean(scores))
    except ImportError:
        return float("nan")


def _prf_macro(y_true, y_pred, n_classes):
    ps, rs, f1s = [], [], []
    for c in range(n_classes):
        tp = ((y_pred==c)&(y_true==c)).sum(); fp = ((y_pred==c)&(y_true!=c)).sum()
        fn = ((y_pred!=c)&(y_true==c)).sum()
        p  = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f  = 2*p*r/(p+r) if (p+r)>0 else 0.0
        ps.append(p); rs.append(r); f1s.append(f)
    return float(np.mean(ps)), float(np.mean(rs)), float(np.mean(f1s))
