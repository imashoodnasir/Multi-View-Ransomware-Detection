
"""
metrics.py
----------
Classification metrics (Accuracy, Precision, Recall, F1, MCC, ROC-AUC)
and hazard prediction metrics (Concordance Index, Brier Score, Calibration Error).
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             matthews_corrcoef, roc_auc_score)


def classification_metrics(y_true, y_pred, y_proba=None, average: str = "macro", num_classes: Optional[int] = None) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    out = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    # ROC-AUC
    if y_proba is not None:
        try:
            if y_proba.ndim == 1 or (num_classes and num_classes == 2):
                auc = roc_auc_score(y_true, y_proba)  # binary
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
            out["roc_auc"] = float(auc)
        except Exception:
            pass
    # MCC
    try:
        out["mcc"] = matthews_corrcoef(y_true, y_pred)
    except Exception:
        pass
    return out


def concordance_index(event_times: np.ndarray, predicted_scores: np.ndarray, event_observed: Optional[np.ndarray] = None) -> float:
    """
    Harrell's C-index for survival ordering. Higher is better.
    event_times: true times
    predicted_scores: risk scores (higher = riskier / shorter time)
    event_observed: 1 if event occurred, 0 if censored. If None, assumes all observed.
    """
    n = len(event_times)
    if event_observed is None:
        event_observed = np.ones(n, dtype=int)
    # Count comparable pairs
    num, den = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if event_times[i] == event_times[j]:
                continue
            # pair is comparable if the shorter time is an observed event
            if event_times[i] < event_times[j] and event_observed[i] == 1:
                den += 1
                if predicted_scores[i] > predicted_scores[j]:
                    num += 1
                elif predicted_scores[i] == predicted_scores[j]:
                    num += 0.5
            elif event_times[j] < event_times[i] and event_observed[j] == 1:
                den += 1
                if predicted_scores[j] > predicted_scores[i]:
                    num += 1
                elif predicted_scores[j] == predicted_scores[i]:
                    num += 0.5
    return float(num / den) if den > 0 else 0.0


def brier_score(y_true_prob: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """
    Brier score for probabilistic predictions in [0,1]. Lower is better.
    Here used as time-agnostic simplification (can be extended to time-dependent BS).
    """
    y_true_prob = np.asarray(y_true_prob, dtype=float)
    y_pred_prob = np.asarray(y_pred_prob, dtype=float)
    return float(np.mean((y_true_prob - y_pred_prob) ** 2))


def mean_absolute_calibration_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Simplified calibration error: mean absolute error between normalized true times and predicted.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-8
    y_true_n = (y_true - y_true.min()) / (y_true.max() - y_true.min() + eps)
    y_pred_n = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + eps)
    return float(np.mean(np.abs(y_true_n - y_pred_n)))
