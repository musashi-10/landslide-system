"""
Model Evaluation Metrics
========================

Computes the full set of classification metrics required for a landslide
early-warning system.

Metrics reported
----------------
- Precision         : fraction of HIGH-risk predictions that are real events
- Recall            : fraction of real events that are flagged as HIGH-risk
- F1 score          : harmonic mean of precision and recall
- ROC-AUC           : discrimination ability across all thresholds
- PR-AUC            : precision-recall curve area (preferred for imbalanced data)
- Confusion matrix  : TP, FP, FN, TN counts
- False negative rate (FNR) : fraction of real events missed
- False positive rate (FPR) : fraction of non-events falsely alarmed

Why PR-AUC over ROC-AUC?
------------------------
Under severe class imbalance (e.g. 1% positive rate), ROC-AUC can appear
high even for a poor classifier because the True Negative Rate component
is always very good.  PR-AUC more directly measures the cost of false
alarms vs missed events, which is the operational concern.

Why report FNR separately?
--------------------------
For an early-warning system, a false negative (missed landslide) is far
more dangerous than a false positive (unnecessary alert).  FNR is
explicitly reported and flagged if it exceeds 30%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import logging
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.5


@dataclass
class MetricsResult:
    """
    Full evaluation metrics for one model on one data split.

    Attributes
    ----------
    model_name : str
    precision : float
    recall : float
    f1 : float
    roc_auc : float
    pr_auc : float        (average precision score)
    confusion : dict      keys: tp, fp, fn, tn
    false_negative_rate : float
    false_positive_rate : float
    brier_score : float   (probability calibration measure)
    threshold : float     (classification threshold used)
    n_samples : int
    n_positives : int
    """

    model_name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion: dict
    false_negative_rate: float
    false_positive_rate: float
    brier_score: float
    threshold: float
    n_samples: int
    n_positives: int

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "confusion": self.confusion,
            "false_negative_rate": self.false_negative_rate,
            "false_positive_rate": self.false_positive_rate,
            "brier_score": self.brier_score,
            "threshold": self.threshold,
            "n_samples": self.n_samples,
            "n_positives": self.n_positives,
        }

    def log(self) -> None:
        """Log a human-readable summary."""
        logger.info(
            f"[{self.model_name}] "
            f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} "
            f"ROC-AUC={self.roc_auc:.3f} PR-AUC={self.pr_auc:.3f} "
            f"FNR={self.false_negative_rate:.3f} "
            f"Brier={self.brier_score:.4f} "
            f"(n={self.n_samples}, pos={self.n_positives})"
        )


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "model",
    threshold: float = _DEFAULT_THRESHOLD,
) -> MetricsResult:
    """
    Compute the full evaluation metric set.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    model_name : str
        Label for logging.
    threshold : float
        Decision threshold for converting probabilities to binary predictions.

    Returns
    -------
    MetricsResult
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_pred = (y_prob >= threshold).astype(int)

    n_samples = len(y_true)
    n_positives = int(y_true.sum())

    if n_positives == 0:
        logger.warning(
            f"[{model_name}] No positive samples in this split. "
            "Metrics may be undefined."
        )
        return MetricsResult(
            model_name=model_name,
            precision=0.0, recall=0.0, f1=0.0,
            roc_auc=0.5, pr_auc=0.0,
            confusion={"tp": 0, "fp": 0, "fn": 0, "tn": n_samples},
            false_negative_rate=1.0, false_positive_rate=0.0,
            brier_score=float(np.mean((y_prob - y_true) ** 2)),
            threshold=threshold,
            n_samples=n_samples, n_positives=0,
        )

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.5  # single class in split

    pr_auc = float(average_precision_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tp = int(cm[0, 0]); fp = fn = tn = 0

    n_neg = n_samples - n_positives
    fnr = fn / n_positives if n_positives > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0

    brier = float(brier_score_loss(y_true, y_prob))

    result = MetricsResult(
        model_name=model_name,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion={"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        false_negative_rate=float(fnr),
        false_positive_rate=float(fpr),
        brier_score=brier,
        threshold=threshold,
        n_samples=n_samples,
        n_positives=n_positives,
    )

    result.log()
    return result
