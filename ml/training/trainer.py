"""
Model Trainer
=============

Trains and compares three baseline models:
  1. Logistic Regression  — linear baseline
  2. Random Forest        — non-linear, robust baseline
  3. XGBoost             — gradient boosting; often best for tabular data

Class imbalance strategy
------------------------
Landslide events are much rarer than non-events.  We do NOT use synthetic
oversampling (e.g. SMOTE) because:
  - Spatially correlated samples would be interpolated nonsensically.
  - Oversampled minority points near class boundaries could introduce noise.

Instead we use:
  - class_weight='balanced' for sklearn models (LR, RF)
  - scale_pos_weight = (n_negatives / n_positives) for XGBoost

This re-weights the loss function without creating synthetic data.

Model selection criterion
--------------------------
We use PR-AUC (area under precision-recall curve) as the primary selection
metric.  Rationale:
  - ROC-AUC is optimistic under heavy class imbalance.
  - Accuracy is dominated by the majority class.
  - PR-AUC focuses on the positive (landslide) class, which is what matters
    for an early-warning system.
  - F1 at a single threshold is less informative than the full curve.

The model with highest val_pr_auc is selected and saved.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False

from ml.config.training_config import TrainingConfig, get_default_config
from ml.evaluation.metrics import compute_metrics, MetricsResult

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """
    Outcome of one complete training run.

    Attributes
    ----------
    model_name : str
        Human-readable model name.
    model : object
        The fitted sklearn-compatible estimator.
    train_metrics : MetricsResult
        Metrics on training data.
    val_metrics : MetricsResult
        Metrics on validation data.
    test_metrics : MetricsResult
        Metrics on held-out test data.
    feature_cols : list[str]
        Feature columns used.
    class_weight_strategy : str
        Description of imbalance handling used.
    training_time_seconds : float
        Wall-clock training time.
    """

    model_name: str
    model: object
    train_metrics: MetricsResult
    val_metrics: MetricsResult
    test_metrics: MetricsResult
    feature_cols: list[str]
    class_weight_strategy: str
    training_time_seconds: float

    def summary_row(self) -> dict:
        """Return a one-row dict for model comparison table."""
        return {
            "model": self.model_name,
            "train_pr_auc": round(self.train_metrics.pr_auc, 4),
            "val_pr_auc": round(self.val_metrics.pr_auc, 4),
            "test_pr_auc": round(self.test_metrics.pr_auc, 4),
            "val_roc_auc": round(self.val_metrics.roc_auc, 4),
            "val_f1": round(self.val_metrics.f1, 4),
            "val_recall": round(self.val_metrics.recall, 4),
            "val_precision": round(self.val_metrics.precision, 4),
            "val_fnr": round(self.val_metrics.false_negative_rate, 4),
            "train_time_s": round(self.training_time_seconds, 2),
        }


class ModelTrainer:
    """
    Trains multiple models, compares them, and returns the best one.

    Parameters
    ----------
    config : TrainingConfig, optional
        Training configuration.  Uses default if not provided.
    """

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        self.config = config or get_default_config()

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_cols: list[str],
    ) -> tuple[list[TrainingResult], TrainingResult]:
        """
        Train all configured models, compare them, and select the best.

        Parameters
        ----------
        X_train, y_train : training features and labels
        X_val, y_val     : validation features and labels
        X_test, y_test   : held-out test features and labels
        feature_cols      : ordered list of feature names

        Returns
        -------
        all_results : list[TrainingResult]
            Results for every trained model.
        best : TrainingResult
            The model with highest val_pr_auc.
        """
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())

        if n_pos == 0:
            raise ValueError(
                "Training data contains no positive samples.  "
                "Cannot train a landslide classifier."
            )

        scale_pos_weight = n_neg / n_pos
        logger.info(
            f"Training — {n_pos} positives, {n_neg} negatives, "
            f"scale_pos_weight={scale_pos_weight:.2f}"
        )

        models_to_train = [
            ("LogisticRegression", self._build_lr()),
            ("RandomForest", self._build_rf()),
        ]
        if _HAS_XGB:
            models_to_train.append(
                ("XGBoost", self._build_xgb(scale_pos_weight))
            )
        else:
            logger.warning("XGBoost not available; skipping XGBoost model.")

        all_results: list[TrainingResult] = []
        for name, model in models_to_train:
            result = self._train_one(
                name=name,
                model=model,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test,
                feature_cols=feature_cols,
                scale_pos_weight=scale_pos_weight,
            )
            all_results.append(result)

        # ── Model comparison ─────────────────────────────────────────────────
        comparison = pd.DataFrame([r.summary_row() for r in all_results])
        logger.info(f"\nModel comparison:\n{comparison.to_string(index=False)}")

        # Select by val PR-AUC
        best = max(all_results, key=lambda r: r.val_metrics.pr_auc)
        logger.info(f"Selected model: {best.model_name} (val PR-AUC={best.val_metrics.pr_auc:.4f})")

        # Log false-negative rate warning
        fnr = best.test_metrics.false_negative_rate
        if fnr > 0.30:
            logger.warning(
                f"⚠️  Best model false-negative rate on test set = {fnr:.2%}. "
                "For an early-warning system, this means 1 in "
                f"{1/fnr:.0f} real events may be missed.  "
                "Consider threshold adjustment."
            )

        return all_results, best

    def _train_one(
        self,
        name: str,
        model: object,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_cols: list[str],
        scale_pos_weight: float,
    ) -> TrainingResult:
        logger.info(f"Training {name} …")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        train_metrics = compute_metrics(y_train, model.predict_proba(X_train)[:, 1], name)
        val_metrics = compute_metrics(y_val, model.predict_proba(X_val)[:, 1], name)
        test_metrics = compute_metrics(y_test, model.predict_proba(X_test)[:, 1], name)

        logger.info(
            f"{name} — val PR-AUC={val_metrics.pr_auc:.4f}, "
            f"val ROC-AUC={val_metrics.roc_auc:.4f}, "
            f"val F1={val_metrics.f1:.4f} ({elapsed:.1f}s)"
        )

        return TrainingResult(
            model_name=name,
            model=model,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            feature_cols=feature_cols,
            class_weight_strategy=(
                "balanced class_weight"
                if name != "XGBoost"
                else f"scale_pos_weight={scale_pos_weight:.2f}"
            ),
            training_time_seconds=elapsed,
        )

    def _build_lr(self) -> LogisticRegression:
        cfg = self.config.logistic_regression
        return LogisticRegression(
            max_iter=cfg.get("max_iter", 1000),
            class_weight=cfg.get("class_weight", "balanced"),
            C=cfg.get("C", 1.0),
            solver=cfg.get("solver", "lbfgs"),
            random_state=self.config.random_seed,
        )

    def _build_rf(self) -> RandomForestClassifier:
        cfg = self.config.random_forest
        return RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth", None),
            min_samples_leaf=cfg.get("min_samples_leaf", 5),
            class_weight=cfg.get("class_weight", "balanced"),
            n_jobs=cfg.get("n_jobs", -1),
            random_state=self.config.random_seed,
        )

    def _build_xgb(self, scale_pos_weight: float) -> "XGBClassifier":
        cfg = self.config.xgboost
        return XGBClassifier(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", 6),
            learning_rate=cfg.get("learning_rate", 0.05),
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            scale_pos_weight=scale_pos_weight,
            eval_metric=cfg.get("eval_metric", "aucpr"),
            random_state=self.config.random_seed,
            verbosity=0,
        )
