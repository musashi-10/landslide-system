"""
Model Explainability
====================

Provides global and local feature importance using SHAP.

Why SHAP?
---------
SHAP (SHapley Additive exPlanations) provides theoretically-grounded,
consistent feature attributions:
  - Global importance: mean |SHAP| across the dataset
  - Local importance: SHAP values for one prediction

Unlike permutation importance or built-in tree importance:
  - SHAP values are additive and sum to the model's output
  - Local SHAP values directly explain "why this prediction"
  - They can be compared across models

Engineer 6 integration
-----------------------
The Dashboard engineer does NOT need to import model internals.
This module returns plain Python dicts/lists that serialise to JSON.

Output format
-------------
    [
      {"feature": "rainfall_24h",         "shap_value": +0.42, "label": "high_24h_rainfall"},
      {"feature": "slope",                "shap_value": +0.31, "label": "steep_slope"},
      {"feature": "static_susceptibility", "shap_value": +0.18, "label": "high_susceptibility"},
      ...
    ]

Text summary
------------
    Prediction: HIGH

    rainfall_24h          +++ (0.42)
    slope                 ++  (0.31)
    static_susceptibility ++  (0.18)
    antecedent_rainfall   +   (0.09)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import SHAP; fall back to built-in importance if unavailable
try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:  # pragma: no cover
    _HAS_SHAP = False
    logger.warning(
        "SHAP not installed.  Falling back to built-in feature importance. "
        "Install with: pip install shap"
    )

_FACTOR_LABELS = {
    "rainfall_24h": "high_24h_rainfall",
    "rainfall_3d": "high_3day_rainfall",
    "rainfall_7d": "high_7day_rainfall",
    "rainfall_1h": "high_hourly_rainfall",
    "slope": "steep_slope",
    "elevation": "high_elevation",
    "static_susceptibility_score": "high_susceptibility",
    "rain_x_slope": "high_rainfall_on_steep_slope",
    "rain_x_susceptibility": "high_rainfall_in_susceptible_area",
    "antecedent_rainfall": "high_antecedent_moisture",
    "rainfall_intensity": "high_rainfall_intensity",
    "moisture_indicator": "high_soil_moisture",
    "historical_landslide": "prior_landslide_history",
    "aspect": "aspect",
    "forecast_rainfall": "high_forecast_rainfall",
}


@dataclass
class FeatureImportance:
    """One feature's importance entry."""

    feature: str        # internal feature name
    label: str          # human-readable label
    shap_value: float   # SHAP value (signed for local; absolute for global)
    rank: int           # 1-indexed rank (1 = most important)

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "label": self.label,
            "shap_value": round(self.shap_value, 6),
            "rank": self.rank,
        }


class LandslideExplainer:
    """
    SHAP-based explainability for the landslide risk model.

    Parameters
    ----------
    model : fitted sklearn/XGBoost estimator
    feature_cols : list[str]
        Feature column names in the order the model expects.
    X_background : np.ndarray, optional
        Background dataset for SHAP KernelExplainer (used for LR).
        Not needed for tree-based models.
    """

    def __init__(
        self,
        model: Any,
        feature_cols: list[str],
        X_background: Optional[np.ndarray] = None,
    ) -> None:
        self._model = model
        self._feature_cols = feature_cols
        self._explainer = None

        if _HAS_SHAP:
            self._explainer = _build_shap_explainer(model, X_background)
            if self._explainer is not None:
                logger.info(f"SHAP explainer ready: {type(self._explainer).__name__}")
        else:
            logger.warning("Using fallback feature importance (SHAP not available).")

    def global_importance(self, X: np.ndarray, n_top: int = 10) -> list[FeatureImportance]:
        """
        Compute global feature importance over a dataset.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        n_top : int
            Number of top features to return.

        Returns
        -------
        list[FeatureImportance]
            Sorted by absolute mean SHAP value (descending).
        """
        if _HAS_SHAP and self._explainer is not None:
            try:
                shap_values = self._explainer.shap_values(X)
                # For classifiers with two outputs, take the positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                return _rank_importance(mean_abs_shap, self._feature_cols, n_top)
            except Exception as exc:
                logger.warning(f"SHAP global importance failed: {exc}. Using fallback.")

        # Fallback: built-in importance
        return _fallback_importance(self._model, self._feature_cols, n_top)

    def local_explanation(
        self, row_df: pd.DataFrame, n_top: int = 5
    ) -> list[FeatureImportance]:
        """
        Compute local feature importance for a single prediction.

        Parameters
        ----------
        row_df : pd.DataFrame
            Single-row DataFrame with the preprocessed features.
        n_top : int
            Number of top factors to return.

        Returns
        -------
        list[FeatureImportance]
            Sorted by |SHAP value| descending.
        """
        X = _df_to_array(row_df, self._feature_cols)

        if _HAS_SHAP and self._explainer is not None:
            try:
                shap_values = self._explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                return _rank_importance(shap_values[0], self._feature_cols, n_top)
            except Exception as exc:
                logger.warning(f"SHAP local explanation failed: {exc}. Using fallback.")

        # Fallback: return feature values as "importance"
        return _fallback_importance(self._model, self._feature_cols, n_top)

    def text_summary(
        self, risk_level: str, importances: list[FeatureImportance]
    ) -> str:
        """
        Format a human-readable text explanation.

        Example
        -------
            Prediction: HIGH

            rainfall_24h          +++ (0.42)
            slope                 ++  (0.31)
            high_susceptibility   ++  (0.18)
        """
        lines = [f"Prediction: {risk_level}", ""]
        for imp in importances:
            bars = _magnitude_bars(imp.shap_value)
            lines.append(f"  {imp.label:<28} {bars:<6} ({imp.shap_value:+.2f})")
        return "\n".join(lines)

    def top_factor_names(self, importances: list[FeatureImportance]) -> list[str]:
        """Extract human-readable factor labels from a list of FeatureImportance."""
        return [imp.label for imp in importances]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _build_shap_explainer(model: Any, X_background: Optional[np.ndarray]) -> Any:
    """Choose the appropriate SHAP explainer for the model type."""
    model_name = type(model).__name__
    try:
        if "RandomForest" in model_name or "XGB" in model_name or "GBM" in model_name:
            return _shap.TreeExplainer(model)
        elif "LogisticRegression" in model_name or "Linear" in model_name:
            if X_background is not None:
                bg = _shap.maskers.Independent(X_background, max_samples=100)
                return _shap.LinearExplainer(model, bg)
            else:
                return _shap.LinearExplainer(model, _shap.maskers.Independent(
                    np.zeros((1, 1))  # placeholder; will work for linear models
                ))
        else:
            logger.warning(
                f"Unknown model type {model_name}; using KernelExplainer (slow)."
            )
            if X_background is not None:
                summary = _shap.kmeans(X_background, 10)
                return _shap.KernelExplainer(model.predict_proba, summary)
            return None
    except Exception as exc:
        logger.warning(f"Could not build SHAP explainer: {exc}")
        return None


def _rank_importance(
    values: np.ndarray, feature_cols: list[str], n_top: int
) -> list[FeatureImportance]:
    """Sort features by absolute value and return top n_top."""
    n = min(len(feature_cols), len(values))
    ranked = sorted(range(n), key=lambda i: abs(values[i]), reverse=True)
    result = []
    for rank, idx in enumerate(ranked[:n_top], start=1):
        name = feature_cols[idx]
        result.append(FeatureImportance(
            feature=name,
            label=_FACTOR_LABELS.get(name, name),
            shap_value=float(values[idx]),
            rank=rank,
        ))
    return result


def _fallback_importance(
    model: Any, feature_cols: list[str], n_top: int
) -> list[FeatureImportance]:
    """Extract built-in feature importance from tree models."""
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            # LogisticRegression: use |coefficients|
            coef = getattr(model, "coef_", None)
            if coef is not None:
                importances = np.abs(coef[0])
        if importances is not None:
            return _rank_importance(importances, feature_cols, n_top)
    except Exception as exc:
        logger.warning(f"Fallback importance failed: {exc}")
    return []


def _df_to_array(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Select feature columns from df and convert to numpy array."""
    present = [c for c in feature_cols if c in df.columns]
    if not present:
        return np.zeros((1, 1))
    return df[present].values.astype(float)


def _magnitude_bars(value: float, max_bars: int = 5) -> str:
    """Convert a SHAP value to a bar string like '+++'."""
    magnitude = min(max_bars, max(1, int(abs(value) * 10)))
    char = "+" if value >= 0 else "-"
    return char * magnitude
