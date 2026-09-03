"""
Predictor
=========

Clean inference interface for the landslide risk model.

Design goals
------------
- Engineer 6 (dashboard) calls predict() with a features dict.
- No training code needs to be imported.
- The model is loaded ONCE (singleton pattern) — not per request.
- All errors are caught and returned as structured error responses.
- The preprocessor and model are loaded from the model registry.

Usage
-----
    predictor = Predictor.from_registry(model_dir="ml/models/artifacts", version="v1")
    result = predictor.predict({
        "location_id": "LOC_001",
        "timestamp_utc": "2026-09-02T12:00:00Z",
        "features": {
            "elevation": 1240.5,
            "slope": 32.4,
            ...
        }
    })

The result is a dict that serialises directly to the API contract.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from shared.schemas import RiskPrediction
from ml.inference.risk_classifier import RiskClassifier, RiskLevel
from ml.features.engineering import FeatureEngineer
from ml.models.registry import ModelRegistry, ModelArtifact

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Raised when inference fails in a recoverable way."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


class Predictor:
    """
    Production inference interface.

    Wraps the loaded model artifact + preprocessor + risk classifier.
    The model is loaded once and cached; there is no per-request loading.

    Parameters
    ----------
    artifact : ModelArtifact
        Loaded model artifact from ModelRegistry.
    risk_classifier : RiskClassifier, optional
        Risk level classifier.  Uses default thresholds if not provided.
    explainer : optional
        LandslideExplainer instance (loaded separately to keep this class lean).
    """

    def __init__(
        self,
        artifact: ModelArtifact,
        risk_classifier: Optional[RiskClassifier] = None,
        explainer: Optional[Any] = None,
    ) -> None:
        self._artifact = artifact
        self._model = artifact.model
        self._preprocessor = artifact.preprocessor
        self._feature_cols = artifact.feature_cols
        self._risk_classifier = risk_classifier or RiskClassifier()
        self._explainer = explainer
        self._feature_engineer = FeatureEngineer()

        logger.info(
            f"Predictor ready — model={artifact.model_type}, "
            f"version={artifact.version}, "
            f"features={len(self._feature_cols)}"
        )

    @classmethod
    def from_registry(
        cls,
        model_dir: str = "ml/models/artifacts",
        version: Optional[str] = None,
        risk_classifier: Optional[RiskClassifier] = None,
    ) -> "Predictor":
        """
        Load a Predictor from the model registry.

        Parameters
        ----------
        model_dir : str
        version : str, optional
            If None, loads the latest available version.
        risk_classifier : RiskClassifier, optional

        Raises
        ------
        PredictionError : if no model is available.
        """
        registry = ModelRegistry(model_dir=model_dir)
        if version is None:
            version = registry.latest_version()
        if version is None:
            raise PredictionError(
                "MODEL_UNAVAILABLE",
                "No trained model found.  Run training first or use MockPredictor.",
            )
        if not registry.exists(version):
            raise PredictionError(
                "MODEL_VERSION_NOT_FOUND",
                f"Model version '{version}' not found in {model_dir}.",
            )

        artifact = registry.load(version)
        return cls(artifact=artifact, risk_classifier=risk_classifier)

    def predict(self, request: dict) -> RiskPrediction:
        """
        Run inference on a single prediction request.

        Parameters
        ----------
        request : dict
            Must contain: location_id, timestamp_utc, features (dict).

        Returns
        -------
        RiskPrediction

        Raises
        ------
        PredictionError : on validation or inference failure.
        """
        # ── Validate request structure ────────────────────────────────────────
        _validate_request(request)

        location_id: str = request["location_id"]
        timestamp_utc: str = request["timestamp_utc"]
        raw_features: dict = request["features"]

        # ── Build feature row ─────────────────────────────────────────────────
        try:
            row_df = _features_to_dataframe(raw_features, location_id, timestamp_utc)
        except Exception as exc:
            raise PredictionError(
                "FEATURE_PREPARATION_ERROR",
                f"Failed to build feature row: {exc}",
            ) from exc

        # ── Feature engineering ───────────────────────────────────────────────
        try:
            row_df = self._feature_engineer.transform(row_df)
        except Exception as exc:
            raise PredictionError(
                "FEATURE_ENGINEERING_ERROR",
                f"Feature engineering failed: {exc}",
            ) from exc

        # ── Preprocessing ─────────────────────────────────────────────────────
        try:
            X = self._preprocessor.get_feature_matrix(row_df)
        except Exception as exc:
            raise PredictionError(
                "PREPROCESSING_ERROR",
                f"Preprocessing failed: {exc}",
            ) from exc

        # ── Model prediction ──────────────────────────────────────────────────
        try:
            prob_array = self._model.predict_proba(X)
            risk_probability = float(prob_array[0, 1])
        except Exception as exc:
            raise PredictionError(
                "MODEL_INFERENCE_ERROR",
                f"Model prediction failed: {exc}",
            ) from exc

        # Sanity check
        if not (0.0 <= risk_probability <= 1.0):
            raise PredictionError(
                "INVALID_PROBABILITY",
                f"Model returned invalid probability: {risk_probability}",
            )

        # ── Risk classification ────────────────────────────────────────────────
        risk_level: RiskLevel = self._risk_classifier.classify(risk_probability)

        # ── Explainability ────────────────────────────────────────────────────
        top_risk_factors = _compute_top_factors(
            row_df=row_df,
            X=X,
            feature_cols=self._feature_cols,
            explainer=self._explainer,
        )

        return RiskPrediction(
            location_id=location_id,
            timestamp_utc=timestamp_utc,
            risk_probability=risk_probability,
            risk_level=risk_level.value,
            top_risk_factors=top_risk_factors,
        )

    @property
    def model_version(self) -> str:
        return self._artifact.version

    @property
    def model_type(self) -> str:
        return self._artifact.model_type


# ── Helpers ───────────────────────────────────────────────────────────────────


def _validate_request(request: dict) -> None:
    """Validate the prediction request dict structure."""
    if not isinstance(request, dict):
        raise PredictionError("MALFORMED_REQUEST", "Request must be a JSON object.")

    for field in ("location_id", "timestamp_utc", "features"):
        if field not in request:
            raise PredictionError(
                "MISSING_FIELD",
                f"Required field '{field}' is missing from the request.",
            )

    if not isinstance(request["features"], dict):
        raise PredictionError(
            "INVALID_FEATURES",
            "'features' must be a JSON object (dict).",
        )


def _features_to_dataframe(
    features: dict, location_id: str, timestamp_utc: str
) -> pd.DataFrame:
    """Convert a features dict to a single-row DataFrame."""
    row = {
        "location_id": location_id,
        "timestamp_utc": timestamp_utc,
        **features,
    }
    return pd.DataFrame([row])


def _compute_top_factors(
    row_df: pd.DataFrame,
    X: np.ndarray,
    feature_cols: list[str],
    explainer: Optional[Any],
    n_top: int = 5,
) -> list[str]:
    """
    Derive top_risk_factors for the API response.

    If an explainer is available, uses SHAP values.
    Otherwise, falls back to feature values × provisional importance weights.
    """
    if explainer is not None:
        try:
            local_importance = explainer.local_explanation(row_df)
            return [item["feature"] for item in local_importance[:n_top]]
        except Exception as exc:
            logger.warning(f"Explainer failed, falling back to value-based factors: {exc}")

    # Fallback: rank features by absolute value (normalised)
    if X.shape[1] == 0 or not feature_cols:
        return []

    values = np.abs(X[0])
    n_cols = min(len(feature_cols), len(values))
    ranked = sorted(
        range(n_cols),
        key=lambda i: values[i],
        reverse=True,
    )
    raw_factors = [feature_cols[i] for i in ranked[:n_top]]
    return [_humanise_factor(f) for f in raw_factors]


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
}


def _humanise_factor(feature_name: str) -> str:
    """Convert internal feature name to a human-readable risk factor label."""
    return _FACTOR_LABELS.get(feature_name, feature_name)
