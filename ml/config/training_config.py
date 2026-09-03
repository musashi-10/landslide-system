"""
ML Training Configuration
=========================

Single source of truth for all training hyperparameters, feature lists,
risk thresholds, file paths, and random seeds.

All configuration is serialisable to JSON so that every training run can
be fully reproduced from a saved config file.

Usage
-----
    from ml.config import get_default_config

    cfg = get_default_config()
    cfg.random_seed = 42
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class RiskThresholds:
    """
    Configurable probability thresholds for risk level classification.

    ⚠️  IMPORTANT: These thresholds are provisional starting values.
    They have NOT been scientifically validated against operational data.
    Final thresholds must be selected using:
      - Precision-recall tradeoff analysis on the validation set
      - Domain-expert review
      - Operational false-alarm rate constraints

    Attributes
    ----------
    low_max : float
        Probabilities strictly below this value → LOW risk.
    moderate_max : float
        Probabilities in [low_max, moderate_max) → MODERATE risk.
    high_max : float
        Probabilities in [moderate_max, high_max) → HIGH risk.
    critical_min : float
        Probabilities ≥ critical_min → CRITICAL risk.
        (Must equal high_max.)
    """

    low_max: float = 0.30
    moderate_max: float = 0.55
    high_max: float = 0.75
    critical_min: float = 0.75  # must equal high_max

    def validate(self) -> None:
        """Raise ValueError if thresholds are inconsistent."""
        if not (0.0 < self.low_max < self.moderate_max < self.high_max <= 1.0):
            raise ValueError("Risk thresholds must be strictly increasing in (0, 1].")
        if self.critical_min != self.high_max:
            raise ValueError("critical_min must equal high_max.")


@dataclass
class TrainingConfig:
    """
    Complete training configuration for one ML training run.

    All fields are serialisable so the config can be saved alongside
    the model artifact for full reproducibility.
    """

    # ── Reproducibility ────────────────────────────────────────────────
    random_seed: int = 42
    model_version: str = "v1"
    dataset_version: str = "v1"
    preprocessing_version: str = "1.0.0"

    # ── Feature lists ─────────────────────────────────────────────────
    static_features: list[str] = field(default_factory=lambda: [
        "elevation",
        "slope",
        "aspect",
        "historical_landslide",
    ])

    categorical_features: list[str] = field(default_factory=lambda: [
        "soil_type",
        "geology",
        "land_cover",
    ])

    dynamic_features: list[str] = field(default_factory=lambda: [
        "rainfall_1h",
        "rainfall_6h",
        "rainfall_24h",
        "rainfall_3d",
        "rainfall_7d",
        "forecast_rainfall",
        "moisture_indicator",
    ])

    satellite_features: list[str] = field(default_factory=lambda: [
        # Optional — model must run without these
        "ndvi",
        "vegetation_change",
        "bare_surface_indicator",
        "change_indicator",
    ])

    engineered_features: list[str] = field(default_factory=lambda: [
        "rainfall_intensity",
        "antecedent_rainfall",
        "static_susceptibility_score",
        "rain_x_slope",
        "rain_x_susceptibility",
    ])

    # ── Validation strategy ────────────────────────────────────────────
    # See ml/training/validation_strategy.py for rationale
    temporal_test_fraction: float = 0.20   # last 20% of timeline → test set
    temporal_val_fraction: float = 0.10    # preceding 10% → validation set
    n_spatial_cv_folds: int = 5            # spatial block CV folds

    # ── Model hyperparameters ──────────────────────────────────────────
    logistic_regression: dict = field(default_factory=lambda: {
        "max_iter": 1000,
        "class_weight": "balanced",
        "C": 1.0,
        "solver": "lbfgs",
    })

    random_forest: dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "n_jobs": -1,
    })

    xgboost: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        # scale_pos_weight is set dynamically from class ratio in trainer
    })

    # ── Risk classification ────────────────────────────────────────────
    risk_thresholds: RiskThresholds = field(default_factory=RiskThresholds)

    # ── Paths ──────────────────────────────────────────────────────────
    model_dir: str = "ml/models/artifacts"
    config_dir: str = "ml/config"
    data_dir: str = "ml/data"

    # ── Target definition ──────────────────────────────────────────────
    # See ml/training/target.py for full documentation
    event_buffer_days: int = 7       # temporal buffer around events
    event_buffer_km: float = 5.0     # spatial buffer around events (km)
    min_negative_distance_km: float = 5.0  # min distance from any event for negatives

    # ── Class imbalance ────────────────────────────────────────────────
    # Strategy: use class_weight / scale_pos_weight rather than synthetic
    # oversampling, to avoid amplifying spatially-correlated noise.
    # See ml/training/trainer.py for implementation.
    use_class_weights: bool = True

    def feature_columns(self, include_satellite: bool = False) -> list[str]:
        """Return the full ordered feature column list used by the model."""
        cols = (
            self.static_features
            + self.categorical_features
            + self.dynamic_features
            + self.engineered_features
        )
        if include_satellite:
            cols = cols + self.satellite_features
        return cols

    def to_dict(self) -> dict:
        """Return JSON-serialisable representation."""
        d = asdict(self)
        return d

    def save(self, path: Optional[str] = None) -> Path:
        """Persist config to JSON file."""
        if path is None:
            out = Path(self.config_dir) / "training_run.json"
        else:
            out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        # Reconstruct RiskThresholds
        if "risk_thresholds" in data:
            data["risk_thresholds"] = RiskThresholds(**data["risk_thresholds"])
        return cls(**data)


def get_default_config() -> TrainingConfig:
    """Return a default TrainingConfig with validated thresholds."""
    cfg = TrainingConfig()
    cfg.risk_thresholds.validate()
    return cfg
