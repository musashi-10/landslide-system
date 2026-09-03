"""
ML Provider abstraction.

The backend never imports Engineer 5's training code directly.
It always talks to ML inference through this abstraction.

Hierarchy
---------
MLProvider (ABC)
    MockMLProvider      – deterministic mock; used by default & in tests
    HttpMLProvider      – calls a remote ML service endpoint
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

# Risk levels in ascending order of severity
_RISK_LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

# Feature-name → human-readable factor label
_FACTOR_LABELS: dict[str, str] = {
    "rainfall_24h": "high_24h_rainfall",
    "rainfall_7d": "high_7d_rainfall",
    "slope": "steep_slope",
    "susceptibility": "high_susceptibility",
    "historical_landslide": "historical_landslide_activity",
    "moisture_indicator": "high_soil_moisture",
    "rainfall_1h": "intense_1h_rainfall",
}


class MLProvider(ABC):
    """Abstract ML inference provider."""

    @abstractmethod
    def predict(self, location_id: str, features: dict, timestamp_utc: str | None = None) -> dict:
        """
        Return a risk prediction dict with keys:
            location_id, timestamp_utc, risk_probability, risk_level, top_risk_factors
        """


class MockMLProvider(MLProvider):
    """
    Deterministic mock ML provider.

    Produces stable predictions based on a hash of the location_id so that:
    - Tests always get the same result for the same input.
    - Different locations get different risk levels.
    - The full contract path is exercised end-to-end.

    High-risk locations (those whose hash maps to HIGH/CRITICAL) are:
    any location_id that, when hashed, yields index ≥ 2 in _RISK_LEVELS.
    """

    # Named locations used by tests and the end-to-end scenario
    _SEEDED: dict[str, tuple[float, str, list[str]]] = {
        "LOC_TEST_CRITICAL": (
            0.92,
            "CRITICAL",
            ["high_24h_rainfall", "steep_slope", "high_susceptibility"],
        ),
        "LOC_TEST_HIGH": (
            0.78,
            "HIGH",
            ["high_24h_rainfall", "steep_slope"],
        ),
        "LOC_TEST_MODERATE": (
            0.45,
            "MODERATE",
            ["high_soil_moisture"],
        ),
        "LOC_TEST_LOW": (
            0.12,
            "LOW",
            [],
        ),
    }

    def predict(self, location_id: str, features: dict, timestamp_utc: str | None = None) -> dict:
        ts = timestamp_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Seeded overrides take priority
        if location_id in self._SEEDED:
            prob, level, factors = self._SEEDED[location_id]
        else:
            prob, level, factors = self._derive_from_hash(location_id, features)

        return {
            "location_id": location_id,
            "timestamp_utc": ts,
            "risk_probability": prob,
            "risk_level": level,
            "top_risk_factors": factors,
        }

    @staticmethod
    def _derive_from_hash(location_id: str, features: dict) -> tuple[float, str, list[str]]:
        """Produce a stable but varied prediction from location_id hash."""
        digest = int(hashlib.md5(location_id.encode()).hexdigest(), 16)  # noqa: S324
        level_idx = digest % len(_RISK_LEVELS)
        level = _RISK_LEVELS[level_idx]

        # Probability varies within the level's natural band
        band_starts = {0: 0.05, 1: 0.35, 2: 0.65, 3: 0.85}
        base = band_starts[level_idx]
        prob = round(base + (digest % 100) / 1000, 3)
        prob = min(max(prob, 0.0), 1.0)

        # Pick top risk factors from features present
        factors: list[str] = []
        for feat, label in _FACTOR_LABELS.items():
            val = features.get(feat)
            if val is not None and isinstance(val, (int, float)) and val > 0:
                factors.append(label)
            if len(factors) >= 3:
                break

        return prob, level, factors[:3]


class HttpMLProvider(MLProvider):
    """
    Calls a remote ML inference service over HTTP.

    The remote service must implement POST /predict following
    docs/api-contract.md Section 8.
    """

    def __init__(self, base_url: str, timeout: int = 10) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def predict(self, location_id: str, features: dict, timestamp_utc: str | None = None) -> dict:
        ts = timestamp_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        payload = {
            "location_id": location_id,
            "timestamp_utc": ts,
            "features": features,
        }
        try:
            resp = httpx.post(
                f"{self._base_url}/predict",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            logger.error("ML service call failed: %s", exc)
            raise RuntimeError(f"ML service unavailable: {exc}") from exc
