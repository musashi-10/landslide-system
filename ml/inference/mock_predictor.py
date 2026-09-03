"""
Mock Predictor
==============

Deterministic mock predictor for parallel development by Engineer 6.

Provides realistic-looking RiskPrediction outputs without requiring a
trained model.  Mock outputs follow the exact same schema as production.

Usage
-----
    from ml.inference.mock_predictor import MockPredictor

    mock = MockPredictor()
    result = mock.predict({
        "location_id": "LOC_TEST_001",
        "timestamp_utc": "2026-09-02T12:00:00Z",
        "features": {}
    })

The mock is deterministic: the same location_id always returns the same
risk level, allowing Engineer 6 to test the full dashboard rendering.

Known mock locations
--------------------
  LOC_TEST_001  → HIGH    (0.78)
  LOC_TEST_002  → MODERATE (0.45)
  LOC_TEST_003  → CRITICAL (0.91)
  LOC_TEST_004  → LOW     (0.12)
  (any other)   → MODERATE (0.42)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from shared.schemas import RiskPrediction

logger = logging.getLogger(__name__)

# Deterministic mock responses keyed by location_id
_MOCK_RESPONSES: dict[str, tuple[float, str, list[str]]] = {
    "LOC_TEST_001": (
        0.78,
        "HIGH",
        ["high_24h_rainfall", "steep_slope", "high_susceptibility"],
    ),
    "LOC_TEST_002": (
        0.45,
        "MODERATE",
        ["high_antecedent_moisture", "high_elevation"],
    ),
    "LOC_TEST_003": (
        0.91,
        "CRITICAL",
        ["high_24h_rainfall", "steep_slope", "prior_landslide_history", "high_soil_moisture"],
    ),
    "LOC_TEST_004": (
        0.12,
        "LOW",
        ["high_elevation"],
    ),
}

_DEFAULT_MOCK = (0.42, "MODERATE", ["high_antecedent_moisture"])


class MockPredictor:
    """
    Deterministic mock predictor for Engineer 6 integration testing.

    Returns the same prediction for the same location_id every time.
    Does not require a trained model or any data.

    Parameters
    ----------
    mock_responses : dict, optional
        Custom {location_id: (probability, risk_level, factors)} dict.
        If not provided, uses the built-in test set.
    """

    def __init__(self, mock_responses: Optional[dict] = None) -> None:
        self._responses = mock_responses or dict(_MOCK_RESPONSES)
        logger.info(
            f"MockPredictor initialised with {len(self._responses)} mock locations."
        )

    def predict(self, request: dict) -> RiskPrediction:
        """
        Return a deterministic mock RiskPrediction.

        Parameters
        ----------
        request : dict
            Expected keys: location_id, timestamp_utc, features.
            The features dict is accepted but ignored.

        Returns
        -------
        RiskPrediction
        """
        location_id = request.get("location_id", "UNKNOWN")
        timestamp_utc = request.get(
            "timestamp_utc",
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        probability, risk_level, factors = self._responses.get(location_id, _DEFAULT_MOCK)

        return RiskPrediction(
            location_id=location_id,
            timestamp_utc=timestamp_utc,
            risk_probability=probability,
            risk_level=risk_level,
            top_risk_factors=factors,
        )

    def predict_batch(self, requests: list[dict]) -> list[RiskPrediction]:
        """Predict for multiple locations."""
        return [self.predict(r) for r in requests]

    def list_mock_locations(self) -> list[str]:
        """Return all location_ids with custom mock responses."""
        return list(self._responses.keys())

    def add_mock(
        self,
        location_id: str,
        probability: float,
        risk_level: str,
        factors: list[str],
    ) -> None:
        """Add or override a mock response."""
        self._responses[location_id] = (probability, risk_level, factors)
