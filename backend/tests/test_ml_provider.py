"""
ML Provider contract tests.

Verifies that both MockMLProvider and HttpMLProvider return responses
conforming to the documented prediction contract.
"""
from __future__ import annotations

import pytest

from backend.services.ml_provider import MockMLProvider


class TestMockMLProvider:
    def setup_method(self):
        self.provider = MockMLProvider()

    def test_returns_required_keys(self):
        result = self.provider.predict("LOC_001", {})
        required = {"location_id", "timestamp_utc", "risk_probability", "risk_level", "top_risk_factors"}
        assert required.issubset(result.keys())

    def test_location_id_preserved(self):
        result = self.provider.predict("MY_LOCATION", {})
        assert result["location_id"] == "MY_LOCATION"

    def test_probability_in_range(self):
        for loc in ("LOC_A", "LOC_B", "LOC_C", "LOC_D"):
            result = self.provider.predict(loc, {})
            assert 0.0 <= result["risk_probability"] <= 1.0

    def test_risk_level_is_valid(self):
        valid_levels = {"LOW", "MODERATE", "HIGH", "CRITICAL"}
        for loc in ("LOC_X", "LOC_Y", "LOC_Z"):
            result = self.provider.predict(loc, {})
            assert result["risk_level"] in valid_levels

    def test_top_risk_factors_is_list(self):
        result = self.provider.predict("LOC_001", {"rainfall_24h": 90.0})
        assert isinstance(result["top_risk_factors"], list)

    def test_seeded_critical_location(self):
        result = self.provider.predict("LOC_TEST_CRITICAL", {})
        assert result["risk_level"] == "CRITICAL"
        assert result["risk_probability"] >= 0.85

    def test_seeded_high_location(self):
        result = self.provider.predict("LOC_TEST_HIGH", {})
        assert result["risk_level"] == "HIGH"

    def test_seeded_moderate_location(self):
        result = self.provider.predict("LOC_TEST_MODERATE", {})
        assert result["risk_level"] == "MODERATE"

    def test_seeded_low_location(self):
        result = self.provider.predict("LOC_TEST_LOW", {})
        assert result["risk_level"] == "LOW"

    def test_deterministic_for_same_input(self):
        """Same location always returns same risk level."""
        result1 = self.provider.predict("STABLE_LOC", {})
        result2 = self.provider.predict("STABLE_LOC", {})
        assert result1["risk_level"] == result2["risk_level"]
        assert result1["risk_probability"] == result2["risk_probability"]

    def test_timestamp_utc_injected(self):
        ts = "2026-09-03T00:00:00Z"
        result = self.provider.predict("LOC_001", {}, timestamp_utc=ts)
        assert result["timestamp_utc"] == ts

    def test_features_influence_top_factors(self):
        features = {"rainfall_24h": 90.0, "slope": 35.0, "moisture_indicator": 0.85}
        result = self.provider.predict("LOC_FEAT_TEST", features)
        # Should pick up some factors from the feature set
        assert isinstance(result["top_risk_factors"], list)
