"""
Tests for ml.inference.mock_predictor — MockPredictor.

Tests:
  - predict() returns valid RiskPrediction
  - probability in [0.0, 1.0]
  - all four risk levels reachable via known test locations
  - same location → same result (deterministic)
  - unknown location → default response (no crash)
  - predict_batch() works
  - add_mock() overrides correctly
  - list_mock_locations() returns list
"""

import pytest

from shared.schemas import RiskPrediction
from ml.inference.mock_predictor import MockPredictor

VALID_LEVELS = {"LOW", "MODERATE", "HIGH", "CRITICAL"}


class TestMockPredictor:

    def setup_method(self):
        self.mock = MockPredictor()

    def _req(self, loc="LOC_TEST_001"):
        return {
            "location_id": loc,
            "timestamp_utc": "2026-09-02T12:00:00Z",
            "features": {"slope": 32.4},
        }

    def test_returns_risk_prediction_instance(self):
        result = self.mock.predict(self._req())
        assert isinstance(result, RiskPrediction)

    def test_probability_in_range(self):
        for loc in ["LOC_TEST_001", "LOC_TEST_002", "LOC_TEST_003", "LOC_TEST_004"]:
            r = self.mock.predict(self._req(loc))
            assert 0.0 <= r.risk_probability <= 1.0, f"probability out of range for {loc}"

    def test_risk_level_valid(self):
        for loc in ["LOC_TEST_001", "LOC_TEST_002", "LOC_TEST_003", "LOC_TEST_004"]:
            r = self.mock.predict(self._req(loc))
            assert r.risk_level in VALID_LEVELS

    def test_all_four_levels_reachable(self):
        levels = set()
        for loc in ["LOC_TEST_001", "LOC_TEST_002", "LOC_TEST_003", "LOC_TEST_004"]:
            r = self.mock.predict(self._req(loc))
            levels.add(r.risk_level)
        assert levels == VALID_LEVELS

    def test_deterministic_same_location(self):
        r1 = self.mock.predict(self._req("LOC_TEST_001"))
        r2 = self.mock.predict(self._req("LOC_TEST_001"))
        assert r1.risk_probability == r2.risk_probability
        assert r1.risk_level == r2.risk_level
        assert r1.top_risk_factors == r2.top_risk_factors

    def test_location_id_in_response(self):
        r = self.mock.predict(self._req("LOC_TEST_001"))
        assert r.location_id == "LOC_TEST_001"

    def test_timestamp_in_response(self):
        r = self.mock.predict(self._req())
        assert r.timestamp_utc == "2026-09-02T12:00:00Z"

    def test_top_risk_factors_is_list_of_strings(self):
        r = self.mock.predict(self._req("LOC_TEST_001"))
        assert isinstance(r.top_risk_factors, list)
        for f in r.top_risk_factors:
            assert isinstance(f, str)

    def test_unknown_location_no_crash(self):
        r = self.mock.predict(self._req("COMPLETELY_UNKNOWN_LOCATION_XYZ"))
        assert isinstance(r, RiskPrediction)
        assert r.risk_level in VALID_LEVELS

    def test_predict_batch_length(self):
        requests = [self._req(f"LOC_TEST_00{i}") for i in range(1, 5)]
        results = self.mock.predict_batch(requests)
        assert len(results) == 4

    def test_predict_batch_all_valid(self):
        requests = [self._req(f"LOC_TEST_00{i}") for i in range(1, 5)]
        for r in self.mock.predict_batch(requests):
            assert isinstance(r, RiskPrediction)
            assert 0.0 <= r.risk_probability <= 1.0
            assert r.risk_level in VALID_LEVELS

    def test_add_mock_overrides(self):
        self.mock.add_mock("LOC_CUSTOM_999", 0.62, "HIGH", ["custom_factor"])
        r = self.mock.predict(self._req("LOC_CUSTOM_999"))
        assert r.risk_probability == 0.62
        assert r.risk_level == "HIGH"
        assert "custom_factor" in r.top_risk_factors

    def test_list_mock_locations_contains_defaults(self):
        locs = self.mock.list_mock_locations()
        assert "LOC_TEST_001" in locs
        assert "LOC_TEST_002" in locs
        assert "LOC_TEST_003" in locs
        assert "LOC_TEST_004" in locs

    def test_custom_responses_dict(self):
        custom = {"CUSTOM_LOC": (0.95, "CRITICAL", ["factor_a", "factor_b"])}
        mock = MockPredictor(mock_responses=custom)
        r = mock.predict(self._req("CUSTOM_LOC"))
        assert r.risk_probability == 0.95
        assert r.risk_level == "CRITICAL"

    def test_missing_features_no_crash(self):
        """Features dict is accepted but ignored — must not crash on empty."""
        req = {"location_id": "LOC_TEST_001", "timestamp_utc": "2026-09-01T00:00:00Z", "features": {}}
        r = self.mock.predict(req)
        assert isinstance(r, RiskPrediction)
