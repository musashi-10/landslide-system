"""
Tests for ml.inference.predictor — Predictor and PredictionError.

Tests:
  - MockPredictor.predict() returns valid RiskPrediction schema
  - probability in [0.0, 1.0]
  - risk_level is one of LOW/MODERATE/HIGH/CRITICAL
  - top_risk_factors is a non-empty list of strings
  - Malformed request (missing location_id) raises PredictionError
  - Malformed request (features not a dict) raises PredictionError
  - Missing 'features' key raises PredictionError
  - Missing 'timestamp_utc' raises PredictionError
  - Predictor.from_registry() raises PredictionError when no model present
"""

import pytest

from shared.schemas import RiskPrediction
from ml.inference.predictor import PredictionError, _validate_request
from ml.inference.mock_predictor import MockPredictor


# ── MockPredictor tests (also validates the RiskPrediction output shape) ──────


class TestMockPredictorInference:

    def setup_method(self):
        self.mock = MockPredictor()

    def _make_request(self, location_id="LOC_TEST_001"):
        return {
            "location_id": location_id,
            "timestamp_utc": "2026-09-02T12:00:00Z",
            "features": {
                "elevation": 1240.5,
                "slope": 32.4,
                "rainfall_24h": 91.5,
            },
        }

    def test_returns_risk_prediction_schema(self):
        result = self.mock.predict(self._make_request())
        assert isinstance(result, RiskPrediction)

    def test_probability_in_valid_range(self):
        result = self.mock.predict(self._make_request())
        assert 0.0 <= result.risk_probability <= 1.0

    def test_risk_level_is_valid(self):
        valid_levels = {"LOW", "MODERATE", "HIGH", "CRITICAL"}
        result = self.mock.predict(self._make_request())
        assert result.risk_level in valid_levels

    def test_top_risk_factors_is_list(self):
        result = self.mock.predict(self._make_request())
        assert isinstance(result.top_risk_factors, list)

    def test_top_risk_factors_contains_strings(self):
        result = self.mock.predict(self._make_request("LOC_TEST_001"))
        for f in result.top_risk_factors:
            assert isinstance(f, str)

    def test_location_id_preserved(self):
        result = self.mock.predict(self._make_request("LOC_TEST_001"))
        assert result.location_id == "LOC_TEST_001"

    def test_timestamp_preserved(self):
        req = self._make_request()
        result = self.mock.predict(req)
        assert result.timestamp_utc == "2026-09-02T12:00:00Z"

    def test_all_four_risk_levels_reachable(self):
        levels = set()
        for loc_id in ["LOC_TEST_001", "LOC_TEST_002", "LOC_TEST_003", "LOC_TEST_004"]:
            r = self.mock.predict(self._make_request(loc_id))
            levels.add(r.risk_level)
        assert "HIGH" in levels
        assert "MODERATE" in levels
        assert "CRITICAL" in levels
        assert "LOW" in levels

    def test_deterministic_output(self):
        """Same location → same output every time."""
        r1 = self.mock.predict(self._make_request("LOC_TEST_001"))
        r2 = self.mock.predict(self._make_request("LOC_TEST_001"))
        assert r1.risk_probability == r2.risk_probability
        assert r1.risk_level == r2.risk_level

    def test_unknown_location_returns_default(self):
        r = self.mock.predict(self._make_request("COMPLETELY_UNKNOWN_LOC"))
        assert isinstance(r, RiskPrediction)
        assert 0.0 <= r.risk_probability <= 1.0

    def test_predict_batch(self):
        requests = [self._make_request(f"LOC_TEST_00{i}") for i in range(1, 4)]
        results = self.mock.predict_batch(requests)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, RiskPrediction)

    def test_list_mock_locations(self):
        locations = self.mock.list_mock_locations()
        assert isinstance(locations, list)
        assert "LOC_TEST_001" in locations

    def test_add_mock(self):
        self.mock.add_mock("LOC_CUSTOM", 0.55, "HIGH", ["custom_factor"])
        r = self.mock.predict(self._make_request("LOC_CUSTOM"))
        assert r.risk_level == "HIGH"
        assert r.risk_probability == 0.55


# ── Request validation tests ──────────────────────────────────────────────────


class TestRequestValidation:

    def test_valid_request_does_not_raise(self):
        req = {
            "location_id": "LOC_001",
            "timestamp_utc": "2026-09-02T12:00:00Z",
            "features": {"slope": 32.4},
        }
        _validate_request(req)  # should not raise

    def test_missing_location_id_raises(self):
        req = {"timestamp_utc": "2026-09-02T12:00:00Z", "features": {}}
        with pytest.raises(PredictionError) as exc_info:
            _validate_request(req)
        assert exc_info.value.code == "MISSING_FIELD"

    def test_missing_timestamp_raises(self):
        req = {"location_id": "LOC_001", "features": {}}
        with pytest.raises(PredictionError) as exc_info:
            _validate_request(req)
        assert exc_info.value.code == "MISSING_FIELD"

    def test_missing_features_raises(self):
        req = {"location_id": "LOC_001", "timestamp_utc": "2026-09-02T12:00:00Z"}
        with pytest.raises(PredictionError) as exc_info:
            _validate_request(req)
        assert exc_info.value.code == "MISSING_FIELD"

    def test_features_not_dict_raises(self):
        req = {
            "location_id": "LOC_001",
            "timestamp_utc": "2026-09-02T12:00:00Z",
            "features": "not_a_dict",
        }
        with pytest.raises(PredictionError) as exc_info:
            _validate_request(req)
        assert exc_info.value.code == "INVALID_FEATURES"

    def test_non_dict_request_raises(self):
        with pytest.raises(PredictionError) as exc_info:
            _validate_request("not a dict")
        assert exc_info.value.code == "MALFORMED_REQUEST"

    def test_prediction_error_to_dict(self):
        err = PredictionError("TEST_CODE", "test message")
        d = err.to_dict()
        assert "error" in d
        assert d["error"]["code"] == "TEST_CODE"
        assert d["error"]["message"] == "test message"


# ── Predictor.from_registry raises without trained model ─────────────────────


class TestPredictorWithoutModel:

    def test_from_registry_raises_without_model(self, tmp_path):
        from ml.inference.predictor import Predictor
        with pytest.raises(PredictionError) as exc_info:
            Predictor.from_registry(model_dir=str(tmp_path), version=None)
        assert exc_info.value.code == "MODEL_UNAVAILABLE"
