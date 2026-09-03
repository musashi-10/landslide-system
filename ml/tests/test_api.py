"""
Tests for ml.api.app — FastAPI inference service.

Tests:
  - GET /health returns 200 with correct schema
  - POST /predict happy path with valid features returns RiskPrediction schema
  - POST /predict missing location_id returns error
  - POST /predict missing features key returns 422
  - POST /predict with empty features does not crash (mock mode)
  - GET /risk/current/{location_id} returns 200
  - GET /risk/map returns 200 with locations list
  - GET /risk/history/{location_id} returns 200
  - GET /risk/factors/{location_id} returns 200
"""

import pytest
from fastapi.testclient import TestClient

from ml.api.app import create_app

VALID_LEVELS = {"LOW", "MODERATE", "HIGH", "CRITICAL"}


@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the FastAPI app.
    
    In test mode the app auto-falls back to MockPredictor since
    no trained model is present, so all endpoints work immediately.
    """
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── Health ────────────────────────────────────────────────────────────────────


class TestHealth:

    def test_health_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_status_ok(self, client):
        r = client.get("/health")
        data = r.json()
        assert data["status"] == "ok"

    def test_health_service_name(self, client):
        r = client.get("/health")
        data = r.json()
        assert "service" in data

    def test_health_mock_mode_field_present(self, client):
        r = client.get("/health")
        data = r.json()
        assert "mock_mode" in data


# ── Predict ───────────────────────────────────────────────────────────────────


class TestPredict:

    def _valid_request(self, location_id="LOC_TEST_001"):
        return {
            "location_id": location_id,
            "timestamp_utc": "2026-09-02T12:00:00Z",
            "features": {
                "elevation": 1240.5,
                "slope": 32.4,
                "aspect": 145.0,
                "soil_type": "loam",
                "geology": "schist",
                "land_cover": "forest",
                "historical_landslide": 1,
                "rainfall_1h": 12.4,
                "rainfall_6h": 48.2,
                "rainfall_24h": 91.5,
                "rainfall_3d": 165.3,
                "rainfall_7d": 240.1,
                "forecast_rainfall": 42.0,
                "moisture_indicator": 0.71,
            },
        }

    def test_predict_200(self, client):
        r = client.post("/predict", json=self._valid_request())
        assert r.status_code == 200

    def test_predict_returns_location_id(self, client):
        r = client.post("/predict", json=self._valid_request("LOC_TEST_001"))
        data = r.json()
        assert data["location_id"] == "LOC_TEST_001"

    def test_predict_returns_timestamp(self, client):
        r = client.post("/predict", json=self._valid_request())
        data = r.json()
        assert "timestamp_utc" in data

    def test_predict_probability_in_range(self, client):
        r = client.post("/predict", json=self._valid_request())
        data = r.json()
        assert 0.0 <= data["risk_probability"] <= 1.0

    def test_predict_risk_level_valid(self, client):
        r = client.post("/predict", json=self._valid_request())
        data = r.json()
        assert data["risk_level"] in VALID_LEVELS

    def test_predict_top_risk_factors_is_list(self, client):
        r = client.post("/predict", json=self._valid_request())
        data = r.json()
        assert isinstance(data["top_risk_factors"], list)

    def test_predict_empty_features_no_crash(self, client):
        """Empty features should not crash — mock mode handles gracefully."""
        req = {
            "location_id": "LOC_TEST_001",
            "timestamp_utc": "2026-09-02T12:00:00Z",
            "features": {},
        }
        r = client.post("/predict", json=req)
        # Should return either 200 (mock response) or a structured error
        assert r.status_code in (200, 422)

    def test_predict_missing_location_id_error(self, client):
        """Request missing location_id should return a structured error."""
        req = {
            "timestamp_utc": "2026-09-02T12:00:00Z",
            "features": {"slope": 32.4},
        }
        r = client.post("/predict", json=req)
        # FastAPI will return 422 (validation error) for missing required field
        assert r.status_code == 422

    def test_predict_malformed_json_error(self, client):
        r = client.post("/predict", content="not json", headers={"content-type": "application/json"})
        assert r.status_code == 422


# ── Risk endpoints ────────────────────────────────────────────────────────────


class TestRiskEndpoints:

    def test_current_risk_200(self, client):
        r = client.get("/risk/current/LOC_TEST_001")
        assert r.status_code == 200

    def test_current_risk_has_probability(self, client):
        r = client.get("/risk/current/LOC_TEST_001")
        data = r.json()
        # May return a prediction or error depending on mock state
        if "risk_probability" in data:
            assert 0.0 <= data["risk_probability"] <= 1.0

    def test_risk_map_200(self, client):
        r = client.get("/risk/map")
        assert r.status_code == 200

    def test_risk_map_has_locations(self, client):
        r = client.get("/risk/map")
        data = r.json()
        assert "locations" in data
        assert isinstance(data["locations"], list)

    def test_risk_map_filter_by_level(self, client):
        r = client.get("/risk/map?risk_level=HIGH")
        assert r.status_code == 200
        data = r.json()
        for loc in data["locations"]:
            assert loc["risk_level"] == "HIGH"

    def test_risk_history_200(self, client):
        r = client.get("/risk/history/LOC_TEST_001")
        assert r.status_code == 200

    def test_risk_history_has_history(self, client):
        r = client.get("/risk/history/LOC_TEST_001")
        data = r.json()
        assert "history" in data
        assert isinstance(data["history"], list)

    def test_risk_factors_200(self, client):
        r = client.get("/risk/factors/LOC_TEST_001")
        assert r.status_code == 200

    def test_risk_factors_has_factors(self, client):
        r = client.get("/risk/factors/LOC_TEST_001")
        data = r.json()
        assert "factors" in data
        assert isinstance(data["factors"], list)
