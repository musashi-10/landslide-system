"""
Backend API tests.

Uses FastAPI's TestClient (synchronous) to test all endpoints.
Dependencies are overridden with isolated in-memory instances so tests
are fully self-contained and require no external services.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.dependencies import get_risk_service, get_alert_service
from backend.services.risk_service import RiskService
from backend.services.alert_service import AlertService
from backend.services.ml_provider import MockMLProvider
from backend.services.notification_service import MockNotificationProvider
from backend.repositories.risk_repository import RiskRepository
from backend.repositories.alert_repository import AlertRepository
from alerts.config import AlertConfig
from alerts.engine import AlertEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """Return a TestClient with fresh in-memory repositories."""
    risk_repo = RiskRepository()
    alert_repo = AlertRepository()
    notifier = MockNotificationProvider()
    alert_cfg = AlertConfig(min_risk_level="HIGH", cooldown_seconds=0, persistence_count=1)
    engine = AlertEngine(config=alert_cfg, alert_repository=alert_repo, notification_provider=notifier)
    alert_svc = AlertService(engine=engine, repository=alert_repo)
    risk_svc = RiskService(ml_provider=MockMLProvider(), risk_repository=risk_repo, alert_engine=engine)

    app = create_app()
    app.dependency_overrides[get_risk_service] = lambda: risk_svc
    app.dependency_overrides[get_alert_service] = lambda: alert_svc

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "backend"


# ── Current Risk ──────────────────────────────────────────────────────────────

class TestCurrentRisk:
    def test_known_location_returns_prediction(self, client):
        resp = client.get("/risk/current/LOC_TEST_HIGH")
        assert resp.status_code == 200
        data = resp.json()
        assert data["location_id"] == "LOC_TEST_HIGH"
        assert "risk_probability" in data
        assert data["risk_level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")
        assert isinstance(data["top_risk_factors"], list)

    def test_critical_location(self, client):
        resp = client.get("/risk/current/LOC_TEST_CRITICAL")
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_level"] == "CRITICAL"
        assert data["risk_probability"] >= 0.85

    def test_low_risk_location(self, client):
        resp = client.get("/risk/current/LOC_TEST_LOW")
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_level"] == "LOW"

    def test_response_shape_matches_contract(self, client):
        resp = client.get("/risk/current/LOC_001")
        assert resp.status_code == 200
        data = resp.json()
        required_keys = {"location_id", "timestamp_utc", "risk_probability", "risk_level", "top_risk_factors"}
        assert required_keys.issubset(data.keys())

    def test_probability_in_valid_range(self, client):
        resp = client.get("/risk/current/LOC_002")
        assert resp.status_code == 200
        prob = resp.json()["risk_probability"]
        assert 0.0 <= prob <= 1.0


# ── Risk Map ──────────────────────────────────────────────────────────────────

class TestRiskMap:
    def test_returns_locations(self, client):
        resp = client.get("/risk/map")
        assert resp.status_code == 200
        data = resp.json()
        assert "locations" in data
        assert "timestamp_utc" in data
        assert len(data["locations"]) > 0

    def test_location_shape(self, client):
        resp = client.get("/risk/map")
        loc = resp.json()["locations"][0]
        for key in ("location_id", "latitude", "longitude", "risk_probability", "risk_level"):
            assert key in loc

    def test_risk_level_filter(self, client):
        # Seed a CRITICAL prediction first
        client.get("/risk/current/LOC_TEST_CRITICAL")
        resp = client.get("/risk/map?risk_level=CRITICAL")
        assert resp.status_code == 200
        locations = resp.json()["locations"]
        for loc in locations:
            assert loc["risk_level"] in ("CRITICAL",)

    def test_bbox_filter(self, client):
        # bbox that includes LOC_001 (27.123, 88.456)
        resp = client.get("/risk/map?bbox=88.4,27.1,88.5,27.2")
        assert resp.status_code == 200
        data = resp.json()
        loc_ids = [l["location_id"] for l in data["locations"]]
        assert "LOC_001" in loc_ids

    def test_bbox_filter_excludes_outside(self, client):
        # tiny bbox in the middle of nowhere
        resp = client.get("/risk/map?bbox=0.0,0.0,0.1,0.1")
        assert resp.status_code == 200
        assert len(resp.json()["locations"]) == 0


# ── Risk History ──────────────────────────────────────────────────────────────

class TestRiskHistory:
    def test_history_returned(self, client):
        # Generate some history
        client.get("/risk/current/LOC_001")
        resp = client.get("/risk/history/LOC_001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["location_id"] == "LOC_001"
        assert isinstance(data["history"], list)

    def test_history_entry_shape(self, client):
        client.get("/risk/current/LOC_001")
        resp = client.get("/risk/history/LOC_001")
        entry = resp.json()["history"][0]
        for key in ("timestamp_utc", "risk_probability", "risk_level"):
            assert key in entry

    def test_empty_history_is_valid(self, client):
        resp = client.get("/risk/history/BRAND_NEW_LOCATION")
        assert resp.status_code == 200
        # May have 1 entry because get_current auto-generates
        data = resp.json()
        assert isinstance(data["history"], list)


# ── Risk Factors ──────────────────────────────────────────────────────────────

class TestRiskFactors:
    def test_factors_returned(self, client):
        client.get("/risk/current/LOC_TEST_CRITICAL")
        resp = client.get("/risk/factors/LOC_TEST_CRITICAL")
        assert resp.status_code == 200
        data = resp.json()
        assert data["location_id"] == "LOC_TEST_CRITICAL"
        assert isinstance(data["factors"], list)

    def test_factor_shape(self, client):
        client.get("/risk/current/LOC_TEST_CRITICAL")
        resp = client.get("/risk/factors/LOC_TEST_CRITICAL")
        factors = resp.json()["factors"]
        if factors:
            f = factors[0]
            assert "feature" in f


# ── Alerts ────────────────────────────────────────────────────────────────────

class TestAlerts:
    def test_alerts_endpoint_returns_list(self, client):
        resp = client.get("/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert "alerts" in data
        assert isinstance(data["alerts"], list)

    def test_high_risk_creates_alert(self, client):
        # LOC_TEST_CRITICAL should always fire an alert
        client.get("/risk/current/LOC_TEST_CRITICAL")
        resp = client.get("/alerts")
        alerts = resp.json()["alerts"]
        loc_ids = [a["location_id"] for a in alerts]
        assert "LOC_TEST_CRITICAL" in loc_ids

    def test_alert_shape(self, client):
        client.get("/risk/current/LOC_TEST_HIGH")
        resp = client.get("/alerts")
        alerts = resp.json()["alerts"]
        if alerts:
            a = alerts[0]
            for key in ("alert_id", "location_id", "timestamp_utc", "risk_level", "status"):
                assert key in a

    def test_low_risk_does_not_create_alert(self, client):
        # LOW risk should NOT create an alert (threshold is HIGH)
        client.get("/risk/current/LOC_TEST_LOW")
        resp = client.get("/alerts")
        alerts = resp.json()["alerts"]
        low_alerts = [a for a in alerts if a["location_id"] == "LOC_TEST_LOW"]
        assert len(low_alerts) == 0


# ── Predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_valid_response(self, client):
        payload = {
            "location_id": "LOC_PREDICT_TEST",
            "timestamp_utc": "2026-09-03T10:00:00Z",
            "features": {
                "elevation": 1200.0,
                "slope": 35.0,
                "rainfall_24h": 90.0,
            },
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["location_id"] == "LOC_PREDICT_TEST"
        assert "risk_probability" in data
        assert 0.0 <= data["risk_probability"] <= 1.0
        assert data["risk_level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")

    def test_predict_missing_location_id_returns_422(self, client):
        payload = {"timestamp_utc": "2026-09-03T10:00:00Z", "features": {}}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_empty_features_is_valid(self, client):
        """Empty features are allowed — the ML provider handles missing values."""
        payload = {
            "location_id": "LOC_EMPTY_FEAT",
            "timestamp_utc": "2026-09-03T10:00:00Z",
            "features": {},
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200


# ── Error Handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_invalid_path_returns_404(self, client):
        resp = client.get("/risk/nonexistent_endpoint")
        assert resp.status_code in (404, 422)

    def test_malformed_predict_body(self, client):
        resp = client.post("/predict", json={"bad": "data"})
        assert resp.status_code == 422
