"""
End-to-End Platform Integration Test.

Scenario:
    Mock environmental data
        → MockMLProvider (seeded CRITICAL location)
        → RiskService
        → Risk API  (GET /risk/current)
        → AlertEngine
        → Alert API (GET /alerts)

Expected:
    risk_probability = 0.92
    risk_level = CRITICAL
    alert status = ACTIVE

This test verifies that all platform components communicate correctly
end-to-end without any external services.
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


@pytest.fixture()
def e2e_client():
    """
    Full platform stack with isolated in-memory state.
    Alert threshold is HIGH so CRITICAL locations always generate alerts.
    """
    risk_repo = RiskRepository()
    alert_repo = AlertRepository()
    notifier = MockNotificationProvider()
    alert_config = AlertConfig(
        min_risk_level="HIGH",
        cooldown_seconds=0,
        persistence_count=1,
        enabled=True,
    )
    engine = AlertEngine(
        config=alert_config,
        alert_repository=alert_repo,
        notification_provider=notifier,
    )
    alert_svc = AlertService(engine=engine, repository=alert_repo)
    risk_svc = RiskService(
        ml_provider=MockMLProvider(),
        risk_repository=risk_repo,
        alert_engine=engine,
    )

    app = create_app()
    app.dependency_overrides[get_risk_service] = lambda: risk_svc
    app.dependency_overrides[get_alert_service] = lambda: alert_svc

    with TestClient(app) as client:
        yield client, notifier

    app.dependency_overrides.clear()


class TestEndToEndPlatform:
    """Full pipeline: mock data → ML → API → alert → notification."""

    def test_health_is_ok(self, e2e_client):
        client, _ = e2e_client
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_critical_risk_prediction(self, e2e_client):
        """LOC_TEST_CRITICAL always returns CRITICAL risk at 0.92 probability."""
        client, _ = e2e_client
        resp = client.get("/risk/current/LOC_TEST_CRITICAL")
        assert resp.status_code == 200
        data = resp.json()
        assert data["location_id"] == "LOC_TEST_CRITICAL"
        assert data["risk_level"] == "CRITICAL"
        assert data["risk_probability"] == 0.92

    def test_critical_risk_triggers_active_alert(self, e2e_client):
        """After a CRITICAL prediction, an ACTIVE alert must exist."""
        client, _ = e2e_client
        client.get("/risk/current/LOC_TEST_CRITICAL")

        resp = client.get("/alerts")
        assert resp.status_code == 200
        alerts = resp.json()["alerts"]

        critical_alerts = [
            a for a in alerts
            if a["location_id"] == "LOC_TEST_CRITICAL" and a["status"] == "ACTIVE"
        ]
        assert len(critical_alerts) >= 1
        assert critical_alerts[0]["risk_level"] == "CRITICAL"

    def test_notification_dispatched(self, e2e_client):
        """MockNotificationProvider must have received the alert."""
        client, notifier = e2e_client
        client.get("/risk/current/LOC_TEST_CRITICAL")
        assert len(notifier.sent) >= 1
        assert notifier.sent[0].risk_level == "CRITICAL"

    def test_risk_history_populated_after_prediction(self, e2e_client):
        """After predicting, history endpoint must return at least one entry."""
        client, _ = e2e_client
        client.get("/risk/current/LOC_TEST_CRITICAL")
        resp = client.get("/risk/history/LOC_TEST_CRITICAL")
        assert resp.status_code == 200
        assert len(resp.json()["history"]) >= 1

    def test_risk_map_includes_all_known_locations(self, e2e_client):
        """Risk map returns multiple locations (pre-seeded)."""
        client, _ = e2e_client
        resp = client.get("/risk/map")
        assert resp.status_code == 200
        locs = resp.json()["locations"]
        assert len(locs) >= 5

    def test_risk_factors_explain_critical_risk(self, e2e_client):
        """Factors endpoint returns the known top risk factors."""
        client, _ = e2e_client
        client.get("/risk/current/LOC_TEST_CRITICAL")
        resp = client.get("/risk/factors/LOC_TEST_CRITICAL")
        assert resp.status_code == 200
        data = resp.json()
        assert data["location_id"] == "LOC_TEST_CRITICAL"
        assert len(data["factors"]) > 0

    def test_predict_endpoint_stores_and_alerts(self, e2e_client):
        """POST /predict triggers the full pipeline."""
        client, notifier = e2e_client
        payload = {
            "location_id": "LOC_TEST_CRITICAL",
            "timestamp_utc": "2026-09-03T10:00:00Z",
            "features": {
                "rainfall_24h": 90.0,
                "slope": 35.0,
                "elevation": 1200.0,
            },
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_level"] == "CRITICAL"
        assert data["risk_probability"] == 0.92

    def test_low_risk_location_no_alert(self, e2e_client):
        """LOC_TEST_LOW must not trigger any ACTIVE alert."""
        client, _ = e2e_client
        client.get("/risk/current/LOC_TEST_LOW")
        resp = client.get("/alerts")
        alerts = resp.json()["alerts"]
        low_alerts = [a for a in alerts if a["location_id"] == "LOC_TEST_LOW"]
        assert len(low_alerts) == 0

    def test_ml_unavailability_returns_503(self, e2e_client):
        """When ML provider raises RuntimeError, the API returns 503."""
        client, _ = e2e_client

        from backend.services.ml_provider import MLProvider

        class BrokenMLProvider(MLProvider):
            def predict(self, location_id, features, timestamp_utc=None):
                raise RuntimeError("ML service down")

        from backend.main import create_app as _create_app
        from backend.services.risk_service import RiskService as _RS
        from backend.repositories.risk_repository import RiskRepository as _RR

        broken_svc = _RS(
            ml_provider=BrokenMLProvider(),
            risk_repository=_RR(),
            alert_engine=None,
        )
        app2 = _create_app()
        app2.dependency_overrides[get_risk_service] = lambda: broken_svc

        with TestClient(app2, raise_server_exceptions=False) as c2:
            resp = c2.post("/predict", json={
                "location_id": "LOC_X",
                "timestamp_utc": "2026-09-03T10:00:00Z",
                "features": {},
            })
        assert resp.status_code == 503
        body = resp.json()
        # FastAPI wraps HTTPException detail; accept either shape
        assert "error" in body or ("detail" in body and "error" in body["detail"])

        app2.dependency_overrides.clear()
