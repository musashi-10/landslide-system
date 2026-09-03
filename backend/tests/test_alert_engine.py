"""
Alert engine unit tests.

Tests threshold logic, cooldown, persistence, and lifecycle transitions.
"""
from __future__ import annotations

import time
import pytest

from alerts.config import AlertConfig
from alerts.engine import AlertEngine
from alerts.models import Alert
from backend.repositories.alert_repository import AlertRepository
from backend.services.notification_service import MockNotificationProvider


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_engine(
    min_risk_level="HIGH",
    cooldown_seconds=0,
    persistence_count=1,
    enabled=True,
):
    repo = AlertRepository()
    notifier = MockNotificationProvider()
    config = AlertConfig(
        min_risk_level=min_risk_level,
        cooldown_seconds=cooldown_seconds,
        persistence_count=persistence_count,
        enabled=enabled,
    )
    return AlertEngine(config=config, alert_repository=repo, notification_provider=notifier), repo, notifier


def pred(location_id="LOC_001", risk_level="HIGH", probability=0.80):
    return {
        "location_id": location_id,
        "risk_level": risk_level,
        "risk_probability": probability,
        "timestamp_utc": "2026-09-03T10:00:00Z",
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAlertThreshold:
    def test_high_risk_creates_alert(self):
        engine, repo, _ = make_engine(min_risk_level="HIGH")
        alert = engine.process(pred(risk_level="HIGH"))
        assert alert is not None
        assert alert.risk_level == "HIGH"
        assert alert.status == "ACTIVE"

    def test_critical_risk_creates_alert(self):
        engine, repo, _ = make_engine(min_risk_level="HIGH")
        alert = engine.process(pred(risk_level="CRITICAL", probability=0.95))
        assert alert is not None
        assert alert.risk_level == "CRITICAL"

    def test_moderate_risk_below_high_threshold_no_alert(self):
        engine, repo, _ = make_engine(min_risk_level="HIGH")
        alert = engine.process(pred(risk_level="MODERATE"))
        assert alert is None

    def test_low_risk_no_alert(self):
        engine, repo, _ = make_engine(min_risk_level="HIGH")
        alert = engine.process(pred(risk_level="LOW", probability=0.1))
        assert alert is None

    def test_moderate_threshold_triggers_on_moderate(self):
        engine, repo, _ = make_engine(min_risk_level="MODERATE")
        alert = engine.process(pred(risk_level="MODERATE", probability=0.45))
        assert alert is not None


class TestAlertDisabled:
    def test_disabled_engine_creates_no_alerts(self):
        engine, repo, _ = make_engine(enabled=False)
        alert = engine.process(pred(risk_level="CRITICAL", probability=0.99))
        assert alert is None
        assert len(repo.list_all()) == 0


class TestAlertPersistence:
    def test_persistence_1_fires_immediately(self):
        engine, repo, _ = make_engine(persistence_count=1)
        alert = engine.process(pred(risk_level="HIGH"))
        assert alert is not None

    def test_persistence_2_needs_two_consecutive(self):
        engine, repo, _ = make_engine(persistence_count=2)
        # First prediction — no alert
        alert1 = engine.process(pred(risk_level="HIGH"))
        assert alert1 is None
        # Second consecutive — alert fires
        alert2 = engine.process(pred(risk_level="HIGH"))
        assert alert2 is not None

    def test_persistence_resets_on_low_risk(self):
        engine, repo, _ = make_engine(persistence_count=2)
        engine.process(pred(risk_level="HIGH"))   # count = 1
        engine.process(pred(risk_level="LOW"))    # resets to 0
        alert = engine.process(pred(risk_level="HIGH"))  # count = 1 again → no alert
        assert alert is None


class TestAlertCooldown:
    def test_zero_cooldown_allows_immediate_repeat(self):
        engine, repo, _ = make_engine(cooldown_seconds=0)
        alert1 = engine.process(pred(risk_level="HIGH"))
        assert alert1 is not None
        # Resolve first alert so a new one can fire
        repo.update_status(alert1.alert_id, "RESOLVED")
        alert2 = engine.process(pred(risk_level="HIGH"))
        assert alert2 is not None

    def test_cooldown_suppresses_duplicate_alert(self):
        engine, repo, _ = make_engine(cooldown_seconds=9999)
        alert1 = engine.process(pred(risk_level="HIGH"))
        assert alert1 is not None
        # Second prediction within cooldown → no new alert
        alert2 = engine.process(pred(risk_level="HIGH"))
        assert alert2 is None


class TestAlertNotification:
    def test_notification_sent_on_alert(self):
        engine, repo, notifier = make_engine()
        engine.process(pred(risk_level="HIGH"))
        assert len(notifier.sent) == 1
        assert notifier.sent[0].risk_level == "HIGH"

    def test_no_notification_below_threshold(self):
        engine, repo, notifier = make_engine(min_risk_level="HIGH")
        engine.process(pred(risk_level="LOW"))
        assert len(notifier.sent) == 0


class TestAlertLifecycle:
    def test_acknowledge_transitions_status(self):
        engine, repo, _ = make_engine()
        alert = engine.process(pred(risk_level="HIGH"))
        assert alert is not None
        success = engine.acknowledge(alert.alert_id)
        assert success is True
        updated = repo.get(alert.alert_id)
        assert updated.status == "ACKNOWLEDGED"

    def test_resolve_transitions_status(self):
        engine, repo, _ = make_engine()
        alert = engine.process(pred(risk_level="HIGH"))
        success = engine.resolve(alert.alert_id)
        assert success is True
        assert repo.get(alert.alert_id).status == "RESOLVED"

    def test_acknowledge_unknown_id_returns_false(self):
        engine, _, _ = make_engine()
        result = engine.acknowledge("ALT_NONEXISTENT")
        assert result is False


class TestAlertRepository:
    def test_alert_stored_correctly(self):
        engine, repo, _ = make_engine()
        alert = engine.process(pred(location_id="LOC_STORE_TEST", risk_level="CRITICAL"))
        assert alert is not None
        found = repo.get(alert.alert_id)
        assert found is not None
        assert found.location_id == "LOC_STORE_TEST"

    def test_list_active_excludes_resolved(self):
        engine, repo, _ = make_engine(cooldown_seconds=0)
        alert = engine.process(pred(risk_level="HIGH"))
        engine.resolve(alert.alert_id)
        active = repo.list_active()
        assert not any(a.alert_id == alert.alert_id for a in active)

    def test_get_active_by_location(self):
        engine, repo, _ = make_engine()
        engine.process(pred(location_id="LOC_FIND_ME", risk_level="HIGH"))
        found = repo.get_active_by_location("LOC_FIND_ME")
        assert found is not None
        assert found.location_id == "LOC_FIND_ME"
