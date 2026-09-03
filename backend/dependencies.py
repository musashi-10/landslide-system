"""
FastAPI dependency injection.

All shared service instances are created here and injected into route
handlers via Depends().  This makes testing easy: replace any provider
by overriding the dependency.
"""
from __future__ import annotations

from functools import lru_cache

from backend.config import BackendSettings, get_backend_settings
from backend.repositories.alert_repository import AlertRepository
from backend.repositories.risk_repository import RiskRepository
from backend.services.alert_service import AlertService
from backend.services.ml_provider import MockMLProvider, HttpMLProvider, MLProvider
from backend.services.notification_service import (
    MockNotificationProvider,
    EmailNotificationProvider,
    NotificationProvider,
)
from backend.services.risk_service import RiskService
from alerts.config import AlertConfig
from alerts.engine import AlertEngine


# ── Singletons ────────────────────────────────────────────────────────────────
# Created once per process; thread-safe because repositories use locks.

@lru_cache(maxsize=1)
def _risk_repository() -> RiskRepository:
    return RiskRepository()


@lru_cache(maxsize=1)
def _alert_repository() -> AlertRepository:
    return AlertRepository()


@lru_cache(maxsize=1)
def _ml_provider() -> MLProvider:
    s = get_backend_settings()
    if s.use_mock_ml:
        return MockMLProvider()
    return HttpMLProvider(base_url=s.ml_service_url, timeout=s.ml_service_timeout)


@lru_cache(maxsize=1)
def _notification_provider() -> NotificationProvider:
    s = get_backend_settings()
    if s.notification_provider == "email":
        recipients = [r.strip() for r in s.alert_email_recipients.split(",") if r.strip()]
        return EmailNotificationProvider(
            smtp_host=s.smtp_host,
            smtp_port=s.smtp_port,
            smtp_user=s.smtp_user,
            smtp_password=s.smtp_password,
            recipients=recipients,
        )
    return MockNotificationProvider()


@lru_cache(maxsize=1)
def _alert_engine() -> AlertEngine:
    s = get_backend_settings()
    config = AlertConfig(
        min_risk_level=s.alert_min_risk_level,
        cooldown_seconds=s.alert_cooldown_seconds,
        persistence_count=s.alert_persistence_count,
        enabled=s.alert_enabled,
    )
    return AlertEngine(
        config=config,
        alert_repository=_alert_repository(),
        notification_provider=_notification_provider(),
    )


@lru_cache(maxsize=1)
def _alert_service() -> AlertService:
    return AlertService(
        engine=_alert_engine(),
        repository=_alert_repository(),
    )


@lru_cache(maxsize=1)
def _risk_service() -> RiskService:
    return RiskService(
        ml_provider=_ml_provider(),
        risk_repository=_risk_repository(),
        alert_engine=_alert_engine(),
    )


# ── FastAPI Depends callables ─────────────────────────────────────────────────

def get_risk_service() -> RiskService:
    return _risk_service()


def get_alert_service() -> AlertService:
    return _alert_service()


def get_settings() -> BackendSettings:
    return get_backend_settings()
