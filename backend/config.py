"""
Backend-specific configuration.

Extends the shared Settings with platform-specific fields.
All values are read from environment variables / .env file.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    # ── Application ────────────────────────────────────────────────
    app_name: str = "landslide-backend"
    environment: str = "development"
    log_level: str = "INFO"

    # ── HTTP server ────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── CORS ───────────────────────────────────────────────────────
    # Comma-separated list of allowed origins for the frontend dev server
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ── Database (optional – not required for in-memory mode) ──────
    postgres_db: str = "landslide"
    postgres_user: str = "landslide_user"
    postgres_password: str = "change_me"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    use_database: bool = False  # set True to enable SQLAlchemy persistence

    # ── ML service ─────────────────────────────────────────────────
    use_mock_ml: bool = True          # False = call ml_service_url
    ml_service_url: str = "http://localhost:8001"
    ml_service_timeout: int = 10      # seconds

    # ── Alert engine ────────────────────────────────────────────────
    alert_enabled: bool = True
    # Minimum risk_level that triggers an alert: LOW | MODERATE | HIGH | CRITICAL
    alert_min_risk_level: str = "HIGH"
    # Seconds before the same location can generate a new alert
    alert_cooldown_seconds: int = 3600
    # How many consecutive HIGH+ predictions before alert fires (persistence)
    alert_persistence_count: int = 1

    # ── Notifications ──────────────────────────────────────────────
    # mock | email | (future providers)
    notification_provider: str = "mock"
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_email_recipients: str = ""  # comma-separated

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_backend_settings() -> BackendSettings:
    return BackendSettings()
