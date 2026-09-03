"""
Alert Engine.

Processes a RiskPrediction, applies configured thresholds, manages
alert lifecycle (ACTIVE → ACKNOWLEDGED → RESOLVED), and dispatches
notifications through an injected NotificationProvider.

Design
------
- Stateless engine; state lives in AlertRepository.
- Cooldown and persistence checks happen here.
- No provider-specific logic; uses NotificationProvider abstraction.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from alerts.config import AlertConfig
from alerts.models import Alert, risk_level_value

logger = logging.getLogger(__name__)


class AlertEngine:
    """
    Core alert processing engine.

    Parameters
    ----------
    config : AlertConfig
        Threshold / cooldown / persistence configuration.
    alert_repository
        Object with add(alert), list_active(), get_by_location(location_id)
        interface.  Accepts in-memory or DB-backed implementations.
    notification_provider
        Object with send(alert) interface.
    """

    def __init__(self, config: AlertConfig, alert_repository, notification_provider):
        self.config = config
        self._repo = alert_repository
        self._notifier = notification_provider
        # persistence counter: location_id → consecutive qualifying count
        self._persistence: dict[str, int] = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def process(self, prediction: dict) -> Alert | None:
        """
        Evaluate a risk prediction and optionally create an alert.

        Parameters
        ----------
        prediction : dict
            Must contain keys: location_id, risk_level, risk_probability,
            timestamp_utc.

        Returns
        -------
        Alert | None
            The newly created Alert, or None if no alert was warranted.
        """
        if not self.config.enabled:
            return None

        location_id: str = prediction["location_id"]
        risk_level: str = prediction.get("risk_level", "LOW").upper()
        risk_prob: float = prediction.get("risk_probability", 0.0)

        min_val = risk_level_value(self.config.min_risk_level)
        pred_val = risk_level_value(risk_level)

        # ── Below threshold → reset persistence counter ─────────────────────
        if pred_val < min_val:
            self._persistence[location_id] = 0
            logger.debug(
                "Location %s: risk %s below threshold %s — no alert.",
                location_id, risk_level, self.config.min_risk_level,
            )
            return None

        # ── Persistence check ────────────────────────────────────────────────
        self._persistence[location_id] = self._persistence.get(location_id, 0) + 1
        if self._persistence[location_id] < self.config.persistence_count:
            logger.debug(
                "Location %s: persistence %d/%d — holding.",
                location_id,
                self._persistence[location_id],
                self.config.persistence_count,
            )
            return None

        # ── Cooldown check ───────────────────────────────────────────────────
        existing = self._repo.get_active_by_location(location_id)
        if existing:
            created = datetime.fromisoformat(
                existing.timestamp_utc.replace("Z", "+00:00")
            )
            elapsed = (datetime.now(timezone.utc) - created).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                logger.debug(
                    "Location %s: alert in cooldown (%.0fs remaining).",
                    location_id,
                    self.config.cooldown_seconds - elapsed,
                )
                return None

        # ── Create alert ─────────────────────────────────────────────────────
        message = self._build_message(location_id, risk_level, risk_prob)
        alert = Alert(
            location_id=location_id,
            risk_level=risk_level,
            message=message,
        )
        self._repo.add(alert)
        self._persistence[location_id] = 0   # reset after firing

        try:
            self._notifier.send(alert)
        except Exception as exc:  # noqa: BLE001
            logger.error("Notification failed for alert %s: %s", alert.alert_id, exc)

        logger.info(
            "Alert created: %s for location %s at risk_level %s.",
            alert.alert_id, location_id, risk_level,
        )
        return alert

    def acknowledge(self, alert_id: str) -> bool:
        """Transition an ACTIVE alert to ACKNOWLEDGED."""
        return self._repo.update_status(alert_id, "ACKNOWLEDGED")

    def resolve(self, alert_id: str) -> bool:
        """Transition an ACTIVE or ACKNOWLEDGED alert to RESOLVED."""
        return self._repo.update_status(alert_id, "RESOLVED")

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_message(location_id: str, risk_level: str, probability: float) -> str:
        pct = int(probability * 100)
        return (
            f"{risk_level} landslide risk at {location_id} "
            f"(probability {pct}%). Immediate attention required."
        )
