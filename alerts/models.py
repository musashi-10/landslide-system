"""
Alert data models.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

AlertStatus = Literal["ACTIVE", "ACKNOWLEDGED", "RESOLVED"]
RiskLevel = Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]

RISK_LEVEL_ORDER: dict[str, int] = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}


def risk_level_value(level: str) -> int:
    """Return numeric ordering for a risk level string (case-insensitive)."""
    return RISK_LEVEL_ORDER.get(level.upper(), -1)


@dataclass
class Alert:
    location_id: str
    risk_level: str
    message: str
    status: AlertStatus = "ACTIVE"
    alert_id: str = field(default_factory=lambda: f"ALT_{uuid.uuid4().hex[:8].upper()}")
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "location_id": self.location_id,
            "timestamp_utc": self.timestamp_utc,
            "risk_level": self.risk_level,
            "status": self.status,
            "message": self.message,
        }
