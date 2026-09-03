"""
In-memory alert repository.

Manages alert lifecycle: ACTIVE → ACKNOWLEDGED → RESOLVED.
Interface mirrors what an ORM-backed implementation would expose.
"""
from __future__ import annotations

import threading
from typing import List, Optional

from alerts.models import Alert


class AlertRepository:
    """Thread-safe in-memory store for Alert objects."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, Alert] = {}   # alert_id → Alert

    def add(self, alert: Alert) -> None:
        with self._lock:
            self._store[alert.alert_id] = alert

    def get(self, alert_id: str) -> Optional[Alert]:
        with self._lock:
            return self._store.get(alert_id)

    def list_all(self) -> List[Alert]:
        with self._lock:
            return list(self._store.values())

    def list_active(self) -> List[Alert]:
        with self._lock:
            return [a for a in self._store.values() if a.status == "ACTIVE"]

    def get_active_by_location(self, location_id: str) -> Optional[Alert]:
        """Return the most-recent ACTIVE alert for a location, or None."""
        with self._lock:
            candidates = [
                a for a in self._store.values()
                if a.location_id == location_id and a.status == "ACTIVE"
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.timestamp_utc)

    def update_status(self, alert_id: str, status: str) -> bool:
        with self._lock:
            alert = self._store.get(alert_id)
            if alert is None:
                return False
            alert.status = status  # type: ignore[assignment]
            return True

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
