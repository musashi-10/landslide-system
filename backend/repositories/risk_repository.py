"""
In-memory risk repository.

Stores risk predictions and history per location.
Interface is designed so a SQLAlchemy-backed implementation can replace
this class without any changes to service or API layers.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Optional


class RiskRepository:
    """Thread-safe in-memory store for risk predictions and history."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # latest prediction per location_id
        self._current: dict[str, dict] = {}
        # ordered history per location_id (newest last)
        self._history: dict[str, list[dict]] = defaultdict(list)
        # known locations metadata: location_id → {latitude, longitude}
        self._locations: dict[str, dict] = {}

    # ── Locations ─────────────────────────────────────────────────────────────

    def register_location(self, location_id: str, latitude: float, longitude: float) -> None:
        with self._lock:
            self._locations[location_id] = {
                "location_id": location_id,
                "latitude": latitude,
                "longitude": longitude,
            }

    def list_locations(self) -> list[dict]:
        with self._lock:
            return list(self._locations.values())

    # ── Predictions ───────────────────────────────────────────────────────────

    def save_prediction(self, prediction: dict) -> None:
        """Store or replace the current prediction for a location."""
        location_id = prediction["location_id"]
        with self._lock:
            self._current[location_id] = dict(prediction)
            self._history[location_id].append(dict(prediction))
            # cap history to last 100 entries per location
            if len(self._history[location_id]) > 100:
                self._history[location_id] = self._history[location_id][-100:]

    def get_current(self, location_id: str) -> Optional[dict]:
        with self._lock:
            return self._current.get(location_id)

    def list_current_all(self) -> list[dict]:
        """Return latest prediction for every known location."""
        with self._lock:
            return list(self._current.values())

    def get_history(self, location_id: str, limit: int = 50) -> List[dict]:
        with self._lock:
            return list(self._history[location_id][-limit:])

    def clear(self) -> None:
        """Reset all stored state (useful in tests)."""
        with self._lock:
            self._current.clear()
            self._history.clear()
            self._locations.clear()
