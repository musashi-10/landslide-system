"""
Risk Service.

Orchestrates:
  1. Receiving a prediction request (location_id + features)
  2. Calling the ML provider
  3. Persisting the result in RiskRepository
  4. Triggering the AlertEngine
  5. Returning the prediction dict

All business logic lives here, not in the API routers.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from backend.services.ml_provider import MLProvider
from backend.repositories.risk_repository import RiskRepository

logger = logging.getLogger(__name__)

# Default locations seeded into the in-memory repository on startup.
# In production these would come from the GIS pipeline parquet output.
_DEFAULT_LOCATIONS: list[dict] = [
    {"location_id": "LOC_001", "latitude": 27.123, "longitude": 88.456},
    {"location_id": "LOC_002", "latitude": 27.234, "longitude": 88.567},
    {"location_id": "LOC_003", "latitude": 27.345, "longitude": 88.678},
    {"location_id": "LOC_004", "latitude": 27.456, "longitude": 88.789},
    {"location_id": "LOC_005", "latitude": 27.567, "longitude": 88.890},
    {"location_id": "LOC_TEST_CRITICAL", "latitude": 27.100, "longitude": 88.100},
    {"location_id": "LOC_TEST_HIGH",     "latitude": 27.110, "longitude": 88.110},
    {"location_id": "LOC_TEST_MODERATE", "latitude": 27.120, "longitude": 88.120},
    {"location_id": "LOC_TEST_LOW",      "latitude": 27.130, "longitude": 88.130},
]


class RiskService:
    """
    Core risk business logic.

    Parameters
    ----------
    ml_provider    : ML inference abstraction
    risk_repository: prediction + history storage
    alert_engine   : alert processing (optional; pass None to disable)
    """

    def __init__(
        self,
        ml_provider: MLProvider,
        risk_repository: RiskRepository,
        alert_engine=None,
    ) -> None:
        self._ml = ml_provider
        self._repo = risk_repository
        self._alerts = alert_engine
        self._seed_locations()

    # ── Public ────────────────────────────────────────────────────────────────

    def predict(self, location_id: str, features: dict, timestamp_utc: str | None = None) -> dict:
        """
        Request a new ML prediction, persist it, trigger alert evaluation,
        and return the prediction dict.
        """
        ts = timestamp_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            prediction = self._ml.predict(location_id, features, ts)
        except RuntimeError as exc:
            logger.error("ML provider error: %s", exc)
            raise

        self._repo.save_prediction(prediction)

        # Trigger alert engine if configured
        if self._alerts is not None:
            try:
                self._alerts.process(prediction)
            except Exception as exc:  # noqa: BLE001
                logger.error("Alert engine error: %s", exc)

        return prediction

    def get_current(self, location_id: str) -> Optional[dict]:
        """
        Return the latest prediction for a location.

        If no prediction exists yet, generate one via the mock/ML provider
        so the API always returns something useful during development.
        """
        current = self._repo.get_current(location_id)
        if current is None:
            # Auto-generate a prediction using empty features (mock returns seeded values)
            current = self.predict(location_id, {})
        return current

    def get_history(self, location_id: str, limit: int = 50) -> list[dict]:
        return self._repo.get_history(location_id, limit=limit)

    def get_risk_map(
        self,
        risk_level_filter: str | None = None,
        bbox: str | None = None,
    ) -> list[dict]:
        """
        Return current risk for all known locations, with optional filters.

        Parameters
        ----------
        risk_level_filter : e.g. "HIGH" — returns only locations at this level+
        bbox : "west,south,east,north" string
        """
        locations = self._repo.list_locations()
        results = []

        for loc in locations:
            lid = loc["location_id"]
            pred = self._repo.get_current(lid)
            if pred is None:
                pred = self.predict(lid, {})

            entry = {
                "location_id": lid,
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "risk_probability": pred["risk_probability"],
                "risk_level": pred["risk_level"],
            }

            # Apply risk_level filter
            if risk_level_filter:
                from alerts.models import risk_level_value
                if risk_level_value(pred["risk_level"]) < risk_level_value(risk_level_filter):
                    continue

            # Apply bbox filter: west,south,east,north
            if bbox:
                try:
                    west, south, east, north = [float(x) for x in bbox.split(",")]
                    lat, lon = loc["latitude"], loc["longitude"]
                    if not (south <= lat <= north and west <= lon <= east):
                        continue
                except (ValueError, TypeError):
                    pass  # malformed bbox → ignore filter

            results.append(entry)

        return results

    def get_factors(self, location_id: str) -> list[dict]:
        """Return risk factor breakdown for a location."""
        pred = self._repo.get_current(location_id)
        if pred is None:
            pred = self.predict(location_id, {})

        factors = []
        for i, factor_name in enumerate(pred.get("top_risk_factors", [])):
            factors.append({
                "feature": factor_name,
                "value": None,          # ML model does not expose raw values by default
                "importance": round(0.5 - i * 0.1, 2),  # mock importance ordering
            })
        return factors

    # ── Private ───────────────────────────────────────────────────────────────

    def _seed_locations(self) -> None:
        """Register default locations so the map API always has data."""
        for loc in _DEFAULT_LOCATIONS:
            self._repo.register_location(
                loc["location_id"], loc["latitude"], loc["longitude"]
            )
