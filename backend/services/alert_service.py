"""
Alert Service facade.

Thin wrapper that connects AlertEngine + AlertRepository to the API layer.
Keeps the API routers free of alert business logic.
"""
from __future__ import annotations

from typing import List

from alerts.engine import AlertEngine
from backend.repositories.alert_repository import AlertRepository


class AlertService:
    def __init__(self, engine: AlertEngine, repository: AlertRepository) -> None:
        self._engine = engine
        self._repo = repository

    def list_active(self) -> List[dict]:
        """Return all ACTIVE alerts as dicts."""
        return [a.to_dict() for a in self._repo.list_active()]

    def list_all(self) -> List[dict]:
        """Return all alerts (any status) as dicts."""
        return [a.to_dict() for a in self._repo.list_all()]

    def acknowledge(self, alert_id: str) -> bool:
        return self._engine.acknowledge(alert_id)

    def resolve(self, alert_id: str) -> bool:
        return self._engine.resolve(alert_id)
