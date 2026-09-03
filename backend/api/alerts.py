"""
GET /alerts
"""
from fastapi import APIRouter, Depends

from backend.dependencies import get_alert_service
from backend.schemas import AlertEntry, AlertsResponse
from backend.services.alert_service import AlertService

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=AlertsResponse)
def get_alerts(
    alert_service: AlertService = Depends(get_alert_service),
) -> AlertsResponse:
    """Return all currently ACTIVE alerts."""
    alerts = alert_service.list_active()
    return AlertsResponse(alerts=[AlertEntry(**a) for a in alerts])
