"""
Risk endpoints.

GET /risk/current/{location_id}
GET /risk/map
GET /risk/history/{location_id}
GET /risk/factors/{location_id}
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.dependencies import get_risk_service
from backend.schemas import (
    RiskCurrentResponse,
    RiskFactorsResponse,
    RiskFactor,
    RiskHistoryEntry,
    RiskHistoryResponse,
    RiskMapLocation,
    RiskMapResponse,
)
from backend.services.risk_service import RiskService

router = APIRouter(prefix="/risk", tags=["risk"])


@router.get("/current/{location_id}", response_model=RiskCurrentResponse)
def get_current_risk(
    location_id: str,
    risk_service: RiskService = Depends(get_risk_service),
) -> RiskCurrentResponse:
    """Return the latest risk prediction for a location."""
    prediction = risk_service.get_current(location_id)
    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "LOCATION_NOT_FOUND", "message": f"No data for location '{location_id}'."}},
        )
    return RiskCurrentResponse(**prediction)


@router.get("/map", response_model=RiskMapResponse)
def get_risk_map(
    risk_level: Optional[str] = Query(None, description="Filter by minimum risk level (LOW|MODERATE|HIGH|CRITICAL)"),
    bbox: Optional[str] = Query(None, description="Bounding box: west,south,east,north"),
    risk_service: RiskService = Depends(get_risk_service),
) -> RiskMapResponse:
    """Return risk summary for all locations, with optional filters."""
    locations = risk_service.get_risk_map(
        risk_level_filter=risk_level,
        bbox=bbox,
    )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return RiskMapResponse(
        timestamp_utc=ts,
        locations=[RiskMapLocation(**loc) for loc in locations],
    )


@router.get("/history/{location_id}", response_model=RiskHistoryResponse)
def get_risk_history(
    location_id: str,
    limit: int = Query(50, ge=1, le=500, description="Maximum number of history entries"),
    risk_service: RiskService = Depends(get_risk_service),
) -> RiskHistoryResponse:
    """Return risk time series for a location."""
    history = risk_service.get_history(location_id, limit=limit)
    return RiskHistoryResponse(
        location_id=location_id,
        history=[
            RiskHistoryEntry(
                timestamp_utc=h["timestamp_utc"],
                risk_probability=h["risk_probability"],
                risk_level=h["risk_level"],
            )
            for h in history
        ],
    )


@router.get("/factors/{location_id}", response_model=RiskFactorsResponse)
def get_risk_factors(
    location_id: str,
    risk_service: RiskService = Depends(get_risk_service),
) -> RiskFactorsResponse:
    """Return feature importance / risk factor breakdown for a location."""
    factors = risk_service.get_factors(location_id)
    return RiskFactorsResponse(
        location_id=location_id,
        factors=[RiskFactor(**f) for f in factors],
    )
