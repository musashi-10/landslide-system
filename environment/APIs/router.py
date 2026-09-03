"""
environment/APIs/router.py

FastAPI router for the dynamic environment API.

Endpoints
---------
GET /environment/health
    Health check.

GET /environment/{location_id}
    Current dynamic conditions for a location.

GET /environment/{location_id}/history
    Time-series of dynamic records for a location.

Integration with Engineer 6
----------------------------
These endpoints conform to the API contract (docs/api-contract.md §9 error
format).  Engineer 6 can mount this router in the main backend application.

Example mount (in backend main.py):
    from environment.APIs.router import router as env_router
    app.include_router(env_router, prefix="/environment", tags=["environment"])

Authentication / rate limiting
-------------------------------
Not implemented in prototype.  Add middleware at the backend level.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from environment.aggregation.feature_builder import build_dynamic_features, dynamic_record_to_dict
from environment.aggregation.pipeline import DynamicFeaturePipeline, LocationSpec
from environment.caching.local_cache import LocalCache
from environment.config import DEFAULT_CONFIG
from environment.providers import MockWeatherProvider, get_provider
from environment.schemas.dynamic_record import utc_now_iso

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level defaults (can be overridden in tests)
_provider = MockWeatherProvider()  # Replace with get_provider("open_meteo") in production
_cache = LocalCache(ttl_seconds=DEFAULT_CONFIG.cache_ttl_seconds)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class EnvironmentResponse(BaseModel):
    """Response schema for GET /environment/{location_id}."""

    location_id: str
    timestamp_utc: str
    rainfall_1h: Optional[float]
    rainfall_6h: Optional[float]
    rainfall_24h: Optional[float]
    rainfall_3d: Optional[float]
    rainfall_7d: Optional[float]
    forecast_rainfall: Optional[float]
    moisture_indicator: Optional[float]
    data_source: str


class EnvironmentHistoryItem(BaseModel):
    timestamp_utc: str
    rainfall_1h: Optional[float]
    rainfall_24h: Optional[float]
    rainfall_7d: Optional[float]
    forecast_rainfall: Optional[float]
    moisture_indicator: Optional[float]


class EnvironmentHistoryResponse(BaseModel):
    location_id: str
    history: List[EnvironmentHistoryItem]


class HealthResponse(BaseModel):
    status: str
    service: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_location_id(location_id: str):
    """
    Parse a location_id into (lat, lon).

    Format: LOC_{lat:+.4f}_{lon:+08.4f}
    Example: LOC_+27.1230_+088.4560
    """
    try:
        parts = location_id.split("_")
        # parts: ['LOC', '+27.1230', '+088.4560']
        lat = float(parts[1])
        lon = float(parts[2])
        return lat, lon
    except (IndexError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_LOCATION_ID", "message": str(exc)}},
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["environment"])
def health_check():
    """Health check endpoint. Returns service status."""
    return HealthResponse(status="ok", service="environment")


@router.get("/{location_id}", response_model=EnvironmentResponse, tags=["environment"])
def get_current_conditions(
    location_id: str,
    timestamp_utc: Optional[str] = Query(
        default=None,
        description="ISO 8601 UTC timestamp. Defaults to current time.",
    ),
):
    """
    Get current dynamic environmental conditions for a location.

    Returns dynamic rainfall features and moisture indicator at the
    requested timestamp (defaults to now).

    Response conforms to the data contract §6 feature names.
    """
    ts = timestamp_utc or utc_now_iso()
    lat, lon = _parse_location_id(location_id)

    try:
        record = build_dynamic_features(
            location_id=location_id,
            latitude=lat,
            longitude=lon,
            timestamp_utc=ts,
            provider=_provider,
            config=DEFAULT_CONFIG,
            cache=_cache,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_INPUT", "message": str(exc)}},
        )
    except Exception as exc:
        logger.exception("Unexpected error building features for %s", location_id)
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "INTERNAL_ERROR", "message": str(exc)}},
        )

    return EnvironmentResponse(
        location_id=record.location_id,
        timestamp_utc=record.timestamp_utc,
        rainfall_1h=record.rainfall_1h,
        rainfall_6h=record.rainfall_6h,
        rainfall_24h=record.rainfall_24h,
        rainfall_3d=record.rainfall_3d,
        rainfall_7d=record.rainfall_7d,
        forecast_rainfall=record.forecast_rainfall,
        moisture_indicator=record.moisture_indicator,
        data_source=record.data_source,
    )


@router.get("/{location_id}/history", response_model=EnvironmentHistoryResponse, tags=["environment"])
def get_conditions_history(
    location_id: str,
    hours: int = Query(default=72, ge=1, le=720, description="How many hours of history to return"),
    step_hours: int = Query(default=1, ge=1, le=24, description="Time step between records (hours)"),
):
    """
    Get a time-series of dynamic conditions for a location.

    Returns records at ``step_hours`` intervals over the last ``hours`` period.
    Useful for Engineer 6's dashboard time-series charts.
    """
    lat, lon = _parse_location_id(location_id)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    history: List[EnvironmentHistoryItem] = []

    for h in range(hours, 0, -step_hours):
        ts_dt = now - __import__("datetime").timedelta(hours=h)
        ts_str = ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            record = build_dynamic_features(
                location_id=location_id,
                latitude=lat,
                longitude=lon,
                timestamp_utc=ts_str,
                provider=_provider,
                config=DEFAULT_CONFIG,
                cache=_cache,
            )
            history.append(
                EnvironmentHistoryItem(
                    timestamp_utc=record.timestamp_utc,
                    rainfall_1h=record.rainfall_1h,
                    rainfall_24h=record.rainfall_24h,
                    rainfall_7d=record.rainfall_7d,
                    forecast_rainfall=record.forecast_rainfall,
                    moisture_indicator=record.moisture_indicator,
                )
            )
        except Exception as exc:
            logger.warning("History point %s failed for %s: %s", ts_str, location_id, exc)
            # Skip failed points — don't abort entire history

    return EnvironmentHistoryResponse(location_id=location_id, history=history)
