"""
Pydantic response/request schemas for the Backend API.

These models exactly mirror docs/api-contract.md and must not be changed
without updating that document first.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ── Auth ──────────────────────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    name: str
    email: str
    mobile: str
    password: str
    location_id: Optional[str] = None


class UserProfile(BaseModel):
    id: str
    name: str
    email: str
    mobile: str
    role: str
    location_id: Optional[str] = None


class AuthResponse(BaseModel):
    token: str
    user: UserProfile


# ── Health ──────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    service: str


# ── Current Risk ─────────────────────────────────────────────────────────────


class RiskCurrentResponse(BaseModel):
    location_id: str
    timestamp_utc: str
    risk_probability: float = Field(ge=0.0, le=1.0)
    risk_level: str
    top_risk_factors: List[str] = []


# ── Risk Map ─────────────────────────────────────────────────────────────────


class RiskMapLocation(BaseModel):
    location_id: str
    latitude: float
    longitude: float
    risk_probability: float = Field(ge=0.0, le=1.0)
    risk_level: str


class RiskMapResponse(BaseModel):
    timestamp_utc: str
    locations: List[RiskMapLocation]


# ── Risk History ─────────────────────────────────────────────────────────────


class RiskHistoryEntry(BaseModel):
    timestamp_utc: str
    risk_probability: float = Field(ge=0.0, le=1.0)
    risk_level: str


class RiskHistoryResponse(BaseModel):
    location_id: str
    history: List[RiskHistoryEntry]


# ── Risk Factors ─────────────────────────────────────────────────────────────


class RiskFactor(BaseModel):
    feature: str
    value: Optional[float] = None
    importance: Optional[float] = None


class RiskFactorsResponse(BaseModel):
    location_id: str
    factors: List[RiskFactor]


# ── Alerts ───────────────────────────────────────────────────────────────────


class AlertEntry(BaseModel):
    alert_id: str
    location_id: str
    timestamp_utc: str
    risk_level: str
    status: str
    message: Optional[str] = None


class AlertsResponse(BaseModel):
    alerts: List[AlertEntry]


# ── Predict ───────────────────────────────────────────────────────────────────


class FeatureSet(BaseModel):
    elevation: Optional[float] = None
    slope: Optional[float] = None
    aspect: Optional[float] = None
    soil_type: Optional[str] = None
    geology: Optional[str] = None
    land_cover: Optional[str] = None
    historical_landslide: Optional[int] = None
    rainfall_1h: Optional[float] = None
    rainfall_6h: Optional[float] = None
    rainfall_24h: Optional[float] = None
    rainfall_3d: Optional[float] = None
    rainfall_7d: Optional[float] = None
    forecast_rainfall: Optional[float] = None
    moisture_indicator: Optional[float] = None


class PredictRequest(BaseModel):
    location_id: str
    timestamp_utc: str
    features: FeatureSet


class PredictResponse(BaseModel):
    location_id: str
    timestamp_utc: str
    risk_probability: float = Field(ge=0.0, le=1.0)
    risk_level: str
    top_risk_factors: List[str] = []


# ── Error ─────────────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
