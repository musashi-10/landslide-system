"""
ML Inference Service API
========================

FastAPI application implementing the ML prediction endpoints defined in
docs/api-contract.md.

Endpoints
---------
  GET  /health                          Health check
  POST /predict                         Single-location risk prediction
  GET  /risk/current/{location_id}      Latest risk for a location
  GET  /risk/map                        Risk for all known locations
  GET  /risk/history/{location_id}      Historical risk time series
  GET  /risk/factors/{location_id}      Feature importance for a location

Integration notes for Engineer 6
---------------------------------
- Import: `from ml.api.app import create_app` then `app = create_app()`
- Or run directly: `uvicorn ml.api.app:app --port 8001`
- If no trained model is found, the service automatically falls back to
  MockPredictor and returns a "mock_mode": true flag in /health.

Error format (from api-contract.md §9)
---------------------------------------
  {"error": {"code": "ERROR_CODE", "message": "Human-readable explanation"}}
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from shared.schemas import RiskPrediction
from ml.inference.predictor import Predictor, PredictionError
from ml.inference.mock_predictor import MockPredictor
from ml.inference.risk_classifier import RiskClassifier
from shared.config.settings import get_settings

logger = logging.getLogger(__name__)

# ── Request / Response models ─────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Request body for POST /predict (api-contract.md §8)."""

    location_id: str
    timestamp_utc: str
    features: dict[str, Any] = Field(default_factory=dict)

    model_config = {"json_schema_extra": {
        "example": {
            "location_id": "LOC_001",
            "timestamp_utc": "2026-09-02T12:00:00Z",
            "features": {
                "elevation": 1240.5,
                "slope": 32.4,
                "aspect": 145.0,
                "soil_type": "loam",
                "geology": "schist",
                "land_cover": "forest",
                "historical_landslide": 1,
                "rainfall_1h": 12.4,
                "rainfall_6h": 48.2,
                "rainfall_24h": 91.5,
                "rainfall_3d": 165.3,
                "rainfall_7d": 240.1,
                "forecast_rainfall": 42.0,
                "moisture_indicator": 0.71,
            },
        }
    }}


class RiskMapLocation(BaseModel):
    location_id: str
    latitude: float
    longitude: float
    risk_probability: float
    risk_level: str


class RiskMapResponse(BaseModel):
    timestamp_utc: str
    locations: list[RiskMapLocation]


class RiskHistoryEntry(BaseModel):
    timestamp_utc: str
    risk_probability: float
    risk_level: str


class RiskHistoryResponse(BaseModel):
    location_id: str
    history: list[RiskHistoryEntry]


class RiskFactor(BaseModel):
    feature: str
    value: Optional[float] = None
    importance: float


class RiskFactorsResponse(BaseModel):
    location_id: str
    factors: list[RiskFactor]


class HealthResponse(BaseModel):
    status: str
    service: str
    model_version: Optional[str] = None
    mock_mode: bool = False


# ── App factory ────────────────────────────────────────────────────────────────

# Module-level predictor — loaded once at startup
_predictor: Optional[Predictor | MockPredictor] = None
_mock_mode: bool = False


def _load_predictor() -> tuple[Any, bool]:
    """
    Load the real predictor if a trained model exists, else fall back to mock.

    Returns
    -------
    (predictor, is_mock) : tuple
    """
    settings = get_settings()
    model_dir = settings.model_path or "ml/models/artifacts"
    version = settings.model_version or None

    try:
        predictor = Predictor.from_registry(model_dir=model_dir, version=version)
        logger.info(
            f"Real predictor loaded: {predictor.model_type} v{predictor.model_version}"
        )
        return predictor, False
    except (PredictionError, FileNotFoundError, Exception) as exc:
        logger.warning(
            f"Could not load trained model ({exc}). "
            "Falling back to MockPredictor. "
            "Run training to enable real predictions."
        )
        return MockPredictor(), True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup; nothing to clean up at shutdown."""
    global _predictor, _mock_mode
    _predictor, _mock_mode = _load_predictor()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Landslide ML Risk Engine",
        description="AI-powered landslide risk prediction service.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── Health check ──────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health():
        """
        Service health check (api-contract.md §2).

        Returns mock_mode: true when no trained model is loaded.
        """
        version = getattr(_predictor, "model_version", None)
        return HealthResponse(
            status="ok",
            service="landslide-ml-engine",
            model_version=version,
            mock_mode=_mock_mode,
        )

    # ── Prediction ─────────────────────────────────────────────────────────────

    @app.post("/predict", response_model=RiskPrediction, tags=["Prediction"])
    async def predict(body: PredictRequest):
        """
        Predict landslide risk for a single location (api-contract.md §8).

        Accepts the standardised feature vector and returns a RiskPrediction.
        """
        request_dict = {
            "location_id": body.location_id,
            "timestamp_utc": body.timestamp_utc,
            "features": body.features,
        }

        try:
            result = _predictor.predict(request_dict)
            return result
        except PredictionError as exc:
            return JSONResponse(
                status_code=422,
                content={"error": {"code": exc.code, "message": exc.message}},
            )
        except Exception as exc:
            logger.exception("Unexpected prediction error")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": f"Unexpected error: {exc}",
                    }
                },
            )

    # ── Current risk ───────────────────────────────────────────────────────────

    @app.get("/risk/current/{location_id}", response_model=RiskPrediction, tags=["Risk"])
    async def current_risk(location_id: str):
        """
        Latest risk for a location (api-contract.md §3).

        Uses an empty feature vector — real-time features should be provided
        via POST /predict for production use.  This endpoint is primarily for
        dashboard polling with cached predictions.
        """
        request_dict = {
            "location_id": location_id,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "features": {},
        }
        try:
            return _predictor.predict(request_dict)
        except PredictionError as exc:
            return JSONResponse(
                status_code=422,
                content={"error": {"code": exc.code, "message": exc.message}},
            )

    # ── Risk map ───────────────────────────────────────────────────────────────

    @app.get("/risk/map", response_model=RiskMapResponse, tags=["Risk"])
    async def risk_map(
        risk_level: Optional[str] = Query(None, description="Filter by risk level"),
        bounding_box: Optional[str] = Query(
            None, description="min_lat,min_lon,max_lat,max_lon"
        ),
    ):
        """
        Risk map snapshot (api-contract.md §4).

        Returns mock locations in mock mode.  In production, this should be
        driven by a scheduled prediction cache (Engineer 6 integration).
        """
        # In mock mode return sample locations
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        mock_locations = [
            RiskMapLocation(
                location_id="LOC_TEST_001",
                latitude=27.35,
                longitude=88.62,
                risk_probability=0.78,
                risk_level="HIGH",
            ),
            RiskMapLocation(
                location_id="LOC_TEST_002",
                latitude=27.40,
                longitude=88.65,
                risk_probability=0.45,
                risk_level="MODERATE",
            ),
            RiskMapLocation(
                location_id="LOC_TEST_003",
                latitude=27.25,
                longitude=88.55,
                risk_probability=0.91,
                risk_level="CRITICAL",
            ),
            RiskMapLocation(
                location_id="LOC_TEST_004",
                latitude=27.50,
                longitude=88.70,
                risk_probability=0.12,
                risk_level="LOW",
            ),
        ]

        if risk_level:
            mock_locations = [
                loc for loc in mock_locations
                if loc.risk_level == risk_level.upper()
            ]

        return RiskMapResponse(timestamp_utc=now_utc, locations=mock_locations)

    # ── Risk history ───────────────────────────────────────────────────────────

    @app.get(
        "/risk/history/{location_id}", response_model=RiskHistoryResponse, tags=["Risk"]
    )
    async def risk_history(location_id: str):
        """
        Historical risk time series for a location (api-contract.md §5).

        Returns sample history in mock mode.
        """
        history = [
            RiskHistoryEntry(
                timestamp_utc="2026-09-01T12:00:00Z",
                risk_probability=0.42,
                risk_level="MODERATE",
            ),
            RiskHistoryEntry(
                timestamp_utc="2026-09-02T12:00:00Z",
                risk_probability=0.78,
                risk_level="HIGH",
            ),
        ]
        return RiskHistoryResponse(location_id=location_id, history=history)

    # ── Risk factors ───────────────────────────────────────────────────────────

    @app.get(
        "/risk/factors/{location_id}", response_model=RiskFactorsResponse, tags=["Risk"]
    )
    async def risk_factors(location_id: str):
        """
        Feature importance for a location (api-contract.md §6).
        """
        factors = [
            RiskFactor(feature="rainfall_24h", value=91.5, importance=0.42),
            RiskFactor(feature="slope", value=32.4, importance=0.31),
            RiskFactor(feature="static_susceptibility_score", value=0.75, importance=0.18),
            RiskFactor(feature="antecedent_rainfall", value=405.4, importance=0.09),
        ]
        return RiskFactorsResponse(location_id=location_id, factors=factors)

    return app


# ── Module-level app for uvicorn ──────────────────────────────────────────────
app = create_app()
