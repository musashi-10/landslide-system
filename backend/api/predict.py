"""
POST /predict

Proxy endpoint that accepts environmental features, calls the ML provider,
and returns a risk prediction.  The backend does NOT implement the model —
it delegates to the configured MLProvider.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_risk_service
from backend.schemas import PredictRequest, PredictResponse
from backend.services.risk_service import RiskService

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    risk_service: RiskService = Depends(get_risk_service),
) -> PredictResponse:
    """
    Submit environmental features for a risk prediction.

    The backend forwards the request to the ML provider (mock or real).
    Results are stored and alert evaluation is triggered automatically.
    """
    try:
        prediction = risk_service.predict(
            location_id=request.location_id,
            features=request.features.model_dump(exclude_none=True),
            timestamp_utc=request.timestamp_utc,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "ML_UNAVAILABLE",
                    "message": str(exc),
                }
            },
        ) from exc

    return PredictResponse(**prediction)
