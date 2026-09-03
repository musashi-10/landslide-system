"""GET /health"""
from fastapi import APIRouter
from backend.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Service liveness probe."""
    return HealthResponse(status="ok", service="backend")
