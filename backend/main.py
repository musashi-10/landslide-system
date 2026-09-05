"""
Landslide Early Warning System — Backend API

Start with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Or:
    python -m backend.main
"""
from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import health, risk, alerts, predict, auth
from backend.config import get_backend_settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_backend_settings()

    app = FastAPI(
        title="Landslide Early Warning — Backend API",
        description=(
            "Platform API for the AI-powered landslide early-warning system. "
            "Provides risk maps, predictions, history, factors, and alerts."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(risk.router)
    app.include_router(alerts.router)
    app.include_router(predict.router)
    app.include_router(auth.router)

    # ── Global error handler ─────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception for %s: %s", request.url, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred. Please try again later.",
                }
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_backend_settings()
    logging.basicConfig(level=settings.log_level)
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
    )
