"""
FastAPI Main Application Factory.
"""

import logging
import os
from typing import List, Optional
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router

logger = logging.getLogger(__name__)


def create_app(allowed_origins: Optional[List[str]] = None) -> FastAPI:
    """
    Construct and configure the FastAPI application.

    Args:
        allowed_origins: Optional list of explicit frontend origins. Unrestricted wildcard CORS is forbidden.
    """
    app = FastAPI(
        title="Kerala Government Orders Search & Verification API",
        description=(
            "FastAPI HTTP gateway for the LangGraph Multi-Agent Architecture analyzing "
            "Kerala Government Orders with evidence-based fact-checking and relationship investigation."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS strictly: no unrestricted wildcard allowed
    if allowed_origins is None:
        env_origins = os.getenv("ALLOWED_ORIGINS")
        if env_origins and env_origins.strip():
            allowed_origins = [o.strip() for o in env_origins.split(",") if o.strip()]

    if allowed_origins:
        if "*" in allowed_origins:
            raise ValueError("Unrestricted wildcard '*' is forbidden in ALLOWED_ORIGINS.")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "X-Admin-API-Key", "Authorization", "Accept"],
        )

    # Global validation error handler
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning("Validation error on path %s: %s", request.url.path, exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error_type": "VALIDATION_ERROR",
                "detail": jsonable_encoder(exc.errors(), custom_encoder={Exception: str}),
            },
        )

    # Global sanitized exception handler
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server exception on path %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_type": "INTERNAL_SERVER_ERROR",
                "detail": "An unexpected error occurred while processing the request.",
            },
        )

    # Mount API routes
    app.include_router(router)

    return app


# Default app instance for ASGI servers (e.g. uvicorn app.api.main:app)
app = create_app()
