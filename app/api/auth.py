"""
Authentication and security dependencies for Administrative Endpoints.
"""

import logging
import os
import secrets
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# Header schema extracting X-Admin-API-Key from HTTP requests
ADMIN_API_KEY_HEADER = APIKeyHeader(name="X-Admin-API-Key", auto_error=False)


def verify_admin_api_key(
    provided_key: Optional[str] = Security(ADMIN_API_KEY_HEADER),
) -> bool:
    """
    Validate administrative API key header using constant-time comparison.

    Security Policies:
    1. Fail-closed: If ADMIN_API_KEY is unset or empty on the server, fail closed with HTTP 503.
    2. Header enforcement: Rejects requests missing the X-Admin-API-Key header with HTTP 403.
    3. Side-channel protection: Uses secrets.compare_digest for constant-time comparison.
    """
    expected_key = os.getenv("ADMIN_API_KEY")

    # 1. Fail closed if server has no key configured (HTTP 503 Service Unavailable)
    if not expected_key or not expected_key.strip():
        logger.error(
            "ADMIN_API_KEY environment variable is not configured on the server. "
            "Administrative services are unavailable."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin authentication is not configured on the server. Administrative services are unavailable.",
        )

    # 2. Reject missing or empty client header (HTTP 403 Forbidden)
    if not provided_key or not provided_key.strip():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing required administrative header: X-Admin-API-Key.",
        )

    # 3. Constant-time digest comparison to prevent timing side-channel attacks
    if not secrets.compare_digest(provided_key.strip(), expected_key.strip()):
        logger.warning("Unauthorized access attempt with invalid X-Admin-API-Key.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid administrative API key.",
        )

    return True
