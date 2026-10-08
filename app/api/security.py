"""Authentication for operator-only endpoints."""

import hmac

from fastapi import Header, HTTPException

from app.helpers.config import get_settings


def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    """Protect endpoints that write to the property index.

    Unauthenticated writes would let anyone poison listings or inject
    instructions into the grounded generator, so these endpoints are disabled
    until ADMIN_API_KEY is configured.
    """
    expected = get_settings().ADMIN_API_KEY
    if not expected:
        raise HTTPException(status_code=503, detail="Admin endpoints are disabled; set ADMIN_API_KEY")
    if not hmac.compare_digest(x_admin_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid admin key")
