"""Authentication for operator-only endpoints."""

import hmac
import logging
from typing import Any

from fastapi import Header, HTTPException, Request

from app.api.ratelimit import client_ip, limit_admin, record_admin_failure
from app.helpers.config import get_settings


logger = logging.getLogger(__name__)


def require_admin(request: Request, x_admin_key: str | None = Header(default=None)) -> None:
    """Protect endpoints that write to the property index or manage bookings.

    Unauthenticated writes would let anyone poison listings or read customers'
    contact details, so these endpoints stay disabled until ADMIN_API_KEY is set.
    Calls are rate-limited and a client that keeps sending a wrong key is locked
    out for a while, which makes guessing the key impractical.
    """
    expected = get_settings().ADMIN_API_KEY
    if not expected:
        raise HTTPException(status_code=503, detail="Admin endpoints are disabled; set ADMIN_API_KEY")
    limit_admin(request)
    if not hmac.compare_digest((x_admin_key or "").encode(), expected.encode()):
        record_admin_failure(request)
        logger.warning("Rejected admin request | client=%s path=%s", client_ip(request), request.url.path)
        raise HTTPException(status_code=401, detail="Invalid admin key")


def audit(request: Request, action: str, target: str | None = None, detail: dict[str, Any] | None = None) -> None:
    """Record an operator action. Failing to audit must not block the action."""
    from app.database import models as db
    from app.database.session import session_scope

    try:
        with session_scope() as session:
            session.add(
                db.AdminAudit(
                    action=action[:60],
                    target=target[:100] if target else None,
                    client=client_ip(request),
                    detail=detail,
                )
            )
    except Exception:
        logger.exception("Could not write admin audit record | action=%s", action)
