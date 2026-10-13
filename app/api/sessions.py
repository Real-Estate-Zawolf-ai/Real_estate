"""Signed website session ids and short-lived voice tokens.

A website conversation is identified by a server-issued, HMAC-signed session
id. Text chat and voice both resolve it to the same LangGraph thread, and a
client cannot address another customer's conversation without the server key.

A session id is ``<random>.<issued-at>.<signature>``. It stops being accepted
``SESSION_MAX_AGE_DAYS`` after it was issued, and the server re-signs it with a
fresh timestamp once it is a day old, so an active customer keeps their
conversation while a leaked id eventually dies.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from functools import lru_cache
from uuid import uuid4

from app.database import replay
from app.helpers.config import get_settings


logger = logging.getLogger(__name__)

_REFRESH_AFTER_SECONDS = 24 * 3600


@lru_cache(maxsize=1)
def _secret() -> bytes:
    configured = get_settings().SESSION_SECRET
    if configured:
        return configured.encode()
    logger.warning("SESSION_SECRET is not set; using a per-process key (sessions reset on restart).")
    return secrets.token_bytes(32)


def reset_secret_cache() -> None:
    _secret.cache_clear()


def _sign(value: str) -> str:
    return hmac.new(_secret(), value.encode(), hashlib.sha256).hexdigest()[:32]


def _b36(number: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while number:
        number, remainder = divmod(number, 36)
        result = digits[remainder] + result
    return result or "0"


def _sign_session(raw: str, issued_at: int) -> str:
    stamp = _b36(issued_at)
    return f"{raw}.{stamp}.{_sign(f'session:{raw}.{stamp}')}"


def issue_session_id(raw: str | None = None, *, now: float | None = None) -> str:
    """Issue a signed id for a new conversation, or re-sign an existing one."""
    return _sign_session(raw or uuid4().hex, int(now if now is not None else time.time()))


def inspect_session_id(value: str | None, *, now: float | None = None) -> tuple[str, int] | None:
    """Return ``(raw_id, age_seconds)`` for a valid, unexpired id, else None."""
    if not value or value.count(".") != 2 or len(value) > 100:
        return None
    raw, stamp, signature = value.split(".")
    if not hmac.compare_digest(signature, _sign(f"session:{raw}.{stamp}")):
        return None
    try:
        issued_at = int(stamp, 36)
    except ValueError:
        return None
    age = int(now if now is not None else time.time()) - issued_at
    if age < -300 or age > get_settings().SESSION_MAX_AGE_DAYS * 86400:
        return None
    return raw, max(age, 0)


def verify_session_id(value: str | None) -> str | None:
    """Return the raw session id when the signature is valid and not expired."""
    verified = inspect_session_id(value)
    return verified[0] if verified else None


def needs_refresh(age_seconds: int) -> bool:
    return age_seconds >= _REFRESH_AFTER_SECONDS


def website_thread(raw_session_id: str) -> str:
    return f"web:{raw_session_id}"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def issue_voice_token(raw_session_id: str, language: str) -> str:
    claims = {
        "sid": raw_session_id,
        "lang": language if language in {"ar", "en"} else "en",
        "exp": int(time.time()) + get_settings().VOICE_TOKEN_TTL_SECONDS,
        "nonce": secrets.token_hex(8),
    }
    body = _b64(json.dumps(claims, separators=(",", ":")).encode())
    return f"{body}.{_sign('voice:' + body)}"


def verify_voice_token(token: str | None) -> dict | None:
    if not token or token.count(".") != 1 or len(token) > 512:
        return None
    body, signature = token.split(".")
    if not hmac.compare_digest(signature, _sign("voice:" + body)):
        return None
    try:
        claims = json.loads(_unb64(body))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(claims, dict) or int(claims.get("exp", 0)) < time.time():
        return None
    if not isinstance(claims.get("nonce"), str) or not isinstance(claims.get("sid"), str):
        return None
    return claims


def consume_voice_token(token: str | None) -> dict | None:
    """Verify a voice token and reject any second use of it.

    The single-use guarantee lives in the database, so it holds across API
    workers and restarts (a captured token cannot be replayed on another worker).
    """
    claims = verify_voice_token(token)
    if claims is None:
        return None
    remaining = max(int(claims["exp"] - time.time()), 1)
    if not replay.claim("voice_token", claims["nonce"], ttl_seconds=remaining + 60):
        return None
    return claims


def origin_allowed(origin: str | None) -> bool:
    allowed = get_settings().cors_origins
    if "*" in allowed:
        return True
    return bool(origin) and origin.rstrip("/") in allowed
