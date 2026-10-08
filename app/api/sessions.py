"""Signed website session ids and short-lived voice tokens.

A website conversation is identified by a server-issued, HMAC-signed session
id. Text chat and voice both resolve it to the same LangGraph thread, and a
client cannot address another customer's conversation without the server key.
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

from app.helpers.config import get_settings


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _secret() -> bytes:
    configured = get_settings().SESSION_SECRET
    if configured:
        return configured.encode()
    logger.warning("SESSION_SECRET is not set; using a per-process key (sessions reset on restart).")
    return secrets.token_bytes(32)


def _sign(value: str) -> str:
    return hmac.new(_secret(), value.encode(), hashlib.sha256).hexdigest()[:32]


def issue_session_id() -> str:
    raw = uuid4().hex
    return f"{raw}.{_sign('session:' + raw)}"


def verify_session_id(value: str | None) -> str | None:
    """Return the raw session id when the signature is valid."""
    if not value or value.count(".") != 1 or len(value) > 100:
        return None
    raw, signature = value.split(".")
    if hmac.compare_digest(signature, _sign("session:" + raw)):
        return raw
    return None


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
    return claims


_used_nonces: dict[str, float] = {}


def consume_voice_token(token: str | None) -> dict | None:
    """Verify a voice token and reject any second use of it."""
    claims = verify_voice_token(token)
    if claims is None:
        return None
    now = time.time()
    for nonce, expires in list(_used_nonces.items()):
        if expires < now:
            _used_nonces.pop(nonce, None)
    if claims["nonce"] in _used_nonces:
        return None
    _used_nonces[claims["nonce"]] = float(claims["exp"])
    return claims


def origin_allowed(origin: str | None) -> bool:
    allowed = get_settings().cors_origins
    if "*" in allowed:
        return True
    return bool(origin) and origin.rstrip("/") in allowed
