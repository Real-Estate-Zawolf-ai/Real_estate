"""Request rate limiting.

Sliding-window counters kept in process memory. The free-tier deployment runs a
single instance, so per-process limits are the limits; with several workers each
enforces its own budget (document this when scaling out - limits are then
per-worker). Memory is bounded: idle keys are purged and, if an attacker rotates
identities faster than they expire, the oldest keys are evicted.

Client identity comes from the socket peer, or from ``X-Forwarded-For`` counted
from the *right* by ``TRUSTED_PROXY_HOPS``. Counting from the right matters: the
left-most entries are client-controlled, so trusting them would let anyone dodge
a limit just by sending their own header.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from fastapi import HTTPException, Request, WebSocket

from app.helpers.config import get_settings


logger = logging.getLogger(__name__)

_MAX_KEYS = 50_000


class SlidingWindowLimiter:

    def __init__(self, max_keys: int = _MAX_KEYS) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._max_keys = max_keys
        self._last_purge = time.monotonic()

    def hit(self, key: str, limit: int, window: float, *, record: bool = True) -> float:
        """Count one event. Returns 0 when allowed, else seconds until a retry can succeed."""
        now = time.monotonic()
        with self._lock:
            self._maybe_purge(now, window)
            events = self._hits.get(key)
            if events is None:
                events = self._hits[key] = deque()
            cutoff = now - window
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                return max(events[0] + window - now, 0.001)
            if record:
                events.append(now)
            return 0.0

    def count(self, key: str, window: float) -> int:
        now = time.monotonic()
        with self._lock:
            events = self._hits.get(key)
            if not events:
                return 0
            return sum(1 for moment in events if moment > now - window)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def _maybe_purge(self, now: float, window: float) -> None:
        if now - self._last_purge < 60 and len(self._hits) < self._max_keys:
            return
        self._last_purge = now
        horizon = now - max(window, 900)
        for key in [k for k, events in self._hits.items() if not events or events[-1] <= horizon]:
            del self._hits[key]
        overflow = len(self._hits) - self._max_keys
        if overflow > 0:  # still too many live keys: drop the least recently active
            for key in sorted(self._hits, key=lambda k: self._hits[k][-1] if self._hits[k] else 0)[:overflow]:
                del self._hits[key]


_limiter = SlidingWindowLimiter()


def reset_limits() -> None:
    """Forget all counters (tests, and admin lockout reset)."""
    _limiter.reset()


# ---------------------------------------------------------------------------
# Client identity
# ---------------------------------------------------------------------------


def _forwarded_for(header: str | None, hops: int) -> str | None:
    if not header or hops < 1:
        return None
    parts = [part.strip() for part in header.split(",") if part.strip()]
    if not parts:
        return None
    # The entry added by the closest trusted proxy is `hops` from the right.
    return parts[-hops] if len(parts) >= hops else parts[0]


def client_ip(connection: Request | WebSocket) -> str:
    hops = get_settings().TRUSTED_PROXY_HOPS
    forwarded = _forwarded_for(connection.headers.get("x-forwarded-for"), hops)
    if forwarded:
        return forwarded[:64]
    peer = connection.client.host if connection.client else "unknown"
    return peer[:64]


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


def _reject(retry_after: float, what: str) -> HTTPException:
    seconds = max(int(retry_after + 0.999), 1)
    logger.warning("Rate limit hit | scope=%s retry_after=%ss", what, seconds)
    return HTTPException(
        status_code=429,
        detail="Too many requests. Please slow down and try again shortly.",
        headers={"Retry-After": str(seconds)},
    )


def _enforce(scope: str, key: str, limit: int, window: float) -> None:
    settings = get_settings()
    if not settings.RATE_LIMIT_ENABLED or limit <= 0:
        return
    retry = _limiter.hit(f"{scope}:{key}", limit, window)
    if retry:
        raise _reject(retry, scope)


MINUTE = 60.0
HOUR = 3600.0


def limit_chat(request: Request, raw_session: str) -> None:
    s = get_settings()
    _enforce("chat-ip", client_ip(request), s.RATE_LIMIT_CHAT_PER_MINUTE, MINUTE)
    _enforce("chat-session", raw_session, s.RATE_LIMIT_CHAT_SESSION_PER_MINUTE, MINUTE)


def limit_session_create(request: Request) -> None:
    _enforce("session-create", client_ip(request), get_settings().RATE_LIMIT_SESSION_CREATE_PER_HOUR, HOUR)


def limit_search(request: Request) -> None:
    _enforce("search", client_ip(request), get_settings().RATE_LIMIT_SEARCH_PER_MINUTE, MINUTE)


def limit_voice_token(request: Request, raw_session: str) -> None:
    s = get_settings()
    _enforce("voice-token-ip", client_ip(request), s.RATE_LIMIT_VOICE_TOKEN_PER_MINUTE, MINUTE)
    _enforce("voice-token-session", raw_session, s.RATE_LIMIT_VOICE_TOKEN_PER_MINUTE, MINUTE)


def allow_voice_connect(websocket: WebSocket) -> bool:
    """WebSocket handshakes cannot raise HTTP errors, so this returns a verdict."""
    s = get_settings()
    if not s.RATE_LIMIT_ENABLED:
        return True
    return _limiter.hit(f"voice-connect:{client_ip(websocket)}", s.RATE_LIMIT_VOICE_CONNECT_PER_MINUTE, MINUTE) == 0


def limit_leads(request: Request) -> None:
    _enforce("leads", client_ip(request), get_settings().RATE_LIMIT_LEADS_PER_HOUR, HOUR)


def limit_webhook(request: Request) -> None:
    _enforce("webhook", client_ip(request), get_settings().RATE_LIMIT_WEBHOOK_PER_MINUTE, MINUTE)


def allow_whatsapp_sender(sender: str) -> bool:
    """Per-sender flood control; over-limit messages are dropped, not errored."""
    s = get_settings()
    if not s.RATE_LIMIT_ENABLED:
        return True
    return _limiter.hit(f"wa-sender:{sender}", s.RATE_LIMIT_WHATSAPP_SENDER_PER_MINUTE, MINUTE) == 0


# --- operator endpoints -----------------------------------------------------

_ADMIN_FAILURE_WINDOW = 15 * MINUTE


def limit_admin(request: Request) -> None:
    """Throttle operator calls and lock out clients that keep sending a wrong key."""
    s = get_settings()
    if not s.RATE_LIMIT_ENABLED:
        return
    ip = client_ip(request)
    failures = _limiter.count(f"admin-fail:{ip}", _ADMIN_FAILURE_WINDOW)
    if s.RATE_LIMIT_ADMIN_FAILURES_PER_15MIN > 0 and failures >= s.RATE_LIMIT_ADMIN_FAILURES_PER_15MIN:
        raise _reject(_ADMIN_FAILURE_WINDOW / 3, "admin-lockout")
    _enforce("admin", ip, s.RATE_LIMIT_ADMIN_PER_MINUTE, MINUTE)


def record_admin_failure(request: Request) -> None:
    if get_settings().RATE_LIMIT_ENABLED:
        _limiter.hit(f"admin-fail:{client_ip(request)}", 10**9, _ADMIN_FAILURE_WINDOW)
