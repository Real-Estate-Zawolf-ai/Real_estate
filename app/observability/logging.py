"""Application logging: structured, request-correlated and secret-free."""

from __future__ import annotations

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

# Set per request by the middleware in app.main; shows up on every log line.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Last line of defence: scrub anything that looks like a credential before it is
# written, in case a library logs a URL or header it should not.
_REDACTIONS = (
    (re.compile(r"(?i)(token|api[_-]?key|secret|password|authorization|x-admin-key)(=|:\s*|\"\s*:\s*\")([^\s&\"',;]+)"), r"\1\2[redacted]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"), "Bearer [redacted]"),
    (re.compile(r"\bgsk_[A-Za-z0-9]{10,}"), "[redacted-groq-key]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"), "[redacted-google-key]"),
    (re.compile(r"\bEAA[A-Za-z0-9]{20,}"), "[redacted-meta-token]"),
    (re.compile(r"/bot\d+:[A-Za-z0-9_\-]{20,}"), "/bot[redacted]"),
    (re.compile(r"postgres(?:ql)?(?:\+\w+)?://[^\s:@/]+:[^\s@]+@"), "postgresql://[redacted]@"),
)


def redact(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        scrubbed = redact(message)
        if scrubbed != message:
            record.msg, record.args = scrubbed, None
        if record.exc_info and record.exc_info[1] is not None:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = record.exc_text or redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        request_id = request_id_var.get()
        base = super().format(record)
        return f"{base} [req={request_id}]" if request_id else base


# Libraries that log full URLs / payloads at INFO.
_QUIET_LOGGERS = ("httpx", "httpcore", "urllib3", "google_genai", "google.genai", "websockets", "qdrant_client", "groq", "uvicorn.access")


def configure_logging(level: str | int | None = None, fmt: str | None = None) -> None:
    """Configure process-wide logging (idempotent)."""
    from app.helpers.config import get_settings

    settings = get_settings()
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RedactingFilter())
    use_json = (fmt or settings.LOG_FORMAT) == "json"
    handler.setFormatter(
        JsonFormatter() if use_json else _TextFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level or settings.LOG_LEVEL)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # uvicorn logs rejected/accepted WebSocket upgrades (with the query string, i.e.
    # the single-use voice token) through its own handler, which bypasses the filter
    # above. Filter at the logger so those lines are scrubbed too.
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        if not any(isinstance(f, RedactingFilter) for f in uvicorn_logger.filters):
            uvicorn_logger.addFilter(RedactingFilter())
