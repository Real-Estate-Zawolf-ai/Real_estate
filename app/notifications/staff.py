"""Staff notifications with a durable outbox.

Every alert is stored in ``staff_notifications`` before any delivery is tried, so
it is visible through the admin API and can be retried even when every provider
is down. Delivery fans out to each configured channel (webhook, Telegram,
WhatsApp, e-mail); an alert counts as sent when at least one channel accepts it.

Nothing here may raise into a customer request: a failed alert must never turn a
successful booking into an error.
"""

from __future__ import annotations

import logging
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any, Callable

import httpx
from sqlalchemy import select

from app.database import models as db
from app.database.session import session_scope
from app.helpers.config import get_settings


logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 5
MAX_MESSAGE_CHARS = 3_500
_TIMEOUT = httpx.Timeout(10.0)


def _describe_error(error: Exception) -> str:
    """Error text that is safe to store and log.

    httpx puts the full request URL in its messages, and Telegram's URL carries
    the bot token, so only the exception class and HTTP status are kept.
    """
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return type(error).__name__


# ---------------------------------------------------------------------------
# Channels: each takes the message text and raises on failure
# ---------------------------------------------------------------------------


def _send_webhook(message: str) -> None:
    url = get_settings().STAFF_WEBHOOK_URL
    # "text" is Slack/Mattermost's field, "content" is Discord's.
    response = httpx.post(url, json={"text": message, "content": message}, timeout=_TIMEOUT)
    response.raise_for_status()


def _send_telegram(message: str) -> None:
    settings = get_settings()
    response = httpx.post(
        f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage",
        json={"chat_id": settings.TELEGRAM_CHAT_ID, "text": message, "disable_web_page_preview": True},
        timeout=_TIMEOUT,
    )
    response.raise_for_status()


def _send_whatsapp(message: str) -> None:
    settings = get_settings()
    response = httpx.post(
        f"https://graph.facebook.com/{settings.WHATSAPP_API_VERSION}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages",
        headers={"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}"},
        json={
            "messaging_product": "whatsapp",
            "to": "".join(ch for ch in settings.STAFF_WHATSAPP_NUMBER if ch.isdigit()),
            "type": "text",
            "text": {"body": message[:4096]},
        },
        timeout=_TIMEOUT,
    )
    response.raise_for_status()


def _send_email(message: str) -> None:
    settings = get_settings()
    email = EmailMessage()
    email["From"] = settings.SMTP_FROM
    email["To"] = settings.STAFF_EMAIL_TO
    email["Subject"] = message.splitlines()[0][:120] if message else "Aqar alert"
    email.set_content(message)
    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as smtp:
        smtp.starttls()
        if settings.SMTP_USERNAME and settings.SMTP_PASSWORD:
            smtp.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
        smtp.send_message(email)


_CHANNELS: dict[str, Callable[[str], None]] = {
    "webhook": _send_webhook,
    "telegram": _send_telegram,
    "whatsapp": _send_whatsapp,
    "email": _send_email,
}


def _deliver(message: str) -> tuple[bool, str | None, bool]:
    """Try every configured channel. Returns (any_sent, error_summary, had_channels)."""
    channels = get_settings().staff_channels
    if not channels:
        return False, None, False
    errors: list[str] = []
    sent = False
    for name in channels:
        try:
            _CHANNELS[name](message)
            sent = True
        except Exception as error:
            errors.append(f"{name}: {_describe_error(error)}")
            logger.warning("Staff notification via %s failed: %s", name, _describe_error(error))
    return sent, "; ".join(errors) or None, True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _attempt(notification_id: int) -> str:
    """Deliver one stored notification and record the outcome."""
    with session_scope() as session:
        row = session.get(db.StaffNotification, notification_id)
        if row is None or row.status == "sent":
            return "missing" if row is None else "sent"
        sent, error, had_channels = _deliver(row.message)
        row.attempts += 1
        if sent:
            row.status, row.sent_at, row.last_error = "sent", datetime.now(timezone.utc), error
        elif not had_channels:
            row.status = "no_channel"
        else:
            row.status = "failed" if row.attempts < MAX_ATTEMPTS else "abandoned"
            row.last_error = error
        return row.status


def notify_staff(
    kind: str,
    message: str,
    *,
    ref_type: str | None = None,
    ref_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> int | None:
    """Store an alert and try to deliver it. Never raises; returns the outbox id."""
    try:
        with session_scope() as session:
            row = db.StaffNotification(
                kind=kind,
                ref_type=ref_type,
                ref_id=ref_id,
                message=message[:MAX_MESSAGE_CHARS],
                payload=payload,
                status="pending",
                attempts=0,
            )
            session.add(row)
            session.flush()
            notification_id = row.id
    except Exception:
        logger.exception("Could not store staff notification | kind=%s", kind)
        return None
    try:
        _attempt(notification_id)
    except Exception:
        logger.exception("Staff notification delivery crashed | id=%s", notification_id)
    return notification_id


def retry_pending(limit: int = 20) -> int:
    """Retry undelivered alerts (failed, or stored while no channel existed)."""
    if not get_settings().staff_channels:
        return 0
    try:
        with session_scope() as session:
            ids = list(
                session.scalars(
                    select(db.StaffNotification.id)
                    .where(db.StaffNotification.status.in_(("pending", "failed", "no_channel")))
                    .order_by(db.StaffNotification.id)
                    .limit(limit)
                )
            )
    except Exception:
        logger.exception("Could not list pending staff notifications")
        return 0
    delivered = 0
    for notification_id in ids:
        try:
            if _attempt(notification_id) == "sent":
                delivered += 1
        except Exception:
            logger.exception("Staff notification retry crashed | id=%s", notification_id)
    return delivered
