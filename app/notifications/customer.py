"""Proactive messages to customers (booking confirmed / cancelled by staff).

Only WhatsApp customers can be reached this way: a website visitor has no push
channel, so they see the new status when they return to the chat or when the
advisor phones them. Meta only allows free-form messages inside the 24-hour
window after the customer last wrote; outside it the send is rejected and we
log the failure rather than pretend it worked.
"""

from __future__ import annotations

import logging

import httpx

from app.helpers.config import get_settings
from app.helpers.phone import mask_phone


logger = logging.getLogger(__name__)


def send_to_customer(external_id: str, text: str) -> bool:
    """Deliver ``text`` to the customer's channel. Returns True if sent."""
    channel, _, address = external_id.partition(":")
    settings = get_settings()
    if channel != "whatsapp" or not address:
        logger.info("Customer %s has no push channel; notice not sent", channel)
        return False
    if not (settings.WHATSAPP_ACCESS_TOKEN and settings.WHATSAPP_PHONE_NUMBER_ID):
        logger.warning("WhatsApp sending is not configured; customer notice not sent")
        return False
    try:
        response = httpx.post(
            f"https://graph.facebook.com/{settings.WHATSAPP_API_VERSION}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages",
            headers={"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}"},
            json={
                "messaging_product": "whatsapp",
                "to": "".join(ch for ch in address if ch.isdigit()),
                "type": "text",
                "text": {"body": text[:4096]},
            },
            timeout=httpx.Timeout(15.0),
        )
        response.raise_for_status()
        return True
    except Exception as error:
        status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else type(error).__name__
        logger.warning("WhatsApp customer notice to %s failed: %s", mask_phone(address), status)
        return False
