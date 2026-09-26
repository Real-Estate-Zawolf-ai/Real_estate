"""Meta WhatsApp Cloud API webhook channel.

The webhook deliberately acknowledges messages quickly. A background worker
should download and transcode inbound voice notes before passing PCM audio to
``app.voice.gemini_live``; doing that work in the webhook request risks Meta
retrying the event.
"""

import hashlib
import hmac
from typing import Any

from helpers.config import get_settings


def verify_webhook_challenge(
    mode: str | None, verify_token: str | None, challenge: str | None
) -> str | None:
    """Return Meta's verification challenge when the request is valid."""
    settings = get_settings()
    if (
        mode == "subscribe"
        and challenge is not None
        and settings.WHATSAPP_VERIFY_TOKEN
        and hmac.compare_digest(verify_token or "", settings.WHATSAPP_VERIFY_TOKEN)
    ):
        return challenge
    return None


def is_valid_signature(body: bytes, signature: str | None) -> bool:
    """Check Meta's SHA-256 webhook signature without exposing its secret."""
    secret = get_settings().WHATSAPP_APP_SECRET
    if not secret:
        return False

    expected = "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    return bool(signature and hmac.compare_digest(signature, expected))


def extract_incoming_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Extract supported inbound WhatsApp messages from a webhook payload."""
    messages: list[dict[str, str]] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for message in value.get("messages", []):
                message_type = message.get("type")
                sender = message.get("from")
                message_id = message.get("id")
                if message_type not in {"audio", "text"} or not sender or not message_id:
                    continue

                extracted = {"id": message_id, "from": sender, "type": message_type}
                if message_type == "audio":
                    # The media id lives under the "audio" object, distinct
                    # from the message's own id above. A background worker
                    # needs it to download the voice note via the Graph API
                    # before handing PCM audio to app.voice.gemini_live.
                    media_id = message.get("audio", {}).get("id")
                    if not media_id:
                        continue
                    extracted["media_id"] = media_id

                messages.append(extracted)
    return messages
