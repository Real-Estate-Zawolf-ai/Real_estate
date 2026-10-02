"""Meta WhatsApp Cloud API webhook channel.

The webhook deliberately acknowledges messages quickly. A background worker
should download and transcode inbound voice notes before passing PCM audio to
``app.voice.gemini_live``; doing that work in the webhook request risks Meta
retrying the event.
"""

import hashlib
import hmac
from typing import Any

import httpx

from ..helpers.config import get_settings


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
                if message_type == "text":
                    text = message.get("text", {}).get("body")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    extracted["text"] = text.strip()
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


async def send_text_message(recipient: str, text: str) -> None:
    """Send a WhatsApp Cloud API text response."""
    settings = get_settings()
    if not settings.WHATSAPP_ACCESS_TOKEN or not settings.WHATSAPP_PHONE_NUMBER_ID:
        raise RuntimeError("WhatsApp sending credentials are not configured")
    url = (
        f"https://graph.facebook.com/{settings.WHATSAPP_API_VERSION}/"
        f"{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    )
    headers = {"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}"}
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient,
        "type": "text",
        "text": {"body": text[:4096]},
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()


async def download_media(media_id: str) -> bytes:
    """Resolve and download WhatsApp media using the configured access token."""
    settings = get_settings()
    if not settings.WHATSAPP_ACCESS_TOKEN:
        raise RuntimeError("WHATSAPP_ACCESS_TOKEN is not configured")
    headers = {"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}"}
    metadata_url = f"https://graph.facebook.com/{settings.WHATSAPP_API_VERSION}/{media_id}"
    async with httpx.AsyncClient(timeout=60) as client:
        metadata_response = await client.get(metadata_url, headers=headers)
        metadata_response.raise_for_status()
        media_url = metadata_response.json().get("url")
        if not media_url:
            raise RuntimeError("WhatsApp media response did not include a URL")
        media_response = await client.get(media_url, headers=headers)
        media_response.raise_for_status()
        return media_response.content


def transcribe_audio(audio: bytes) -> str:
    """Transcribe a downloaded WhatsApp voice note through Groq Whisper."""
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")
    from groq import Groq

    transcription = Groq(api_key=settings.GROQ_API_KEY).audio.transcriptions.create(
        file=("voice-note.ogg", audio),
        model=settings.WHATSAPP_TRANSCRIPTION_MODEL,
        response_format="text",
    )
    text = transcription if isinstance(transcription, str) else getattr(transcription, "text", "")
    if not text or not text.strip():
        raise RuntimeError("Voice-note transcription was empty")
    return text.strip()
