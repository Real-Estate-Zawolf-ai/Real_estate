"""Webhook endpoints for external communication providers.

Two WhatsApp transports are supported and both run the shared LangGraph
agent workflow with one conversation thread per phone number:

* ``/whatsapp/webhook`` — Meta WhatsApp Cloud API (official, signed webhooks).
* ``/internal/whatsapp/*`` — the optional open-wa gateway in
  ``app/channels/whatsapp.ts`` (bearer-secret authenticated).

Inbound voice notes are transcribed (Groq Whisper) and answered as text;
real-time WhatsApp calls are not supported by either transport.
"""

import base64
import binascii
import hmac
import logging
import time
from collections import OrderedDict
from threading import Lock
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...channels.whatsapp import (
    extract_incoming_messages,
    download_media,
    is_valid_signature,
    send_text_message,
    transcribe_audio,
    verify_webhook_challenge,
)
from app.agents.messages import say
from app.agents.service import conversation_service
from app.helpers.config import get_settings
from app.observability.metrics import record_request


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])
gateway_router = APIRouter(prefix="/internal/whatsapp", tags=["whatsapp"], include_in_schema=False)

MAX_AUDIO_BYTES = 16 * 1024 * 1024


class _RecentIds:
    """Bounded memory of processed message ids (providers retry deliveries)."""

    def __init__(self, size: int = 5_000) -> None:
        self._ids: OrderedDict[str, None] = OrderedDict()
        self._size = size
        self._lock = Lock()

    def first_time(self, message_id: str) -> bool:
        with self._lock:
            if message_id in self._ids:
                return False
            self._ids[message_id] = None
            if len(self._ids) > self._size:
                self._ids.popitem(last=False)
            return True


_seen_messages = _RecentIds()


def whatsapp_thread(sender: str) -> str:
    return "whatsapp:" + "".join(ch for ch in sender if ch.isdigit() or ch == "+")


def run_whatsapp_turn(sender: str, text: str, modality: str) -> str:
    """Run one WhatsApp message through the agent graph and return the reply."""
    thread_id = whatsapp_thread(sender)
    started = time.monotonic()
    try:
        result = conversation_service.run_turn(
            thread_id,
            text,
            channel="whatsapp",
            customer_id=thread_id,
            modality=modality,
        )
    except Exception:
        record_request("whatsapp_turn", "error", time.monotonic() - started)
        raise
    record_request("whatsapp_turn", "degraded" if result.error else "ok", time.monotonic() - started)
    return result.response


async def process_whatsapp_message(message: dict[str, str]) -> None:
    """Run an accepted Cloud API message through the agents and send the answer."""
    try:
        if message["type"] == "audio":
            audio = await download_media(message["media_id"])
            customer_text = await run_in_threadpool(transcribe_audio, audio)
            modality = "voice_note"
        else:
            customer_text = message["text"]
            modality = "text"

        reply = await run_in_threadpool(run_whatsapp_turn, message["from"], customer_text, modality)
        await send_text_message(message["from"], reply)
    except Exception:
        logger.exception("WhatsApp workflow failed | message_id=%s", message["id"])
        # Tell the customer instead of leaving them without any answer.
        try:
            notice = f"{say('ar', 'search_unavailable')}\n{say('en', 'search_unavailable')}"
            await send_text_message(message["from"], notice)
        except Exception:
            logger.exception("WhatsApp failure notice could not be sent | message_id=%s", message["id"])


@router.get("/webhook", response_class=Response)
async def verify_whatsapp_webhook(
    mode: str | None = Query(default=None, alias="hub.mode"),
    verify_token: str | None = Query(default=None, alias="hub.verify_token"),
    challenge: str | None = Query(default=None, alias="hub.challenge"),
) -> Response:
    """Complete Meta's webhook-verification handshake."""
    verified_challenge = verify_webhook_challenge(mode, verify_token, challenge)
    if verified_challenge is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid verify token")
    return Response(content=verified_challenge, media_type="text/plain")


@router.post("/webhook", status_code=status.HTTP_200_OK)
async def receive_whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Validate and acknowledge WhatsApp Cloud API events quickly."""
    body = await request.body()
    if not is_valid_signature(body, request.headers.get("X-Hub-Signature-256")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    try:
        payload: dict[str, Any] = await request.json()
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from error

    for message in extract_incoming_messages(payload):
        if not _seen_messages.first_time(message["id"]):
            logger.info("Skipping duplicate WhatsApp delivery | message_id=%s", message["id"])
            continue
        logger.info(
            "Received WhatsApp %s message (id=%s)",
            message["type"],
            message["id"],
        )
        background_tasks.add_task(process_whatsapp_message, message)

    return {"status": "accepted"}


# ---------------------------------------------------------------------------
# open-wa gateway (app/channels/whatsapp.ts)
# ---------------------------------------------------------------------------


class GatewayText(BaseModel):
    message_id: str = Field(min_length=1, max_length=200)
    sender: str = Field(min_length=3, max_length=64)
    type: str = "text"
    text: str = Field(min_length=1, max_length=4_000)


class GatewayAudio(BaseModel):
    message_id: str = Field(min_length=1, max_length=200)
    sender: str = Field(min_length=3, max_length=64)
    mime_type: str = Field(default="audio/ogg", max_length=100)
    audio_base64: str = Field(min_length=1, max_length=MAX_AUDIO_BYTES * 4 // 3 + 4)


def _require_gateway(authorization: str | None) -> None:
    secret = get_settings().WHATSAPP_GATEWAY_SECRET
    if not secret:
        raise HTTPException(status_code=503, detail="WhatsApp gateway is not configured")
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, secret):
        raise HTTPException(status_code=401, detail="Invalid gateway credentials")


def _gateway_sender(sender: str) -> str:
    # open-wa ids look like "201001234567@c.us".
    return sender.split("@", 1)[0]


@gateway_router.post("/message")
async def gateway_message(payload: GatewayText, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_gateway(authorization)
    if not _seen_messages.first_time(payload.message_id):
        return {"reply": None, "duplicate": True}
    reply = await run_in_threadpool(run_whatsapp_turn, _gateway_sender(payload.sender), payload.text, "text")
    return {"reply": reply}


@gateway_router.post("/audio")
async def gateway_audio(payload: GatewayAudio, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_gateway(authorization)
    if not payload.mime_type.startswith("audio/"):
        raise HTTPException(status_code=415, detail="Unsupported media type")
    if not _seen_messages.first_time(payload.message_id):
        return {"reply": None, "duplicate": True}
    try:
        audio = base64.b64decode(payload.audio_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=422, detail="Invalid audio encoding") from error
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Voice note too large")
    try:
        text = await run_in_threadpool(transcribe_audio, audio)
    except Exception as error:
        logger.exception("Voice-note transcription failed | message_id=%s", payload.message_id)
        raise HTTPException(status_code=503, detail="Transcription unavailable") from error
    reply = await run_in_threadpool(run_whatsapp_turn, _gateway_sender(payload.sender), text, "voice_note")
    return {"reply": reply, "transcript": text}
