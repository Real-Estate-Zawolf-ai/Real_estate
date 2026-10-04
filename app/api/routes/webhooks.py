"""Webhook endpoints for external communication providers."""

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, Response, status
from starlette.concurrency import run_in_threadpool

from ...channels.whatsapp import (
    extract_incoming_messages,
    download_media,
    is_valid_signature,
    send_text_message,
    transcribe_audio,
    verify_webhook_challenge,
)
from app.agents.tools.crm import crm_service
from app.workflows.rag import answer_query


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])


async def process_whatsapp_message(message: dict[str, str]) -> None:
    """Run an accepted text message through RAG and send the answer."""
    try:
        if message["type"] == "audio":
            audio = await download_media(message["media_id"])
            customer_text = await run_in_threadpool(transcribe_audio, audio)
        else:
            customer_text = message["text"]

        result = await run_in_threadpool(answer_query, customer_text, {})
        crm_service.record_interaction(
            customer_id=message["from"],
            channel="whatsapp",
            content=customer_text,
            response=result["answer"],
        )
        await send_text_message(message["from"], result["answer"])
    except Exception:
        logger.exception("WhatsApp workflow failed | message_id=%s", message["id"])


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
        logger.info(
            "Received WhatsApp %s message from %s (id=%s)",
            message["type"],
            message["from"],
            message["id"],
        )
        background_tasks.add_task(process_whatsapp_message, message)

    return {"status": "accepted"}
