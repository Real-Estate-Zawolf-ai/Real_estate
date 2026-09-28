"""Webhook endpoints for external communication providers."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from ...channels.whatsapp import (
    extract_incoming_messages,
    is_valid_signature,
    verify_webhook_challenge,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])


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
async def receive_whatsapp_webhook(request: Request) -> dict[str, str]:
    """Validate and acknowledge WhatsApp Cloud API events quickly."""
    body = await request.body()
    if not is_valid_signature(body, request.headers.get("X-Hub-Signature-256")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    try:
        payload: dict[str, Any] = await request.json()
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from error

    for message in extract_incoming_messages(payload):
        # Replace this log with a durable job queue before production. The
        # request must return quickly so Meta does not redeliver the webhook.
        logger.info(
            "Received WhatsApp %s message from %s (id=%s)",
            message["type"],
            message["from"],
            message["id"],
        )

    return {"status": "accepted"}
