"""Website text chat, backed by the shared LangGraph agent workflow."""

import logging
import time

from fastapi import APIRouter, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from app.agents.service import TurnResult, conversation_service
from app.api.ratelimit import limit_chat, limit_session_create
from app.api.sessions import (
    inspect_session_id,
    issue_session_id,
    needs_refresh,
    website_thread,
)
from app.observability.metrics import record_request
from app.schemas.interaction import ChatRequest, ChatResponse, SessionRequest


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def resolve_session(session_id: str | None, request: Request | None = None) -> tuple[str, str]:
    """Return (signed id, raw id).

    A valid id continues its conversation (re-signed when it is a day old so
    active customers do not expire); an unknown, tampered or expired id starts a
    new one - creating sessions is rate-limited per client.
    """
    verified = inspect_session_id(session_id)
    if verified is not None:
        raw, age = verified
        return (issue_session_id(raw) if needs_refresh(age) else session_id), raw
    if request is not None:
        limit_session_create(request)
    session_id = issue_session_id()
    verified = inspect_session_id(session_id)
    assert verified is not None
    return session_id, verified[0]


def to_response(session_id: str, result: TurnResult) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        intent=result.intent,
        agent=result.agent,
        response=result.response,
        language=result.language,
        candidates=result.properties,
        cited_property_ids=result.cited_property_ids,
        grounding_property_ids=result.grounding_property_ids,
        requirements=result.requirements,
        qualification=result.qualification,
        selected_property_id=result.selected_property_id,
        appointment=result.appointment,
        handoff=result.handoff,
        scheduling_stage=result.scheduling_stage,
        degraded=bool(result.error),
    )


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    session_id, raw = resolve_session(request.session_id, http_request)
    limit_chat(http_request, raw)
    thread_id = website_thread(raw)
    started = time.monotonic()
    try:
        result = await run_in_threadpool(
            conversation_service.run_turn,
            thread_id,
            request.query,
            channel="website",
            customer_id=thread_id,
            modality="text",
            language=request.language,
            clear_requirements=list(request.clear_requirements),
        )
    except ValueError as error:
        record_request("chat", "invalid", time.monotonic() - started)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Chat turn failed | session=%s", raw[:8])
        record_request("chat", "error", time.monotonic() - started)
        raise HTTPException(status_code=503, detail="Assistant is temporarily unavailable") from error
    record_request("chat", "degraded" if result.error else "ok", time.monotonic() - started)
    return to_response(session_id, result)


@router.post("/reset", status_code=status.HTTP_200_OK)
async def reset_search(request: SessionRequest, http_request: Request) -> dict[str, str]:
    """Clear search filters and results; keep the customer's profile."""
    session_id, raw = resolve_session(request.session_id, http_request)
    limit_chat(http_request, raw)
    await run_in_threadpool(conversation_service.reset_search, website_thread(raw))
    return {"session_id": session_id, "status": "reset"}
