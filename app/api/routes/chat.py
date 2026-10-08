"""Website text chat, backed by the shared LangGraph agent workflow."""

import logging
import time

from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.agents.service import TurnResult, conversation_service
from app.api.sessions import issue_session_id, verify_session_id, website_thread
from app.observability.metrics import record_request
from app.schemas.interaction import ChatRequest, ChatResponse, SessionRequest


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def resolve_session(session_id: str | None) -> tuple[str, str]:
    """Return (signed id, raw id); unknown or tampered ids start a new session."""
    raw = verify_session_id(session_id)
    if raw is None:
        session_id = issue_session_id()
        raw = verify_session_id(session_id)
    return session_id, raw


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
async def chat(request: ChatRequest) -> ChatResponse:
    session_id, raw = resolve_session(request.session_id)
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
async def reset_search(request: SessionRequest) -> dict[str, str]:
    """Clear search filters and results; keep the customer's profile."""
    session_id, raw = resolve_session(request.session_id)
    await run_in_threadpool(conversation_service.reset_search, website_thread(raw))
    return {"session_id": session_id, "status": "reset"}
