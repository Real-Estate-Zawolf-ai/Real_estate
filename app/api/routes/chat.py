"""Lightweight conversational endpoint for local and integration use."""

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.agents.graph import graph
from app.schemas.interaction import ChatRequest, ChatResponse


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    state = {
        "query": request.query,
        "requirements": request.requirements,
        "channel": request.channel,
        "messages": request.history,
    }
    try:
        result = await run_in_threadpool(graph.invoke, state)
    except Exception as error:
        raise HTTPException(status_code=503, detail="Chat assistant is unavailable") from error

    cited_property_ids = result.get("cited_property_ids")
    grounding_property_ids = result.get("grounding_property_ids")
    properties = result.get("candidates")

    return ChatResponse(
        answer=result.get("response", ""),
        response_type=(
            "property" if result.get("intent") == "property_search" else "conversational"
        ),
        cited_property_ids=cited_property_ids if cited_property_ids is not None else [],
        grounding_property_ids=(
            grounding_property_ids if grounding_property_ids is not None else []
        ),
        properties=properties if properties is not None else [],
    )
