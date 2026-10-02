"""Lightweight conversational endpoint for local and integration use."""

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.agents.state import route_agent, router as classify_intent
from app.schemas.interaction import ChatRequest, ChatResponse
from app.workflows.rag import answer_query


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    state = {"query": request.query, "channel": request.channel}
    intent_state = classify_intent(state)
    intent = intent_state["intent"]
    destination = route_agent(intent_state)

    if destination == "property":
        try:
            result = await run_in_threadpool(answer_query, request.query, request.requirements)
        except Exception as error:
            raise HTTPException(status_code=503, detail="Property assistant is unavailable") from error
        return ChatResponse(intent=intent, response=result["answer"], candidates=result["properties"])

    if destination == "scheduling":
        response = "Please share your preferred viewing date in YYYY-MM-DD format."
    elif destination == "qualification":
        response = "Please share your budget, purpose, and preferred timeline."
    else:
        response = "How can I help with your property search today?"

    return ChatResponse(intent=intent, response=response)
