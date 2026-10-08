"""Single entry point that runs one customer turn through the agent graph.

Website text, website voice (via the Gemini Live tool bridge) and WhatsApp
all call ``conversation_service.run_turn`` with a per-customer ``thread_id``,
so every channel shares one LangGraph workflow and one checkpointed state.

Storage note: the default checkpointer is LangGraph's in-process
``MemorySaver``. State therefore survives across turns and channels but not
a process restart, and is not shared between multiple API workers. Pass a
durable checkpointer (e.g. ``langgraph-checkpoint-postgres``) for production.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agents.graph import build_graph
from app.agents.state import describe_requirements, property_label
from app.agents.tools.crm import crm_service


logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 2_000


@dataclass
class TurnResult:
    thread_id: str
    intent: str
    agent: str
    response: str
    language: str
    properties: list[dict[str, Any]] = field(default_factory=list)
    cited_property_ids: list[str] = field(default_factory=list)
    grounding_property_ids: list[str] = field(default_factory=list)
    requirements: dict[str, Any] = field(default_factory=dict)
    qualification: dict[str, Any] = field(default_factory=dict)
    selected_property_id: str | None = None
    appointment: dict[str, Any] | None = None
    handoff: dict[str, Any] | None = None
    scheduling_stage: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ConversationService:

    def __init__(self, checkpointer: Any = None) -> None:
        self._graph = build_graph(checkpointer or MemorySaver())
        self._locks: dict[str, Lock] = {}
        self._locks_guard = Lock()

    def _lock_for(self, thread_id: str) -> Lock:
        with self._locks_guard:
            return self._locks.setdefault(thread_id, Lock())

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def run_turn(
        self,
        thread_id: str,
        text: str,
        *,
        channel: str,
        customer_id: str,
        modality: str = "text",
        language: str | None = None,
        clear_requirements: list[str] | None = None,
    ) -> TurnResult:
        message = (text or "").strip()
        if not message:
            raise ValueError("message cannot be blank")
        message = message[:MAX_MESSAGE_CHARS]

        turn_input: dict[str, Any] = {
            "messages": [HumanMessage(content=message)],
            "query": message,
            "channel": channel,
            "modality": modality,
            "customer_id": customer_id,
            "clear_requirements": list(clear_requirements or []),
            # Per-turn outputs start empty so stale results are never re-sent.
            "response": "",
            "turn_properties": [],
            "cited_property_ids": [],
            "grounding_property_ids": [],
            "error": None,
        }
        if language in {"ar", "en"}:
            turn_input["language"] = language

        # One turn at a time per conversation: a voice tool call and a typed
        # message for the same customer must not interleave state updates.
        with self._lock_for(thread_id):
            state = self._graph.invoke(turn_input, self._config(thread_id))

        result = self._result(thread_id, state)
        try:
            crm_service.record_interaction(
                customer_id=customer_id,
                channel=f"{channel}:{modality}",
                content=message,
                response=result.response,
            )
        except Exception:
            logger.exception("Failed to record interaction | thread=%s", thread_id)
        return result

    @staticmethod
    def _result(thread_id: str, state: dict[str, Any]) -> TurnResult:
        selected = state.get("selected_property") or {}
        return TurnResult(
            thread_id=thread_id,
            intent=state.get("intent", "conversational"),
            agent=state.get("active_agent", "conversational"),
            response=state.get("response", ""),
            language=state.get("language", "en"),
            properties=list(state.get("turn_properties") or []),
            cited_property_ids=list(state.get("cited_property_ids") or []),
            grounding_property_ids=list(state.get("grounding_property_ids") or []),
            requirements=dict(state.get("requirements") or {}),
            qualification=dict(state.get("qualification") or {}),
            selected_property_id=str(selected["property_id"]) if selected.get("property_id") is not None else None,
            appointment=state.get("appointment"),
            handoff=state.get("handoff"),
            scheduling_stage=(state.get("scheduling") or {}).get("stage"),
            error=state.get("error"),
        )

    def get_state(self, thread_id: str) -> dict[str, Any]:
        snapshot = self._graph.get_state(self._config(thread_id))
        return dict(snapshot.values or {})

    def reset_search(self, thread_id: str) -> None:
        """Start a new search but keep the customer's profile and contact."""
        if not self.get_state(thread_id):
            return
        with self._lock_for(thread_id):
            self._graph.update_state(
                self._config(thread_id),
                {"requirements": {}, "candidates": [], "selected_property": None, "scheduling": {}},
                as_node="record_reply",
            )

    def context_summary(self, thread_id: str, *, max_messages: int = 6) -> str:
        """Plain-text recap of a conversation, used to brief the voice model."""
        state = self.get_state(thread_id)
        if not state:
            return ""
        lines: list[str] = []
        known = describe_requirements(state.get("requirements") or {}, state.get("qualification") or {}, "en")
        if known:
            lines.append(f"Known customer requirements: {known}.")
        if state.get("selected_property"):
            lines.append(f"Property currently discussed: {property_label(state['selected_property'])}.")
        if state.get("candidates"):
            listed = "; ".join(f"{index}. {property_label(item)}" for index, item in enumerate(state["candidates"][:5], start=1))
            lines.append(f"Last properties shown: {listed}.")
        recent = []
        for message in (state.get("messages") or [])[-max_messages:]:
            role = {"human": "Customer", "ai": "Assistant"}.get(getattr(message, "type", ""), "")
            if role:
                recent.append(f"{role}: {str(message.content)[:300]}")
        if recent:
            lines.append("Recent conversation:\n" + "\n".join(recent))
        return "\n".join(lines)


conversation_service = ConversationService()
