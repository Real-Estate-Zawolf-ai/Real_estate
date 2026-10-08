"""Conversation-channel schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RequirementKey = Literal[
    "property_type",
    "district",
    "transaction_type",
    "bedrooms",
    "min_bedrooms",
    "min_price",
    "max_price",
    "delivery",
]


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    # Signed id returned by a previous /chat or /voice/token response.
    # Omit it to start a new conversation.
    session_id: str | None = Field(default=None, max_length=100)
    channel: Literal["website"] = "website"
    language: Literal["ar", "en"] | None = None
    # Requirement keys to drop before this turn (removable filter chips).
    clear_requirements: list[RequirementKey] = Field(default_factory=list, max_length=8)


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    agent: str
    response: str
    language: str
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    cited_property_ids: list[str] = Field(default_factory=list)
    grounding_property_ids: list[str] = Field(default_factory=list)
    requirements: dict[str, Any] = Field(default_factory=dict)
    qualification: dict[str, Any] = Field(default_factory=dict)
    selected_property_id: str | None = None
    appointment: dict[str, Any] | None = None
    handoff: dict[str, Any] | None = None
    scheduling_stage: str | None = None
    # True when a backing service failed and the reply is a recovery message.
    degraded: bool = False


class SessionRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=100)
    language: Literal["ar", "en"] = "en"
