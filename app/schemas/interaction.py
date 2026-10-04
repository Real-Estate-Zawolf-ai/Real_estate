"""Conversation-channel schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    customer_id: str | None = None
    channel: Literal["website", "whatsapp"] = "website"
    requirements: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    response_type: Literal["property", "conversational"] = "conversational"
    cited_property_ids: list[str] = Field(default_factory=list)
    grounding_property_ids: list[str] = Field(default_factory=list)
    properties: list[dict[str, Any]] = Field(default_factory=list)
