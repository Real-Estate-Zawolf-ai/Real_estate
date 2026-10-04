"""Conversation-channel schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    customer_id: str | None = None
    channel: Literal["website", "whatsapp"] = "website"
    requirements: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    intent: str
    response: str
    candidates: list[dict[str, Any]] = Field(default_factory=list)
