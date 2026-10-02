"""Lead qualification API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LeadQualification(BaseModel):
    budget: float | str | None = None
    purpose: str | None = None
    timeline: str | None = None
    preferred_location: str | None = None
    property_type: str | None = None
    financing_ready: bool = False
    requested_viewing: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class LeadScoreResponse(BaseModel):
    score: int = Field(ge=0, le=100)
    band: str
    qualified: bool
    missing_fields: list[str]


class LeadCaptureRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    phone: str = Field(min_length=3, max_length=30)
    message: str = Field(default="", max_length=2_000)
    property: str | None = Field(default=None, max_length=100)
