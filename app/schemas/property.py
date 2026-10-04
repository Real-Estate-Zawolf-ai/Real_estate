"""Public property API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PropertyRecord(BaseModel):
    property_id: str = Field(min_length=1, max_length=100)
    text: str = Field(default="", max_length=20_000)
    property_type: str | None = None
    location: str | None = None
    price: float | None = Field(default=None, ge=0)
    bedrooms: int | None = Field(default=None, ge=0)
    bathrooms: int | None = Field(default=None, ge=0)
    area: float | None = Field(default=None, ge=0)
    availability: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PropertySearchRequest(BaseModel):
    query: str = Field(default="", max_length=2_000)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=10, ge=1, le=100)


class PropertySearchResponse(BaseModel):
    results: list[dict[str, Any]]
    count: int = Field(ge=0)
