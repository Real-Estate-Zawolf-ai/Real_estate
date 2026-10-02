"""Shared customer request and response schemas."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class CustomerCreate(BaseModel):
    full_name: str = Field(min_length=1, max_length=200)
    phone_number: str | None = Field(default=None, max_length=30)
    email: str | None = Field(default=None, max_length=320)
    preferred_language: str = Field(default="en", min_length=2, max_length=10)


class CustomerRead(CustomerCreate):
    id: UUID
