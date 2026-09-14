"""Lead records and sales-qualification operations."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.crm.client import CRMClient


class LeadStatus(StrEnum):
    NEW = "new"
    QUALIFYING = "qualifying"
    QUALIFIED = "qualified"
    VIEWING_BOOKED = "viewing_booked"
    WON = "won"
    LOST = "lost"


class CreateLead(BaseModel):
    customer_id: UUID
    source: str = Field(default="website", min_length=1, max_length=100)
    property_id: UUID | None = None
    budget_min: int | None = Field(default=None, ge=0)
    budget_max: int | None = Field(default=None, ge=0)
    notes: str | None = Field(default=None, max_length=2_000)


class Lead(CreateLead):
    id: UUID = Field(default_factory=uuid4)
    status: LeadStatus = LeadStatus.NEW
    score: int = Field(default=0, ge=0, le=100)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LeadNotFoundError(LookupError):
    """Raised when a lead cannot be found."""


class LeadService:
    def __init__(self, client: CRMClient) -> None:
        self.client = client

    def create(self, request: CreateLead) -> Lead:
        """Create a lead attached to a known customer."""
        if self.client.get("customers", request.customer_id) is None:
            raise ValueError("A lead must reference an existing customer.")
        lead = Lead(**request.model_dump())
        return self.client.save("leads", lead.id, lead)

    def update_status(self, lead_id: UUID, status: LeadStatus) -> Lead:
        """Move a lead through the sales pipeline."""
        lead = self.client.get("leads", lead_id)
        if lead is None:
            raise LeadNotFoundError(f"Lead {lead_id} was not found.")
        lead.status = status
        lead.updated_at = datetime.now(UTC)
        return self.client.save("leads", lead.id, lead)

    def update_score(self, lead_id: UUID, score: int) -> Lead:
        """Set a normalized lead score between 0 and 100."""
        if not 0 <= score <= 100:
            raise ValueError("Lead score must be between 0 and 100.")
        lead = self.client.get("leads", lead_id)
        if lead is None:
            raise LeadNotFoundError(f"Lead {lead_id} was not found.")
        lead.score = score
        lead.updated_at = datetime.now(UTC)
        return self.client.save("leads", lead.id, lead)
