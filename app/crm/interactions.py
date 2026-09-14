"""Conversation and contact history for CRM customers and leads."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.crm.client import CRMClient


class InteractionChannel(StrEnum):
    WEBSITE = "website"
    WHATSAPP = "whatsapp"
    PHONE = "phone"
    EMAIL = "email"


class CreateInteraction(BaseModel):
    customer_id: UUID
    channel: InteractionChannel
    content: str = Field(min_length=1, max_length=10_000)
    lead_id: UUID | None = None
    direction: str = Field(default="inbound", pattern="^(inbound|outbound)$")


class Interaction(CreateInteraction):
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class InteractionService:
    def __init__(self, client: CRMClient) -> None:
        self.client = client

    def record(self, request: CreateInteraction) -> Interaction:
        """Store a customer interaction after checking referenced records."""
        if self.client.get("customers", request.customer_id) is None:
            raise ValueError("An interaction must reference an existing customer.")
        if request.lead_id and self.client.get("leads", request.lead_id) is None:
            raise ValueError("The referenced lead was not found.")
        interaction = Interaction(**request.model_dump())
        return self.client.save("interactions", interaction.id, interaction)

    def list_for_customer(self, customer_id: UUID) -> list[Interaction]:
        """Return chronological interaction history for one customer."""
        interactions = [
            interaction
            for interaction in self.client.list("interactions")
            if interaction.customer_id == customer_id
        ]
        return sorted(interactions, key=lambda interaction: interaction.created_at)
