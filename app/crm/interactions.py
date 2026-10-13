"""Conversation and contact history for CRM customers and leads (PostgreSQL)."""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.crm.client import CRMClient
from app.database import models as db


class InteractionChannel(str, Enum):
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
    created_at: datetime | None = None


class InteractionService:
    def __init__(self, client: CRMClient | None = None) -> None:
        self.client = client or CRMClient()

    def record(self, request: CreateInteraction) -> Interaction:
        """Store a customer interaction after checking referenced records."""
        with self.client.session() as session:
            customer = session.scalar(select(db.Customer).where(db.Customer.public_id == request.customer_id))
            if customer is None:
                raise ValueError("An interaction must reference an existing customer.")
            lead_pk = None
            if request.lead_id:
                lead_pk = session.scalar(select(db.Lead.id).where(db.Lead.public_id == request.lead_id))
                if lead_pk is None:
                    raise ValueError("The referenced lead was not found.")
            row = db.Interaction(
                customer_id=customer.id,
                lead_id=lead_pk,
                channel=request.channel.value,
                direction=request.direction,
                content=request.content,
            )
            session.add(row)
            session.flush()
            session.refresh(row)
            return Interaction(
                id=row.public_id,
                customer_id=request.customer_id,
                channel=request.channel,
                content=row.content,
                lead_id=request.lead_id,
                direction=row.direction,
                created_at=row.created_at,
            )

    def list_for_customer(self, customer_id: UUID) -> list[Interaction]:
        """Return chronological interaction history for one customer."""
        with self.client.session() as session:
            customer = session.scalar(select(db.Customer).where(db.Customer.public_id == customer_id))
            if customer is None:
                return []
            rows = session.execute(
                select(db.Interaction, db.Lead.public_id)
                .outerjoin(db.Lead, db.Lead.id == db.Interaction.lead_id)
                .where(db.Interaction.customer_id == customer.id)
                .order_by(db.Interaction.created_at, db.Interaction.id)
            ).all()
            history = []
            for row, lead_public_id in rows:
                base = row.channel.split(":", 1)[0]
                history.append(
                    Interaction(
                        id=row.public_id,
                        customer_id=customer_id,
                        channel=InteractionChannel(base) if base in {c.value for c in InteractionChannel} else InteractionChannel.WEBSITE,
                        content=row.content,
                        lead_id=lead_public_id,
                        direction=row.direction,
                        created_at=row.created_at,
                    )
                )
            return history
