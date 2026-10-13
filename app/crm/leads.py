"""Lead records and sales-qualification operations (PostgreSQL)."""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.crm.client import CRMClient
from app.database import models as db


class LeadStatus(str, Enum):
    NEW = "new"
    QUALIFYING = "qualifying"
    QUALIFIED = "qualified"
    VIEWING_BOOKED = "viewing_booked"
    WON = "won"
    LOST = "lost"


class CreateLead(BaseModel):
    customer_id: UUID
    source: str = Field(default="website", min_length=1, max_length=100)
    # Listing id from the property index (a string such as "501189949").
    property_id: str | None = Field(default=None, max_length=100)
    budget_min: int | None = Field(default=None, ge=0)
    budget_max: int | None = Field(default=None, ge=0)
    notes: str | None = Field(default=None, max_length=2_000)


class Lead(CreateLead):
    id: UUID = Field(default_factory=uuid4)
    status: str = LeadStatus.NEW.value
    score: int = Field(default=0, ge=0, le=100)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class LeadNotFoundError(LookupError):
    """Raised when a lead cannot be found."""


def _entity(row: db.Lead, customer_public_id: UUID) -> Lead:
    return Lead(
        id=row.public_id,
        customer_id=customer_public_id,
        source=row.source or "website",
        property_id=row.property_ref,
        budget_min=int(row.budget_min) if row.budget_min is not None else None,
        budget_max=int(row.budget_max) if row.budget_max is not None else None,
        notes=row.notes,
        status=row.status,
        score=row.score,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class LeadService:
    def __init__(self, client: CRMClient | None = None) -> None:
        self.client = client or CRMClient()

    def _row(self, session, lead_id: UUID) -> tuple[db.Lead, UUID]:
        row = session.execute(
            select(db.Lead, db.Customer.public_id)
            .join(db.Customer, db.Customer.id == db.Lead.customer_id)
            .where(db.Lead.public_id == lead_id)
        ).first()
        if row is None:
            raise LeadNotFoundError(f"Lead {lead_id} was not found.")
        return row[0], row[1]

    def create(self, request: CreateLead) -> Lead:
        """Create a lead attached to a known customer."""
        with self.client.session() as session:
            customer = session.scalar(select(db.Customer).where(db.Customer.public_id == request.customer_id))
            if customer is None:
                raise ValueError("A lead must reference an existing customer.")
            row = db.Lead(
                customer_id=customer.id,
                status=LeadStatus.NEW.value,
                source=request.source,
                property_ref=request.property_id,
                budget_min=request.budget_min,
                budget_max=request.budget_max,
                notes=request.notes,
                score=0,
            )
            session.add(row)
            session.flush()
            session.refresh(row)
            return _entity(row, customer.public_id)

    def update_status(self, lead_id: UUID, status: LeadStatus | str) -> Lead:
        """Move a lead through the sales pipeline."""
        with self.client.session() as session:
            row, customer_public_id = self._row(session, lead_id)
            row.status = status.value if isinstance(status, LeadStatus) else str(status)
            session.flush()
            session.refresh(row)
            return _entity(row, customer_public_id)

    def update_score(self, lead_id: UUID, score: int) -> Lead:
        """Set a normalized lead score between 0 and 100."""
        if not 0 <= score <= 100:
            raise ValueError("Lead score must be between 0 and 100.")
        with self.client.session() as session:
            row, customer_public_id = self._row(session, lead_id)
            row.score = score
            session.flush()
            session.refresh(row)
            return _entity(row, customer_public_id)
