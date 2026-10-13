"""Human-handoff queue: requests for a sales advisor and their lifecycle."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.crm.client import CRMClient
from app.database import models as db


HANDOFF_STATUSES = ("open", "claimed", "resolved")

# A customer who asks twice within this window gets the existing ticket back
# instead of a duplicate (voice retries, double submits, impatient taps).
DEDUP_WINDOW = timedelta(minutes=10)


class HandoffNotFoundError(LookupError):
    """Raised when a handoff does not exist."""


class InvalidHandoffTransition(ValueError):
    """Raised for an impossible status change (e.g. resolving a resolved ticket)."""


def serialize(handoff: db.Handoff, customer: db.Customer, lead: db.Lead | None) -> dict[str, Any]:
    return {
        "id": str(handoff.public_id),
        "customer_id": customer.external_id,
        "customer_ref": str(customer.public_id),
        "lead_id": str(lead.public_id) if lead else None,
        "status": handoff.status,
        "reason": handoff.reason,
        "priority": handoff.priority,
        "summary": handoff.summary or {},
        "claimed_by": handoff.claimed_by,
        "created_at": handoff.created_at.isoformat() if handoff.created_at else None,
        "resolved_at": handoff.resolved_at.isoformat() if handoff.resolved_at else None,
    }


class HandoffService:
    def __init__(self, client: CRMClient | None = None) -> None:
        self.client = client or CRMClient()

    def list(self, status: str | None = None, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.client.session() as session:
            query = (
                select(db.Handoff, db.Customer, db.Lead)
                .join(db.Customer, db.Customer.id == db.Handoff.customer_id)
                .outerjoin(db.Lead, db.Lead.id == db.Handoff.lead_id)
            )
            if status:
                query = query.where(db.Handoff.status == status)
            rows = session.execute(query.order_by(db.Handoff.id.desc()).limit(limit)).all()
            return [serialize(*row) for row in reversed(rows)]

    def _load(self, session, handoff_id: UUID) -> tuple[db.Handoff, db.Customer, db.Lead | None]:
        row = session.execute(
            select(db.Handoff, db.Customer, db.Lead)
            .join(db.Customer, db.Customer.id == db.Handoff.customer_id)
            .outerjoin(db.Lead, db.Lead.id == db.Handoff.lead_id)
            .where(db.Handoff.public_id == handoff_id)
        ).first()
        if row is None:
            raise HandoffNotFoundError(f"Handoff {handoff_id} was not found.")
        return row[0], row[1], row[2]

    def get(self, handoff_id: UUID) -> dict[str, Any]:
        with self.client.session() as session:
            return serialize(*self._load(session, handoff_id))

    def claim(self, handoff_id: UUID, by: str) -> dict[str, Any]:
        with self.client.session() as session:
            handoff, customer, lead = self._load(session, handoff_id)
            if handoff.status == "resolved":
                raise InvalidHandoffTransition("A resolved handoff cannot be claimed.")
            handoff.status = "claimed"
            handoff.claimed_by = by[:100]
            session.flush()
            return serialize(handoff, customer, lead)

    def resolve(self, handoff_id: UUID, by: str) -> dict[str, Any]:
        with self.client.session() as session:
            handoff, customer, lead = self._load(session, handoff_id)
            if handoff.status == "resolved":
                raise InvalidHandoffTransition("This handoff is already resolved.")
            handoff.status = "resolved"
            handoff.claimed_by = handoff.claimed_by or by[:100]
            handoff.resolved_at = datetime.now(timezone.utc)
            if lead is not None and lead.status == "handoff_requested":
                lead.status = "contacted"
            session.flush()
            return serialize(handoff, customer, lead)
