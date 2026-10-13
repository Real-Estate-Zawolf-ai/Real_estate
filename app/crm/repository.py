"""Low-level CRM queries shared by the services and the agent facade."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.database import models as db
from app.database.upsert import insert_ignore
from app.helpers.phone import normalize_phone


OPEN_LEAD_EXCLUDED = ("won", "lost", "closed")


def channel_of(external_id: str) -> str:
    return external_id.split(":", 1)[0] if ":" in external_id else "unknown"


def get_customer_row(session: Session, external_id: str) -> db.Customer | None:
    return session.scalar(select(db.Customer).where(db.Customer.external_id == external_id))


def get_or_create_customer(session: Session, external_id: str) -> db.Customer:
    row = get_customer_row(session, external_id)
    if row is not None:
        return row
    insert_ignore(
        session,
        db.Customer,
        {
            "external_id": external_id,
            "source": channel_of(external_id),
            "preferred_language": "en",
            "phone_verified": False,
        },
        ["external_id"],
    )
    row = get_customer_row(session, external_id)
    assert row is not None
    return row


def latest_lead(session: Session, customer_pk: int, *, open_only: bool = False) -> db.Lead | None:
    query = select(db.Lead).where(db.Lead.customer_id == customer_pk)
    if open_only:
        query = query.where(db.Lead.status.not_in(OPEN_LEAD_EXCLUDED))
    return session.scalar(query.order_by(db.Lead.id.desc()).limit(1))


def ensure_open_lead(session: Session, customer: db.Customer, source: str) -> db.Lead:
    lead = latest_lead(session, customer.id, open_only=True)
    if lead is None:
        lead = db.Lead(customer_id=customer.id, status="new", source=source, score=0)
        session.add(lead)
        session.flush()
    return lead


def apply_phone(
    session: Session,
    customer: db.Customer,
    raw_phone: str | None,
    *,
    verified: bool | None = None,
    method: str | None = None,
) -> str | None:
    """Store a normalised phone on the customer; returns the number saved, if any.

    Verification belongs to one specific number: changing the number clears it,
    and it is only set again by an explicit confirmation or a channel guarantee
    (the WhatsApp sender id).
    """
    phone = normalize_phone(raw_phone)
    if phone is None:
        return None
    if customer.phone != phone:
        customer.phone = phone
        customer.phone_verified = False
        customer.phone_verified_at = None
        customer.phone_verification_method = None
    if verified:
        customer.phone_verified = True
        customer.phone_verified_at = datetime.now(timezone.utc)
        customer.phone_verification_method = method or "readback"
    session.flush()
    return phone


def to_decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value)) if value not in (None, "") else None
    except (InvalidOperation, ValueError):
        return None


def mark_lead_status(session: Session, lead_pk: int, status: str) -> None:
    session.execute(update(db.Lead).where(db.Lead.id == lead_pk).values(status=status))
