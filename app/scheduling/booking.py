"""Booking service for property-viewing appointments (PostgreSQL).

Lifecycle::

    pending_confirmation --(staff)--> confirmed --(staff)--> completed
            |                              |
            +--(customer/staff/system)-----+--> cancelled
            +--(hold timed out)--> expired

A customer's request is only a *hold*: an advisor must confirm it, and the
assistant never tells a customer a visit is confirmed before that. Double
booking is prevented twice - an overlap check in code and, on PostgreSQL, an
exclusion constraint that holds across workers and processes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import and_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.crm.client import CRMClient
from app.database import models as db
from app.helpers.config import get_settings
from app.helpers.phone import normalize_phone
from app.scheduling.availability import TimeSlot, overlaps


class BookingStatus(str, Enum):
    PENDING = "pending_confirmation"
    HOLD = "pending_confirmation"  # earlier name for the same state
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    COMPLETED = "completed"


ACTIVE = (BookingStatus.PENDING.value, BookingStatus.CONFIRMED.value)


class CreateBooking(BaseModel):
    """Information required to reserve a property-viewing time slot."""

    property_id: str = Field(min_length=1, max_length=100)
    # Public id of the customer record (see CRM).
    customer_id: UUID
    agent_id: str | None = Field(default=None, max_length=100)
    slot: TimeSlot
    # The number an advisor will call. It must match the customer's *verified* number.
    contact_phone: str
    notes: str | None = Field(default=None, max_length=1_000)

    @field_validator("contact_phone")
    @classmethod
    def valid_phone(cls, value: str) -> str:
        normalized = normalize_phone(value)
        if normalized is None:
            raise ValueError("contact_phone is not a valid phone number")
        return normalized


class Booking(BaseModel):
    """A viewing appointment and its lifecycle state."""

    id: UUID = Field(default_factory=uuid4)
    property_id: str
    customer_id: UUID
    customer_ref: str | None = None
    agent_id: str | None = None
    slot: TimeSlot
    status: BookingStatus = BookingStatus.PENDING
    contact_phone: str = ""
    notes: str | None = None
    hold_expires_at: datetime | None = None
    confirmed_at: datetime | None = None
    confirmed_by: str | None = None
    cancelled_at: datetime | None = None
    cancelled_by: str | None = None
    cancel_reason: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class BookingConflictError(ValueError):
    """Raised when an appointment overlaps an active booking."""


class BookingNotFoundError(LookupError):
    """Raised when a requested booking does not exist."""


class InvalidBookingTransition(ValueError):
    """Raised when a booking cannot move to the requested status."""


class PhoneNotVerifiedError(ValueError):
    """Raised when a booking is attempted without a verified phone number."""


def _entity(row: db.Booking, customer_public_id: UUID, customer_external: str | None = None) -> Booking:
    return Booking(
        id=row.public_id,
        property_id=row.property_id,
        customer_id=customer_public_id,
        customer_ref=customer_external,
        agent_id=row.agent_id,
        slot=TimeSlot(starts_at=row.starts_at, ends_at=row.ends_at),
        status=BookingStatus(row.status),
        contact_phone=row.contact_phone,
        notes=row.notes,
        hold_expires_at=row.hold_expires_at,
        confirmed_at=row.confirmed_at,
        confirmed_by=row.confirmed_by,
        cancelled_at=row.cancelled_at,
        cancelled_by=row.cancelled_by,
        cancel_reason=row.cancel_reason,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _is_overlap_violation(error: IntegrityError) -> bool:
    origin = getattr(error, "orig", None)
    return getattr(origin, "sqlstate", None) == "23P01" or "bookings_no_overlap" in str(error)


class BookingService:
    """Coordinates booking state and conflict detection."""

    def __init__(self, client: CRMClient | None = None) -> None:
        self.client = client or CRMClient()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @staticmethod
    def expire_stale_holds(session: Session, now: datetime) -> int:
        """Release unconfirmed holds whose time ran out so their slot reopens."""
        result = session.execute(
            update(db.Booking)
            .where(
                db.Booking.status == BookingStatus.PENDING.value,
                db.Booking.hold_expires_at.is_not(None),
                db.Booking.hold_expires_at < now,
            )
            .values(status=BookingStatus.EXPIRED.value)
        )
        return result.rowcount or 0

    def list_active(self, property_id: str, *, now: datetime | None = None) -> list[Booking]:
        now = now or datetime.now(timezone.utc)
        with self.client.session() as session:
            self.expire_stale_holds(session, now)
            rows = session.execute(
                select(db.Booking, db.Customer.public_id)
                .join(db.Customer, db.Customer.id == db.Booking.customer_id)
                .where(db.Booking.property_id == property_id, db.Booking.status.in_(ACTIVE))
                .order_by(db.Booking.starts_at)
            ).all()
            return [_entity(row, public_id) for row, public_id in rows]

    def get(self, booking_id: UUID) -> Booking | None:
        with self.client.session() as session:
            found = self._load(session, booking_id)
            return _entity(found[0], found[1].public_id, found[1].external_id) if found else None

    def list_for_customer(self, customer_public_id: UUID, *, active_only: bool = False) -> list[Booking]:
        with self.client.session() as session:
            query = (
                select(db.Booking, db.Customer.public_id, db.Customer.external_id)
                .join(db.Customer, db.Customer.id == db.Booking.customer_id)
                .where(db.Customer.public_id == customer_public_id)
                .order_by(db.Booking.starts_at)
            )
            if active_only:
                query = query.where(db.Booking.status.in_(ACTIVE))
            return [_entity(row, public_id, external) for row, public_id, external in session.execute(query).all()]

    def list(
        self,
        *,
        status: str | None = None,
        upcoming_only: bool = False,
        limit: int = 100,
        now: datetime | None = None,
    ) -> list[Booking]:
        """Bookings for the staff dashboard, soonest first."""
        now = now or datetime.now(timezone.utc)
        with self.client.session() as session:
            self.expire_stale_holds(session, now)
            query = (
                select(db.Booking, db.Customer.public_id, db.Customer.external_id)
                .join(db.Customer, db.Customer.id == db.Booking.customer_id)
                .order_by(db.Booking.starts_at)
                .limit(limit)
            )
            if status:
                query = query.where(db.Booking.status == status)
            if upcoming_only:
                query = query.where(db.Booking.ends_at >= now)
            return [_entity(row, public_id, external) for row, public_id, external in session.execute(query).all()]

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def create(self, request: CreateBooking, *, now: datetime | None = None) -> Booking:
        """Reserve a hold if the property and (optional) advisor are free."""
        now = now or datetime.now(timezone.utc)
        settings = get_settings()
        if request.slot.starts_at <= now:
            raise ValueError("Appointments must be in the future.")

        with self.client.session() as session:
            customer = session.scalar(select(db.Customer).where(db.Customer.public_id == request.customer_id))
            if customer is None:
                raise ValueError("A booking must reference an existing customer.")
            # Hard gate: this number must have been confirmed by the customer (or
            # guaranteed by the channel). Checked here so no caller can skip it.
            if not customer.phone_verified or normalize_phone(customer.phone) != request.contact_phone:
                raise PhoneNotVerifiedError("The customer's phone number has not been verified.")

            self.expire_stale_holds(session, now)
            clash = [db.Booking.property_id == request.property_id]
            if request.agent_id:
                clash = [(db.Booking.property_id == request.property_id) | (db.Booking.agent_id == request.agent_id)]
            conflict = session.scalar(
                select(db.Booking.id)
                .where(
                    *clash,
                    db.Booking.status.in_(ACTIVE),
                    and_(db.Booking.starts_at < request.slot.ends_at, db.Booking.ends_at > request.slot.starts_at),
                )
                .limit(1)
            )
            if conflict is not None:
                raise BookingConflictError("This appointment time is no longer available.")

            hold_until = min(now + timedelta(hours=settings.BOOKING_HOLD_TTL_HOURS), request.slot.starts_at)
            row = db.Booking(
                property_id=request.property_id,
                customer_id=customer.id,
                agent_id=request.agent_id,
                starts_at=request.slot.starts_at,
                ends_at=request.slot.ends_at,
                status=BookingStatus.PENDING.value,
                contact_phone=request.contact_phone,
                notes=request.notes,
                hold_expires_at=hold_until,
            )
            session.add(row)
            try:
                session.flush()
            except IntegrityError as error:
                session.rollback()
                if _is_overlap_violation(error):
                    raise BookingConflictError("This appointment time is no longer available.") from error
                raise
            session.refresh(row)
            return _entity(row, customer.public_id, customer.external_id)

    def confirm(self, booking_id: UUID, *, by: str = "staff", now: datetime | None = None) -> Booking:
        """An advisor accepts a hold. Idempotent for an already-confirmed booking."""
        now = now or datetime.now(timezone.utc)
        with self.client.session() as session:
            row, customer = self._require(session, booking_id)
            if row.status == BookingStatus.CONFIRMED.value:
                return _entity(row, customer.public_id, customer.external_id)
            if row.status != BookingStatus.PENDING.value:
                raise InvalidBookingTransition(f"A {row.status} booking cannot be confirmed.")
            if row.hold_expires_at and row.hold_expires_at < now:
                row.status = BookingStatus.EXPIRED.value
                session.flush()
                raise InvalidBookingTransition("This hold has expired; the customer needs to book again.")
            row.status = BookingStatus.CONFIRMED.value
            row.confirmed_at = now
            row.confirmed_by = by[:100]
            row.hold_expires_at = None
            session.flush()
            return _entity(row, customer.public_id, customer.external_id)

    def cancel(
        self,
        booking_id: UUID,
        *,
        by: str = "customer",
        reason: str | None = None,
        now: datetime | None = None,
    ) -> Booking:
        """Cancel a booking and free its slot. Idempotent for a cancelled booking."""
        now = now or datetime.now(timezone.utc)
        with self.client.session() as session:
            row, customer = self._require(session, booking_id)
            if row.status == BookingStatus.CANCELLED.value:
                return _entity(row, customer.public_id, customer.external_id)
            if row.status not in ACTIVE:
                raise InvalidBookingTransition(f"A {row.status} booking cannot be cancelled.")
            row.status = BookingStatus.CANCELLED.value
            row.cancelled_at = now
            row.cancelled_by = by[:20]
            row.cancel_reason = (reason or "")[:255] or None
            session.flush()
            return _entity(row, customer.public_id, customer.external_id)

    def complete(self, booking_id: UUID) -> Booking:
        """Mark a confirmed viewing as held."""
        with self.client.session() as session:
            row, customer = self._require(session, booking_id)
            if row.status != BookingStatus.CONFIRMED.value:
                raise InvalidBookingTransition("Only a confirmed booking can be completed.")
            row.status = BookingStatus.COMPLETED.value
            session.flush()
            return _entity(row, customer.public_id, customer.external_id)

    # ------------------------------------------------------------------

    @staticmethod
    def _load(session: Session, booking_id: UUID) -> tuple[db.Booking, db.Customer] | None:
        found = session.execute(
            select(db.Booking, db.Customer)
            .join(db.Customer, db.Customer.id == db.Booking.customer_id)
            .where(db.Booking.public_id == booking_id)
        ).first()
        return (found[0], found[1]) if found else None

    def _require(self, session: Session, booking_id: UUID) -> tuple[db.Booking, db.Customer]:
        found = self._load(session, booking_id)
        if found is None:
            raise BookingNotFoundError(f"Booking {booking_id} was not found.")
        return found


__all__ = [
    "Booking",
    "BookingConflictError",
    "BookingNotFoundError",
    "BookingService",
    "BookingStatus",
    "CreateBooking",
    "InvalidBookingTransition",
    "PhoneNotVerifiedError",
    "overlaps",
]
