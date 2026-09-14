"""Booking service for property-viewing appointments.

The in-memory repository is useful for development and tests. Replace it with
a database repository implementing the same methods before production.
"""

from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from scheduling.availability import TimeSlot, overlaps


class BookingStatus(StrEnum):
    HOLD = "hold"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


class CreateBooking(BaseModel):
    """Information required to reserve a property-viewing time slot."""

    property_id: UUID
    customer_id: UUID
    agent_id: UUID | None = None
    slot: TimeSlot
    notes: str | None = Field(default=None, max_length=1_000)


class Booking(BaseModel):
    """A viewing appointment and its lifecycle state."""

    id: UUID = Field(default_factory=uuid4)
    property_id: UUID
    customer_id: UUID
    agent_id: UUID | None = None
    slot: TimeSlot
    status: BookingStatus = BookingStatus.HOLD
    notes: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BookingConflictError(ValueError):
    """Raised when an appointment overlaps an active booking."""


class BookingNotFoundError(LookupError):
    """Raised when a requested booking does not exist."""


class InMemoryBookingRepository:
    """Thread-safe development repository for appointment records."""

    def __init__(self) -> None:
        self._bookings: dict[UUID, Booking] = {}
        self._lock = Lock()

    def list_active(self, property_id: UUID, agent_id: UUID | None) -> list[Booking]:
        with self._lock:
            return [
                booking
                for booking in self._bookings.values()
                if booking.status in {BookingStatus.HOLD, BookingStatus.CONFIRMED}
                and (
                    booking.property_id == property_id
                    or (agent_id is not None and booking.agent_id == agent_id)
                )
            ]

    def get(self, booking_id: UUID) -> Booking | None:
        with self._lock:
            return self._bookings.get(booking_id)

    def save(self, booking: Booking) -> Booking:
        with self._lock:
            self._bookings[booking.id] = booking
            return booking

    def reserve(self, booking: Booking) -> Booking:
        """Atomically check conflicts and save an active booking."""
        with self._lock:
            for existing in self._bookings.values():
                is_active = existing.status in {
                    BookingStatus.HOLD,
                    BookingStatus.CONFIRMED,
                }
                blocks_property = existing.property_id == booking.property_id
                blocks_agent = (
                    booking.agent_id is not None
                    and existing.agent_id == booking.agent_id
                )
                if is_active and (blocks_property or blocks_agent) and overlaps(
                    booking.slot, existing.slot
                ):
                    raise BookingConflictError(
                        "This appointment time is no longer available."
                    )
            self._bookings[booking.id] = booking
            return booking


class BookingService:
    """Coordinates booking state and conflict detection."""

    def __init__(self, repository: InMemoryBookingRepository | None = None) -> None:
        self.repository = repository or InMemoryBookingRepository()

    def create(self, request: CreateBooking) -> Booking:
        """Create a temporary hold if the property and agent are both free."""
        return self.repository.reserve(
            Booking(
                property_id=request.property_id,
                customer_id=request.customer_id,
                agent_id=request.agent_id,
                slot=request.slot,
                notes=request.notes,
            )
        )

    def confirm(self, booking_id: UUID) -> Booking:
        """Confirm a hold after the customer accepts the appointment."""
        booking = self._get_required(booking_id)
        if booking.status == BookingStatus.CANCELLED:
            raise ValueError("Cancelled bookings cannot be confirmed.")
        booking.status = BookingStatus.CONFIRMED
        booking.updated_at = datetime.now(UTC)
        return self.repository.save(booking)

    def cancel(self, booking_id: UUID) -> Booking:
        """Cancel a booking and free its slot for future reservations."""
        booking = self._get_required(booking_id)
        booking.status = BookingStatus.CANCELLED
        booking.updated_at = datetime.now(UTC)
        return self.repository.save(booking)

    def _get_required(self, booking_id: UUID) -> Booking:
        booking = self.repository.get(booking_id)
        if booking is None:
            raise BookingNotFoundError(f"Booking {booking_id} was not found.")
        return booking
