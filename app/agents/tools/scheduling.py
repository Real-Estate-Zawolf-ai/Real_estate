"""Viewing scheduling used by the scheduling agent.

Wraps ``app.scheduling.BookingService`` so availability and conflict
detection come from the same booking records. A booking created here is a
*hold*: the customer has confirmed the slot, but a sales advisor still has to
confirm it, so the assistant never tells a customer a visit is confirmed.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from threading import Lock
from uuid import NAMESPACE_URL, UUID, uuid5
from zoneinfo import ZoneInfo

from app.helpers.config import get_settings
from app.scheduling.availability import TimeSlot, build_available_slots
from app.scheduling.booking import BookingService, BookingStatus, CreateBooking


def _stable_uuid(kind: str, value: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"aqar:{kind}:{value}")


class SchedulingService:

    def __init__(self, booking_service: BookingService | None = None) -> None:
        self._bookings = booking_service or BookingService()
        self._references: dict[UUID, dict[str, str]] = {}
        self._lock = Lock()

    @property
    def timezone(self) -> ZoneInfo:
        return ZoneInfo(get_settings().VIEWING_TIMEZONE)

    def today(self) -> date:
        return datetime.now(self.timezone).date()

    def get_available_slots(
        self,
        property_id: str,
        day: date,
        *,
        now: datetime | None = None,
    ) -> list[dict]:
        """Return free viewing slots for one property on one local day."""
        settings = get_settings()
        tz = self.timezone
        window = TimeSlot(
            starts_at=datetime.combine(day, time(settings.VIEWING_DAY_START_HOUR), tzinfo=tz),
            ends_at=datetime.combine(day, time(settings.VIEWING_DAY_END_HOUR), tzinfo=tz),
        )
        active = self._bookings.repository.list_active(_stable_uuid("property", property_id), None)
        slots = build_available_slots(
            [window],
            [booking.slot for booking in active],
            duration_minutes=settings.VIEWING_SLOT_MINUTES,
            interval_minutes=settings.VIEWING_SLOT_MINUTES,
        )
        # Leave at least an hour of notice for the advisor.
        earliest = (now or datetime.now(tz)) + timedelta(hours=1)
        return [
            {"starts_at": slot.starts_at.isoformat(), "ends_at": slot.ends_at.isoformat()}
            for slot in slots
            if slot.starts_at >= earliest
        ]

    def request_viewing(
        self,
        *,
        property_id: str,
        customer_id: str,
        starts_at: str,
        ends_at: str,
        notes: str | None = None,
    ) -> dict:
        """Place a hold on a slot. Raises BookingConflictError when taken."""
        booking = self._bookings.create(
            CreateBooking(
                property_id=_stable_uuid("property", property_id),
                customer_id=_stable_uuid("customer", customer_id),
                slot=TimeSlot(
                    starts_at=datetime.fromisoformat(starts_at),
                    ends_at=datetime.fromisoformat(ends_at),
                ),
                notes=notes,
            )
        )
        with self._lock:
            self._references[booking.id] = {"property_id": property_id, "customer_id": customer_id}
        return self._serialize(booking)

    def cancel(self, appointment_id: str) -> dict:
        return self._serialize(self._bookings.cancel(UUID(appointment_id)))

    def list_for_customer(self, customer_id: str) -> list[dict]:
        with self._lock:
            ids = [booking_id for booking_id, ref in self._references.items() if ref["customer_id"] == customer_id]
        return [self._serialize(self._bookings.repository.get(booking_id)) for booking_id in ids]

    def _serialize(self, booking) -> dict:
        with self._lock:
            reference = self._references.get(booking.id, {})
        return {
            "id": str(booking.id),
            "property_id": reference.get("property_id"),
            "customer_id": reference.get("customer_id"),
            "starts_at": booking.slot.starts_at.isoformat(),
            "ends_at": booking.slot.ends_at.isoformat(),
            # "hold" = requested by the customer, awaiting advisor confirmation.
            "status": "pending_confirmation" if booking.status == BookingStatus.HOLD else booking.status.value,
        }


scheduling_service = SchedulingService()
