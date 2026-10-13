"""Viewing scheduling used by the scheduling agent and the staff API.

Wraps ``app.scheduling.BookingService`` so availability, conflict detection and
the booking lifecycle all come from the same PostgreSQL records. A booking made
here is a *hold*: the customer confirmed the slot, but a sales advisor still has
to confirm it, so the assistant never tells a customer a visit is confirmed.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.agents.tools.crm import crm_service
from app.crm import repository as repo
from app.crm.client import CRMClient
from app.database import models as db
from app.helpers.config import get_settings
from app.helpers.phone import mask_phone
from app.scheduling.availability import TimeSlot, build_available_slots
from app.scheduling.booking import (
    Booking,
    BookingConflictError,
    BookingService,
    BookingStatus,
    CreateBooking,
    PhoneNotVerifiedError,
)


logger = logging.getLogger(__name__)


class SchedulingService:

    def __init__(self, booking_service: BookingService | None = None, client: CRMClient | None = None) -> None:
        self._client = client or CRMClient()
        self._bookings = booking_service or BookingService(self._client)

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    @property
    def timezone(self) -> ZoneInfo:
        return ZoneInfo(get_settings().VIEWING_TIMEZONE)

    def today(self) -> date:
        return datetime.now(self.timezone).date()

    def _local(self, moment: datetime) -> datetime:
        return moment.astimezone(self.timezone)

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

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
        now = now or datetime.now(tz)
        active = self._bookings.list_active(str(property_id), now=now)
        slots = build_available_slots(
            [window],
            [booking.slot for booking in active],
            duration_minutes=settings.VIEWING_SLOT_MINUTES,
            interval_minutes=settings.VIEWING_SLOT_MINUTES,
        )
        earliest = now + timedelta(hours=settings.BOOKING_MIN_NOTICE_HOURS)
        return [
            {"starts_at": slot.starts_at.isoformat(), "ends_at": slot.ends_at.isoformat()}
            for slot in slots
            if slot.starts_at >= earliest
        ]

    # ------------------------------------------------------------------
    # Customer-driven operations
    # ------------------------------------------------------------------

    def request_viewing(
        self,
        *,
        property_id: str,
        customer_id: str,
        starts_at: str,
        ends_at: str,
        notes: str | None = None,
        property_label: str | None = None,
        now: datetime | None = None,
    ) -> dict:
        """Place a hold on a slot.

        Raises ``BookingConflictError`` when the slot is gone (or was never an
        offered slot) and ``PhoneNotVerifiedError`` when the customer's number
        has not been confirmed.
        """
        start = datetime.fromisoformat(starts_at)
        end = datetime.fromisoformat(ends_at)
        now = now or datetime.now(self.timezone)

        # Only slots the calendar actually offers can be booked, whatever the
        # caller (or a model) asks for.
        offered = self.get_available_slots(property_id, self._local(start).date(), now=now)
        if not any(
            datetime.fromisoformat(slot["starts_at"]) == start and datetime.fromisoformat(slot["ends_at"]) == end
            for slot in offered
        ):
            raise BookingConflictError("This appointment time is no longer available.")

        with self._client.session() as session:
            customer = repo.get_customer_row(session, customer_id)
            if customer is None or not customer.phone_verified or not customer.phone:
                raise PhoneNotVerifiedError("The customer's phone number has not been verified.")
            customer_public_id, phone, name = customer.public_id, customer.phone, customer.name

        booking = self._bookings.create(
            CreateBooking(
                property_id=str(property_id),
                customer_id=customer_public_id,
                slot=TimeSlot(starts_at=start, ends_at=end),
                contact_phone=phone,
                notes=notes,
            ),
            now=now,
        )

        try:
            crm_service.ensure_lead(customer_id, source=customer_id.split(":", 1)[0])
            crm_service.update_lead(
                customer_id,
                {"status": "viewing_requested", "appointment_id": str(booking.id), "property_ref": str(property_id)},
            )
        except Exception:
            logger.exception("Could not update the lead for a new booking | booking=%s", booking.id)

        self._alert_staff_new(booking, customer_id, name, phone, property_label)
        return self._serialize(booking, customer_id)

    def cancel(self, appointment_id: str, *, customer_id: str | None = None, by: str = "customer", reason: str | None = None) -> dict:
        """Cancel a booking. A customer can only cancel their own."""
        booking_id = UUID(appointment_id)
        existing = self._bookings.get(booking_id)
        if existing is None:
            raise LookupError(f"Booking {appointment_id} was not found.")
        if customer_id is not None and existing.customer_ref != customer_id:
            # Same answer as "not found": never reveal other customers' bookings.
            raise LookupError(f"Booking {appointment_id} was not found.")
        was_active = existing.status in (BookingStatus.PENDING, BookingStatus.CONFIRMED)
        booking = self._bookings.cancel(booking_id, by=by, reason=reason)
        if was_active:
            self._alert_staff_cancelled(booking, by)
        return self._serialize(booking, booking.customer_ref)

    def list_for_customer(self, customer_id: str, *, active_only: bool = False) -> list[dict]:
        with self._client.session() as session:
            customer = repo.get_customer_row(session, customer_id)
            public_id = customer.public_id if customer else None
        if public_id is None:
            return []
        return [self._serialize(item, customer_id) for item in self._bookings.list_for_customer(public_id, active_only=active_only)]

    # ------------------------------------------------------------------
    # Staff operations (admin API)
    # ------------------------------------------------------------------

    def staff_list(self, *, status: str | None = None, upcoming_only: bool = False, limit: int = 100) -> list[dict]:
        return [
            self._serialize(item, item.customer_ref, include_contact=True)
            for item in self._bookings.list(status=status, upcoming_only=upcoming_only, limit=limit)
        ]

    def staff_get(self, booking_id: UUID) -> dict | None:
        booking = self._bookings.get(booking_id)
        return self._serialize(booking, booking.customer_ref, include_contact=True) if booking else None

    def staff_confirm(self, booking_id: UUID, *, by: str = "staff") -> dict:
        booking = self._bookings.confirm(booking_id, by=by)
        self._notify_customer(booking, "booking_confirmed_notice")
        return self._serialize(booking, booking.customer_ref, include_contact=True)

    def staff_cancel(self, booking_id: UUID, *, by: str = "staff", reason: str | None = None) -> dict:
        existing = self._bookings.get(booking_id)
        booking = self._bookings.cancel(booking_id, by="staff", reason=reason or f"cancelled by {by}")
        if existing and existing.status in (BookingStatus.PENDING, BookingStatus.CONFIRMED):
            self._notify_customer(booking, "booking_cancelled_notice")
        return self._serialize(booking, booking.customer_ref, include_contact=True)

    def staff_complete(self, booking_id: UUID) -> dict:
        booking = self._bookings.complete(booking_id)
        return self._serialize(booking, booking.customer_ref, include_contact=True)

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _when(self, booking: Booking) -> str:
        return self._local(booking.slot.starts_at).strftime("%a %d %b %Y %H:%M")

    def _alert_staff_new(self, booking: Booking, customer_id: str, name: str | None, phone: str, label: str | None) -> None:
        from app.notifications.staff import notify_staff

        message = (
            "New viewing request - needs confirmation\n"
            f"Property: {label or booking.property_id} (ID {booking.property_id})\n"
            f"When: {self._when(booking)} ({get_settings().VIEWING_TIMEZONE})\n"
            f"Customer: {name or 'unknown'} | phone: {phone} (verified)\n"
            f"Channel: {customer_id.split(':', 1)[0]}\n"
            f"Booking: {str(booking.id)[:8]}"
        )
        notify_staff("booking_requested", message, ref_type="booking", ref_id=str(booking.id))

    def _alert_staff_cancelled(self, booking: Booking, by: str) -> None:
        from app.notifications.staff import notify_staff

        message = (
            f"Viewing cancelled by {by}\n"
            f"Property: {booking.property_id}\n"
            f"Was: {self._when(booking)} ({get_settings().VIEWING_TIMEZONE})\n"
            f"Phone: {booking.contact_phone}\n"
            f"Booking: {str(booking.id)[:8]}"
        )
        notify_staff("booking_cancelled", message, ref_type="booking", ref_id=str(booking.id))

    def _notify_customer(self, booking: Booking, phrase: str) -> None:
        """Tell a WhatsApp customer their booking changed. Website users see it on return."""
        from app.agents.messages import say
        from app.notifications.customer import send_to_customer

        try:
            with self._client.session() as session:
                language = session.scalar(
                    select(db.Customer.preferred_language).where(db.Customer.public_id == booking.customer_id)
                ) or "en"
            text = say(language, phrase, property=booking.property_id, when=self._when(booking), ref=str(booking.id)[:8])
            if booking.customer_ref:
                send_to_customer(booking.customer_ref, text)
        except Exception:
            logger.exception("Could not notify customer of booking change | booking=%s", booking.id)

    # ------------------------------------------------------------------

    def _serialize(self, booking: Booking, customer_id: str | None, *, include_contact: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": str(booking.id),
            "ref": str(booking.id)[:8],
            "property_id": booking.property_id,
            "customer_id": customer_id,
            "starts_at": self._local(booking.slot.starts_at).isoformat(),
            "ends_at": self._local(booking.slot.ends_at).isoformat(),
            "status": booking.status.value,
            "notes": booking.notes,
            "cancel_reason": booking.cancel_reason,
        }
        if include_contact:
            data["contact_phone"] = booking.contact_phone
            data["confirmed_by"] = booking.confirmed_by
            data["cancelled_by"] = booking.cancelled_by
            data["hold_expires_at"] = booking.hold_expires_at.isoformat() if booking.hold_expires_at else None
        return data

    def masked(self, phone: str) -> str:
        return mask_phone(phone)


scheduling_service = SchedulingService()
