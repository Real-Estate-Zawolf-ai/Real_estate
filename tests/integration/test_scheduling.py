"""Booking lifecycle against the SQL-backed booking service."""

from datetime import datetime, timedelta, timezone

import pytest

from app.crm.customers import CreateCustomer, CustomerService
from app.database import models as db
from app.database.session import session_scope
from app.scheduling.availability import TimeSlot, build_available_slots
from app.scheduling.booking import (
    BookingConflictError,
    BookingNotFoundError,
    BookingService,
    BookingStatus,
    CreateBooking,
    InvalidBookingTransition,
    PhoneNotVerifiedError,
)

PHONE = "+201012345678"


def future(hours: int = 48) -> datetime:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).replace(minute=0, second=0, microsecond=0)


def make_customer(*, verified: bool = True, phone: str = PHONE):
    customer = CustomerService().create(CreateCustomer(full_name="Nora Ali", phone_number=phone))
    if verified:
        with session_scope() as session:
            row = session.query(db.Customer).filter_by(public_id=customer.id).one()
            row.phone_verified = True
    return customer


def request_for(customer, start: datetime, property_id: str = "501189949", minutes: int = 30, phone: str = PHONE):
    return CreateBooking(
        property_id=property_id,
        customer_id=customer.id,
        slot=TimeSlot(starts_at=start, ends_at=start + timedelta(minutes=minutes)),
        contact_phone=phone,
    )


def test_booking_conflict_and_cancellation_workflow():
    service = BookingService()
    customer, other = make_customer(), make_customer(phone="+201098765432")
    start = future()
    booking = service.create(request_for(customer, start))
    assert booking.status == BookingStatus.PENDING

    with pytest.raises(BookingConflictError):
        service.create(request_for(other, start, phone="+201098765432"))
    with pytest.raises(BookingConflictError):  # partial overlap is a conflict too
        service.create(request_for(other, start + timedelta(minutes=15), phone="+201098765432"))
    # A different property at the same time is fine.
    service.create(request_for(other, start, property_id="999", phone="+201098765432"))

    assert service.cancel(booking.id).status == "cancelled"
    # The freed slot can be booked again.
    assert service.create(request_for(other, start, phone="+201098765432")).status == BookingStatus.PENDING


def test_available_slots_exclude_blocked_interval():
    start = datetime(2026, 10, 2, 10, tzinfo=timezone.utc)
    window = TimeSlot(starts_at=start, ends_at=start + timedelta(hours=1))
    blocked = TimeSlot(starts_at=start, ends_at=start + timedelta(minutes=30))
    slots = build_available_slots([window], [blocked])
    assert len(slots) == 1
    assert slots[0].starts_at == start + timedelta(minutes=30)


def test_booking_requires_a_verified_phone_number():
    service = BookingService()
    unverified = make_customer(verified=False)
    with pytest.raises(PhoneNotVerifiedError):
        service.create(request_for(unverified, future()))

    # Verified, but the booking names a different number than the verified one.
    verified = make_customer(phone="+201055555555")
    with pytest.raises(PhoneNotVerifiedError):
        service.create(request_for(verified, future(), phone=PHONE))


def test_invalid_phone_number_is_rejected_before_it_reaches_the_database():
    customer = make_customer()
    for bad in ("12345", "abc", "0101234", "+999"):
        with pytest.raises(ValueError):
            request_for(customer, future(), phone=bad)


def test_lifecycle_confirm_complete_and_invalid_transitions():
    service = BookingService()
    customer = make_customer()
    booking = service.create(request_for(customer, future()))

    confirmed = service.confirm(booking.id, by="Sara")
    assert (confirmed.status, confirmed.confirmed_by) == (BookingStatus.CONFIRMED, "Sara")
    assert service.confirm(booking.id).status == BookingStatus.CONFIRMED  # idempotent

    done = service.complete(booking.id)
    assert done.status == BookingStatus.COMPLETED
    for action in (service.cancel, service.confirm):
        with pytest.raises(InvalidBookingTransition):
            action(booking.id)
    with pytest.raises(BookingNotFoundError):
        service.confirm(customer.id)  # a customer id is not a booking id


def test_cancel_is_idempotent_and_records_who_and_why():
    service = BookingService()
    booking = service.create(request_for(make_customer(), future()))
    cancelled = service.cancel(booking.id, by="staff", reason="agent unavailable")
    assert (cancelled.cancelled_by, cancelled.cancel_reason) == ("staff", "agent unavailable")
    assert service.cancel(booking.id, by="customer").cancelled_by == "staff"  # unchanged


def test_unconfirmed_hold_expires_and_frees_the_slot(settings_env):
    settings_env(BOOKING_HOLD_TTL_HOURS="1")
    service = BookingService()
    start = future(72)
    first = service.create(request_for(make_customer(), start))
    assert first.hold_expires_at is not None

    later = datetime.now(timezone.utc) + timedelta(hours=2)
    assert service.list_active("501189949", now=later) == []  # expired hold no longer blocks
    other = make_customer(phone="+201098765432")
    assert service.create(request_for(other, start, phone="+201098765432"), now=later) is not None
    with pytest.raises(InvalidBookingTransition):
        service.confirm(first.id, now=later)


def test_cannot_book_in_the_past():
    with pytest.raises(ValueError):
        BookingService().create(request_for(make_customer(), datetime.now(timezone.utc) - timedelta(hours=1)))
