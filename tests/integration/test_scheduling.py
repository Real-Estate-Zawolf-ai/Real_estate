from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from app.scheduling.availability import TimeSlot, build_available_slots
from app.scheduling.booking import BookingConflictError, BookingService, CreateBooking


def test_booking_conflict_and_cancellation_workflow():
    service = BookingService()
    start = datetime(2026, 10, 2, 10, tzinfo=timezone.utc)
    request = CreateBooking(
        property_id=uuid4(),
        customer_id=uuid4(),
        slot=TimeSlot(starts_at=start, ends_at=start + timedelta(minutes=30)),
    )
    booking = service.create(request)
    with pytest.raises(BookingConflictError):
        service.create(request.model_copy(update={"customer_id": uuid4()}))
    assert service.cancel(booking.id).status == "cancelled"


def test_available_slots_exclude_blocked_interval():
    start = datetime(2026, 10, 2, 10, tzinfo=timezone.utc)
    window = TimeSlot(starts_at=start, ends_at=start + timedelta(hours=1))
    blocked = TimeSlot(starts_at=start, ends_at=start + timedelta(minutes=30))
    slots = build_available_slots([window], [blocked])
    assert len(slots) == 1
    assert slots[0].starts_at == start + timedelta(minutes=30)
