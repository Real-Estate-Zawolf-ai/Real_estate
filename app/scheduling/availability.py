"""Availability calculations for property-viewing appointments."""

from datetime import datetime, timedelta
from typing import Iterable

from pydantic import BaseModel, model_validator


class TimeSlot(BaseModel):
    """A timezone-aware interval that can be offered for a viewing."""

    starts_at: datetime
    ends_at: datetime

    @model_validator(mode="after")
    def validate_interval(self) -> "TimeSlot":
        if self.starts_at.tzinfo is None or self.ends_at.tzinfo is None:
            raise ValueError("Appointment times must include a timezone.")
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at must be later than starts_at.")
        return self


def overlaps(first: TimeSlot, second: TimeSlot) -> bool:
    """Return whether two half-open time intervals intersect."""
    return first.starts_at < second.ends_at and second.starts_at < first.ends_at


def is_available(candidate: TimeSlot, unavailable: Iterable[TimeSlot]) -> bool:
    """Return whether a candidate slot has no conflict with unavailable time."""
    return not any(overlaps(candidate, blocked) for blocked in unavailable)


def build_available_slots(
    windows: Iterable[TimeSlot],
    unavailable: Iterable[TimeSlot],
    duration_minutes: int = 30,
    interval_minutes: int = 30,
) -> list[TimeSlot]:
    """Split availability windows into bookable, non-conflicting slots.

    ``unavailable`` should include confirmed bookings, temporary holds, agent
    calendar events, and property blackout periods.
    """
    if duration_minutes <= 0 or interval_minutes <= 0:
        raise ValueError("Slot duration and interval must be positive.")

    blocked = list(unavailable)
    duration = timedelta(minutes=duration_minutes)
    interval = timedelta(minutes=interval_minutes)
    slots: list[TimeSlot] = []

    for window in windows:
        starts_at = window.starts_at
        while starts_at + duration <= window.ends_at:
            candidate = TimeSlot(starts_at=starts_at, ends_at=starts_at + duration)
            if is_available(candidate, blocked):
                slots.append(candidate)
            starts_at += interval

    return slots
