from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4


class SchedulingService:

    def __init__(self) -> None:
        self._appointments: dict[str, dict] = {}
        self._lock = Lock()

    def get_available_slots(
        self,
        date: str | None,
    ) -> list[dict]:
        try:
            day = datetime.fromisoformat(date).date() if date else datetime.now(timezone.utc).date()
        except ValueError as error:
            raise ValueError("date must use ISO format YYYY-MM-DD") from error
        return [
            {
                "starts_at": datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).isoformat(),
                "ends_at": datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).replace(minute=30).isoformat(),
            }
            for hour in (10, 12, 14, 16)
        ]

    def book(
        self,
        customer_id: str,
        slot: dict,
    ) -> dict:
        appointment = {
            "id": str(uuid4()),
            "customer_id": customer_id,
            "slot": dict(slot),
            "status": "confirmed",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._appointments[appointment["id"]] = appointment
        return dict(appointment)

    def cancel(
        self,
        appointment_id: str,
    ) -> None:
        with self._lock:
            appointment = self._appointments.get(appointment_id)
            if appointment is None:
                raise KeyError(f"Appointment {appointment_id} was not found")
            appointment["status"] = "cancelled"


scheduling_service = SchedulingService()
