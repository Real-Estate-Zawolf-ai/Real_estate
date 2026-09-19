class SchedulingService:

    def get_available_slots(
        self,
        date: str,
    ) -> list[dict]:
        ...

    def book(
        self,
        customer_id: str,
        slot: dict,
    ) -> dict:
        ...

    def cancel(
        self,
        appointment_id: str,
    ) -> None:
        ...


scheduling_service = SchedulingService()