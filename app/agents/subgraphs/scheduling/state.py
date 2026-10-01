from app.agents.state import ConversationState
from app.agents.tools.scheduling import scheduling_service


class SchedulingState(
    ConversationState,
    total=False,
):

    appointment: dict | None

    requested_date: str
    requested_time: str

    available_slots: list[dict]

    selected_slot: dict | None

def check_availability(state: SchedulingState) -> dict:

    slots = scheduling_service.get_available_slots(
        date=state.get("requested_date", ""),
    ) or []

    return {
        "available_slots": slots,
    }

def select_slot(
    state: SchedulingState,
) -> dict:

    slots = state.get("available_slots", [])

    if not slots:
        return {
            "selected_slot": None,
        }

    selected_slot = slots[0]

    return {
        "selected_slot": selected_slot,
    }

def book_appointment(
    state: SchedulingState,
) -> dict:

    slot = state.get("selected_slot")

    if not slot:
        return {
            "appointment": None,
        }

    appointment = scheduling_service.book(
        customer_id=state.get("customer_id", ""),
        slot=slot,
    )

    return {
        "appointment": appointment,
    }
