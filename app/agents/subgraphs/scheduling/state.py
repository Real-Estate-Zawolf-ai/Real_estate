from app.graph.state import ConversationState


class SchedulingState(
    ConversationState,
    total=False,
):

    appointment: dict | None

    requested_date: str
    requested_time: str

    available_slots: list[dict]

    selected_slot: dict | None

def check_availability(
    state: SchedulingState,scheduling_service
) -> dict:

    slots = scheduling_service.get_available_slots(
        date=state.get("requested_date"),
    )

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
    scheduling_service,
) -> dict:

    slot = state.get("selected_slot")

    if not slot:
        return {
            "appointment": None,
        }

    appointment = scheduling_service.book(
        customer_id="customer_123",
        slot=slot,
    )

    return {
        "appointment": appointment,
    }