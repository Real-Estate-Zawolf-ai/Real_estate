"""Viewing-scheduling agent.

Multi-turn flow persisted in ``state["scheduling"]``::

    need_property -> need_date -> choose_slot -> need_contact -> verify_contact
        -> confirm -> booked

A booking hold is created only after the customer explicitly confirms *and* has
explicitly verified their phone number, and the customer is told an advisor
still has to confirm the visit - the assistant never claims a confirmed visit on
its own. The same agent cancels a viewing (``cancel_pick`` -> ``cancel_confirm``)
and reports booking status.

Phone verification: a number counts as verified only after the customer says
"yes" to a read-back of that exact number (digit by digit on voice calls), or it
is the sender number of a WhatsApp conversation. Changing the number clears the
verification. The booking service independently refuses unverified numbers.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from app.agents.messages import say
from app.agents.state import ConversationState, property_label
from app.agents.tools.crm import crm_service
from app.agents.tools.scheduling import scheduling_service
from app.agents.understanding import (
    extract_date,
    extract_qualification,
    extract_time,
    is_abort,
    is_affirmative,
    is_negative,
    resolve_property_reference,
)
from app.helpers.phone import display_phone, spoken_phone
from app.scheduling.booking import BookingConflictError, PhoneNotVerifiedError


logger = logging.getLogger(__name__)

_AR_WEEKDAYS = ["الاتنين", "التلات", "الأربع", "الخميس", "الجمعة", "السبت", "الحد"]

# Stages that belong to creating a booking (as opposed to cancelling one).
_BOOKING_STAGES = {"need_property", "need_date", "choose_slot", "need_contact", "verify_contact", "confirm"}
_CANCEL_STAGES = {"cancel_pick", "cancel_confirm"}


class SchedulingState(
    ConversationState,
    total=False,
):

    appointment: dict | None

    scheduling: dict


def _format_day(day: date, language: str) -> str:
    if language == "ar":
        return f"{_AR_WEEKDAYS[day.weekday()]} {day.day}/{day.month}"
    return day.strftime("%A %d %B")


def _format_time(slot: dict) -> str:
    return datetime.fromisoformat(slot["starts_at"]).strftime("%H:%M")


def _format_slots(slots: list[dict]) -> str:
    return ", ".join(_format_time(slot) for slot in slots[:6])


def _when(slot: dict, language: str) -> str:
    starts = datetime.fromisoformat(slot["starts_at"])
    return f"{_format_day(starts.date(), language)} {starts.strftime('%H:%M')}"


def _pick_slot(text: str, slots: list[dict], *, allow_position: bool) -> dict | None:
    # "The first one" means a slot only once slots have been offered; in the
    # turn that picks the property ("the second apartment") it is the property.
    if allow_position:
        by_position = resolve_property_reference(text, [{"property_id": None, **slot} for slot in slots])
        if by_position:
            by_position.pop("property_id", None)
            return by_position
    wanted = extract_time(text)
    if wanted is None:
        return None
    for slot in slots:
        starts = datetime.fromisoformat(slot["starts_at"])
        if (starts.hour, starts.minute) == wanted:
            return slot
    return None


# ---------------------------------------------------------------------------
# Contact verification
# ---------------------------------------------------------------------------


def _contact_stage(customer: dict) -> str:
    """What the customer's phone still needs before a booking may be made."""
    if not customer.get("phone"):
        return "need_contact"
    return "ok" if customer.get("phone_verified") else "verify_contact"


def _phone_for_readback(customer: dict, state: SchedulingState) -> str:
    phone = customer["phone"]
    return spoken_phone(phone) if state.get("modality") == "voice" else display_phone(phone)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def advance_scheduling(state: SchedulingState) -> dict:
    language = state.get("language", "en")
    query = state.get("query", "")
    intent = state.get("intent")
    flow = dict(state.get("scheduling") or {})
    selected = state.get("selected_property")
    customer = dict(state.get("customer") or {})

    def reply(text: str, **extra: object) -> dict:
        return {"response": text, "scheduling": flow, "active_agent": "scheduling", **extra}

    # Cancelling and status questions are independent of the booking flow.
    if flow.get("stage") in _CANCEL_STAGES or intent == "cancel_visit":
        return _cancel_flow(state, flow, language, reply)
    if intent == "booking_status":
        return _status(state, flow, language, reply)

    # "Cancel" / "never mind" at any point abandons the booking in progress.
    if flow.get("stage") in _BOOKING_STAGES and is_abort(query):
        return reply(say(language, "schedule_cancelled"), scheduling={})

    # 1. Which property?
    if not selected:
        if state.get("candidates"):
            flow["stage"] = "need_property"
            return reply(say(language, "schedule_need_property"))
        flow.clear()
        return reply(say(language, "schedule_need_search"))
    if flow.get("property_id") != selected.get("property_id"):
        flow = {"property_id": selected.get("property_id")}
    label = property_label(selected)

    # The customer answers the read-back of their number. A message that carries
    # a new number is not an answer about the old one: it is re-verified below.
    gave_new_number = "phone" in extract_qualification(query)
    if flow.get("stage") == "verify_contact" and customer.get("phone") and not gave_new_number:
        if is_affirmative(query):
            customer = {**customer, "phone_verified": True, "phone_verification_method": "readback"}
        elif is_negative(query):
            customer = {key: value for key, value in customer.items() if key not in {"phone", "phone_verified", "phone_verification_method"}}
            flow["stage"] = "need_contact"
            return reply(say(language, "schedule_phone_rejected"), customer=customer)

    if flow.get("stage") == "confirm":
        if is_negative(query):
            return reply(say(language, "schedule_cancelled"), scheduling={})
        if is_affirmative(query) and _contact_stage(customer) == "ok":
            return _book(state, flow, selected, language, reply, customer)
        # Anything else (e.g. a different time) falls through to re-selection.

    try:
        # 2. Which day?
        today = scheduling_service.today()
        requested = extract_date(query, today)
        if requested:
            if requested < today:
                flow["stage"] = "need_date"
                return reply(say(language, "schedule_bad_date"))
            flow["day"] = requested.isoformat()
            flow.pop("slot", None)
            flow["slots"] = scheduling_service.get_available_slots(selected["property_id"], requested)
        if not flow.get("day"):
            flow["stage"] = "need_date"
            return reply(say(language, "schedule_need_date", property=label))

        day = date.fromisoformat(flow["day"])
        slots = flow.get("slots") or []
        if not slots:
            flow["stage"] = "need_date"
            flow.pop("day", None)
            return reply(say(language, "schedule_no_slots", day=_format_day(day, language)))

        # 3. Which time? (A newly named time also replaces an earlier choice.)
        named_time = extract_time(query)
        if not flow.get("slot") or named_time or flow.get("stage") == "confirm":
            chosen = _pick_slot(query, slots, allow_position=flow.get("stage") in {"choose_slot", "confirm"})
            if chosen is not None:
                flow["slot"] = chosen
            elif flow.get("slot") and not named_time:
                pass  # nothing about the time was said (a corrected number, "ok"...): keep the chosen slot
            else:
                key = "schedule_slot_unavailable" if named_time else "schedule_offer_slots"
                flow["stage"] = "choose_slot"
                flow.pop("slot", None)
                return reply(say(language, key, day=_format_day(day, language), slots=_format_slots(slots)))
    except Exception:
        logger.exception("Scheduling availability lookup failed")
        return reply(say(language, "schedule_unavailable"))

    # 4. A valid, explicitly verified phone number is mandatory.
    stage = _contact_stage(customer)
    if stage == "need_contact":
        flow["stage"] = "need_contact"
        key = "schedule_phone_invalid" if customer.get("phone_invalid") else "schedule_need_contact"
        return reply(say(language, key), customer=customer)
    if stage == "verify_contact":
        flow["stage"] = "verify_contact"
        return reply(
            say(language, "schedule_verify_phone", phone=_phone_for_readback(customer, state)),
            customer=customer,
        )

    # 5. Explicit confirmation before anything is reserved.
    flow["stage"] = "confirm"
    return reply(say(language, "schedule_confirm", property=label, when=_when(flow["slot"], language)), customer=customer)


def _book(state: SchedulingState, flow: dict, selected: dict, language: str, reply, customer: dict) -> dict:
    slot = flow["slot"]
    customer_id = state.get("customer_id") or "anonymous"
    try:
        # Persist the (verified) contact first: the booking service checks the
        # stored customer record, not the chat state.
        crm_service.upsert_customer(customer_id, language=language, **customer)
        appointment = scheduling_service.request_viewing(
            property_id=str(selected["property_id"]),
            customer_id=customer_id,
            starts_at=slot["starts_at"],
            ends_at=slot["ends_at"],
            notes=f"Requested via {state.get('channel', 'website')} ({state.get('modality', 'text')})",
            property_label=property_label(selected),
        )
    except BookingConflictError:
        day = date.fromisoformat(flow["day"])
        flow["slots"] = scheduling_service.get_available_slots(selected["property_id"], day)
        flow["stage"] = "choose_slot"
        flow.pop("slot", None)
        return reply(
            say(language, "schedule_conflict", day=_format_day(day, language), slots=_format_slots(flow["slots"])),
            customer=customer,
        )
    except PhoneNotVerifiedError:
        flow["stage"] = "verify_contact"
        customer = {**customer, "phone_verified": False}
        return reply(
            say(language, "schedule_verify_phone", phone=_phone_for_readback(customer, state)),
            customer=customer,
        )
    except Exception:
        logger.exception("Viewing booking failed")
        return reply(say(language, "schedule_unavailable"), customer=customer)

    text = say(
        language,
        "schedule_booked",
        property=property_label(selected),
        when=_when(slot, language),
        ref=appointment["id"][:8],
    )
    return reply(text, scheduling={"stage": "booked"}, appointment=appointment, customer=customer)


# ---------------------------------------------------------------------------
# Cancellation and status
# ---------------------------------------------------------------------------


def _booking_label(appointment: dict, state: SchedulingState) -> str:
    known = [state.get("selected_property"), *(state.get("candidates") or [])]
    for item in known:
        if item and str(item.get("property_id")) == str(appointment["property_id"]):
            return property_label(item)
    return f"ID {appointment['property_id']}"


def _when_appointment(appointment: dict, language: str) -> str:
    return _when({"starts_at": appointment["starts_at"]}, language)


def _cancel_flow(state: SchedulingState, flow: dict, language: str, reply) -> dict:
    query = state.get("query", "")
    customer_id = state.get("customer_id") or "anonymous"
    stage = flow.get("stage")

    # "Cancel" in the middle of making a new booking abandons that booking.
    if state.get("intent") == "cancel_visit" and stage in _BOOKING_STAGES:
        return reply(say(language, "schedule_cancelled"), scheduling={})

    try:
        active = scheduling_service.list_for_customer(customer_id, active_only=True)
    except Exception:
        logger.exception("Could not load bookings to cancel")
        return reply(say(language, "cancel_failed"), scheduling={})

    if stage == "cancel_confirm":
        target = next((item for item in active if item["id"] == flow.get("cancel_id")), None)
        if target is None:
            return reply(say(language, "cancel_none"), scheduling={})
        if is_negative(query):
            return reply(say(language, "cancel_kept"), scheduling={})
        if is_affirmative(query):
            try:
                cancelled = scheduling_service.cancel(
                    target["id"], customer_id=customer_id, by="customer", reason="cancelled by the customer in chat"
                )
            except Exception:
                logger.exception("Customer cancellation failed")
                return reply(say(language, "cancel_failed"), scheduling={})
            text = say(
                language,
                "cancel_done",
                property=_booking_label(target, state),
                when=_when_appointment(target, language),
            )
            return reply(text, scheduling={}, appointment=cancelled)
        return reply(_confirm_cancel_text(target, state, language))

    if stage == "cancel_pick":
        candidates = [{"property_id": None, **item} for item in active]
        picked = resolve_property_reference(query, candidates)
        if picked is None:
            return reply(say(language, "cancel_pick", items=_list_bookings(active, state, language)))
        target = next(item for item in active if item["id"] == picked["id"])
        flow.update(stage="cancel_confirm", cancel_id=target["id"])
        return reply(_confirm_cancel_text(target, state, language))

    if not active:
        return reply(say(language, "cancel_none"), scheduling={})
    if len(active) == 1:
        flow.update(stage="cancel_confirm", cancel_id=active[0]["id"])
        return reply(_confirm_cancel_text(active[0], state, language))
    flow["stage"] = "cancel_pick"
    return reply(say(language, "cancel_pick", items=_list_bookings(active, state, language)))


def _confirm_cancel_text(appointment: dict, state: SchedulingState, language: str) -> str:
    return say(
        language,
        "cancel_confirm",
        property=_booking_label(appointment, state),
        when=_when_appointment(appointment, language),
        ref=appointment["ref"],
    )


def _list_bookings(bookings: list[dict], state: SchedulingState, language: str) -> str:
    return "\n".join(
        f"{index}. {_booking_label(item, state)} - {_when_appointment(item, language)}"
        for index, item in enumerate(bookings[:5], start=1)
    )


def _status(state: SchedulingState, flow: dict, language: str, reply) -> dict:
    customer_id = state.get("customer_id") or "anonymous"
    try:
        active = scheduling_service.list_for_customer(customer_id, active_only=True)
    except Exception:
        logger.exception("Could not load booking status")
        return reply(say(language, "schedule_unavailable"))
    if not active:
        return reply(say(language, "booking_status_none"))
    lines = []
    for item in active[:5]:
        key = "booking_status_confirmed" if item["status"] == "confirmed" else "booking_status_pending"
        lines.append(
            say(language, key, property=_booking_label(item, state), when=_when_appointment(item, language), ref=item["ref"])
        )
    return reply("\n".join(lines))
