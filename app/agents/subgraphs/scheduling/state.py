"""Viewing-scheduling agent.

Multi-turn flow persisted in ``state["scheduling"]``:

    need_property -> need_date -> choose_slot -> need_contact -> confirm -> booked

A booking hold is created only after the customer explicitly confirms, and the
customer is told an advisor still has to confirm it — the assistant never
claims a confirmed visit on its own.
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
    extract_time,
    is_affirmative,
    is_negative,
    resolve_property_reference,
)
from app.scheduling.booking import BookingConflictError


logger = logging.getLogger(__name__)

_AR_WEEKDAYS = ["الاتنين", "التلات", "الأربع", "الخميس", "الجمعة", "السبت", "الحد"]


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


def advance_scheduling(state: SchedulingState) -> dict:
    language = state.get("language", "en")
    query = state.get("query", "")
    flow = dict(state.get("scheduling") or {})
    selected = state.get("selected_property")
    customer = state.get("customer") or {}

    def reply(text: str, **extra: object) -> dict:
        return {"response": text, "scheduling": flow, "active_agent": "scheduling", **extra}

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

    if flow.get("stage") == "confirm":
        if is_negative(query):
            return reply(say(language, "schedule_cancelled"), scheduling={})
        if is_affirmative(query):
            return _book(state, flow, selected, language, reply)
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
            if chosen is None:
                key = "schedule_slot_unavailable" if named_time else "schedule_offer_slots"
                flow["stage"] = "choose_slot"
                flow.pop("slot", None)
                return reply(say(language, key, day=_format_day(day, language), slots=_format_slots(slots)))
            flow["slot"] = chosen
    except Exception:
        logger.exception("Scheduling availability lookup failed")
        return reply(say(language, "schedule_unavailable"))

    # 4. How can the advisor reach the customer? (WhatsApp ids are phone numbers.)
    if not customer.get("phone") and state.get("channel") != "whatsapp":
        flow["stage"] = "need_contact"
        return reply(say(language, "schedule_need_contact"))

    # 5. Explicit confirmation before anything is reserved.
    flow["stage"] = "confirm"
    return reply(say(language, "schedule_confirm", property=label, when=_when(flow["slot"], language)))


def _book(state: SchedulingState, flow: dict, selected: dict, language: str, reply) -> dict:
    slot = flow["slot"]
    customer_id = state.get("customer_id") or "anonymous"
    try:
        appointment = scheduling_service.request_viewing(
            property_id=str(selected["property_id"]),
            customer_id=customer_id,
            starts_at=slot["starts_at"],
            ends_at=slot["ends_at"],
            notes=f"Requested via {state.get('channel', 'website')} ({state.get('modality', 'text')})",
        )
    except BookingConflictError:
        day = date.fromisoformat(flow["day"])
        flow["slots"] = scheduling_service.get_available_slots(selected["property_id"], day)
        flow["stage"] = "choose_slot"
        flow.pop("slot", None)
        return reply(say(language, "schedule_conflict", day=_format_day(day, language), slots=_format_slots(flow["slots"])))
    except Exception:
        logger.exception("Viewing booking failed")
        return reply(say(language, "schedule_unavailable"))

    crm_service.ensure_lead(customer_id, source=state.get("channel", "website"))
    crm_service.upsert_customer(customer_id, **(state.get("customer") or {}))
    crm_service.update_lead(customer_id, {"status": "viewing_requested", "appointment_id": appointment["id"]})
    text = say(
        language,
        "schedule_booked",
        property=property_label(selected),
        when=_when(slot, language),
        ref=appointment["id"][:8],
    )
    return reply(text, scheduling={"stage": "booked"}, appointment=appointment)
