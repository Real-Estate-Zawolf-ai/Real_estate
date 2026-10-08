"""Human-handoff node: package the collected context for a sales advisor."""

from __future__ import annotations

from app.agents.messages import say
from app.agents.state import ConversationState
from app.agents.tools.crm import crm_service
from app.leads.handoff import evaluate_handoff


def handoff_to_human(state: ConversationState) -> dict:
    language = state.get("language", "en")
    channel = state.get("channel", "website")
    customer_id = state.get("customer_id") or "anonymous"
    customer = state.get("customer") or {}
    phone = customer.get("phone") or (customer_id.split(":", 1)[-1] if channel == "whatsapp" else None)

    if not phone:
        # Remember the request so the next message (the number) completes it.
        return {
            "response": say(language, "handoff_need_contact"),
            "handoff": {"status": "awaiting_contact"},
            "active_agent": "handoff",
        }

    qualification = state.get("qualification") or {}
    decision = evaluate_handoff({**qualification, "requested_human": True})
    selected = state.get("selected_property") or {}
    handoff = crm_service.record_handoff(
        customer_id,
        source=channel,
        reason=decision.reason or "customer_requested_human",
        priority=decision.priority,
        summary={
            "contact": {**customer, "phone": phone},
            "requirements": state.get("requirements") or {},
            "qualification": qualification,
            "selected_property_id": selected.get("property_id"),
            "appointment": state.get("appointment"),
            "recent_messages": [
                {"role": getattr(message, "type", ""), "content": getattr(message, "content", "")}
                for message in (state.get("messages") or [])[-6:]
            ],
        },
    )
    crm_service.upsert_customer(customer_id, **customer)
    contact = f" ({phone})"
    return {
        "response": say(language, "handoff_done", contact=contact),
        "handoff": {"status": "requested", "id": handoff["id"], "priority": handoff["priority"]},
        "lead_id": handoff["lead_id"],
        "active_agent": "handoff",
    }
