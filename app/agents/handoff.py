"""Human-handoff node: package the collected context for a sales advisor."""

from __future__ import annotations

import logging

from app.agents.messages import say
from app.agents.state import ConversationState
from app.agents.tools.crm import crm_service
from app.helpers.phone import display_phone
from app.leads.handoff import evaluate_handoff
from app.leads.scoring import calculate_lead_score


logger = logging.getLogger(__name__)


def handoff_to_human(state: ConversationState) -> dict:
    language = state.get("language", "en")
    channel = state.get("channel", "website")
    customer_id = state.get("customer_id") or "anonymous"
    customer = dict(state.get("customer") or {})
    # E.164 and already verified on WhatsApp (the router pre-fills the sender).
    phone = customer.get("phone")

    if not phone:
        # Remember the request so the next message (the number) completes it.
        key = "schedule_phone_invalid" if customer.get("phone_invalid") else "handoff_need_contact"
        return {
            "response": say(language, key),
            "handoff": {"status": "awaiting_contact"},
            "active_agent": "handoff",
        }

    qualification = state.get("qualification") or {}
    decision = evaluate_handoff({**qualification, "requested_human": True})
    selected = state.get("selected_property") or {}
    try:
        # Store the contact before the ticket so the staff alert has a name.
        crm_service.upsert_customer(customer_id, language=language, **customer)
        handoff = crm_service.record_handoff(
            customer_id,
            source=channel,
            reason=decision.reason or "customer_requested_human",
            priority=decision.priority,
            summary={
                "contact": {
                    "name": customer.get("name"),
                    "phone": phone,
                    "phone_verified": bool(customer.get("phone_verified")),
                    "email": customer.get("email"),
                },
                "requirements": state.get("requirements") or {},
                "qualification": qualification,
                "lead_score": calculate_lead_score(qualification),
                "selected_property_id": selected.get("property_id"),
                "appointment": state.get("appointment"),
                "recent_messages": [
                    {"role": getattr(message, "type", ""), "content": str(getattr(message, "content", ""))[:500]}
                    for message in (state.get("messages") or [])[-6:]
                ],
            },
        )
    except Exception:
        logger.exception("Handoff could not be recorded | customer=%s", customer_id[:24])
        return {
            "response": say(language, "schedule_unavailable"),
            "error": "handoff_unavailable",
            "active_agent": "handoff",
        }
    return {
        "response": say(language, "handoff_done", contact=f" ({display_phone(phone)})"),
        "handoff": {"status": "requested", "id": handoff["id"], "priority": handoff["priority"]},
        "lead_id": handoff["lead_id"],
        "active_agent": "handoff",
    }
