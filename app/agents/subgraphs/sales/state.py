from __future__ import annotations

import logging
from typing import Literal

from app.agents.messages import say
from app.agents.state import ConversationState, conversation_history
from app.agents.subgraphs.property.state import search_filters
from app.agents.tools.crm import crm_service


logger = logging.getLogger(__name__)


SalesIntent = Literal[
    "sales_inquiry",
    "price_inquiry",
    "payment_inquiry",
    "offer_inquiry",
    "objection",
    "follow_up",
    "purchase_intent",
]


class SalesState(
    ConversationState,
    total=False,
):
    # --------------------------------------------------------
    # Sales intent
    # --------------------------------------------------------

    sales_intent: SalesIntent

    # --------------------------------------------------------
    # Customer context
    # --------------------------------------------------------

    customer_profile: dict

    # --------------------------------------------------------
    # Property context
    # --------------------------------------------------------

    selected_property: dict | None

    # --------------------------------------------------------
    # Sales context
    # --------------------------------------------------------

    sales_context: dict


    # --------------------------------------------------------
    # Objections
    # --------------------------------------------------------

    objection: str | None

    objection_type: str | None

    # --------------------------------------------------------
    # Offers / payment plans
    # --------------------------------------------------------

    offer: dict | None

    payment_plan: dict | None

    # --------------------------------------------------------
    # Sales action
    # --------------------------------------------------------

    next_action: str | None

    # Example:
    #
    # "provide_payment_plan"
    # "handle_objection"
    # "schedule_viewing"
    # "handoff_to_sales_rep"
    # "continue_conversation"

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    response: str


def understand_sales_intent(
    state: SalesState,
) -> dict:
    """
    Understand the customer's current sales intent.
    """

    query = state.get(
        "query",
        "",
    )

    normalized = query.lower()
    if any(word in normalized for word in ("payment", "installment", "تقسيط")):
        sales_intent = "payment_inquiry"
    elif any(word in normalized for word in ("price", "cost", "سعر")):
        sales_intent = "price_inquiry"
    elif any(word in normalized for word in ("offer", "discount", "عرض", "خصم")):
        sales_intent = "offer_inquiry"
    elif any(word in normalized for word in ("buy", "reserve", "شراء", "احجز", "اشتري")):
        sales_intent = "purchase_intent"
    else:
        sales_intent = "sales_inquiry"

    return {
        "sales_intent": sales_intent,
    }


def load_sales_context(
    state: SalesState,
) -> dict:
    """
    Load customer and sales context.
    """

    customer_id = state.get(
        "customer_id"
    )

    if not customer_id:
        return {}

    customer_profile = crm_service.get_customer(
        customer_id
    )

    return {
        "customer_profile": (
            customer_profile or {}
        ),
    }


def handle_sales_request(
    state: SalesState,
) -> dict:
    """
    Determine the appropriate sales action.
    """

    sales_intent = state.get(
        "sales_intent"
    )

    if sales_intent == "objection":
        next_action = "handle_objection"

    elif sales_intent == "price_inquiry":
        next_action = "provide_price"

    elif sales_intent == "payment_inquiry":
        next_action = "provide_payment_plan"

    elif sales_intent == "offer_inquiry":
        next_action = "provide_offer"

    elif sales_intent == "purchase_intent":
        next_action = "move_to_conversion"

    elif sales_intent == "follow_up":
        next_action = "continue_follow_up"

    else:
        next_action = "continue_sales_conversation"

    return {
        "next_action": next_action,
    }


def generate_sales_response(
    state: SalesState,
) -> dict:
    """
    Generate a grounded sales response.
    """

    from app.workflows.rag import answer_about_properties, answer_query

    language = state.get("language", "en")
    action = state.get("next_action", "continue_sales_conversation")
    selected = state.get("selected_property")
    candidates = state.get("candidates") or []
    history = conversation_history(state)
    requirements = state.get("requirements") or {}

    try:
        # Prices, payment plans and offers are answered only from listing
        # evidence; the generator refuses to calculate or invent them.
        if selected or candidates:
            result = answer_about_properties(
                state.get("query", ""),
                [selected] if selected else candidates[:5],
                requirements,
                conversation_history=history,
            )
        else:
            result = answer_query(
                state.get("query", ""),
                search_filters(requirements),
                conversation_history=history,
            )
    except Exception as error:
        logger.exception("Sales workflow failed")
        return {
            "response": say(language, "search_unavailable"),
            "error": f"sales_unavailable: {type(error).__name__}",
            "active_agent": "sales",
        }

    response = result["answer"]
    update: dict = {}
    if not (selected or candidates):
        # This turn ran a search: keep the results so "the first one" works next.
        update["candidates"] = result["properties"]
    customer_id = state.get("customer_id")
    if action in {"move_to_conversion", "provide_offer"}:
        response = f"{response}\n\n{say(language, 'sales_handoff_offer')}"
        if customer_id:
            update["lead_id"] = crm_service.ensure_lead(customer_id, source=state.get("channel", "website"))
            crm_service.update_lead(customer_id, {"status": "hot", "sales_intent": state.get("sales_intent")})

    return {
        "response": response,
        "turn_properties": result["properties"],
        "cited_property_ids": result["cited_property_ids"],
        "grounding_property_ids": result["grounding_property_ids"],
        "active_agent": "sales",
        **update,
    }
