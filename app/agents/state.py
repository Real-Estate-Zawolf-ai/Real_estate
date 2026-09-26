from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


Channel = Literal[
    "website",
    "whatsapp",
]

Language = Literal[
    "ar",
    "en",
]


class ConversationState(TypedDict, total=False):


    messages: Annotated[list[BaseMessage], add_messages]

    query: str
    rewritten_query: str

    channel: Channel
    language: Language

    # --------------------------------------------------------
    # Routing
    # --------------------------------------------------------

    intent: str
    next_agent: str

    # --------------------------------------------------------
    # Customer
    # --------------------------------------------------------

    customer_id: str

    # --------------------------------------------------------
    # Property
    # --------------------------------------------------------

    requirements: dict
    candidates: list[dict]
    selected_property: dict | None

    # --------------------------------------------------------
    # Qualification
    # --------------------------------------------------------

    qualification: dict

    # --------------------------------------------------------
    # Scheduling
    # --------------------------------------------------------

    appointment: dict | None

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    response: str

def router (state: ConversationState) -> dict:
    """
    Determine the next agent to handle the conversation
    based on the current state.
    """

    return {
        "intent": "poperty_search"
    }

def route_agent(state: ConversationState) -> str:


    intent = state.get("intent")

    if intent == "property_search":
        return "property"

    if intent == "qualification":
        return "qualification"

    if intent == "scheduling":
        return "scheduling"

    return "conversational"