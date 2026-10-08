from __future__ import annotations

import logging

from app.agents.messages import say
from app.agents.state import (
    FILTER_KEYS,
    ConversationState,
    conversation_history,
    describe_requirements,
)


logger = logging.getLogger(__name__)

# Intents answered from the already-selected listing rather than a new search.
DETAIL_INTENTS = {"property_information", "availability", "project_information"}


class PropertyState(
    ConversationState,
    total=False,
):
    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    rewritten_query: str

    # --------------------------------------------------------
    # Structured requirements
    # --------------------------------------------------------

    requirements: dict

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    candidates: list[dict]

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    selected_property: dict | None

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    response: str


def search_filters(requirements: dict) -> dict:
    """Hard metadata filters understood by ``build_metadata_filter``."""
    return {key: requirements[key] for key in FILTER_KEYS if requirements.get(key) not in (None, "")}


def rewrite_query(state: PropertyState) -> str:
    """Carry earlier requirements into the retrieval query text."""
    query = state.get("query", "").strip()
    carried = describe_requirements(state.get("requirements") or {}, {}, "en")
    return f"{query} — {carried}" if carried and carried.lower() not in query.lower() else query


def run_property_workflow(state: PropertyState) -> dict:
    """Execute the same RAG workflow used by HTTP and channel adapters."""
    from app.workflows.rag import answer_about_properties, answer_query

    language = state.get("language", "en")
    requirements = state.get("requirements") or {}
    history = conversation_history(state)
    selected = state.get("selected_property")

    try:
        if state.get("intent") in DETAIL_INTENTS and selected:
            result = answer_about_properties(
                state.get("query", ""),
                [selected],
                requirements,
                conversation_history=history,
            )
            return {
                "response": result["answer"],
                "turn_properties": result["properties"],
                "cited_property_ids": result["cited_property_ids"],
                "grounding_property_ids": result["grounding_property_ids"],
                "active_agent": "property",
            }

        rewritten = rewrite_query(state)
        result = answer_query(
            rewritten,
            search_filters(requirements),
            conversation_history=history,
        )
    except Exception as error:  # provider, vector DB or timeout failures
        logger.exception("Property workflow failed")
        return {
            "response": say(language, "search_unavailable"),
            "error": f"property_search_unavailable: {type(error).__name__}",
            "active_agent": "property",
        }

    return {
        "rewritten_query": rewritten,
        "response": result["answer"],
        "candidates": result["properties"],
        "turn_properties": result["properties"],
        "cited_property_ids": result["cited_property_ids"],
        "grounding_property_ids": result["grounding_property_ids"],
        # A new result list invalidates the previous selection.
        "selected_property": None,
        "active_agent": "property",
    }
