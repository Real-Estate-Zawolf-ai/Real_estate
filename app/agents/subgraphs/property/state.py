from __future__ import annotations

from typing import Any

from app.agents.state import ConversationState
from app.agents.tools.property import property_search_service


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


def extract_requirements(
    state: PropertyState,
    property_llm: Any | None = None,
) -> dict:
    """
    Extract structured property requirements
    from the user's query.
    """

    query = state.get("query", "")

    if property_llm is None:
        return {"requirements": state.get("requirements", {})}

    requirements = property_llm.extract_requirements(query=query)

    return {
        "requirements": requirements,
    }


def rewrite_query(
    state: PropertyState,
    property_llm: Any | None = None,
) -> dict:
    """
    Rewrite the user's query into a retrieval-friendly query.
    """

    query = state.get("query", "")

    requirements = state.get(
        "requirements",
        {},
    )

    if property_llm is None:
        return {"rewritten_query": query}

    rewritten_query = property_llm.rewrite_query(query=query, requirements=requirements)

    return {
        "rewritten_query": rewritten_query,
    }


def retrieve_properties(
    state: PropertyState,
) -> dict:
    """
    Retrieve candidate properties using
    the property search service.
    """

    query = state.get(
        "rewritten_query",
        state.get("query", ""),
    )

    requirements = state.get(
        "requirements",
        {},
    )

    candidates = property_search_service.search(
        query=query,
        filters=requirements,
    )

    return {
        "candidates": candidates,
    }


def select_property(
    state: PropertyState,
    property_llm: Any | None = None,
) -> dict:
    """
    Select the most relevant property from
    the retrieved candidates.
    """

    candidates = state.get(
        "candidates",
        [],
    )

    if not candidates:
        return {
            "selected_property": None,
        }

    if property_llm is None:
        return {"selected_property": candidates[0]}

    selected_property = property_llm.select_property(
        candidates=candidates,
        requirements=state.get("requirements", {}),
    )

    return {
        "selected_property": selected_property,
    }


def generate_property_response(
    state: PropertyState,
    property_llm: Any | None = None,
) -> dict:
    """
    Generate the final grounded response.
    """

    selected_property = state.get(
        "selected_property"
    )

    if not selected_property:
        return {
            "response": "No suitable property found."
        }

    if property_llm is None:
        return {"response": "A matching property was found."}

    response = property_llm.generate_response(
        selected_property=selected_property,
        requirements=state.get("requirements", {}),
    )

    return {
        "response": response,
    }
