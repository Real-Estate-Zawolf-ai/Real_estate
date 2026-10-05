from __future__ import annotations

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


def run_property_workflow(state: PropertyState) -> dict:
    """Execute the same RAG workflow used by HTTP and channel adapters."""
    from app.workflows.rag import answer_query

    result = answer_query(
        state.get("query", ""),
        state.get("requirements", {}),
    )
    return {
        "response": result["answer"],
        "candidates": result["properties"],
        "cited_property_ids": result["cited_property_ids"],
        "grounding_property_ids": result["grounding_property_ids"],
    }


def extract_requirements(
    state: PropertyState,
) -> dict:
    """
    Extract structured property requirements
    from the user's query.
    """

    query = state.get("query", "")

    requirements = property_search_service.extract_requirements(query)

    return {
        "requirements": requirements,
    }


def rewrite_query(
    state: PropertyState,
) -> dict:
    """
    Rewrite the user's query into a retrieval-friendly query.
    """

    query = state.get("query", "")

    requirements = state.get(
        "requirements",
        {},
    )

    rewritten_query = property_search_service.rewrite_query(query, requirements)

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

    selected_property = property_search_service.select(
        candidates,
        state.get("requirements", {}),
    )

    return {
        "selected_property": selected_property,
    }


def generate_property_response(
    state: PropertyState,
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

    response = property_search_service.format_response(selected_property)

    return {
        "response": response,
    }
