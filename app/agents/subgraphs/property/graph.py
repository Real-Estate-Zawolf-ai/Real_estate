# app/graph/subgraphs/property/graph.py

from langgraph.graph import START, END, StateGraph
from .state import extract_requirements
from .state import PropertyState
from .state import (
    rewrite_query,
    retrieve_properties,
    select_property,
    generate_property_response,
)


def build_property_graph():

    builder = StateGraph(PropertyState)

    # Nodes
    builder.add_node(
    "extract_requirements",
    extract_requirements,
    )

    builder.add_node(
        "rewrite_query",
        rewrite_query,
    )

    builder.add_node(
        "retrieve_properties",
        retrieve_properties,
    )

    builder.add_node(
        "select_property",
        select_property,
    )

    builder.add_node(
        "generate_property_response",
        generate_property_response,
    )
    # Flow
    builder.add_edge(START, "extract_requirements")
    builder.add_edge("extract_requirements", "rewrite_query")
    builder.add_edge("rewrite_query", "retrieve_properties")
    builder.add_edge("retrieve_properties", "select_property")
    builder.add_edge("select_property", "generate_property_response")
    builder.add_edge("generate_property_response", END)
    

    return builder.compile()


property_graph = build_property_graph()


