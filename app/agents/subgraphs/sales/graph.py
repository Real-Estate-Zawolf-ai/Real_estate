from langgraph.graph import START, END, StateGraph
from __future__ import annotations
from .state import SalesState
from .state import (
    understand_sales_intent,
    load_sales_context,
    handle_sales_request,
    generate_sales_response
    )


def build_sales_graph():

    builder = StateGraph(
        SalesState
    )

    builder.add_node(
        "understand_sales_intent",
        understand_sales_intent,
    )

    builder.add_node(
        "load_sales_context",
        load_sales_context,
    )

    builder.add_node(
        "handle_sales_request",
        handle_sales_request,
    )

    builder.add_edge(
        START,
        "understand_sales_intent",
    )

    builder.add_edge(
        "understand_sales_intent",
        "load_sales_context",
    )

    builder.add_edge(
        "load_sales_context",
        "handle_sales_request",
    )

    builder.add_edge(
        "handle_sales_request",
        generate_sales_response,
    )

    builder.add_edge(
        "handle_sales_request",
        END,)
    

    return builder.compile()


sales_graph = (
    build_sales_graph()
)