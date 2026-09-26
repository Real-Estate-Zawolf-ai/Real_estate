from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .state import SalesState
from .state import (
    generate_sales_response,
    handle_sales_request,
    load_sales_context,
    understand_sales_intent,
)


def build_sales_graph():
    builder = StateGraph(SalesState)

    builder.add_node("understand_sales_intent", understand_sales_intent)
    builder.add_node("load_sales_context", load_sales_context)
    builder.add_node("handle_sales_request", handle_sales_request)
    builder.add_node("generate_sales_response", generate_sales_response)

    builder.add_edge(START, "understand_sales_intent")
    builder.add_edge("understand_sales_intent", "load_sales_context")
    builder.add_edge("load_sales_context", "handle_sales_request")
    builder.add_edge("handle_sales_request", "generate_sales_response")
    builder.add_edge("generate_sales_response", END)

    return builder.compile()


sales_graph = build_sales_graph()
