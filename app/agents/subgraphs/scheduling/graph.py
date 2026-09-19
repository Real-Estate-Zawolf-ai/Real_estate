# scheduling/graph.py

from langgraph.graph import START, END, StateGraph

from .state import SchedulingState

from .state import (
    check_availability,
    select_slot,
    book_appointment,
)


def build_scheduling_graph():

    builder = StateGraph(
        SchedulingState
    )

    builder.add_node(
        "check_availability",
        check_availability,
    )

    builder.add_node(
        "select_slot",
        select_slot,
    )

    builder.add_node(
        "book",
        book_appointment,
    )

    builder.add_edge(
        START,
        "check_availability",
    )

    builder.add_edge(
        "check_availability",
        "select_slot",
    )

    builder.add_edge(
        "select_slot",
        "book",
    )

    builder.add_edge(
        "book",
        END,
    )

    return builder.compile()


scheduling_graph = build_scheduling_graph()