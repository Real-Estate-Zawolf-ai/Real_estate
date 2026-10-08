# scheduling/graph.py

from langgraph.graph import START, END, StateGraph

from .state import SchedulingState, advance_scheduling


def build_scheduling_graph():

    builder = StateGraph(
        SchedulingState
    )

    # One step per turn: the flow position lives in state["scheduling"],
    # so each customer reply advances the booking by one stage.
    builder.add_node(
        "advance_scheduling",
        advance_scheduling,
    )

    builder.add_edge(
        START,
        "advance_scheduling",
    )

    builder.add_edge(
        "advance_scheduling",
        END,
    )

    return builder.compile()


scheduling_graph = build_scheduling_graph()
