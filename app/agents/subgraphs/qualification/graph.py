# qualification/graph.py

from langgraph.graph import START, END, StateGraph

from .state import QualificationState

from .state import (
    analyze_lead,
    qualify_lead,
    prepare_crm,
)


def build_qualification_graph():

    builder = StateGraph(
        QualificationState
    )

    builder.add_node(
        "analyze_lead",
        analyze_lead,
    )

    builder.add_node(
        "qualify",
        qualify_lead,
    )

    builder.add_node(
        "prepare_crm",
        prepare_crm,
    )

    builder.add_edge(
        START,
        "analyze_lead",
    )

    builder.add_edge(
        "analyze_lead",
        "qualify",
    )

    builder.add_edge(
        "qualify",
        "prepare_crm",
    )

    builder.add_edge(
        "prepare_crm",
        END,
    )

    return builder.compile()


qualification_graph = (
    build_qualification_graph()
)