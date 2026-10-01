# qualification/graph.py

from langgraph.graph import START, END, StateGraph

from .state import QualificationState

from .state import (
    ask_question,
    check_qualification,
)


def build_qualification_graph():

    builder = StateGraph(
        QualificationState
    )

    builder.add_node(
        "check_qualification",
        check_qualification,
    )

    builder.add_node(
        "ask_question",
        ask_question,
    )

    builder.add_edge(
        START,
        "check_qualification",
    )

    builder.add_conditional_edges(
        "check_qualification",
        lambda state: "complete" if state.get("qualification_complete") else "ask_question",
        {
            "complete": END,
            "ask_question": "ask_question",
        },
    )

    builder.add_edge("ask_question", END)

    return builder.compile()


qualification_graph = (
    build_qualification_graph()
)
