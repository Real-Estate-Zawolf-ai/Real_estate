from langgraph.graph import StateGraph, START, END

from app.agents.subgraphs.conversational.state import ConversationalState


def generate_response(
    state: ConversationalState,
) -> dict:

    query = state["query"]

    response = f"Response to: {query}"

    return {
        "response": response,
    }


def build_graph():

    builder = StateGraph(
        ConversationalState
    )

    builder.add_node(
        "generate_response",
        generate_response,
    )

    builder.add_edge(
        START,
        "generate_response",
    )

    builder.add_edge(
        "generate_response",
        END,
    )

    return builder.compile()


conversational_graph = build_graph()
