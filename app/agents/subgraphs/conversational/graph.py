from langgraph.graph import StateGraph, START, END

from app.agents.messages import say
from app.agents.state import describe_requirements
from app.agents.subgraphs.conversational.state import ConversationalState


def generate_response(
    state: ConversationalState,
) -> dict:

    language = state.get("language", "en")
    if state.get("intent") == "greeting":
        response = say(language, "greeting")
    else:
        response = say(language, "fallback")

    # Remind the customer what is already known instead of starting over.
    summary = describe_requirements(state.get("requirements") or {}, state.get("qualification") or {}, language)
    if summary:
        response = f"{response} ({say(language, 'noted', summary=summary).rstrip('.')})"

    return {
        "response": response,
        "active_agent": "conversational",
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
