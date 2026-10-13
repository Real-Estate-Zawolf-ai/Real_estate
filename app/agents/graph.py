from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END

from app.agents.state import ConversationState

from .handoff import handoff_to_human
from .subgraphs.conversational.graph import conversational_graph
from .subgraphs.property.graph import property_graph
from .subgraphs.qualification.graph import qualification_graph
from .subgraphs.sales.graph import sales_graph
from .subgraphs.scheduling.graph import scheduling_graph
from .state import router, route_agent


# Node names carry an "_agent" suffix because LangGraph forbids a node and a
# state key ("qualification", "handoff") sharing a name.
AGENTS = ("conversational", "property", "qualification", "scheduling", "sales", "handoff")


# Older turns are dropped from the stored state: prompts only ever use the last
# few messages, and an unbounded list would bloat every checkpoint.
MAX_STORED_MESSAGES = 60


def record_reply(state: ConversationState) -> dict:
    """Append the agent's reply to the checkpointed conversation history."""
    response = state.get("response") or ""
    update: list = [AIMessage(content=response)] if response else []
    history = state.get("messages") or []
    overflow = len(history) + len(update) - MAX_STORED_MESSAGES
    if overflow > 0:
        update = [RemoveMessage(id=message.id) for message in history[:overflow] if message.id] + update
    return {"messages": update} if update else {}


def build_graph(checkpointer=None):

    builder = StateGraph(ConversationState)

    # --------------------------------------------------------
    # Agents
    # --------------------------------------------------------

    builder.add_node(
        "conversational_agent",
        conversational_graph,
    )

    builder.add_node(
        "property_agent",
        property_graph,
    )

    builder.add_node(
        "qualification_agent",
        qualification_graph,
    )

    builder.add_node(
        "scheduling_agent",
        scheduling_graph,
    )

    builder.add_node(
        "sales_agent",
        sales_graph,
    )

    builder.add_node(
        "handoff_agent",
        handoff_to_human,
    )

    # --------------------------------------------------------
    # Routing
    # --------------------------------------------------------

    builder.add_node(
        "router",
        router,
    )

    builder.add_node(
        "record_reply",
        record_reply,
    )

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        route_agent,
        {agent: f"{agent}_agent" for agent in AGENTS},
    )

    for agent in AGENTS:
        builder.add_edge(f"{agent}_agent", "record_reply")
    builder.add_edge("record_reply", END)

    # With a checkpointer, state is persisted per ``thread_id`` so every turn
    # (text, voice or WhatsApp) continues the same customer conversation.
    return builder.compile(checkpointer=checkpointer)

graph = build_graph()
