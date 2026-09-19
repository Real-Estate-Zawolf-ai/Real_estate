from langgraph.graph import StateGraph, START, END

from app.graph.state import ConversationState

from .subgraphs.conversational.graph import conversational_graph
from .subgraphs.property.graph import property_graph
from .subgraphs.qualification.graph import qualification_graph
from .subgraphs.scheduling.graph import scheduling_graph
from .state import router, route_agent





def build_graph():

    builder = StateGraph(ConversationState)

    # --------------------------------------------------------
    # Agents
    # --------------------------------------------------------

    builder.add_node(
        "conversational",
        conversational_graph,
    )

    builder.add_node(
        "property",
        property_graph,
    )

    builder.add_node(
        "qualification",
        qualification_graph,
    )

    builder.add_node(
        "scheduling",
        scheduling_graph,
    )

    # --------------------------------------------------------
    # Routing
    # --------------------------------------------------------

    builder.add_node(
        "router",
        router,
    )

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        route_agent,
        {
            "conversational": "conversational",
            "property": "property",
            "qualification": "qualification",
            "scheduling": "scheduling",
        },
    )

    builder.add_edge("conversational", END)
    builder.add_edge("property", END)
    builder.add_edge("qualification", END)
    builder.add_edge("scheduling", END)

    return builder.compile()

graph = build_graph()