# app/graph/subgraphs/property/graph.py

from langgraph.graph import START, END, StateGraph
from .state import PropertyState, run_property_workflow


def build_property_graph():

    builder = StateGraph(PropertyState)

    # Nodes
    builder.add_node("property_workflow", run_property_workflow)
    builder.add_edge(START, "property_workflow")
    builder.add_edge("property_workflow", END)
    

    return builder.compile()


property_graph = build_property_graph()


