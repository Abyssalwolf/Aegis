"""
LangGraph graph definition for the Police Investigation System.

Flow:
  START
    → router_node            (decides which agent to call)
    → [specialist agent]     (fir_node | case_diary_node | ... )
    → supervisor_node        (reads blackboard, cross-checks)
  END

The blackboard is NOT a LangGraph edge — it's a side-channel via Redis.
Agents write to Redis; the supervisor reads from Redis.
LangGraph manages the routing logic and state accumulation.
"""

from __future__ import annotations
from typing import Any

from langgraph.graph import StateGraph, START, END

from orchestration.graph.state import InvestigationState, FILE_TYPE
from agents.specialists import AGENT_REGISTRY
from agents.supervisor import SupervisorAgent

# ─── Node functions ───────────────────────────────────────────────────────────

def router_node(state: InvestigationState) -> dict[str, Any]:
    """
    Determines which specialist agent should handle the uploaded file.
    In a real system this could also call an LLM classifier on the file content.
    """
    agent = AGENT_REGISTRY.get(state.file_type)
    if agent is None:
        raise ValueError(f"No agent registered for file_type='{state.file_type}'")
    return {"assigned_agent": agent.agent_id}


def _route_to_agent(state: InvestigationState) -> str:
    """
    LangGraph conditional edge function.
    Returns the node name to transition to based on file_type.
    """
    return f"{state.file_type}_node"


def make_agent_node(file_type: str):
    """Factory: wraps each specialist agent as a named LangGraph node."""
    agent = AGENT_REGISTRY[file_type]

    def node_fn(state: InvestigationState) -> dict[str, Any]:
        return agent(state)

    node_fn.__name__ = f"{file_type}_agent_node"
    return node_fn


# ─── Build graph ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph StateGraph.

    Graph topology:
      START → router → (conditional) → [one of 7 agent nodes] → supervisor → END
    """
    graph = StateGraph(InvestigationState)

    # --- Add nodes ---
    graph.add_node("router", router_node)

    file_types: list[FILE_TYPE] = [
        "fir", "case_diary", "statement", "scene_of_crime",
        "forensic", "seizure", "arrest_remand",
    ]

    for ft in file_types:
        graph.add_node(f"{ft}_node", make_agent_node(ft))

    supervisor = SupervisorAgent()
    graph.add_node("supervisor", supervisor)

    # --- Add edges ---
    graph.add_edge(START, "router")

    # Conditional: router → one of the 7 agent nodes
    graph.add_conditional_edges(
        "router",
        _route_to_agent,
        {f"{ft}_node": f"{ft}_node" for ft in file_types},
    )

    # All agent nodes → supervisor
    for ft in file_types:
        graph.add_edge(f"{ft}_node", "supervisor")

    graph.add_edge("supervisor", END)

    return graph.compile()


# Singleton compiled graph
investigation_graph = build_graph()
