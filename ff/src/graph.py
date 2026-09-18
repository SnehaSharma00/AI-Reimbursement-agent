
from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.models.state import ReimbursementState
from src.nodes.evaluate_policy import evaluate_against_policy
from src.nodes.extract_expenses import extract_expenses
from src.nodes.generate_report import generate_report
from src.nodes.retrieve_policy import retrieve_policy


def build_reimbursement_graph():
    """Construct and compile the reimbursement agent graph."""

    graph = StateGraph(ReimbursementState)

    graph.add_node("ExtractExpenses", extract_expenses)
    graph.add_node("RetrievePolicy", retrieve_policy)
    graph.add_node("EvaluatePolicy", evaluate_against_policy)
    graph.add_node("GenerateReport", generate_report)

    graph.set_entry_point("ExtractExpenses")
    graph.add_edge("ExtractExpenses", "RetrievePolicy")
    graph.add_edge("RetrievePolicy", "EvaluatePolicy")
    graph.add_edge("EvaluatePolicy", "GenerateReport")
    graph.add_edge("GenerateReport", END)

    return graph.compile()
