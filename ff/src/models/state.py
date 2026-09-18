"""
State Definition
=================
TypedDict that flows through every LangGraph node.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


class ReimbursementState(TypedDict):
    """Graph state that flows through every node."""

    user_input: str                          # Raw expense description from the user
    company_policy: str                      # Full text of the company reimbursement policy
    extracted_expenses: Optional[List[dict]] # List of structured expense objects
    evaluation_results: Optional[List[dict]] # Per-item policy verdicts
    policy_snippets: Optional[List[str]]     # snippets retrieved via RAG (added by RetrievePolicy node)
    final_report: Optional[str]             # Human-readable summary
