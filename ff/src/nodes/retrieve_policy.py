"""
Node – Retrieve Policy Snippets
===============================

A simple RAG node that searches the company policy for the current set of
expense claims and stores the top passages in the state.  Downstream nodes can
use these snippets instead of re‑searching the full text.
"""

from __future__ import annotations

import json
from src.models.state import ReimbursementState
from src.utils.rag import retrieve_policy_snippets


def retrieve_policy(state: ReimbursementState) -> dict:
    """Return a dict containing the most relevant policy passages.

    The node turns the list of extracted expenses into a text query and then
    asks the RAG utility to run a similarity search over the stored policy
    document.  Results are saved under ``policy_snippets`` for later nodes.
    """
    expenses_json = json.dumps(state["extracted_expenses"], indent=2, ensure_ascii=False)
    snippets = retrieve_policy_snippets(state["company_policy"], expenses_json)
    return {"policy_snippets": snippets}
