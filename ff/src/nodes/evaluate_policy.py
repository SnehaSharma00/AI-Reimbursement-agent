"""
Node – Evaluate Against Policy
================================
Checks every extracted expense against the company reimbursement policy.
"""

from __future__ import annotations

import json

from src.config import get_llm
from src.models.state import ReimbursementState
from src.utils.json_parser import parse_json
from src.utils.rag import retrieve_policy_snippets


def evaluate_against_policy(state: ReimbursementState) -> dict:
    """Check every extracted expense against the company reimbursement policy."""

    expenses_json = json.dumps(
        state["extracted_expenses"], indent=2, ensure_ascii=False
    )

    # if an earlier graph node already retrieved relevant policy passages
    # we can reuse them.  otherwise fall back to performing the search here
    # directly so that the function is still usable in isolation.
    snippets = state.get("policy_snippets")
    if snippets is None:
        snippets = retrieve_policy_snippets(state["company_policy"], expenses_json)
    snippet_text = "\n\n".join(snippets)[:2000]

    prompt = f"""You are a logical expense auditor. Decide if each expense is APPROVED, REJECTED, or PARTIALLY_APPROVED based strictly on the policy snippets.

RULES:
- Evaluate each expense independently.
- Base decision ONLY on policy snippets below.
- NUMERIC FORMAT: Treat "12,000" as 12000.
- RATE CALCULATION: For multi-night/multi-day expenses, calculate per-unit rate = Total Amount / Quantity.
  - If Rate <= Limit -> APPROVED.
  - If Rate > Limit -> approve (Limit * Quantity), reject excess -> status "partially_approved".
- TEAM MEALS / FIXED LIMITS:
  - If person count known: per-person rate vs limit.
  - If person count unknown: total amount vs limit (approve up to limit, reject excess -> "partially_approved").

RELEVANT POLICY SNIPPETS:
─────────────────────────────────
{snippet_text}
─────────────────────────────────

EXPENSE CLAIMS:
{expenses_json}

RETURN ONLY A VALID JSON ARRAY in this format:
[
  {{
    "expense_index": 0,
    "category": "meal",
    "claimed_amount": 2000,
    "approved_amount": 1500,
    "rejected_amount": 500,
    "amount": 2000,
    "currency": "INR",
    "description": "Team lunch",
    "reason": "Exceeds daily meal limit of 1500; 1500 approved, 500 rejected.",
    "status": "partially_approved",
    "policy_reference": "Section 3.2 – Meal Allowance"
  }}
]
"""
    response = get_llm().invoke(prompt, max_tokens=800)

    try:
        results = parse_json(response.content)
        if isinstance(results, dict):
            results = [results]
    except (json.JSONDecodeError, Exception):
        results = [
            {
                "expense_index": i,
                "category": exp.get("category", "unknown"),
                "amount": exp.get("amount"),
                "currency": exp.get("currency", "INR"),
                "description": exp.get("description", ""),
                "status": "needs_review",
                "reason": "Unable to evaluate – policy check failed. Please review manually.",
                "policy_reference": "N/A",
            }
            for i, exp in enumerate(state["extracted_expenses"])
        ]

    return {"evaluation_results": results, "policy_snippets": snippets}
