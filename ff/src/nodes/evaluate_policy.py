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
    snippet_text = "\n\n".join(snippets)[:3500]

    prompt = f"""You are a logical and precise expense auditor.

You will receive:
  1. A company reimbursement policy.
  2. A list of expense claims.

(For efficiency, a retrieval subsystem has already identified a few
relevant policy passages; they are shown under "RELEVANT POLICY SNIPPETS".)

YOUR TASK:
For EACH expense, decide if it is APPROVED or REJECTED based on the policy.

RULES:
- Evaluate each expense INDEPENDENTLY.
- Base your decision ONLY on what the policy says.
- **NUMERIC FORMAT**: IGNORE commas in numbers. Treat "12,000" as 12000. Treat "11,000" as 11000.
- **RATE CALCULATION**: If an expense is for multiple units (e.g. 3 nights), calculate the PER-UNIT cost:
  - Step 1: Divide Total Amount by Quantity => Per-Unit Rate.
  - Step 2: Find the Policy Limit for that category (ignore commas).
  - Step 3: Compare Rate vs Limit.
    - If Rate <= Limit, Status is APPROVED.
    - If Rate > Limit:
      - Approve up to (Limit × Quantity).
      - Reject only the excess amount.
      - Status should be "partially_approved".
      - Clearly explain approved vs excess amount.

- **TEAM MEALS AND FIXED LIMIT CATEGORIES**:
  - If person count is KNOWN:
      Step 1: Calculate per-person cost.
      Step 2: Compare with per-person limit.
  - If person count is UNKNOWN:
      Compare TOTAL Amount vs Policy Limit.

  - If Amount <= Limit → APPROVED.
  - If Amount > Limit:
      - Approve up to the policy limit.
      - Reject only the excess amount.
      - Status must be "partially_approved".
      - Clearly mention approved and rejected portions.


- Return ONLY a valid JSON array.

EXAMPLES:
Policy: "Hotel limit 12,000/night. Meals 500/day."
Expense: {{"category": "hotel", "amount": 30000, "description": "Hotel for 3 nights"}}
Result: {{"reason": "Rate: 30000/3 = 10000. Limit: 12000. Since 10000 <= 12000, it is approved.", "status": "approved"}}

Expense: {{"category": "hotel", "amount": 28000, "description": "Hotel for 2 nights"}}
Result: {{"reason": "Rate: 28000/2 = 14000. Limit: 12000. Since 14000 > 12000, it is rejected.", "status": "rejected"}}

Expense: {{"category": "meal", "amount": 1500, "description": "Team dinner"}}
Result: {{"reason": "Team dinner count unspecified. Total 1500 exceeds limit of 500. Rejected.", "status": "rejected"}}

Expense: {{"category": "meal", "amount": 400, "description": "Lunch"}}
Result: {{"reason": "Total 400 is within limit of 500. Approved.", "status": "approved"}}

RELEVANT POLICY SNIPPETS (from RAG search):
─────────────────────────────────
{snippet_text}
─────────────────────────────────

EXPENSE CLAIMS:
{expenses_json}

REQUIRED OUTPUT FORMAT (one object per expense, same order):
[
  {{
    "expense_index": 0,
    "category": "meal",
    "claimed_amount": 2000,
    "approved_amount": 1500,
    "rejected_amount": 500,
    "amount": 2000,
    "currency": "INR",
    "description": "Team lunch at restaurant",
    "reason": "Policy allows meal expenses up to INR 1500 per day; this exceeds the limit.",
    "status": "approved | rejected | partially_approved | needs_review",
    "policy_reference": "Section 3.2 – Meal Allowance"
  }}
]
"""
    response = get_llm().invoke(prompt, max_tokens=1500)

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
