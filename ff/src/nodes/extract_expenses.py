"""
Node – Extract Expenses
========================
Parses user input into a list of structured expense objects.
Handles MULTIPLE expenses in a single message.
"""

from __future__ import annotations

import json

from src.config import get_llm
from src.models.state import ReimbursementState
from src.utils.json_parser import parse_json


def extract_expenses(state: ReimbursementState) -> dict:
    """Parse user input into a list of structured expense objects."""

    prompt = f"""You are an expert expense-report parser.

Your job is to extract ALL individual expenses mentioned in the user's text.
A single message may contain one expense or many expenses.

RULES:
- Extract ONLY items where the user is claiming a specific monetary amount.
- **AMOUNT: ALWAYS extract the TOTAL claimed value.**
  - "36000 for 3 nights" -> Amount: 36000
  - "12000 per night for 3 nights" -> Amount: 36000 (12000 * 3)
  - "Taxi up to 500" -> Amount: 500
  - "Meal expense 500 per day for 3 nights" -> Amount: 1500 (500 * 3)
  - "Meal expense 5000 for 4 days" -> Amount: 5000
- **DESCRIPTION**: Include details needed for policy checks (e.g. duration, people count).
  - "36000 for 3 nights" -> Description: "Hotel stay for 3 nights"
  - "Dinner for 5 people" -> Description: "Dinner for 5 team members"
  - "Meal expense 5000 for 4 days" -> Description: "Meal expense for 4 days"
- If an item does NOT have a concrete amount, SKIP it.
- If the user does not specify a currency, default to "INR".
- Do NOT invent expenses.
- Return ONLY a valid JSON array.

EXAMPLES:
Input: "Stayed at Hotel X for 3 nights costing 12000 per night. Lunch was 500."
Output: [
  {{"category": "hotel", "amount": 36000, "currency": "INR", "description": "Hotel X for 3 nights @ 12000/night"}},
  {{"category": "meal", "amount": 500, "currency": "INR", "description": "Lunch"}}
]

USER INPUT:
\"\"\"{state['user_input']}\"\"\"

REQUIRED OUTPUT FORMAT (JSON array):
[
  {{
    "category": "hotel",
    "amount": 36000,
    "currency": "INR",
    "description": "Hotel stay for 3 nights"
  }}
]
"""
    response = get_llm().invoke(prompt, max_tokens=500)

    try:
        expenses = parse_json(response.content)
        if isinstance(expenses, dict):
            expenses = [expenses]
    except (json.JSONDecodeError, Exception):
        expenses = [
            {
                "category": "unknown",
                "amount": None,
                "currency": "INR",
                "description": state["user_input"],
            }
        ]

    return {"extracted_expenses": expenses}
