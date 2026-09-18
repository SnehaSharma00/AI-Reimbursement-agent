"""
Node – Generate Report
========================
Builds a human-readable verdict report from the evaluation results.
"""

from __future__ import annotations

from src.models.state import ReimbursementState


def generate_report(state: ReimbursementState) -> dict:
    """Build a human-readable verdict report from the evaluation results."""

    results = state["evaluation_results"]

    approved = [r for r in results if r.get("status") in ["approved", "partially_approved"]]
    rejected = [r for r in results if r.get("status") == "rejected"]
    needs_review = [r for r in results if r.get("status") == "needs_review"]

    total_claimed = sum(r.get("claimed_amount", 0) or 0 for r in results)
    total_approved = sum(r.get("approved_amount", 0) or 0 for r in results)
    total_rejected = sum(r.get("rejected_amount", 0) or 0 for r in results)
    total_review = sum(r.get("claimed_amount", 0) or 0 for r in needs_review)


    # Overall verdict (based on financial outcome)
    if total_rejected == 0 and total_review == 0:
        overall = "✅ FULLY APPROVED"
    elif total_approved == 0:
        overall = "❌ FULLY REJECTED"
    else:
        overall = "⚠️  PARTIALLY APPROVED"


    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("       REIMBURSEMENT DECISION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Overall Verdict : {overall}")
    lines.append(f"Total Claimed   : {total_claimed:,.2f}")
    lines.append(f"Total Approved  : {total_approved:,.2f}")
    lines.append(f"Total Rejected  : {total_rejected:,.2f}")
    if total_review > 0:
        lines.append(f"Needs Review    : {total_review:,.2f}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("ITEM-WISE BREAKDOWN")
    lines.append("-" * 60)

    for r in results:
        status_icon = {
        "approved": "✅",
        "rejected": "❌",
        "partially_approved": "⚠️",
        "needs_review": "🔍",
    }.get(
            r.get("status", ""), "❓"
        )
        claimed = r.get("claimed_amount", 0)
        approved_amt = r.get("approved_amount", 0)
        rejected_amt = r.get("rejected_amount", 0)

        amt_str = f"Claimed: {claimed:,.2f} | Approved: {approved_amt:,.2f} | Rejected: {rejected_amt:,.2f} {r.get('currency', 'INR')}"


        lines.append("")
        lines.append(f"  {status_icon}  {r.get('category', 'unknown').upper()}")
        lines.append(f"     Amount      : {amt_str}")
        lines.append(f"     Description : {r.get('description', '-')}")
        lines.append(f"     Status      : {r.get('status', 'unknown').upper()}")
        lines.append(f"     Reason      : {r.get('reason', '-')}")
        policy_ref = r.get("policy_reference", "")
        if policy_ref and policy_ref != "N/A":
            lines.append(f"     Policy Ref  : {policy_ref}")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    return {"final_report": report}
