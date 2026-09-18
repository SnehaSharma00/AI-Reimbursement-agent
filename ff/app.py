"""
🤖 AI Reimbursement Agent – Streamlit UI
==========================================
This Streamlit app serves as the user interface for the AI Reimbursement Agent.
This version of the project has been enhanced with retrieval‑augmented
generation (RAG).  A small vector store is built from the company policy so
that the evaluator can quickly fetch relevant passages before asking the
language model to make a decision.  To make this work you'll need:

  * an embeddings provider (the sample uses OpenAI)
  
Run:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.graph import build_reimbursement_graph
from src.utils.pdf_reader import extract_text_from_pdf
from src.default_policy import DEFAULT_POLICY

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Reimbursement Agent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ─────────────────────────────────────────────────────── */
html, body, .stApp {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}

/* ── Sidebar ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* ── Header banner ──────────────────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35);
    animation: heroSlide 0.8s ease-out;
}
@keyframes heroSlide {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.hero-header h1 {
    color: #fff;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hero-header p {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    margin: 0;
    font-weight: 400;
}

/* ── Glass card ─────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.6rem;
    margin-bottom: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeUp 0.6s ease-out both;
}
.glass-card:hover {
    background: rgba(255, 255, 255, 0.07);
    border-color: rgba(255, 255, 255, 0.14);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Summary metrics row ────────────────────────────────────────── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.metric-box {
    flex: 1;
    min-width: 150px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    animation: fadeUp 0.7s ease-out both;
}
.metric-box .metric-label {
    color: rgba(255,255,255,0.5);
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}
.metric-box .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #fff;
}
.metric-box.approved  { border-color: rgba(16,185,129,0.4); }
.metric-box.approved .metric-value  { color: #10b981; }
.metric-box.rejected  { border-color: rgba(239,68,68,0.4); }
.metric-box.rejected .metric-value  { color: #ef4444; }
.metric-box.review    { border-color: rgba(245,158,11,0.4); }
.metric-box.review .metric-value    { color: #f59e0b; }
.metric-box.total     { border-color: rgba(99,102,241,0.4); }
.metric-box.total .metric-value     { color: #818cf8; }

/* ── Expense item card ──────────────────────────────────────────── */
.expense-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: all 0.3s ease;
    animation: fadeUp 0.6s ease-out both;
}
.expense-card:hover {
    background: rgba(255,255,255,0.06);
    transform: translateX(4px);
}
.expense-card.status-approved { border-left: 4px solid #10b981; }
.expense-card.status-rejected { border-left: 4px solid #ef4444; }
.expense-card.status-needs_review { border-left: 4px solid #f59e0b; }

.expense-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.7rem;
}
.expense-category {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2e8f0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge {
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.badge-approved     { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-rejected     { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-needs_review { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }

.expense-detail {
    color: rgba(255,255,255,0.6);
    font-size: 0.88rem;
    line-height: 1.7;
}
.expense-detail strong {
    color: rgba(255,255,255,0.85);
    font-weight: 600;
}

/* ── Sidebar styling ────────────────────────────────────────────── */
.sidebar-title {
    color: #818cf8;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    letter-spacing: -0.3px;
}
.sidebar-subtitle {
    color: rgba(255,255,255,0.45);
    font-size: 0.82rem;
    margin-bottom: 1rem;
}

/* ── Overall verdict banner ─────────────────────────────────────── */
.verdict-banner {
    text-align: center;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    font-size: 1.25rem;
    font-weight: 700;
    animation: fadeUp 0.5s ease-out both;
}
.verdict-approved {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.3);
    color: #10b981;
}
.verdict-rejected {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    color: #ef4444;
}
.verdict-partial {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.3);
    color: #f59e0b;
}

/* ── Processing animation ───────────────────────────────────────── */
.processing-box {
    text-align: center;
    padding: 3rem 2rem;
    color: rgba(255,255,255,0.7);
}
.processing-box .spinner {
    display: inline-block;
    width: 40px; height: 40px;
    border: 3px solid rgba(255,255,255,0.1);
    border-top: 3px solid #818cf8;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 1rem;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Button styling ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(102,126,234,0.45) !important;
}

/* ── Radio button styling ───────────────────────────────────────── */
.stRadio > div {
    display: flex;
    gap: 0.5rem;
}
.stRadio label {
    color: rgba(255,255,255,0.8) !important;
}

/* ── Text area, file uploader ───────────────────────────────────── */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 2px rgba(102,126,234,0.2) !important;
}

/* ── Expander ───────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 8px !important;
    color: rgba(255,255,255,0.8) !important;
}

/* ── Hide default Streamlit chrome ──────────────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ── Section label ──────────────────────────────────────────────── */
.section-label {
    color: rgba(255,255,255,0.4);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)


# ── Hero Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>💰 AI Reimbursement Agent</h1>
    <p>Upload your company policy &amp; expenses — get instant, AI-powered approval decisions</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN AREA – Company Policy
# ══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">📋 Company Policy</div>', unsafe_allow_html=True)

# ── Default policy info banner ────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(102,126,234,0.12) 0%, rgba(118,75,162,0.12) 100%);
    border: 1px solid rgba(102,126,234,0.3);
    border-left: 4px solid #667eea;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    color: rgba(255,255,255,0.8);
    line-height: 1.6;
">
    💡 <strong style="color:#818cf8;">Try it instantly</strong> — A sample company policy is built-in.
    Select <em>"✨ Use Default Policy"</em> below to run the agent without uploading anything.
</div>
""", unsafe_allow_html=True)

policy_mode = st.radio(
    "Choose your policy source:",
    ["✨ Use Default Policy", "📝 Paste Policy Text", "📄 Upload Policy PDF"],
    horizontal=True,
    key="policy_mode",
)

policy_text = ""

if policy_mode == "✨ Use Default Policy":
    policy_text = DEFAULT_POLICY
    st.success("✅ Default company policy loaded and ready!", icon="📋")
    with st.expander("📄 Preview Default Policy", expanded=False):
        st.markdown("""
<div style="
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.7);
    white-space: pre-wrap;
    font-family: 'Inter', monospace;
    line-height: 1.7;
    max-height: 340px;
    overflow-y: auto;
">""", unsafe_allow_html=True)
        st.text(DEFAULT_POLICY.strip())
        st.markdown("</div>", unsafe_allow_html=True)

elif policy_mode == "📝 Paste Policy Text":
    policy_text = st.text_area(
        "Paste your company reimbursement policy here",
        height=200,
        placeholder="Paste the full text of your reimbursement policy…",
        key="policy_text_input",
    )
    if policy_text.strip():
        st.success("✅ Policy text received!")

else:  # Upload PDF
    policy_pdf = st.file_uploader(
        "Upload Policy PDF",
        type=["pdf"],
        key="policy_pdf",
        help="Upload a PDF containing your company's reimbursement policy",
    )
    if policy_pdf is not None:
        try:
            policy_text = extract_text_from_pdf(policy_pdf.read())
            st.success("Policy loaded successfully!", icon="✅")
            with st.expander("📄 View Extracted Policy Text", expanded=False):
                st.text(policy_text[:3000] + ("..." if len(policy_text) > 3000 else ""))
        except Exception as e:
            st.error(f"❌ Failed to read PDF: {e}")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════
# MAIN AREA – Expense Input
# ══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">💳 Expense Input</div>', unsafe_allow_html=True)

input_mode = st.radio(
    "Choose how to provide your expenses:",
    ["📝 Paste Text", "📄 Upload PDF"],
    horizontal=True,
    key="input_mode",
)

expense_text = ""

if input_mode == "📝 Paste Text":
    expense_text = st.text_area(
        "Describe your expenses",
        height=160,
        placeholder="e.g.  I had a team lunch costing ₹2000, took a cab for ₹800 to the client office, "
                    "stayed at a hotel for 2 nights at ₹6000/night, and bought stationery for ₹1500.",
        key="expense_text_input",
    )
else:
    expense_pdf = st.file_uploader(
        "Upload Expense PDF",
        type=["pdf"],
        key="expense_pdf",
        help="Upload a PDF containing your expense details",
    )
    if expense_pdf is not None:
        try:
            expense_text = extract_text_from_pdf(expense_pdf.read())
            st.success("✅ Expense PDF loaded!")
            with st.expander("📄 View Extracted Expense Text", expanded=False):
                st.text(expense_text[:3000] + ("..." if len(expense_text) > 3000 else ""))
        except Exception as e:
            st.error(f"❌ Failed to read expense PDF: {e}")

# ── Process Button ───────────────────────────────────────────────────
st.markdown("")  # spacer
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    process_clicked = st.button("🚀  Process Reimbursement", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PROCESSING & RESULTS
# ══════════════════════════════════════════════════════════════════════
if process_clicked:
    # ── Validation ───────────────────────────────────────────────────
    if not policy_text.strip():
        st.warning("⚠️ Please provide a company policy — choose 'Use Default Policy', paste text, or upload a PDF.")
        st.stop()
    if not expense_text.strip():
        st.warning("⚠️ Please provide your expense details (paste text or upload PDF).")
        st.stop()

    # ── Run Agent ────────────────────────────────────────────────────
    processing_placeholder = st.empty()
    processing_placeholder.markdown("""
    <div class="processing-box">
        <div class="spinner"></div>
        <div style="font-size:1.1rem; font-weight:600; color:#818cf8;">Analyzing your expenses…</div>
        <div style="font-size:0.85rem; margin-top:0.3rem;">Our AI agent is reviewing each item against your policy</div>
    </div>
    """, unsafe_allow_html=True)

    agent = build_reimbursement_graph()
    result = agent.invoke({
        "user_input": expense_text,
        "company_policy": policy_text,
    })

    # Clear the spinner once processing is done
    processing_placeholder.empty()

    # ── Parse results ────────────────────────────────────────────────
    eval_results = result.get("evaluation_results", [])

    approved = [r for r in eval_results if r.get("status") in ["approved", "partially_approved"]]
    rejected = [r for r in eval_results if r.get("status") == "rejected"]
    needs_review = [r for r in eval_results if r.get("status") == "needs_review"]

    total_claimed = sum(r.get("claimed_amount", 0) or 0 for r in eval_results)
    total_approved = sum(r.get("approved_amount", 0) or 0 for r in eval_results)
    total_rejected = sum(r.get("rejected_amount", 0) or 0 for r in eval_results)
    total_review = sum(r.get("claimed_amount", 0) or 0 for r in needs_review)


    # ── Overall Verdict Banner ───────────────────────────────────────
    if total_rejected == 0 and total_review == 0:
        verdict_class, verdict_text = "verdict-approved", "✅  All Expenses Approved"
    elif total_approved == 0:
        verdict_class, verdict_text = "verdict-rejected", "❌  All Expenses Rejected"
    else:
        verdict_class, verdict_text = "verdict-partial", "⚠️  Partially Approved"


    st.markdown("---")
    st.markdown(f'<div class="verdict-banner {verdict_class}">{verdict_text}</div>', unsafe_allow_html=True)

    # ── Summary Metric Cards ─────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box total">
            <div class="metric-label">Total Claimed</div>
            <div class="metric-value">₹{total_claimed:,.0f}</div>
        </div>
        <div class="metric-box approved">
            <div class="metric-label">Approved</div>
            <div class="metric-value">₹{total_approved:,.0f}</div>
        </div>
        <div class="metric-box rejected">
            <div class="metric-label">Rejected</div>
            <div class="metric-value">₹{total_rejected:,.0f}</div>
        </div>
        <div class="metric-box review">
            <div class="metric-label">Needs Review</div>
            <div class="metric-value">₹{total_review:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Item-wise Breakdown ──────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:0.5rem;">📊 Item-wise Breakdown</div>', unsafe_allow_html=True)

    for idx, r in enumerate(eval_results):
        status = r.get("status", "needs_review")
        badge_label = status.replace("_", " ").upper()
        amt = r.get("amount")
        amt_str = f"₹{amt:,.2f}" if amt is not None else "N/A"
        currency = r.get("currency", "INR")

        delay = idx * 0.1  # stagger animation
        st.markdown(f"""
        <div class="expense-card status-{status}" style="animation-delay: {delay}s;">
            <div class="expense-header">
                <span class="expense-category">{r.get('category', 'unknown')}</span>
                <span class="badge badge-{status}">{badge_label}</span>
            </div>
            <div class="expense-detail">
                <strong>Amount:</strong> {amt_str} {currency}<br>
                <strong>Description:</strong> {r.get('description', '-')}<br>
                <strong>Reason:</strong> {r.get('reason', '-')}<br>
                {"<strong>Policy Ref:</strong> " + r.get('policy_reference', '') + "<br>" if r.get('policy_reference') and r.get('policy_reference') != 'N/A' else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Raw Report (collapsible) ─────────────────────────────────────
    with st.expander("📃 View Full Text Report", expanded=False):
        st.code(result.get("final_report", ""), language=None)
