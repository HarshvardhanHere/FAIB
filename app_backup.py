import os
import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from policy_rag import PolicyRAG

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "logistic_regression_pipeline.joblib"
POLICY_PATH = APP_DIR / "lending_policy.md"
DATA_PATH = next(iter(APP_DIR.glob("bharat_horizon_home_loan_dataset_v2*.xlsx")), None)
LOG_PATH = APP_DIR / "review_log.csv"

st.set_page_config(
    page_title="LoanLens | Bharat Horizon Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Theme CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global ─────────────────────────────────────────────────────── */
html, body, .stApp {
    background-color: #F8F3FA !important;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: #30213A;
}
.main .block-container {
    padding-top: 1.25rem;
    padding-bottom: 2.5rem;
    max-width: 1120px;
}

/* ── Hide default Streamlit chrome ───────────────────────────────── */
#MainMenu            { visibility: hidden; }
footer               { visibility: hidden; }
[data-testid="stHeader"]     { display: none !important; }
[data-testid="stToolbar"]    { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(170deg, #4B176F 0%, #370F50 100%);
    border-right: none;
    box-shadow: 3px 0 20px rgba(75, 23, 111, 0.22);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] small {
    color: rgba(255, 255, 255, 0.87) !important;
}
[data-testid="stSidebar"] .stCaption p,
[data-testid="stSidebar"] [data-testid="stCaption"] p {
    color: rgba(255, 255, 255, 0.40) !important;
    font-size: 0.71rem !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.16) !important;
    margin: 0.75rem 0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    padding: 6px 10px;
    border-radius: 8px;
    transition: background 0.18s ease;
    display: block;
    line-height: 1.5;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255, 255, 255, 0.10);
}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
    gap: 6px;
}

/* ── Typography ──────────────────────────────────────────────────── */
h1, h2, h3 {
    color: #30213A;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
h1 { font-size: 1.55rem; font-weight: 700; margin-bottom: 0.1rem; }
h2 { font-size: 1.2rem;  font-weight: 700; margin-bottom: 0.15rem; }
h3 { font-size: 1.0rem;  font-weight: 600; color: #4B176F; margin-bottom: 0.1rem; }

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button {
    background-color: #4B176F;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 0.52rem 1.5rem;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    transition: background-color 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 2px 8px rgba(75, 23, 111, 0.24);
    cursor: pointer;
}
.stButton > button:hover {
    background-color: #8E2485;
    color: #ffffff;
    box-shadow: 0 4px 14px rgba(142, 36, 133, 0.34);
}
.stButton > button:active { background-color: #3a1050; }
.stButton > button:focus  { outline: 2px solid #8E2485; outline-offset: 2px; }

/* Form submit button */
[data-testid="stFormSubmitButton"] > button {
    background-color: #4B176F !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.62rem 2rem !important;
    box-shadow: 0 2px 10px rgba(75, 23, 111, 0.24) !important;
    transition: background-color 0.2s ease, box-shadow 0.2s ease !important;
    cursor: pointer !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #8E2485 !important;
    box-shadow: 0 4px 14px rgba(142, 36, 133, 0.34) !important;
}

/* ── Metric cards ────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E8DDED;
    border-radius: 12px;
    padding: 1rem 1.25rem 0.9rem;
    box-shadow: 0 2px 8px rgba(75, 23, 111, 0.07);
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span {
    color: #756B7C !important;
    font-size: 0.73rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #30213A !important;
    font-size: 1.38rem !important;
    font-weight: 700 !important;
}

/* ── Form container ──────────────────────────────────────────────── */
[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E8DDED;
    border-radius: 14px;
    padding: 1.5rem 1.75rem 1.25rem;
    box-shadow: 0 2px 12px rgba(75, 23, 111, 0.06);
}

/* ── Text / Number inputs and Textarea ───────────────────────────── */
.stTextInput  > div > div > input,
.stNumberInput > div > div > input,
.stTextArea   textarea {
    border: 1.5px solid #E8DDED !important;
    border-radius: 8px !important;
    background: #FDFAFF !important;
    color: #30213A !important;
    font-size: 0.88rem !important;
    padding: 0.42rem 0.75rem !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
.stTextInput  > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea   textarea:focus {
    border-color: #4B176F !important;
    box-shadow: 0 0 0 3px rgba(75, 23, 111, 0.10) !important;
    outline: none !important;
}

/* ── Selectbox ───────────────────────────────────────────────────── */
.stSelectbox > div > div {
    border: 1.5px solid #E8DDED !important;
    border-radius: 8px !important;
    background-color: #FDFAFF !important;
}

/* ── Slider accent ───────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    background-color: #4B176F !important;
    border-color: #4B176F !important;
}

/* ── Expanders ───────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E8DDED !important;
    border-radius: 10px !important;
    background: #FFFFFF;
    overflow: hidden;
    margin-bottom: 0.55rem;
}
[data-testid="stExpander"] summary {
    color: #4B176F !important;
    font-weight: 600 !important;
    font-size: 0.87rem !important;
    padding: 0.62rem 1rem !important;
}
[data-testid="stExpander"] summary:hover {
    background-color: #F8F3FA;
}
[data-testid="stExpander"] details[open] summary {
    border-bottom: 1px solid #E8DDED;
}

/* ── Alerts ──────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.87rem !important;
}

/* ── Dividers ────────────────────────────────────────────────────── */
hr {
    border-color: #E8DDED !important;
    margin: 1.2rem 0 !important;
}

/* ── Captions ────────────────────────────────────────────────────── */
[data-testid="stCaption"] p,
.stCaption p {
    color: #756B7C;
    font-size: 0.78rem;
}

/* ── JSON viewer ─────────────────────────────────────────────────── */
[data-testid="stJson"] {
    border: 1px solid #E8DDED;
    border-radius: 10px;
    background: #FDFAFF;
    padding: 0.5rem 0.75rem;
    font-size: 0.82rem;
}

/* ── Custom helper classes ───────────────────────────────────────── */
.bh-label {
    font-size: 0.67rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #8E2485;
    margin: 0 0 0.1rem 0;
    padding: 0;
    line-height: 1.4;
}
.bh-disclaimer {
    background: #FFF8E1;
    border: 1px solid #FFE082;
    border-left: 4px solid #FFC107;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    margin-bottom: 1.2rem;
    font-size: 0.81rem;
    color: #5D4037;
    line-height: 1.5;
}
.bh-info-row {
    background: #FFFFFF;
    border: 1px solid #E8DDED;
    border-radius: 12px;
    padding: 0.9rem 1.25rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 1px 6px rgba(75, 23, 111, 0.06);
    font-size: 0.88rem;
    color: #30213A;
    line-height: 1.6;
}
.bh-section-gap { margin-top: 1.25rem; }
</style>
""", unsafe_allow_html=True)

# ── App header banner ─────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(95deg, #4B176F 0%, #8E2485 100%);
    border-radius: 14px;
    padding: 1.1rem 1.75rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1.1rem;
    box-shadow: 0 4px 18px rgba(75, 23, 111, 0.22);
">
    <div style="font-size: 2.2rem; line-height: 1; flex-shrink: 0;">🏦</div>
    <div>
        <div style="color: rgba(255,255,255,0.65); font-size: 0.66rem; font-weight: 700;
                    letter-spacing: 0.13em; text-transform: uppercase; line-height: 1;">
            Bharat Horizon Bank
        </div>
        <div style="color: #fff; font-size: 1.45rem; font-weight: 700; line-height: 1.25;
                    font-family: 'Inter', 'Segoe UI', sans-serif; margin-top: 0.14rem;">
            LoanLens
        </div>
        <div style="color: rgba(255,255,255,0.55); font-size: 0.75rem; margin-top: 0.12rem;">
            AI-assisted home-loan assessment and policy explanation
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Educational disclaimer
st.markdown("""
<div class="bh-disclaimer">
    ⚠️ <strong>EDUCATIONAL PROTOTYPE</strong> — Uses synthetic data and a fictional policy.
    Not financial advice and not suitable for real lending decisions.
</div>
""", unsafe_allow_html=True)

# ── File checks ───────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    st.error("Trained model file is missing. Place logistic_regression_pipeline.joblib beside app.py.")
    st.stop()
if not POLICY_PATH.exists():
    st.error("lending_policy.md is missing. Place it beside app.py.")
    st.stop()

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_rag():
    return PolicyRAG(POLICY_PATH)

model = load_model()
rag = load_rag()

# ── Utilities ─────────────────────────────────────────────────────────────────
def fmt_inr(value):
    return f"₹{value:,.0f}"

def policy_checks(applicant):
    """Transparent deterministic policy checks; independent of model prediction."""
    flags = []
    hard_reject = []
    if applicant["monthly_income_inr"] < 25000:
        hard_reject.append("Monthly income is below ₹25,000.")
    if applicant["credit_score"] < 650:
        hard_reject.append("Credit score is below 650.")
    elif applicant["credit_score"] < 750:
        flags.append("Credit score is between 650 and 749.")
    if applicant["foir"] > 50:
        hard_reject.append("FOIR exceeds 50%.")
    elif applicant["foir"] > 40:
        flags.append("FOIR is above 40% and at most 50%.")
    if applicant["loan_to_annual_income_ratio"] > 2:
        hard_reject.append("Loan amount exceeds 2 times annual income.")
    if applicant["past_default"] == "Default within last 12 months":
        hard_reject.append("A default occurred within the last 12 months.")
    elif applicant["past_default"] in ["Default older than 12 months", "Default timing unclear", "Recently settled loan default"]:
        flags.append("Past default requires manual review.")
    if applicant["past_default"] == "Recently settled loan default":
        flags.append("Policy gap: the policy does not define a waiting period for a recently settled loan default.")
    if applicant.get("documents_status") != "Complete":
        flags.append("Required documents are missing or not verified.")
    return flags, hard_reject

def append_review(record):
    row = pd.DataFrame([record])
    if LOG_PATH.exists():
        row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(LOG_PATH, index=False)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1.5rem 1rem 1.1rem; text-align: center;
                border-bottom: 1px solid rgba(255,255,255,0.14); margin-bottom: 0.9rem;">
        <div style="font-size: 1.9rem; line-height: 1;">🏦</div>
        <div style="color: #fff; font-weight: 700; font-size: 1.05rem;
                    margin-top: 0.35rem; letter-spacing: 0.01em;
                    font-family: 'Inter', 'Segoe UI', sans-serif;">
            LoanLens
        </div>
        <div style="color: rgba(255,255,255,0.42); font-size: 0.63rem;
                    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.1rem;">
            Bharat Horizon Bank
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["1. Loan Application", "2. AI Assessment", "3. Policy Explanation", "4. Human Review & Evaluation"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Synthetic dataset: 3,000 records; labels are intentionally noisy.")

# ── Session state ─────────────────────────────────────────────────────────────
if "applicant" not in st.session_state:
    st.session_state.applicant = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "policy_flags" not in st.session_state:
    st.session_state.policy_flags = []
if "hard_reject" not in st.session_state:
    st.session_state.hard_reject = []
if "review_record" not in st.session_state:
    st.session_state.review_record = {}

# ══════════════════════════════════════════════════════════════════════════════
# Screen 1 — Loan Application
# ══════════════════════════════════════════════════════════════════════════════
if page == "1. Loan Application":
    st.markdown('<p class="bh-label">Home Loan Assessment</p>', unsafe_allow_html=True)
    st.header("Loan Application")
    st.write("Enter a fictional applicant profile. The model uses the supplied synthetic dataset's features.")

    with st.form("application_form"):
        st.markdown(
            '<p class="bh-label" style="margin-bottom: 0.6rem;">Applicant & Loan Details</p>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            monthly_income = st.number_input("Monthly income (₹)", min_value=0, max_value=10_000_000, value=75000, step=5000)
            credit_score = st.slider("Credit score", min_value=300, max_value=900, value=760)
            loan_amount = st.number_input("Loan amount requested (₹)", min_value=100000, max_value=100_000_000, value=2500000, step=100000)
            tenure = st.slider("Loan tenure (years)", min_value=1, max_value=30, value=20)
        with c2:
            existing_emi = st.number_input("Existing monthly EMI (₹)", min_value=0, max_value=1_000_000, value=5000, step=1000)
            employment_years = st.number_input("Years in employment/business", min_value=0.0, max_value=60.0, value=5.0, step=0.5)
            employment_type = st.selectbox("Employment type", ["Salaried", "Self-employed"])
            default_status = st.selectbox("Past default history", [
                "No known past default", "Default within last 12 months",
                "Default older than 12 months", "Recently settled loan default",
                "Default timing unclear"
            ])
            documents_status = st.selectbox("Required documents", ["Complete", "Missing / not verified"])
        submitted = st.form_submit_button("Assess application", type="primary", use_container_width=True)

    if submitted:
        if monthly_income <= 0:
            st.error("Monthly income must be greater than zero.")
        else:
            # Dataset FOIR uses an illustrative 8.5% annual interest rate. This is a demo estimate only.
            monthly_rate = 0.085 / 12
            n = int(tenure * 12)
            proposed_emi = loan_amount * monthly_rate * (1 + monthly_rate) ** n / ((1 + monthly_rate) ** n - 1)
            foir = ((existing_emi + proposed_emi) / monthly_income) * 100
            ratio = loan_amount / (monthly_income * 12)
            past_default_feature = 0 if default_status == "No known past default" else 1

            applicant = {
                "monthly_income_inr": float(monthly_income),
                "credit_score": int(credit_score),
                "loan_amount_inr": float(loan_amount),
                "loan_tenure_years": int(tenure),
                "existing_emi_inr": float(existing_emi),
                "employment_years": float(employment_years),
                "past_default": int(past_default_feature),
                "employment_type": employment_type,
                "foir": float(foir),
                "loan_to_annual_income_ratio": float(ratio),
                "default_status_display": default_status,
                "documents_status": documents_status,
                "estimated_proposed_emi_inr": float(proposed_emi),
            }
            X = pd.DataFrame([{k: applicant[k] for k in [
                "monthly_income_inr", "credit_score", "loan_amount_inr",
                "loan_tenure_years", "existing_emi_inr", "employment_years",
                "past_default", "employment_type", "foir",
                "loan_to_annual_income_ratio"
            ]}])
            pred = str(model.predict(X)[0])
            try:
                probs = model.predict_proba(X)[0]
                classes = list(model.classes_)
                approval_probability = float(probs[classes.index("Approved")])
            except Exception:
                approval_probability = 1.0 if pred == "Approved" else 0.0

            flags, hard_reject = policy_checks(applicant)
            review_required = bool(flags or hard_reject or 0.40 <= approval_probability <= 0.60)
            applicant["model_prediction"] = pred
            applicant["approval_probability"] = approval_probability
            applicant["review_required"] = review_required
            applicant["policy_flags"] = flags
            applicant["hard_reject"] = hard_reject
            st.session_state.applicant = applicant
            st.session_state.prediction = pred
            st.session_state.policy_flags = flags
            st.session_state.hard_reject = hard_reject
            st.session_state.review_record = {}
            st.success("Assessment complete. Open AI Assessment to review the result.")

# ══════════════════════════════════════════════════════════════════════════════
# Screen 2 — AI Assessment
# ══════════════════════════════════════════════════════════════════════════════
if page == "2. AI Assessment":
    st.markdown('<p class="bh-label">Assessment Result</p>', unsafe_allow_html=True)
    st.header("AI Assessment")
    applicant = st.session_state.applicant
    if not applicant:
        st.info("Submit an application first in Loan Application.")
    else:
        pred = applicant["model_prediction"]
        prob = applicant["approval_probability"]

        st.markdown('<p class="bh-label bh-section-gap">Model Output</p>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.metric("Model recommendation", pred)
        b.metric("Approval probability", f"{prob:.1%}")
        c.metric("Human review", "Required" if applicant["review_required"] else "Not flagged")
        st.caption("Probability reflects the model's learned synthetic labels; it is not the probability of repayment.")

        st.markdown('<p class="bh-label bh-section-gap">Financial Indicators</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated proposed EMI", fmt_inr(applicant["estimated_proposed_emi_inr"]))
        c2.metric("FOIR", f"{applicant['foir']:.1f}%")
        c3.metric("Loan / annual income", f"{applicant['loan_to_annual_income_ratio']:.2f}×")

        st.markdown('<div class="bh-section-gap"></div>', unsafe_allow_html=True)
        if applicant["policy_flags"]:
            st.warning("Policy/manual-review flags:\n\n" + "\n".join(f"- {x}" for x in applicant["policy_flags"]))
        if applicant["hard_reject"]:
            st.error("Policy rejection indicators:\n\n" + "\n".join(f"- {x}" for x in applicant["hard_reject"]))
        if not applicant["policy_flags"] and not applicant["hard_reject"]:
            st.success("No configured policy flags triggered. This does not guarantee eligibility.")
        st.info("The model prediction and policy checks are separate. Any rejection indicator, model-policy conflict, missing information, or ambiguity must be reviewed by a human. The app does not make an automatic final decision.")

# ══════════════════════════════════════════════════════════════════════════════
# Screen 3 — Policy Explanation
# ══════════════════════════════════════════════════════════════════════════════
if page == "3. Policy Explanation":
    st.markdown('<p class="bh-label">Policy RAG</p>', unsafe_allow_html=True)
    st.header("Policy Explanation (RAG)")
    applicant = st.session_state.applicant
    question_default = "Explain the policy criteria relevant to this application and identify any manual-review requirements."
    question = st.text_area("Question about the policy", value=question_default, height=90)
    if applicant:
        st.caption("Applicant context is added automatically; no direct identifiers are collected.")
    if st.button("Retrieve policy and generate explanation", type="primary"):
        context = {}
        if applicant:
            context = {
                "monthly_income_inr": applicant["monthly_income_inr"],
                "credit_score": applicant["credit_score"],
                "foir_percent": round(applicant["foir"], 2),
                "loan_to_annual_income_ratio": round(applicant["loan_to_annual_income_ratio"], 3),
                "default_status": applicant["default_status_display"],
                "documents_status": applicant["documents_status"],
                "model_recommendation": applicant["model_prediction"],
            }
        result = rag.answer(question, context, top_k=4, use_llm=True)
        st.markdown('<p class="bh-label bh-section-gap">Explanation</p>', unsafe_allow_html=True)
        st.subheader("Explanation")
        st.write(result["answer"])
        if result["citations"]:
            st.markdown('<p class="bh-label bh-section-gap">Retrieved Policy Sections</p>', unsafe_allow_html=True)
            st.subheader("Retrieved policy sections")
            for section in result["retrieved_sections"]:
                with st.expander(f"{section['citation']} · retrieval score {section['score']:.3f}"):
                    st.write(section["text"])
        else:
            st.warning("No supporting policy section retrieved. Human review is required.")
        if result["escalate"]:
            st.warning("Human review required: the question may be ambiguous or not covered by the supplied policy.")
        if result["used_llm"]:
            st.caption("An LLM generated this explanation from retrieved policy sections. Verify citations before relying on it.")
        else:
            st.caption("Extractive fallback used: no LLM-generated answer was returned.")
        st.session_state.last_rag_result = {
            "question": question, "answer": result["answer"],
            "citations": result["citations"], "used_llm": result["used_llm"],
            "escalate": result["escalate"], "reason": result["reason"]
        }
    if not applicant:
        st.info("You can test general policy questions without an application, or submit an application first for applicant-specific context.")

# ══════════════════════════════════════════════════════════════════════════════
# Screen 4 — Human Review & Evaluation
# ══════════════════════════════════════════════════════════════════════════════
if page == "4. Human Review & Evaluation":
    st.markdown('<p class="bh-label">Review & Oversight</p>', unsafe_allow_html=True)
    st.header("Human Review & Evaluation")
    applicant = st.session_state.applicant
    if applicant:
        st.markdown('<p class="bh-label bh-section-gap">Reviewer Checkpoint</p>', unsafe_allow_html=True)
        st.subheader("Reviewer checkpoint")
        st.markdown(
            f'<div class="bh-info-row">'
            f'Model recommendation: <strong>{applicant["model_prediction"]}</strong> &nbsp;·&nbsp; '
            f'Approval probability: <strong>{applicant["approval_probability"]:.1%}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.write("Review the model result, policy flags, required documents, and retrieved clauses before recording a final demo outcome.")
        with st.form("human_review_form"):
            final_outcome = st.radio("Reviewer decision", ["Approve", "Reject", "Request more information"])
            reviewer_reason = st.text_area("Reason for decision / override", placeholder="Briefly explain the reason...")
            reviewer_name = st.text_input("Reviewer initials (optional)", max_chars=10)
            save_review = st.form_submit_button("Record human review", type="primary")
        if save_review:
            if final_outcome == "Request more information" and not reviewer_reason.strip():
                st.error("Please provide a reason when requesting more information.")
            else:
                record = {
                    "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
                    "credit_score": applicant["credit_score"],
                    "foir": round(applicant["foir"], 2),
                    "loan_to_annual_income_ratio": round(applicant["loan_to_annual_income_ratio"], 3),
                    "model_prediction": applicant["model_prediction"],
                    "approval_probability": round(applicant["approval_probability"], 4),
                    "review_required": applicant["review_required"],
                    "policy_flags": " | ".join(applicant["policy_flags"]),
                    "hard_reject_indicators": " | ".join(applicant["hard_reject"]),
                    "reviewer_decision": final_outcome,
                    "reviewer_reason": reviewer_reason,
                    "reviewer_initials": reviewer_name,
                }
                append_review(record)
                st.session_state.review_record = record
                st.success("Human review recorded in the local review log.")
        if st.session_state.review_record:
            st.caption("Most recent review recorded in this session:")
            st.json(st.session_state.review_record)
    else:
        st.info("Submit an application first to use the reviewer checkpoint.")

    st.divider()
    st.markdown('<p class="bh-label">Dataset Overview</p>', unsafe_allow_html=True)
    st.subheader("Model evaluation on held-out synthetic test data")
    if DATA_PATH and DATA_PATH.exists():
        df = pd.read_excel(DATA_PATH, sheet_name="Loan_Dataset")
        target = "loan_outcome"
        if target in df.columns:
            counts = df[target].value_counts()
            st.write(f"Dataset size: **{len(df):,}** records.")
            m1, m2 = st.columns(2)
            m1.metric("Approved labels", f"{int(counts.get('Approved', 0)):,}")
            m2.metric("Rejected labels", f"{int(counts.get('Rejected', 0)):,}")
            st.caption("The complete test metrics and confusion matrix are in model_evaluation.md from Step 2. These dataset counts are not test metrics.")
    st.warning("This MVP's local CSV review log is temporary on some hosted platforms and is not a secure audit system. Do not enter real applicant data.")
    st.caption("Before production, add validated policy tests, privacy controls, secure persistent audit storage, model monitoring, fairness assessment, and qualified compliance review.")

# ── Sidebar footer ────────────────────────────────────────────────────────────
st.sidebar.caption("Bharat Horizon Bank is fictional. Educational demonstration only.")
