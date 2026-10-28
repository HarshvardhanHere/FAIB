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
    initial_sidebar_state="collapsed",
)

# ── Custom Theme CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Global Styles ───────────────────────────────────────────────── */
:root {
    --primary-purple: #4B176F;
    --secondary-purple: #6A168D;
    --magenta-accent: #A71991;
    --bg-lavender: #F8F3FC;
    --card-bg: #FFFFFF;
    --text-plum: #30213A;
    --border-lavender: #E8DDED;
    --text-muted: #756B7C;
}

html, body, .stApp {
    background-color: #F8F3FC !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #30213A !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}

/* ── Remove default Streamlit chrome & sidebar ───────────────────── */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
    visibility: hidden !important;
    display: none !important;
}
header[data-testid="stHeader"] {
    display: none !important;
}
[data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

/* ── Main block container ────────────────────────────────────────── */
.main .block-container {
    padding-top: 0.65rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 1220px !important;
    margin: 0 auto !important;
}

/* ── Top Navigation Bar ──────────────────────────────────────────── */
.st-key-top_navbar {
    background: linear-gradient(90deg, #380E53 0%, #4B176F 40%, #6A168D 100%) !important;
    padding: 0.60rem 1.4rem !important;
    box-shadow: 0 4px 16px rgba(56, 14, 83, 0.24) !important;
    border-radius: 12px !important;
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 0 1.15rem 0 !important;
}
.st-key-top_navbar [data-testid="stHorizontalBlock"] {
    align-items: center !important;
    gap: 12px !important;
}

/* ── Top Navbar Tabs via st.radio ────────────────────────────────── */
.st-key-top_navbar [data-testid="stRadio"] {
    margin: 0 !important;
    padding: 0 !important;
}
.st-key-top_navbar [data-testid="stRadioGroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 6px !important;
    align-items: center !important;
    justify-content: flex-end !important;
    flex-wrap: nowrap !important;
}

/* Option item / label */
.st-key-top_navbar [data-testid="stRadioGroup"] > label {
    background: rgba(255, 255, 255, 0.12) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.st-key-top_navbar [data-testid="stRadioGroup"] > label:hover {
    background: rgba(255, 255, 255, 0.22) !important;
    border-color: rgba(255, 255, 255, 0.35) !important;
}

/* INACTIVE TEXT MUST BE PURE WHITE */
.st-key-top_navbar [data-testid="stRadioGroup"] label,
.st-key-top_navbar [data-testid="stRadioGroup"] label *,
.st-key-top_navbar [data-testid="stRadioGroup"] [data-testid="stRadioOption"],
.st-key-top_navbar [data-testid="stRadioGroup"] [data-testid="stRadioOption"] * {
    color: #FFFFFF !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    white-space: nowrap !important;
    letter-spacing: 0.01em !important;
}

/* HIDE RADIO CIRCLE & NATIVE INPUT COMPLETELY */
.st-key-top_navbar [data-testid="stRadioGroup"] input[type="radio"],
.st-key-top_navbar [data-testid="stRadioGroup"] [data-testid="stRadioOption"] > div:first-child,
.st-key-top_navbar [data-testid="stRadioGroup"] [class*="e1dm8mpf4"],
.st-key-top_navbar [data-testid="stRadioGroup"] [class*="e1dm8mpf5"] {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    visibility: hidden !important;
}

/* ACTIVE TAB: PURE WHITE PILL WITH DEEP PURPLE TEXT */
.st-key-top_navbar [data-testid="stRadioGroup"] label:has(input:checked) {
    background: #FFFFFF !important;
    border: 1px solid #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.20) !important;
}
.st-key-top_navbar [data-testid="stRadioGroup"] label:has(input:checked) *,
.st-key-top_navbar [data-testid="stRadioGroup"] label:has(input:checked) [data-testid="stRadioOption"] * {
    color: #4B176F !important;
    font-weight: 700 !important;
}

/* ── Typography & Headings ───────────────────────────────────────── */
.page-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.05rem;
}
.page-title-group {
    display: flex;
    align-items: center;
    gap: 12px;
}
.page-badge {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background-color: #4B176F;
    color: #FFFFFF;
    font-weight: 700;
    font-size: 0.98rem;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 2px 6px rgba(75, 23, 111, 0.22);
}
.page-title {
    font-size: 1.40rem;
    font-weight: 700;
    color: #30213A;
    line-height: 1.2;
    margin: 0;
}
.page-subtitle {
    font-size: 0.82rem;
    color: #756B7C;
    margin-top: 2px;
    margin-bottom: 0;
}
.page-disclaimer {
    font-size: 0.80rem;
    color: #756B7C;
    white-space: nowrap;
}

/* ── White Cards ─────────────────────────────────────────────────── */
.ll-card {
    background: #FFFFFF;
    border: 1px solid #E8DDED;
    border-radius: 12px;
    padding: 1.15rem 1.35rem;
    box-shadow: 0 2px 8px rgba(75, 23, 111, 0.05);
    margin-bottom: 0.85rem;
}
.ll-card-header {
    font-size: 1.02rem;
    font-weight: 700;
    color: #30213A;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 1.0rem;
}
.ll-sub-header {
    font-size: 0.92rem;
    font-weight: 700;
    color: #30213A;
    display: flex;
    align-items: center;
    gap: 7px;
    margin-top: 1.0rem;
    margin-bottom: 0.75rem;
}

/* Ensure containers styled with border=True match card design */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #E8DDED !important;
    border-radius: 12px !important;
    padding: 1.15rem 1.35rem !important;
    box-shadow: 0 2px 8px rgba(75, 23, 111, 0.05) !important;
}

/* ── Streamlit Form Resets ───────────────────────────────────────── */
[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}

/* ── Inputs, Selectboxes, Slider ─────────────────────────────────── */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background-color: #FDFAFF !important;
    border: 1.5px solid #E8DDED !important;
    border-radius: 8px !important;
    color: #30213A !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
div[data-baseweb="select"] > div {
    background-color: #FDFAFF !important;
    border: 1.5px solid #E8DDED !important;
    border-radius: 8px !important;
    color: #30213A !important;
    font-size: 0.88rem !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus,
div[data-baseweb="select"]:focus-within {
    border-color: #4B176F !important;
    box-shadow: 0 0 0 2px rgba(75, 23, 111, 0.12) !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #30213A !important;
    margin-bottom: 0.20rem !important;
}

/* Slider Track & Thumb */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #4B176F !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 2px 6px rgba(75, 23, 111, 0.35) !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="background-color"] {
    background-color: #4B176F !important;
}

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #4B176F 0%, #A71991 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 0.55rem 1.4rem !important;
    font-size: 0.90rem !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(75, 23, 111, 0.25) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 4px 15px rgba(167, 25, 145, 0.38) !important;
    opacity: 0.95;
    transform: translateY(-1px);
}

/* Edit small button in Screen 4 */
.st-key-edit_app_btn button {
    background: transparent !important;
    color: #6A168D !important;
    border: 1px solid #E8DDED !important;
    box-shadow: none !important;
    padding: 0.25rem 0.75rem !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}
.st-key-edit_app_btn button:hover {
    background: #FAF5FC !important;
    border-color: #6A168D !important;
    transform: none !important;
}

/* ── Reviewer Decision Radio Styling ─────────────────────────────── */
.st-key-reviewer_decision_container [data-testid="stRadio"] {
    margin: 0 !important;
    padding: 0 !important;
}
.st-key-reviewer_decision_container [data-testid="stRadioGroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 10px !important;
    width: 100% !important;
}
.st-key-reviewer_decision_container [data-testid="stRadioGroup"] > label {
    flex: 1 !important;
    background: #FFFFFF !important;
    border: 1.5px solid #E8DDED !important;
    border-radius: 8px !important;
    padding: 8px 10px !important;
    justify-content: center !important;
    align-items: center !important;
    display: flex !important;
    transition: all 0.18s ease !important;
    cursor: pointer !important;
    margin: 0 !important;
}
.st-key-reviewer_decision_container [data-testid="stRadioGroup"] > label:hover {
    border-color: #6A168D !important;
    background: #FAF5FC !important;
}
.st-key-reviewer_decision_container [data-testid="stRadioGroup"] label * {
    color: #30213A !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
}
.st-key-reviewer_decision_container [data-testid="stRadioGroup"] input[type="radio"],
.st-key-reviewer_decision_container [data-testid="stRadioGroup"] [data-testid="stRadioOption"] > div:first-child,
.st-key-reviewer_decision_container [data-testid="stRadioGroup"] [class*="e1dm8mpf4"],
.st-key-reviewer_decision_container [data-testid="stRadioGroup"] [class*="e1dm8mpf5"] {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    visibility: hidden !important;
}
.st-key-reviewer_decision_container [data-testid="stRadioGroup"] label:has(input:checked) {
    background: #F8F3FC !important;
    border: 2px solid #6A168D !important;
    box-shadow: 0 2px 6px rgba(106, 22, 141, 0.12) !important;
}
.st-key-reviewer_decision_container [data-testid="stRadioGroup"] label:has(input:checked) * {
    color: #4B176F !important;
    font-weight: 700 !important;
}

/* ── Expanders ───────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E8DDED !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
    margin-bottom: 0.55rem !important;
}
[data-testid="stExpander"] summary {
    color: #4B176F !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── File Checks ───────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    st.error("Trained model file is missing. Place logistic_regression_pipeline.joblib beside app.py.")
    st.stop()
if not POLICY_PATH.exists():
    st.error("lending_policy.md is missing. Place it beside app.py.")
    st.stop()

# ── Cached Resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_rag():
    return PolicyRAG(POLICY_PATH)

model = load_model()
rag = load_rag()

# ── Utilities & Calculations ──────────────────────────────────────────────────
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
    if applicant["past_default"] == 1 and applicant.get("default_status_display") == "Default within last 12 months":
        hard_reject.append("A default occurred within the last 12 months.")
    elif applicant.get("default_status_display") in ["Default older than 12 months", "Default timing unclear", "Recently settled loan default"]:
        flags.append("Past default requires manual review.")
    if applicant.get("default_status_display") == "Recently settled loan default":
        flags.append("Policy gap: the policy does not define a waiting period for a recently settled loan default.")
    if applicant.get("documents_status") != "Complete":
        flags.append("Required documents are missing or not verified.")
    return flags, hard_reject

def compute_applicant(
    monthly_income=75000,
    credit_score=760,
    loan_amount=2500000,
    tenure=20,
    existing_emi=5000,
    employment_years=5.0,
    employment_type="Salaried",
    default_status="No known past default",
    documents_status="Complete",
    other_obligations=0,
):
    monthly_rate = 0.085 / 12
    n = int(tenure * 12)
    proposed_emi = loan_amount * monthly_rate * (1 + monthly_rate) ** n / ((1 + monthly_rate) ** n - 1)
    foir = ((existing_emi + proposed_emi) / monthly_income) * 100
    ratio = loan_amount / (monthly_income * 12)
    past_default_feature = 0 if default_status in ["No known past default", "No"] else 1

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
        "default_status_display": "No" if default_status in ["No known past default", "No"] else default_status,
        "documents_status": documents_status,
        "other_obligations_inr": float(other_obligations),
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
    return applicant

def append_review(record):
    row = pd.DataFrame([record])
    if LOG_PATH.exists():
        row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(LOG_PATH, index=False)

# ── Session State Management ──────────────────────────────────────────────────
# Handle smooth programmatic navigation transitions
if "target_tab" in st.session_state and st.session_state.target_tab:
    st.session_state["nav_tab"] = st.session_state.target_tab
    st.session_state.target_tab = None

if "nav_tab" not in st.session_state:
    st.session_state["nav_tab"] = "Loan Application"

if "applicant" not in st.session_state or st.session_state.applicant is None:
    st.session_state.applicant = compute_applicant()
    st.session_state.prediction = st.session_state.applicant["model_prediction"]
    st.session_state.policy_flags = st.session_state.applicant["policy_flags"]
    st.session_state.hard_reject = st.session_state.applicant["hard_reject"]

if "review_record" not in st.session_state:
    st.session_state.review_record = {}

TAB_NAMES = ["Loan Application", "AI Assessment", "Policy Explanation", "Human Review"]

# ── Top Navigation Bar ────────────────────────────────────────────────────────
with st.container(key="top_navbar"):
    col_brand, col_tabs, col_avatar = st.columns([3.3, 6.9, 0.6], vertical_alignment="center")

    with col_brand:
        brand_html = """<div style="display: flex; align-items: center; gap: 10px;">
<div style="font-size: 1.45rem; color: #FFFFFF; line-height: 1; flex-shrink: 0;">🏛️</div>
<div style="color: #FFFFFF; font-weight: 700; font-size: 0.94rem; white-space: nowrap; letter-spacing: 0.01em;">Bharat Horizon Bank</div>
<div style="height: 20px; width: 1px; background: rgba(255,255,255,0.25); margin: 0 3px; flex-shrink: 0;"></div>
<div>
<div style="color: #FFFFFF; font-weight: 700; font-size: 1.02rem; line-height: 1.1;">LoanLens</div>
<div style="color: rgba(255,255,255,0.72); font-size: 0.65rem; line-height: 1; margin-top: 1px; white-space: nowrap;">AI-assisted Home Loan Assessment</div>
</div>
</div>"""
        st.html(brand_html)

    with col_tabs:
        def format_nav_tab(t):
            icons = {
                "Loan Application": "📋 Loan Application",
                "AI Assessment": "⚡ AI Assessment",
                "Policy Explanation": "📖 Policy Explanation",
                "Human Review": "👤 Human Review",
            }
            return icons.get(t, t)

        current_idx = TAB_NAMES.index(st.session_state["nav_tab"]) if st.session_state["nav_tab"] in TAB_NAMES else 0
        active_tab = st.radio(
            "Global Navigation",
            options=TAB_NAMES,
            index=current_idx,
            key="nav_tab",
            horizontal=True,
            format_func=format_nav_tab,
            label_visibility="collapsed",
        )

    with col_avatar:
        avatar_html = """<div style="display: flex; justify-content: flex-end; align-items: center;">
<div style="width: 34px; height: 34px; border-radius: 50%; background: #FFFFFF; color: #4B176F; font-weight: 700; font-size: 0.86rem; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 6px rgba(0,0,0,0.18);">HG</div>
</div>"""
        st.html(avatar_html)

applicant = st.session_state.applicant

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — LOAN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
if active_tab == "Loan Application":
    st.html("""<div class="page-header-row">
<div class="page-title-group">
<div class="page-badge">1</div>
<div>
<div class="page-title">Loan Application</div>
<div class="page-subtitle">Enter a fictional applicant profile to assess home loan eligibility.</div>
</div>
</div>
<div class="page-disclaimer">
Educational prototype — Not financial advice.
</div>
</div>""")

    with st.form("loan_application_form"):
        col_applicant, col_loan = st.columns([52, 48], gap="medium")

        # ── Left Column: Applicant Profile ────────────────────────────────────
        with col_applicant:
            with st.container(border=True):
                st.html("""<div class="ll-card-header">
<span style="font-size: 1.15rem; color: #4B176F;">👤</span>
<span>Applicant Profile</span>
</div>""")

                r1_c1, r1_c2 = st.columns(2)
                with r1_c1:
                    monthly_income = st.number_input(
                        "Monthly income (₹)",
                        min_value=0,
                        max_value=10_000_000,
                        value=int(applicant.get("monthly_income_inr", 75000)),
                        step=5000,
                    )
                with r1_c2:
                    existing_emi = st.number_input(
                        "Existing monthly EMI (₹)",
                        min_value=0,
                        max_value=1_000_000,
                        value=int(applicant.get("existing_emi_inr", 5000)),
                        step=1000,
                    )

                r2_c1, r2_c2 = st.columns(2)
                with r2_c1:
                    curr_credit = int(applicant.get("credit_score", 760))
                    st.html(f"""<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: -4px;">
<span style="font-size: 0.82rem; font-weight: 600; color: #30213A;">Credit score</span>
<span style="font-size: 0.82rem; font-weight: 700; color: #4B176F;">{curr_credit}</span>
</div>""")
                    credit_score = st.slider(
                        "Credit score",
                        min_value=300,
                        max_value=900,
                        value=curr_credit,
                        label_visibility="collapsed",
                    )
                with r2_c2:
                    employment_years = st.number_input(
                        "Years in employment/business",
                        min_value=0.0,
                        max_value=60.0,
                        value=float(applicant.get("employment_years", 5.0)),
                        step=0.5,
                    )

                r3_c1, r3_c2 = st.columns(2)
                with r3_c1:
                    emp_type_opts = ["Salaried", "Self-employed"]
                    curr_emp = applicant.get("employment_type", "Salaried")
                    emp_idx = emp_type_opts.index(curr_emp) if curr_emp in emp_type_opts else 0
                    employment_type = st.selectbox("Employment type", emp_type_opts, index=emp_idx)
                with r3_c2:
                    default_opts = [
                        "No",
                        "Default within last 12 months",
                        "Default older than 12 months",
                        "Recently settled loan default",
                        "Default timing unclear",
                    ]
                    curr_def = applicant.get("default_status_display", "No")
                    def_idx = default_opts.index(curr_def) if curr_def in default_opts else 0
                    default_status = st.selectbox("Past default history", default_opts, index=def_idx)

        # ── Right Column: Loan Details ────────────────────────────────────────
        with col_loan:
            with st.container(border=True):
                st.html("""<div class="ll-card-header">
<span style="font-size: 1.15rem; color: #4B176F;">🏠</span>
<span>Loan Details</span>
</div>""")

                loan_amount = st.number_input(
                    "Loan amount requested (₹)",
                    min_value=100_000,
                    max_value=100_000_000,
                    value=int(applicant.get("loan_amount_inr", 2500000)),
                    step=100000,
                )
                loan_tenure = st.number_input(
                    "Loan tenure (years)",
                    min_value=1,
                    max_value=30,
                    value=int(applicant.get("loan_tenure_years", 20)),
                    step=1,
                )

                st.html("""<div class="ll-sub-header">
<span style="font-size: 1.05rem; color: #4B176F;">💳</span>
<span>Additional Financial Information</span>
</div>""")

                other_obligations = st.number_input(
                    "Other monthly obligations (₹)",
                    min_value=0,
                    max_value=1_000_000,
                    value=int(applicant.get("other_obligations_inr", 0)),
                    step=1000,
                )

                doc_opts = ["Complete", "Missing / not verified"]
                curr_doc = applicant.get("documents_status", "Complete")
                doc_idx = doc_opts.index(curr_doc) if curr_doc in doc_opts else 0
                documents_status = st.selectbox("Required documents", doc_opts, index=doc_idx)

        st.html("<div style='height: 8px;'></div>")

        # ── Bottom Action Row ─────────────────────────────────────────────────
        row_info, row_btn = st.columns([65, 35], vertical_alignment="center")
        with row_info:
            st.html("""<div style="display: flex; align-items: center; gap: 12px; background: #FFFFFF; border: 1px solid #E8DDED; border-radius: 10px; padding: 12px 18px; box-shadow: 0 1px 6px rgba(75, 23, 111, 0.04);">
<div style="width: 26px; height: 26px; border-radius: 50%; background: #4B176F; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 0.82rem; font-weight: 700; flex-shrink: 0;">i</div>
<div>
<div style="font-weight: 700; color: #4B176F; font-size: 0.85rem; line-height: 1.2;">Important</div>
<div style="color: #756B7C; font-size: 0.78rem; margin-top: 1px;">This is a fictional application using synthetic data. Not financial advice.</div>
</div>
</div>""")

        with row_btn:
            submitted = st.form_submit_button("Assess Application →", use_container_width=True)

    if submitted:
        if monthly_income <= 0:
            st.error("Monthly income must be greater than zero.")
        else:
            updated_applicant = compute_applicant(
                monthly_income=monthly_income,
                credit_score=credit_score,
                loan_amount=loan_amount,
                tenure=loan_tenure,
                existing_emi=existing_emi,
                employment_years=employment_years,
                employment_type=employment_type,
                default_status=default_status,
                documents_status=documents_status,
                other_obligations=other_obligations,
            )
            st.session_state.applicant = updated_applicant
            st.session_state.prediction = updated_applicant["model_prediction"]
            st.session_state.policy_flags = updated_applicant["policy_flags"]
            st.session_state.hard_reject = updated_applicant["hard_reject"]
            st.session_state.review_record = {}
            st.session_state.target_tab = "AI Assessment"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — AI ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
elif active_tab == "AI Assessment":
    st.html("""<div class="page-header-row">
<div class="page-title-group">
<div class="page-badge">2</div>
<div>
<div class="page-title">AI Assessment</div>
<div class="page-subtitle">Model prediction and key financial indicators.</div>
</div>
</div>
<div class="page-disclaimer">
Educational prototype — Not financial advice.
</div>
</div>""")

    pred = applicant["model_prediction"]
    prob = applicant["approval_probability"]
    is_rejected = pred == "Rejected"
    review_req = applicant["review_required"]

    # ── Top Recommendation Card ───────────────────────────────────────────────
    status_icon_html = '<div style="width: 44px; height: 44px; border-radius: 50%; background: #FEE2E2; color: #DC2626; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; font-weight: 800; flex-shrink: 0;">✕</div>' if is_rejected else '<div style="width: 44px; height: 44px; border-radius: 50%; background: #DCFCE7; color: #16A34A; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; font-weight: 800; flex-shrink: 0;">✓</div>'

    rec_color = "#DC2626" if is_rejected else "#16A34A"
    prob_bar_color = "#DC2626" if is_rejected else "#16A34A"
    prob_bar_width = max(3, min(100, int(prob * 100)))

    review_icon_bg = "#FEF3C7" if review_req else "#DCFCE7"
    review_icon_color = "#D97706" if review_req else "#16A34A"
    review_text = "Required" if review_req else "Not Required"
    review_color = "#D97706" if review_req else "#16A34A"

    top_card_html = f"""<div style="background: #FFFFFF; border: 1px solid #E8DDED; border-radius: 12px; padding: 1.15rem 1.6rem; box-shadow: 0 2px 8px rgba(75, 23, 111, 0.05); margin-bottom: 0.35rem;">
<div style="display: flex; align-items: center; justify-content: space-between; gap: 20px;">
<div style="display: flex; align-items: center; gap: 14px; min-width: 220px;">
{status_icon_html}
<div>
<div style="font-size: 0.74rem; font-weight: 700; color: #756B7C; text-transform: uppercase; letter-spacing: 0.05em;">Model Recommendation</div>
<div style="font-size: 1.55rem; font-weight: 800; color: {rec_color}; line-height: 1.15;">{pred}</div>
</div>
</div>
<div style="flex: 1; max-width: 320px; min-width: 180px;">
<div style="font-size: 0.74rem; font-weight: 700; color: #756B7C; text-transform: uppercase; letter-spacing: 0.05em;">Approval Probability</div>
<div style="font-size: 1.55rem; font-weight: 800; color: #30213A; line-height: 1.15;">{prob:.1%}</div>
<div style="width: 100%; height: 5px; background: #E8DDED; border-radius: 3px; margin-top: 5px; overflow: hidden;">
<div style="width: {prob_bar_width}%; height: 100%; background: {prob_bar_color}; border-radius: 3px;"></div>
</div>
</div>
<div style="display: flex; align-items: center; justify-content: flex-end; gap: 14px; min-width: 200px;">
<div style="display: flex; align-items: center; gap: 11px;">
<div style="width: 40px; height: 40px; border-radius: 50%; background: {review_icon_bg}; color: {review_icon_color}; display: flex; align-items: center; justify-content: center; font-size: 1.15rem; flex-shrink: 0;">👤</div>
<div>
<div style="font-size: 0.74rem; font-weight: 700; color: #756B7C; text-transform: uppercase; letter-spacing: 0.05em;">Human Review</div>
<div style="font-size: 1.55rem; font-weight: 800; color: {review_color}; line-height: 1.15;">{review_text}</div>
</div>
</div>
<div style="color: #B5A8BE; font-size: 1.35rem; font-weight: 300;">›</div>
</div>
</div>
</div>
<div style="font-size: 0.75rem; color: #756B7C; margin-bottom: 1.15rem; padding-left: 2px;">
※ Probability reflects the model's learned patterns; it is not the probability of repayment.
</div>"""
    st.html(top_card_html)

    # ── Key Financial Indicators ──────────────────────────────────────────────
    st.html("""<div style="font-size: 1.05rem; font-weight: 700; color: #30213A; margin-bottom: 0.8rem;">
Key Financial Indicators
</div>""")

    k1, k2, k3 = st.columns(3, gap="medium")

    # Card 1: Estimated Proposed EMI
    proposed_emi = applicant["estimated_proposed_emi_inr"]
    emi_pct = (proposed_emi / applicant["monthly_income_inr"]) * 100 if applicant["monthly_income_inr"] > 0 else 0
    with k1:
        st.html(f"""<div class="ll-card" style="margin-bottom: 0; min-height: 140px; display: flex; flex-direction: column; justify-content: space-between;">
<div style="display: flex; align-items: center; gap: 10px;">
<div style="width: 32px; height: 32px; border-radius: 8px; background: #F3E8FF; color: #7E22CE; display: flex; align-items: center; justify-content: center; font-size: 1.05rem;">💼</div>
<div style="font-size: 0.74rem; font-weight: 700; color: #756B7C; text-transform: uppercase; letter-spacing: 0.04em;">Estimated Proposed EMI</div>
</div>
<div>
<div style="font-size: 1.55rem; font-weight: 800; color: #30213A; line-height: 1.15;">₹{proposed_emi:,.0f}</div>
<div style="width: 100%; height: 4px; background: #E8DDED; border-radius: 2px; margin-top: 6px; overflow: hidden;">
<div style="width: {min(100, int(emi_pct))}%; height: 100%; background: #4B176F; border-radius: 2px;"></div>
</div>
<div style="font-size: 0.76rem; color: #756B7C; margin-top: 5px;">{emi_pct:.1f}% of monthly income</div>
</div>
</div>""")

    # Card 2: FOIR
    foir_val = applicant["foir"]
    foir_bar_color = "#DC2626" if foir_val > 50 else ("#D97706" if foir_val > 40 else "#16A34A")
    foir_msg = "Exceeds 50% policy limit" if foir_val > 50 else ("Manual review required (40–50%)" if foir_val > 40 else "Within 40% policy guideline")
    with k2:
        st.html(f"""<div class="ll-card" style="margin-bottom: 0; min-height: 140px; display: flex; flex-direction: column; justify-content: space-between;">
<div style="display: flex; align-items: center; gap: 10px;">
<div style="width: 32px; height: 32px; border-radius: 8px; background: #F3E8FF; color: #7E22CE; display: flex; align-items: center; justify-content: center; font-size: 1.05rem;">%</div>
<div style="font-size: 0.74rem; font-weight: 700; color: #756B7C; text-transform: uppercase; letter-spacing: 0.04em;">FOIR</div>
</div>
<div>
<div style="font-size: 1.55rem; font-weight: 800; color: #30213A; line-height: 1.15;">{foir_val:.1f}%</div>
<div style="width: 100%; height: 4px; background: #E8DDED; border-radius: 2px; margin-top: 6px; overflow: hidden;">
<div style="width: {min(100, int(foir_val))}%; height: 100%; background: {foir_bar_color}; border-radius: 2px;"></div>
</div>
<div style="font-size: 0.76rem; color: {foir_bar_color}; font-weight: 600; margin-top: 5px;">{foir_msg}</div>
</div>
</div>""")

    # Card 3: Loan / Annual Income
    ratio_val = applicant["loan_to_annual_income_ratio"]
    ratio_bar_color = "#DC2626" if ratio_val > 2.0 else "#16A34A"
    ratio_msg = "Exceeds 2× policy limit" if ratio_val > 2.0 else "Within 2× policy limit"
    with k3:
        st.html(f"""<div class="ll-card" style="margin-bottom: 0; min-height: 140px; display: flex; flex-direction: column; justify-content: space-between;">
<div style="display: flex; align-items: center; gap: 10px;">
<div style="width: 32px; height: 32px; border-radius: 8px; background: #F3E8FF; color: #7E22CE; display: flex; align-items: center; justify-content: center; font-size: 1.05rem;">📊</div>
<div style="font-size: 0.74rem; font-weight: 700; color: #756B7C; text-transform: uppercase; letter-spacing: 0.04em;">Loan / Annual Income</div>
</div>
<div>
<div style="font-size: 1.55rem; font-weight: 800; color: #30213A; line-height: 1.15;">{ratio_val:.2f}×</div>
<div style="width: 100%; height: 4px; background: #E8DDED; border-radius: 2px; margin-top: 6px; overflow: hidden;">
<div style="width: {min(100, int((ratio_val / 4.0) * 100))}%; height: 100%; background: {ratio_bar_color}; border-radius: 2px;"></div>
</div>
<div style="font-size: 0.76rem; color: {ratio_bar_color}; font-weight: 600; margin-top: 5px;">{ratio_msg}</div>
</div>
</div>""")

    st.html("<div style='height: 12px;'></div>")

    # ── Policy Warning Panels ─────────────────────────────────────────────────
    if applicant["hard_reject"]:
        rejection_bullets = "".join(f"<div style='margin-bottom: 3px;'>• {item}</div>" for item in applicant["hard_reject"])
        st.html(f"""<div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 11px; padding: 12px 18px; margin-bottom: 10px;">
<div style="display: flex; align-items: flex-start; gap: 11px;">
<div style="color: #DC2626; font-size: 1.25rem; line-height: 1;">⚠️</div>
<div style="flex: 1;">
<div style="font-size: 0.90rem; font-weight: 700; color: #DC2626; margin-bottom: 3px;">Policy rejection indicators</div>
<div style="font-size: 0.83rem; color: #991B1B; line-height: 1.45;">
{rejection_bullets}
</div>
</div>
</div>
</div>""")
    elif applicant["policy_flags"]:
        flag_bullets = "".join(f"<div style='margin-bottom: 3px;'>• {item}</div>" for item in applicant["policy_flags"])
        st.html(f"""<div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 11px; padding: 12px 18px; margin-bottom: 10px;">
<div style="display: flex; align-items: flex-start; gap: 11px;">
<div style="color: #D97706; font-size: 1.25rem; line-height: 1;">⚠️</div>
<div style="flex: 1;">
<div style="font-size: 0.90rem; font-weight: 700; color: #D97706; margin-bottom: 3px;">Policy / manual-review flags</div>
<div style="font-size: 0.83rem; color: #92400E; line-height: 1.45;">
{flag_bullets}
</div>
</div>
</div>
</div>""")
    else:
        st.html("""<div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 11px; padding: 12px 18px; margin-bottom: 10px;">
<div style="display: flex; align-items: center; gap: 11px;">
<div style="color: #16A34A; font-size: 1.25rem; line-height: 1;">✓</div>
<div style="font-size: 0.85rem; color: #166534; font-weight: 500;">
No configured policy rejection flags triggered. This does not guarantee eligibility.
</div>
</div>
</div>""")

    # Informational card
    st.html("""<div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 11px; padding: 12px 18px;">
<div style="display: flex; align-items: flex-start; gap: 11px;">
<div style="color: #2563EB; font-size: 1.25rem; line-height: 1;">ℹ️</div>
<div style="font-size: 0.82rem; color: #1E40AF; line-height: 1.45;">
The model prediction and policy checks are separate. Any rejection indicator, model-policy conflict, missing information, or ambiguity must be reviewed by a human. The app does not make an automatic final decision.
</div>
</div>
</div>""")

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — POLICY EXPLANATION (RAG)
# ══════════════════════════════════════════════════════════════════════════════
elif active_tab == "Policy Explanation":
    st.html("""<div class="page-header-row">
<div class="page-title-group">
<div class="page-badge">3</div>
<div>
<div class="page-title">Policy Explanation (RAG)</div>
<div class="page-subtitle">Ask questions about the bank's lending policy and get AI-generated answers with citations.</div>
</div>
</div>
<div class="page-disclaimer">
Educational prototype — Not financial advice.
</div>
</div>""")

    # ── Top Question Card ─────────────────────────────────────────────────────
    default_question = "Why was this application rejected?"

    with st.container(border=True):
        st.html("""<div style="font-size: 0.88rem; font-weight: 700; color: #30213A; margin-bottom: 0.35rem;">
Your question
</div>""")

        q_col, btn_col = st.columns([72, 28], vertical_alignment="bottom")
        with q_col:
            user_question = st.text_area(
                "Your question",
                value=default_question,
                height=70,
                label_visibility="collapsed",
            )
            char_count = len(user_question)
            st.html(f"""<div style="text-align: right; font-size: 0.74rem; color: #756B7C; margin-top: -6px;">
{char_count}/500
</div>""")

        with btn_col:
            query_submitted = st.button("Retrieve policy and generate explanation →", use_container_width=True)

    st.html("<div style='height: 12px;'></div>")

    # Execute or load cached RAG result with space-separated context tokens
    context = {
        "monthly income inr": applicant["monthly_income_inr"],
        "credit score": applicant["credit_score"],
        "foir percent": round(applicant["foir"], 1),
        "loan to annual income ratio": round(applicant["loan_to_annual_income_ratio"], 2),
        "loan amount inr": applicant["loan_amount_inr"],
        "default status": applicant["default_status_display"],
        "documents status": applicant["documents_status"],
        "model recommendation": applicant["model_prediction"],
    }
    if applicant.get("hard_reject"):
        context["rejection reasons"] = ", ".join(applicant["hard_reject"])
    if applicant.get("policy_flags"):
        context["policy flags"] = ", ".join(applicant["policy_flags"])

    if "last_rag_result" not in st.session_state or query_submitted:
        result = rag.answer(user_question, context, top_k=3, use_llm=True)
        st.session_state.last_rag_result = result
    else:
        result = st.session_state.last_rag_result

    # ── Two Columns: AI Explanation & Application Context ─────────────────────
    col_exp, col_ctx = st.columns([62, 38], gap="medium")

    # Left Column: AI Explanation & Relevant Policy Clauses
    with col_exp:
        with st.container(border=True):
            st.html("""<div class="ll-card-header">
<span style="font-size: 1.15rem; color: #4B176F;">📄</span>
<span>AI Explanation</span>
</div>""")

            explanation_text = result["answer"]
            st.html(f"""<div style="font-size: 0.88rem; color: #30213A; line-height: 1.55; margin-bottom: 1.1rem;">
{explanation_text}
</div>
<div class="ll-sub-header" style="margin-top: 0.8rem; margin-bottom: 0.75rem;">
<span style="font-size: 1.05rem; color: #4B176F;">📖</span>
<span>Relevant policy clauses</span>
</div>""")

            sections = result.get("retrieved_sections", [])
            if not sections:
                st.info("No direct supporting policy sections retrieved for this question.")
            else:
                for sec in sections:
                    title = sec["title"]
                    citation = sec["citation"].replace("BH-HOME-POLICY § ", "Policy Section ")
                    clean_text = sec["text"].replace("\n", " ").strip()
                    summary_text = clean_text[:120] + "..." if len(clean_text) > 120 else clean_text

                    st.html(f"""<div style="background: #F8F3FC; border: 1px solid #E8DDED; border-radius: 9px; padding: 10px 14px; margin-bottom: 8px;">
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 3px;">
<div style="display: flex; align-items: center; gap: 7px;">
<span style="color: #4B176F; font-size: 0.95rem;">📄</span>
<span style="font-weight: 700; color: #30213A; font-size: 0.86rem;">{title}</span>
</div>
<span style="color: #6A168D; font-weight: 600; font-size: 0.75rem;">{citation}</span>
</div>
<div style="font-size: 0.79rem; color: #756B7C; line-height: 1.40;">
{summary_text}
</div>
</div>""")

            if result.get("escalate"):
                st.html("""<div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 10px 14px; margin-top: 10px; font-size: 0.80rem; color: #92400E;">
⚠️ Human review required: the policy does not fully resolve this scenario.
</div>""")

    # Right Column: Application Context (Auto-filled)
    with col_ctx:
        with st.container(border=True):
            st.html("""<div class="ll-card-header">
<span style="font-size: 1.15rem; color: #4B176F;">📑</span>
<span>Application Context (Auto-filled)</span>
</div>""")

            past_def_disp = applicant["default_status_display"]
            other_ob_disp = fmt_inr(applicant.get("other_obligations_inr", 0))

            st.html(f"""<table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 6px 0; color: #756B7C;">Monthly income</td>
<td style="padding: 6px 0; text-align: right; font-weight: 700; color: #30213A;">₹{applicant["monthly_income_inr"]:,.0f}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 6px 0; color: #756B7C;">Existing EMI</td>
<td style="padding: 6px 0; text-align: right; font-weight: 700; color: #30213A;">₹{applicant["existing_emi_inr"]:,.0f}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 6px 0; color: #756B7C;">Credit score</td>
<td style="padding: 6px 0; text-align: right; font-weight: 700; color: #30213A;">{applicant["credit_score"]}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 6px 0; color: #756B7C;">Loan amount requested</td>
<td style="padding: 6px 0; text-align: right; font-weight: 700; color: #30213A;">₹{applicant["loan_amount_inr"]:,.0f}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 6px 0; color: #756B7C;">Loan tenure</td>
<td style="padding: 6px 0; text-align: right; font-weight: 700; color: #30213A;">{applicant["loan_tenure_years"]} years</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 6px 0; color: #756B7C;">Employment type</td>
<td style="padding: 6px 0; text-align: right; font-weight: 700; color: #30213A;">{applicant["employment_type"]}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 6px 0; color: #756B7C;">Past default history</td>
<td style="padding: 6px 0; text-align: right; font-weight: 700; color: #30213A;">{past_def_disp}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 6px 0; color: #756B7C;">Other monthly obligations</td>
<td style="padding: 6px 0; text-align: right; font-weight: 700; color: #30213A;">{other_ob_disp}</td>
</tr>
<tr>
<td style="padding: 6px 0; color: #756B7C;">Required documents</td>
<td style="padding: 6px 0; text-align: right; font-weight: 700; color: #30213A;">{applicant["documents_status"]}</td>
</tr>
</table>""")

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 4 — HUMAN REVIEW & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif active_tab == "Human Review":
    st.html("""<div class="page-header-row">
<div class="page-title-group">
<div class="page-badge">4</div>
<div>
<div class="page-title">Human Review & Evaluation</div>
<div class="page-subtitle">Review the model result, policy flags, and applicant details before recording a final outcome.</div>
</div>
</div>
<div class="page-disclaimer">
Educational prototype — Not financial advice.
</div>
</div>""")

    col_rev_left, col_rev_right = st.columns([50, 50], gap="medium")

    # ── Left Column: Application Summary ──────────────────────────────────────
    with col_rev_left:
        with st.container(border=True):
            header_col1, header_col2 = st.columns([75, 25], vertical_alignment="center")
            with header_col1:
                st.html("""<div class="ll-card-header" style="margin-bottom: 0;">
<span style="font-size: 1.15rem; color: #4B176F;">👤</span>
<span>Application Summary</span>
</div>""")
            with header_col2:
                if st.button("✏️ Edit", key="edit_app_btn"):
                    st.session_state.target_tab = "Loan Application"
                    st.rerun()

            st.html("<div style='height: 8px;'></div>")

            past_def = applicant["default_status_display"]
            other_ob = fmt_inr(applicant.get("other_obligations_inr", 0))

            st.html(f"""<table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 5px 0; color: #756B7C;">Monthly income (₹)</td>
<td style="padding: 5px 0; text-align: right; font-weight: 700; color: #30213A;">{applicant["monthly_income_inr"]:,.0f}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 5px 0; color: #756B7C;">Existing monthly EMI (₹)</td>
<td style="padding: 5px 0; text-align: right; font-weight: 700; color: #30213A;">{applicant["existing_emi_inr"]:,.0f}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 5px 0; color: #756B7C;">Credit score</td>
<td style="padding: 5px 0; text-align: right; font-weight: 700; color: #30213A;">{applicant["credit_score"]}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 5px 0; color: #756B7C;">Employment type</td>
<td style="padding: 5px 0; text-align: right; font-weight: 700; color: #30213A;">{applicant["employment_type"]}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 5px 0; color: #756B7C;">Loan amount requested (₹)</td>
<td style="padding: 5px 0; text-align: right; font-weight: 700; color: #30213A;">{applicant["loan_amount_inr"]:,.0f}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 5px 0; color: #756B7C;">Loan tenure (years)</td>
<td style="padding: 5px 0; text-align: right; font-weight: 700; color: #30213A;">{applicant["loan_tenure_years"]}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 5px 0; color: #756B7C;">Years in employment</td>
<td style="padding: 5px 0; text-align: right; font-weight: 700; color: #30213A;">{applicant["employment_years"]}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 5px 0; color: #756B7C;">Past default history</td>
<td style="padding: 5px 0; text-align: right; font-weight: 700; color: #30213A;">{past_def}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 5px 0; color: #756B7C;">Other monthly obligations (₹)</td>
<td style="padding: 5px 0; text-align: right; font-weight: 700; color: #30213A;">{other_ob}</td>
</tr>
<tr>
<td style="padding: 5px 0; color: #756B7C;">Required documents</td>
<td style="padding: 5px 0; text-align: right; font-weight: 700; color: #30213A;">{applicant["documents_status"]}</td>
</tr>
</table>""")

    # ── Right Column: Model & Policy Summary + Reviewer Decision ──────────────
    with col_rev_right:
        # Card 1: Model & Policy Summary
        with st.container(border=True):
            st.html("""<div class="ll-card-header" style="margin-bottom: 0.5rem;">
<span style="font-size: 1.15rem; color: #4B176F;">📊</span>
<span>Model & Policy Summary</span>
</div>""")

            pred_pill = '<span style="background: #FEE2E2; color: #DC2626; font-weight: 700; font-size: 0.78rem; padding: 3px 10px; border-radius: 6px;">Rejected</span>' if applicant["model_prediction"] == "Rejected" else '<span style="background: #DCFCE7; color: #16A34A; font-weight: 700; font-size: 0.78rem; padding: 3px 10px; border-radius: 6px;">Approved</span>'

            foir_color = "#DC2626" if applicant["foir"] > 50 else ("#D97706" if applicant["foir"] > 40 else "#16A34A")
            ratio_color = "#DC2626" if applicant["loan_to_annual_income_ratio"] > 2.0 else "#16A34A"

            st.html(f"""<table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 6px 0; color: #756B7C;">Model recommendation</td>
<td style="padding: 6px 0; text-align: right;">{pred_pill}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 6px 0; color: #756B7C;">Approval probability</td>
<td style="padding: 6px 0; text-align: right; font-weight: 700; color: #30213A;">{applicant["approval_probability"]:.1%}</td>
</tr>
<tr style="border-bottom: 1px solid #F1E8F5;">
<td style="padding: 6px 0; color: #756B7C;">FOIR</td>
<td style="padding: 6px 0; text-align: right; font-weight: 700; color: {foir_color};">{applicant["foir"]:.1f}%</td>
</tr>
<tr>
<td style="padding: 6px 0; color: #756B7C;">Loan / Annual income</td>
<td style="padding: 6px 0; text-align: right; font-weight: 700; color: {ratio_color};">{applicant["loan_to_annual_income_ratio"]:.2f}×</td>
</tr>
</table>""")

            with st.expander("View policy details"):
                if applicant["hard_reject"]:
                    st.markdown("**Rejection indicators:**\n" + "\n".join(f"- {x}" for x in applicant["hard_reject"]))
                if applicant["policy_flags"]:
                    st.markdown("**Policy flags:**\n" + "\n".join(f"- {x}" for x in applicant["policy_flags"]))
                if not applicant["hard_reject"] and not applicant["policy_flags"]:
                    st.write("No configured policy rejection indicators or manual-review flags triggered.")

        st.html("<div style='height: 10px;'></div>")

        # Card 2: Reviewer Decision
        with st.container(border=True):
            st.html("""<div class="ll-card-header" style="margin-bottom: 0.65rem;">
<span style="font-size: 1.15rem; color: #4B176F;">👤</span>
<span>Reviewer Decision</span>
</div>""")

            with st.container(key="reviewer_decision_container"):
                final_decision = st.radio(
                    "Decision Choice",
                    options=["✓ Approve", "✓ Reject", "! Request more information"],
                    index=1,
                    horizontal=True,
                    label_visibility="collapsed",
                )

            st.html("<div style='height: 6px;'></div>")

            reviewer_reason = st.text_area(
                "Reason for decision / override",
                placeholder="Briefly explain the reason...",
                height=65,
            )
            char_count_rev = len(reviewer_reason)
            st.html(f"""<div style="text-align: right; font-size: 0.74rem; color: #756B7C; margin-top: -6px; margin-bottom: 8px;">
{char_count_rev}/500
</div>""")

            row_init, row_rec = st.columns([50, 50], vertical_alignment="center")
            with row_init:
                reviewer_initials = st.text_input(
                    "Reviewer initials (optional)",
                    placeholder="e.g. HG",
                    max_chars=10,
                )
            with row_rec:
                save_review = st.button("Record human review →", use_container_width=True)

            if save_review:
                clean_decision = final_decision.replace("✓ ", "").replace("! ", "")
                if clean_decision == "Request more information" and not reviewer_reason.strip():
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
                        "reviewer_decision": clean_decision,
                        "reviewer_reason": reviewer_reason,
                        "reviewer_initials": reviewer_initials,
                    }
                    append_review(record)
                    st.session_state.review_record = record
                    st.success("Human review recorded successfully in local review log.")

            if st.session_state.review_record:
                st.html("<div style='font-size: 0.76rem; color: #756B7C; margin-top: 8px;'>Most recent review recorded in this session:</div>")
                st.json(st.session_state.review_record)

    # ── Educational Prototype & Evaluation Footer ─────────────────────────────
    st.html("<div style='height: 16px;'></div>")
    with st.expander("Model evaluation & dataset context"):
        if DATA_PATH and DATA_PATH.exists():
            df = pd.read_excel(DATA_PATH, sheet_name="Loan_Dataset")
            target = "loan_outcome"
            if target in df.columns:
                counts = df[target].value_counts()
                st.write(f"Synthetic dataset size: **{len(df):,}** records.")
                m1, m2 = st.columns(2)
                m1.metric("Approved labels in dataset", f"{int(counts.get('Approved', 0)):,}")
                m2.metric("Rejected labels in dataset", f"{int(counts.get('Rejected', 0)):,}")
                st.caption("Full metrics and confusion matrix are documented in model_evaluation.md. Dataset counts are not test evaluation metrics.")
        st.warning("This prototype uses a local CSV review log and is not a production audit system. Bharat Horizon Bank is fictional. For educational demonstration only.")
