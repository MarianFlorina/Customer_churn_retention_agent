"""
app.py — Streamlit UI for the Customer Churn Retention Agent System.
Run with: streamlit run app.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PLAYBOOK_PATH = DATA_DIR / "retention_playbook.json"

st.set_page_config(
    page_title="Customer Churn Retention Agent",
    page_icon="🔄",
    layout="wide",
)

st.title("Customer Churn Prediction & Retention Advisory System")
st.markdown("*Multi-Agent AI Pipeline — Orchestrator-Workers Pattern*")


@st.cache_resource
def get_orchestrator_and_index():
    from agents.orchestrator import ChurnOrchestrator
    from pipeline import load_retention_playbook

    orchestrator = ChurnOrchestrator()
    offer_agent = orchestrator._agents["personalised_offer"]
    load_retention_playbook(offer_agent)
    return orchestrator


def run_single_customer(customer_data: dict) -> None:
    from pipeline import run_pipeline_for_customer, build_html_report
    orchestrator = get_orchestrator_and_index()

    with st.spinner(f"Running pipeline for {customer_data.get('customer_id', 'customer')}..."):
        state = run_pipeline_for_customer(customer_data, orchestrator)

    if state.report is None:
        st.error("Pipeline failed — no report generated")
        return

    report = state.report
    col1, col2, col3, col4 = st.columns(4)

    risk_score = report.churn_risk.risk_score if report.churn_risk else 0
    with col1:
        st.metric("Churn Risk Score", f"{risk_score:.0f}/100")
    with col2:
        health = report.engagement_health.health_status if report.engagement_health else "unknown"
        st.metric("Engagement Health", health.upper())
    with col3:
        strategy = report.retention_strategy.strategy_tier if report.retention_strategy else "None"
        st.metric("Strategy", strategy)
    with col4:
        eval_status = "PASSED" if state.passed else "FAILED"
        st.metric("Evaluation", eval_status)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Engagement Analysis", "Churn Risk", "Retention Strategy", "Offer", "Evaluation"
    ])

    with tab1:
        if report.engagement_health:
            st.write(report.engagement_health.summary)
            st.write("**Key Signals:**")
            for signal in report.engagement_health.key_signals:
                st.write(f"- {signal}")

    with tab2:
        if report.churn_risk:
            st.metric("Risk Score", f"{report.churn_risk.risk_score:.1f}/100")
            st.write(report.churn_risk.explanation)
            st.write("**Key Drivers:**")
            for driver in report.churn_risk.key_drivers:
                st.write(f"- {driver}")

    with tab3:
        if report.retention_strategy:
            st.info(f"**{report.retention_strategy.strategy_tier}**")
            st.write(report.retention_strategy.rationale)

    with tab4:
        if report.personalised_offer:
            offer = report.personalised_offer
            st.write(f"**Subject:** {offer.subject_line}")
            st.divider()
            st.write(offer.message_body)
            st.divider()
            st.info(f"**Offer:** {offer.offer_details}")
            if offer.discount_percentage:
                st.write(f"**Discount:** {offer.discount_percentage:.1f}%")
            st.write(f"**Call to Action:** {offer.call_to_action}")

    with tab5:
        if report.evaluation_result:
            eval_result = report.evaluation_result
            color = "green" if eval_result.passed else "red"
            status = "PASSED" if eval_result.passed else "FAILED"
            st.markdown(f"**Status:** :{color}[{status}]")
            st.metric("Score", f"{eval_result.score:.0f}/100")
            st.write(eval_result.feedback)
            if eval_result.issues:
                st.write("**Issues:**")
                for issue in eval_result.issues:
                    st.write(f"- {issue}")

        st.write(f"**Evaluation iterations:** {report.evaluation_iterations}")

    # Full JSON report
    with st.expander("Full JSON Report"):
        st.json(report.model_dump())

    # HTML preview
    html_content = build_html_report(state)
    with st.expander("HTML Report Preview"):
        st.components.v1.html(html_content, height=800, scrolling=True)


# Sidebar: load sample customers
st.sidebar.header("Customer Selection")

sample_path = DATA_DIR / "sample_input.json"
if sample_path.exists():
    with open(sample_path, "r") as f:
        sample_customers = json.load(f)

    customer_ids = [c["customer_id"] for c in sample_customers]
    selected_id = st.sidebar.selectbox("Choose a sample customer", customer_ids)
    selected_customer = next(c for c in sample_customers if c["customer_id"] == selected_id)

    clean_customer = {k: v for k, v in selected_customer.items() if not k.startswith("_")}

    st.sidebar.json(clean_customer)

    if st.sidebar.button("Run Retention Pipeline", type="primary"):
        run_single_customer(clean_customer)

# Custom customer input
st.sidebar.divider()
st.sidebar.header("Custom Customer")

with st.sidebar.form("custom_customer"):
    cid = st.text_input("Customer ID", value="CUST-0001")
    plan_tier = st.selectbox("Plan Tier", ["free", "starter", "pro", "enterprise"])
    monthly_revenue = st.number_input("Monthly Revenue ($)", min_value=0.0, value=299.0)
    login_freq = st.slider("Login Frequency (per week)", 0.0, 14.0, 2.0)
    feature_score = st.slider("Feature Usage Score (0-100)", 0, 100, 35)
    support_tickets = st.number_input("Support Tickets (30 days)", min_value=0, value=2)
    nps = st.slider("NPS Score", -100, 100, 10)
    months = st.number_input("Months Subscribed", min_value=1, value=6)

    submitted = st.form_submit_button("Analyse Custom Customer")

if submitted:
    custom_customer = {
        "customer_id": cid,
        "plan_tier": plan_tier,
        "monthly_revenue": monthly_revenue,
        "login_frequency_per_week": login_freq,
        "feature_usage_score": feature_score,
        "support_tickets_last_30_days": support_tickets,
        "nps_score": nps,
        "months_subscribed": months,
    }
    run_single_customer(custom_customer)
