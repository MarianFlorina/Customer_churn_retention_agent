"""
pipeline.py — Loads sample_input.json, runs the full churn pipeline for each customer,
saves JSON + HTML reports, and opens them in the browser.
"""
from __future__ import annotations

import json
import logging
import os
import webbrowser
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agents.orchestrator import ChurnOrchestrator
from agents.personalised_offer_agent import PersonalisedOfferAgent
from models.schemas import ChurnState, CustomerInput

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PLAYBOOK_PATH = DATA_DIR / "retention_playbook.json"
SAMPLE_INPUT_PATH = DATA_DIR / "sample_input.json"


def load_retention_playbook(offer_agent: PersonalisedOfferAgent) -> None:
    """Index the retention playbook into ChromaDB."""
    with open(PLAYBOOK_PATH, "r", encoding="utf-8") as f:
        playbook = json.load(f)

    documents = [
        {
            "id": entry["id"],
            "text": entry["text"],
            "metadata": entry.get("metadata", {}),
        }
        for entry in playbook
    ]
    offer_agent.index_documents(documents)
    logger.info("Indexed %d retention playbook entries into ChromaDB", len(documents))


def build_html_report(state: ChurnState) -> str:
    """Generate an HTML report from the pipeline state."""
    report = state.report
    if report is None:
        return "<html><body><p>No report generated.</p></body></html>"

    customer = report.customer_input
    engagement = report.engagement_health
    churn_risk = report.churn_risk
    strategy = report.retention_strategy
    offer = report.personalised_offer
    evaluation = report.evaluation_result

    risk_score = churn_risk.risk_score if churn_risk else 0
    risk_color = "#dc3545" if risk_score >= 75 else "#fd7e14" if risk_score >= 50 else "#28a745"

    health_status = engagement.health_status if engagement else "unknown"
    health_color = "#dc3545" if health_status == "critical" else "#fd7e14" if health_status == "declining" else "#28a745"

    eval_color = "#28a745" if (evaluation and evaluation.passed) else "#dc3545"
    eval_status = "PASSED" if (evaluation and evaluation.passed) else "FAILED"

    key_signals_html = ""
    if engagement and engagement.key_signals:
        items = "".join(f"<li>{s}</li>" for s in engagement.key_signals)
        key_signals_html = f"<ul>{items}</ul>"

    key_drivers_html = ""
    if churn_risk and churn_risk.key_drivers:
        items = "".join(f"<li>{d}</li>" for d in churn_risk.key_drivers)
        key_drivers_html = f"<ul>{items}</ul>"

    issues_html = ""
    if evaluation and evaluation.issues:
        items = "".join(f"<li>{i}</li>" for i in evaluation.issues)
        issues_html = f"<ul>{items}</ul>"

    discount_html = ""
    if offer and offer.discount_percentage is not None:
        discount_html = f"<p><strong>Discount Offered:</strong> {offer.discount_percentage:.1f}%</p>"

    offer_section = ""
    if offer:
        message_paragraphs = "".join(
            f"<p>{p}</p>" for p in offer.message_body.split("\n\n") if p.strip()
        )
        offer_section = f"""
        <div class="card">
            <h2>Personalised Retention Offer</h2>
            <p><strong>Subject:</strong> {offer.subject_line}</p>
            <hr/>
            <div class="message-body">{message_paragraphs}</div>
            <div class="offer-box">
                <h3>Offer Details</h3>
                <p>{offer.offer_details}</p>
                {discount_html}
                <p><strong>Call to Action:</strong> {offer.call_to_action}</p>
            </div>
        </div>
        """

    strategy_tier = strategy.strategy_tier if strategy else "Unknown"
    strategy_rationale = strategy.rationale if strategy else ""
    engagement_summary = engagement.summary if engagement else ""
    churn_explanation = churn_risk.explanation if churn_risk else ""
    risk_level = churn_risk.risk_level if churn_risk else "unknown"
    eval_score = evaluation.score if evaluation else 0
    eval_feedback = evaluation.feedback if evaluation else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Churn Retention Report — {report.customer_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                max-width: 900px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 0; }}
        h3 {{ color: #555; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                          gap: 15px; margin: 15px 0; }}
        .metric {{ background: #f8f9fa; border-radius: 6px; padding: 12px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 4px; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px;
                  color: white; font-weight: bold; font-size: 14px; }}
        .risk-score {{ font-size: 48px; font-weight: bold; color: {risk_color}; }}
        .offer-box {{ background: #f0f7ff; border-left: 4px solid #3498db;
                      padding: 15px; margin-top: 15px; border-radius: 4px; }}
        .message-body {{ line-height: 1.7; }}
        .eval-badge {{ background: {eval_color}; }}
        .health-badge {{ background: {health_color}; }}
        ul {{ margin: 8px 0; padding-left: 20px; }}
        li {{ margin: 4px 0; }}
        hr {{ border: none; border-top: 1px solid #eee; margin: 15px 0; }}
        .footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 40px; }}
    </style>
</head>
<body>
    <h1>Customer Churn Retention Report</h1>
    <p><strong>Customer ID:</strong> {report.customer_id} &nbsp;|&nbsp;
       <strong>Plan:</strong> {customer.plan_tier.upper()} &nbsp;|&nbsp;
       <strong>MRR:</strong> ${customer.monthly_revenue:.2f} &nbsp;|&nbsp;
       <strong>Tenure:</strong> {customer.months_subscribed} months &nbsp;|&nbsp;
       <strong>Evaluation Iterations:</strong> {report.evaluation_iterations}</p>

    <div class="card">
        <h2>Customer Metrics</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value">{customer.login_frequency_per_week}</div>
                <div class="metric-label">Logins/Week</div>
            </div>
            <div class="metric">
                <div class="metric-value">{customer.feature_usage_score}</div>
                <div class="metric-label">Feature Usage Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{customer.support_tickets_last_30_days}</div>
                <div class="metric-label">Support Tickets (30d)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{customer.nps_score}</div>
                <div class="metric-label">NPS Score</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Engagement Health &nbsp;
            <span class="badge health-badge">{health_status.upper()}</span>
        </h2>
        <p>{engagement_summary}</p>
        <h3>Key Signals</h3>
        {key_signals_html}
    </div>

    <div class="card">
        <h2>Churn Risk Assessment</h2>
        <div class="risk-score">{risk_score:.0f}/100</div>
        <p><strong>Risk Level:</strong> {risk_level.upper()}</p>
        <p>{churn_explanation}</p>
        <h3>Key Risk Drivers</h3>
        {key_drivers_html}
    </div>

    <div class="card">
        <h2>Retention Strategy</h2>
        <p><strong>Selected Tier:</strong>
           <span class="badge" style="background:#3498db;">{strategy_tier}</span>
        </p>
        <p>{strategy_rationale}</p>
    </div>

    {offer_section}

    <div class="card">
        <h2>Quality Evaluation &nbsp;
            <span class="badge eval-badge">{eval_status}</span>
        </h2>
        <p><strong>Score:</strong> {eval_score:.0f}/100</p>
        <p>{eval_feedback}</p>
        {issues_html}
    </div>

    <div class="footer">
        <p>Generated by Multi-Agent Customer Churn Retention System</p>
    </div>
</body>
</html>"""
    return html


def run_pipeline_for_customer(
    customer_data: dict[str, Any],
    orchestrator: ChurnOrchestrator,
) -> ChurnState:
    """Initialise state and run the full pipeline for one customer."""
    # Filter out _description key
    clean_data = {k: v for k, v in customer_data.items() if not k.startswith("_")}
    customer_input = CustomerInput(**clean_data)

    state = ChurnState(
        customer_input=customer_input,
        metadata={"customer_id": customer_input.customer_id},
    )

    state = orchestrator.run(state)
    return state


def save_reports(state: ChurnState) -> tuple[Path, Path]:
    """Save JSON and HTML reports and return their paths."""
    if state.report is None:
        raise ValueError("No report in state")

    customer_id = state.report.customer_id
    json_path = DATA_DIR / f"report_{customer_id}.json"
    html_path = DATA_DIR / f"report_{customer_id}.html"

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(state.report.model_dump(), f, indent=2, default=str)

    # Save HTML
    html_content = build_html_report(state)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Saved reports: %s, %s", json_path, html_path)
    return json_path, html_path


def run_full_pipeline() -> list[dict[str, Any]]:
    """Run the pipeline for all customers in sample_input.json."""
    # Load sample data
    with open(SAMPLE_INPUT_PATH, "r", encoding="utf-8") as f:
        customers = json.load(f)

    # Initialise orchestrator (registers all agents)
    orchestrator = ChurnOrchestrator()

    # Index the retention playbook into the RAG agent
    offer_agent = orchestrator._agents["personalised_offer"]
    load_retention_playbook(offer_agent)

    results = []
    for customer_data in customers:
        customer_id = customer_data.get("customer_id", "unknown")
        logger.info("=" * 60)
        logger.info("Processing customer: %s", customer_id)
        logger.info("=" * 60)

        try:
            state = run_pipeline_for_customer(customer_data, orchestrator)
            json_path, html_path = save_reports(state)

            result = {
                "customer_id": customer_id,
                "status": "success",
                "passed": state.passed,
                "iterations": state.iteration,
                "json_path": str(json_path),
                "html_path": str(html_path),
            }
            results.append(result)

            # Open HTML report in browser
            webbrowser.open(html_path.as_uri())

        except Exception as exc:
            logger.error("Pipeline failed for customer %s: %s", customer_id, exc)
            results.append({"customer_id": customer_id, "status": "error", "error": str(exc)})

    return results


if __name__ == "__main__":
    results = run_full_pipeline()
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for r in results:
        status = r.get("status", "unknown")
        if status == "success":
            passed = "PASSED" if r.get("passed") else "FAILED"
            print(f"[{r['customer_id']}] {passed} ({r['iterations']} iterations)")
        else:
            print(f"[{r['customer_id']}] ERROR: {r.get('error', 'unknown')}")
