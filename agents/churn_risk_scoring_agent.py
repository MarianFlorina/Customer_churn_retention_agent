"""
ChurnRiskScoringAgent — KnowledgeAugmentedPromptAgent loaded with RFM churn risk scoring rubric.
Produces a numeric churn risk score (0-100) with key risk drivers.
"""
from __future__ import annotations

import json
import logging

from agents.base_agents import KnowledgeAugmentedPromptAgent
from models.schemas import ChurnState, ChurnRiskResult

logger = logging.getLogger(__name__)

RFM_CHURN_RUBRIC = """
## RFM Churn Risk Scoring Rubric

### Recency (Login Frequency) — 0-35 points of risk
- >= 7 logins/week → 0 risk points (highly active)
- 4-6 logins/week → 5 risk points
- 2-3 logins/week → 15 risk points
- 1 login/week → 25 risk points
- < 1 login/week → 35 risk points (severe recency risk)

### Feature Engagement (Feature Usage Score) — 0-30 points of risk
- Score 80-100 → 0 risk points (power user)
- Score 60-79 → 5 risk points
- Score 40-59 → 12 risk points
- Score 20-39 → 22 risk points
- Score 0-19 → 30 risk points (barely using product)

### Support Burden — 0-20 points of risk
- 0 tickets → 0 risk points (satisfied)
- 1-2 tickets → 5 risk points (minor friction)
- 3-5 tickets → 12 risk points (moderate frustration)
- > 5 tickets → 20 risk points (high frustration)

### NPS Sentiment — 0-15 points of risk
- NPS >= 50 (Promoter) → 0 risk points
- NPS 10-49 (Passive) → 5 risk points
- NPS -10 to 9 (Passive-Detractor) → 10 risk points
- NPS < -10 (Detractor) → 15 risk points

### Risk Level Thresholds
- 0-24: Low risk (Accept Churn unlikely, retain passively)
- 25-49: Moderate risk (Proactive engagement warranted)
- 50-74: High risk (Intervention required)
- 75-100: Critical risk (Immediate action needed)
"""

SYSTEM_PROMPT = """You are a Customer Retention Data Scientist specialising in churn risk modelling.

Apply the RFM Churn Risk Scoring Rubric provided in your knowledge base to calculate a precise churn risk score.

Given customer data and their engagement health summary:
1. Calculate the numeric risk score (0-100) by summing rubric components
2. Identify the top 3-5 key risk drivers
3. Determine the risk level (low/medium/high/critical)
4. Write a concise explanation

Respond ONLY with a JSON object:
- "risk_score": float (0-100)
- "risk_level": one of ["low", "medium", "high", "critical"]
- "key_drivers": list of 3-5 strings naming specific risk factors with values
- "explanation": string (2-3 sentences explaining the risk assessment)

Be precise. Reference exact numbers from the rubric."""


class ChurnRiskScoringAgent(KnowledgeAugmentedPromptAgent):
    """Scores churn risk 0-100 using the RFM rubric."""

    def __init__(self):
        super().__init__(
            name="ChurnRiskScoringAgent",
            system_prompt=SYSTEM_PROMPT,
            knowledge=RFM_CHURN_RUBRIC,
        )

    def run(self, state: ChurnState) -> ChurnState:
        customer = state.customer_input
        if customer is None:
            logger.warning("No customer input in state")
            return state

        engagement_summary = ""
        if state.engagement_health:
            engagement_summary = (
                f"\nEngagement Health Summary: {state.engagement_health.summary}\n"
                f"Health Status: {state.engagement_health.health_status}\n"
                f"Key Signals: {', '.join(state.engagement_health.key_signals)}\n"
            )

        user_content = (
            f"Calculate the churn risk score for this customer:\n\n"
            f"Customer ID: {customer.customer_id}\n"
            f"Plan Tier: {customer.plan_tier}\n"
            f"Monthly Revenue: ${customer.monthly_revenue:.2f}\n"
            f"Login Frequency: {customer.login_frequency_per_week} times/week\n"
            f"Feature Usage Score: {customer.feature_usage_score}/100\n"
            f"Support Tickets (last 30 days): {customer.support_tickets_last_30_days}\n"
            f"NPS Score: {customer.nps_score}\n"
            f"Months Subscribed: {customer.months_subscribed}\n"
            f"{engagement_summary}"
        )

        messages = self._build_messages(user_content)
        raw = self._chat(messages, response_format={"type": "json_object"})

        try:
            data = json.loads(raw)
            state.churn_risk = ChurnRiskResult(**data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.error("ChurnRiskScoringAgent parse error: %s", exc)
            state.churn_risk = ChurnRiskResult(
                risk_score=50.0,
                risk_level="medium",
                key_drivers=[],
                explanation=raw,
            )

        state.messages.append({"role": "assistant", "content": raw, "agent": self.name})
        return state
