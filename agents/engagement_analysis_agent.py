"""
EngagementAnalysisAgent — interprets customer behavioural signals and summarises engagement health.
"""
from __future__ import annotations

import json
import logging

from agents.base_agents import AugmentedPromptAgent
from models.schemas import ChurnState, EngagementHealth

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior Customer Success analyst specialising in SaaS engagement health.

Your role is to interpret customer behavioural signals and produce a clear engagement health summary.

Given customer behavioural data, analyse:
- Login frequency relative to plan expectations
- Feature adoption depth (feature_usage_score 0-100)
- Support burden (support tickets as a churn signal)
- Net Promoter Score sentiment
- Tenure and lifecycle stage

Respond ONLY with a JSON object with these keys:
- "summary": string (2-3 sentences describing overall engagement health)
- "health_status": one of ["healthy", "declining", "critical"]
- "key_signals": list of 3-5 specific behavioural observations

Be specific. Reference actual numbers from the data."""


class EngagementAnalysisAgent(AugmentedPromptAgent):
    """Analyses customer behavioural signals and summarises engagement health."""

    def __init__(self):
        super().__init__(
            name="EngagementAnalysisAgent",
            system_prompt=SYSTEM_PROMPT,
        )

    def run(self, state: ChurnState) -> ChurnState:
        customer = state.customer_input
        if customer is None:
            logger.warning("No customer input in state")
            return state

        user_content = (
            f"Analyse the engagement health for this customer:\n\n"
            f"Customer ID: {customer.customer_id}\n"
            f"Plan Tier: {customer.plan_tier}\n"
            f"Monthly Revenue: ${customer.monthly_revenue:.2f}\n"
            f"Login Frequency: {customer.login_frequency_per_week} times/week\n"
            f"Feature Usage Score: {customer.feature_usage_score}/100\n"
            f"Support Tickets (last 30 days): {customer.support_tickets_last_30_days}\n"
            f"NPS Score: {customer.nps_score}\n"
            f"Months Subscribed: {customer.months_subscribed}\n"
        )

        messages = self._build_messages(user_content)
        raw = self._chat(messages, response_format={"type": "json_object"})

        try:
            data = json.loads(raw)
            state.engagement_health = EngagementHealth(**data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.error("EngagementAnalysisAgent parse error: %s", exc)
            state.engagement_health = EngagementHealth(
                summary=raw,
                health_status="declining",
                key_signals=[],
            )

        state.messages.append({"role": "assistant", "content": raw, "agent": self.name})
        return state
