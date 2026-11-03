"""
PersonalisedOfferAgent — RAGKnowledgePromptAgent over retention playbook ChromaDB KB.
Generates personalised retention messages and offers.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from agents.base_agents import RAGKnowledgePromptAgent
from models.schemas import ChurnState, PersonalisedOffer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Customer Success Manager specialising in personalised retention outreach.

Your role is to craft compelling, personalised retention messages and offers that:
1. Directly address the customer's specific pain points and usage patterns
2. Reference concrete value the customer gets (or could get) from the product
3. Include a specific, actionable offer tailored to their plan tier and risk level
4. Use warm, human language — never generic or templated-sounding

Important constraints:
- Discount offers MUST NOT exceed 30% off their current monthly revenue
- Messages must be personalised to the specific customer — no generic copy
- Offers must align with the retention strategy tier selected

Use the retrieved retention playbook content to inform your approach.

Respond ONLY with a JSON object:
- "subject_line": string (email subject, personalised and compelling)
- "message_body": string (3-4 paragraphs of personalised outreach)
- "offer_details": string (specific offer description)
- "discount_percentage": float or null (percentage discount if applicable, max 30.0)
- "call_to_action": string (specific next step for the customer)"""


class PersonalisedOfferAgent(RAGKnowledgePromptAgent):
    """Generates personalised retention messages using RAG over the retention playbook."""

    def __init__(self):
        super().__init__(
            name="PersonalisedOfferAgent",
            system_prompt=SYSTEM_PROMPT,
            collection_name="retention_playbook",
        )

    def run(self, state: ChurnState) -> ChurnState:
        customer = state.customer_input
        churn_risk = state.churn_risk
        engagement = state.engagement_health
        strategy = state.retention_strategy

        if customer is None or churn_risk is None:
            logger.warning("Missing required state for PersonalisedOfferAgent")
            return state

        strategy_tier = strategy.strategy_tier if strategy else "Proactive Engagement"
        health_status = engagement.health_status if engagement else "declining"
        engagement_summary = engagement.summary if engagement else ""
        drivers = ", ".join(churn_risk.key_drivers[:3]) if churn_risk.key_drivers else "low engagement"

        prev_feedback = ""
        if state.evaluation_feedback:
            prev_feedback = f"\n\nIMPORTANT - Previous offer was rejected. Feedback to address:\n{state.evaluation_feedback}"

        user_content = (
            f"Generate a personalised retention offer for this customer.\n\n"
            f"Customer ID: {customer.customer_id}\n"
            f"Plan Tier: {customer.plan_tier}\n"
            f"Monthly Revenue: ${customer.monthly_revenue:.2f}\n"
            f"Months Subscribed: {customer.months_subscribed}\n"
            f"Churn Risk Score: {churn_risk.risk_score:.1f}/100 ({churn_risk.risk_level})\n"
            f"Key Risk Drivers: {drivers}\n"
            f"Engagement Health: {health_status}\n"
            f"Engagement Summary: {engagement_summary}\n"
            f"Login Frequency: {customer.login_frequency_per_week} times/week\n"
            f"Feature Usage Score: {customer.feature_usage_score}/100\n"
            f"NPS Score: {customer.nps_score}\n"
            f"Retention Strategy: {strategy_tier}\n"
            f"Max Allowed Discount: 30% (max ${customer.monthly_revenue * 0.30:.2f}/month savings)"
            f"{prev_feedback}"
        )

        messages = self._build_messages(user_content)
        raw = self._chat(messages, response_format={"type": "json_object"})

        try:
            data = json.loads(raw)
            # Enforce discount cap before storing
            if data.get("discount_percentage") is not None:
                data["discount_percentage"] = min(float(data["discount_percentage"]), 30.0)
            state.personalised_offer = PersonalisedOffer(**data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.error("PersonalisedOfferAgent parse error: %s", exc)
            state.personalised_offer = PersonalisedOffer(
                subject_line="We value your partnership",
                message_body=raw,
                offer_details="Please contact your account manager.",
                discount_percentage=None,
                call_to_action="Schedule a call",
            )

        state.messages.append({"role": "assistant", "content": raw, "agent": self.name})
        return state
