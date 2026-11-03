"""
RetentionStrategyRoutingAgent — RoutingAgent that classifies customers into retention strategy tiers.

Routing thresholds:
- risk_score >= 75 AND plan_tier == 'enterprise' → Account Review Call
- risk_score >= 75 → Discount Offer
- risk_score 50-74 AND feature_usage_score < 40 → Feature Education
- risk_score 50-74 → Proactive Engagement
- risk_score < 50 → Accept Churn
"""
from __future__ import annotations

import logging

from agents.base_agents import RoutingAgent
from models.schemas import ChurnState, RetentionStrategy

logger = logging.getLogger(__name__)

ROUTES = [
    "Account Review Call",
    "Discount Offer",
    "Feature Education",
    "Proactive Engagement",
    "Accept Churn",
]

SYSTEM_PROMPT = """You are a Customer Retention Strategy specialist.

Your role is to classify customers into the most appropriate retention intervention tier.
You receive the customer's risk score, plan tier, and other contextual information.

Always apply the routing rules precisely as defined. Return ONLY a JSON object with:
- "strategy_tier": the exact route name chosen
- "rationale": 1-2 sentences explaining why this tier was chosen
"""


def _apply_routing_rules(risk_score: float, plan_tier: str, feature_usage_score: float) -> str:
    """Deterministic rule-based routing — no LLM needed for the decision."""
    tier_lower = plan_tier.lower()
    if risk_score >= 75 and tier_lower == "enterprise":
        return "Account Review Call"
    if risk_score >= 75:
        return "Discount Offer"
    if 50 <= risk_score < 75 and feature_usage_score < 40:
        return "Feature Education"
    if 50 <= risk_score < 75:
        return "Proactive Engagement"
    return "Accept Churn"


class RetentionStrategyRoutingAgent(RoutingAgent):
    """Classifies customers into retention strategy tiers based on risk score."""

    def __init__(self):
        super().__init__(
            name="RetentionStrategyRoutingAgent",
            system_prompt=SYSTEM_PROMPT,
            routes=ROUTES,
        )

    def run(self, state: ChurnState) -> ChurnState:
        customer = state.customer_input
        churn_risk = state.churn_risk

        if customer is None or churn_risk is None:
            logger.warning("Missing customer input or churn risk in state")
            return state

        # Use deterministic rules for reliable routing
        strategy_tier = _apply_routing_rules(
            risk_score=churn_risk.risk_score,
            plan_tier=customer.plan_tier,
            feature_usage_score=customer.feature_usage_score,
        )

        drivers_str = ", ".join(churn_risk.key_drivers[:3])
        rationale = (
            f"Risk score of {churn_risk.risk_score:.1f} ({churn_risk.risk_level}) "
            f"on {customer.plan_tier} plan with feature usage {customer.feature_usage_score}/100. "
            f"Key drivers: {drivers_str}."
        )

        state.retention_strategy = RetentionStrategy(
            strategy_tier=strategy_tier,
            rationale=rationale,
        )

        state.messages.append({
            "role": "assistant",
            "content": strategy_tier,
            "agent": self.name,
        })
        state.metadata["route"] = strategy_tier
        return state
