"""
RetentionEvaluationAgent — validates personalised offers against quality criteria.

Validates:
- Personalisation relevance (non-generic language)
- Discount within approved thresholds (max 30%)
- Non-generic language
"""
from __future__ import annotations

import json
import logging

from agents.base_agents import EvaluationAgent
from models.schemas import ChurnState, EvaluationResult

logger = logging.getLogger(__name__)

EVALUATION_CRITERIA = """
Evaluate the personalised retention offer against ALL of the following criteria:

1. PERSONALISATION (0-40 points): Does the message reference specific customer details
   (plan tier, usage patterns, specific numbers, tenure)? Is it clearly written FOR this
   specific customer, not a generic template?

2. DISCOUNT COMPLIANCE (0-30 points): If a discount is offered, is it within the 30%
   maximum threshold? A discount > 30% is an automatic FAIL regardless of other scores.
   If no discount is offered, award full 30 points.

3. NON-GENERIC LANGUAGE (0-30 points): Does the message avoid corporate clichés and
   boilerplate phrases? Is it warm, human, and specific? Phrases like "valued customer",
   "we noticed you haven't been using our platform", or "reach out to our team" are
   red flags.

PASS threshold: score >= 70 AND no automatic fails.

Respond ONLY with a JSON object:
- "passed": bool
- "score": float (0-100, sum of the three criteria above)
- "feedback": string (specific actionable feedback for improvement if failed)
- "issues": list of strings (specific issues found, empty if passed)
"""


class RetentionEvaluationAgent(EvaluationAgent):
    """Evaluates personalised retention offers for quality and compliance."""

    def __init__(self):
        super().__init__(
            name="RetentionEvaluationAgent",
            criteria=EVALUATION_CRITERIA,
        )

    def run(self, state: ChurnState) -> ChurnState:
        offer = state.personalised_offer
        customer = state.customer_input

        if offer is None:
            logger.warning("No personalised offer to evaluate")
            return state

        # Hard check: discount > 30% → automatic fail (before LLM)
        if offer.discount_percentage is not None and offer.discount_percentage > 30.0:
            state.evaluation_result = EvaluationResult(
                passed=False,
                score=0.0,
                feedback=(
                    f"AUTOMATIC FAIL: Discount of {offer.discount_percentage:.1f}% "
                    f"exceeds the maximum allowed 30%. Reduce the discount to 30% or below."
                ),
                issues=[f"Discount {offer.discount_percentage:.1f}% exceeds 30% cap"],
            )
            state.passed = False
            state.evaluation_feedback = state.evaluation_result.feedback
            state.iteration += 1
            state.messages.append({
                "role": "assistant",
                "content": state.evaluation_result.model_dump_json(),
                "agent": self.name,
            })
            return state

        customer_id = customer.customer_id if customer else "unknown"
        plan_tier = customer.plan_tier if customer else "unknown"
        months = customer.months_subscribed if customer else 0
        feature_score = customer.feature_usage_score if customer else 0
        login_freq = customer.login_frequency_per_week if customer else 0

        discount_info = f"Discount offered: {offer.discount_percentage}%" if offer.discount_percentage else "No discount offered"

        content_to_eval = (
            f"Customer Context: ID={customer_id}, Plan={plan_tier}, "
            f"Months={months}, Feature Usage={feature_score}/100, "
            f"Logins/week={login_freq}\n\n"
            f"Subject Line: {offer.subject_line}\n\n"
            f"Message Body:\n{offer.message_body}\n\n"
            f"Offer Details: {offer.offer_details}\n"
            f"{discount_info}\n"
            f"Call to Action: {offer.call_to_action}"
        )

        result_dict = self.evaluate(content_to_eval)

        try:
            eval_result = EvaluationResult(**result_dict)
        except Exception as exc:
            logger.error("EvaluationResult parse error: %s", exc)
            eval_result = EvaluationResult(
                passed=False,
                score=0.0,
                feedback=str(result_dict),
                issues=["Parse error in evaluation result"],
            )

        state.evaluation_result = eval_result
        state.passed = eval_result.passed
        state.evaluation_feedback = eval_result.feedback if not eval_result.passed else ""
        state.iteration += 1

        state.messages.append({
            "role": "assistant",
            "content": eval_result.model_dump_json(),
            "agent": self.name,
        })
        return state
