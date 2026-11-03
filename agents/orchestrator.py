"""
ChurnOrchestrator — MultiAgentOrchestrator that manages ChurnState and sequences all workers.
"""
from __future__ import annotations

import logging

from agents.base_agents import MultiAgentOrchestrator
from agents.engagement_analysis_agent import EngagementAnalysisAgent
from agents.churn_risk_scoring_agent import ChurnRiskScoringAgent
from agents.retention_strategy_routing_agent import RetentionStrategyRoutingAgent
from agents.personalised_offer_agent import PersonalisedOfferAgent
from agents.retention_evaluation_agent import RetentionEvaluationAgent
from models.schemas import ChurnState, CustomerReport

logger = logging.getLogger(__name__)

MAX_EVAL_ITERATIONS = 3

PIPELINE_STEPS = [
    "engagement_analysis",
    "churn_risk_scoring",
    "retention_routing",
    "personalised_offer",
    "evaluation",
]


class ChurnOrchestrator(MultiAgentOrchestrator):
    """Orchestrates the full churn prediction and retention advisory pipeline."""

    def __init__(self):
        super().__init__(name="ChurnOrchestrator")
        self.register_agent("engagement_analysis", EngagementAnalysisAgent())
        self.register_agent("churn_risk_scoring", ChurnRiskScoringAgent())
        self.register_agent("retention_routing", RetentionStrategyRoutingAgent())
        self.register_agent("personalised_offer", PersonalisedOfferAgent())
        self.register_agent("evaluation", RetentionEvaluationAgent())

    def route(self, state: ChurnState) -> str:
        """Determine the next step in the pipeline."""
        if state.engagement_health is None:
            return "engagement_analysis"
        if state.churn_risk is None:
            return "churn_risk_scoring"
        if state.retention_strategy is None:
            return "retention_routing"
        if state.personalised_offer is None:
            return "personalised_offer"
        return "evaluation"

    def run(self, state: ChurnState) -> ChurnState:
        """Execute the full pipeline with evaluator-optimizer loop."""
        customer_id = state.customer_input.customer_id if state.customer_input else "unknown"
        logger.info("ChurnOrchestrator: starting pipeline for customer %s", customer_id)

        # Phase 1: Run deterministic pipeline steps
        for step in ["engagement_analysis", "churn_risk_scoring", "retention_routing"]:
            agent = self._agents[step]
            logger.info("Running agent: %s", step)
            state = agent.run(state)

        # Phase 2: Evaluator-Optimizer loop (max 3 iterations)
        for attempt in range(1, MAX_EVAL_ITERATIONS + 1):
            logger.info("Offer generation attempt %d/%d", attempt, MAX_EVAL_ITERATIONS)

            # Generate (or regenerate) offer
            offer_agent = self._agents["personalised_offer"]
            state = offer_agent.run(state)

            # Evaluate
            eval_agent = self._agents["evaluation"]
            state = eval_agent.run(state)

            if state.passed:
                logger.info("Evaluation passed on attempt %d", attempt)
                break

            if attempt < MAX_EVAL_ITERATIONS:
                logger.info(
                    "Evaluation failed (attempt %d). Feedback: %s",
                    attempt,
                    state.evaluation_feedback[:100],
                )
            else:
                logger.warning("Max evaluation iterations reached. Accepting best offer.")

        # Compile final report
        state.report = CustomerReport(
            customer_id=customer_id,
            customer_input=state.customer_input,
            engagement_health=state.engagement_health,
            churn_risk=state.churn_risk,
            retention_strategy=state.retention_strategy,
            personalised_offer=state.personalised_offer,
            evaluation_result=state.evaluation_result,
            evaluation_iterations=state.iteration,
        )

        logger.info(
            "ChurnOrchestrator: pipeline complete for %s. Passed=%s, Iterations=%d",
            customer_id,
            state.passed,
            state.iteration,
        )
        return state
