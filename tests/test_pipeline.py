"""
Tests for the Customer Churn Retention Agent Pipeline.

Covers:
- Routing threshold logic
- Discount cap enforcement
- End-to-end pipeline execution (mocked)
- Evaluation loop termination within 3 iterations
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers: build a mock OpenAI response
# ---------------------------------------------------------------------------
def _make_openai_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_chroma_mock():
    """Return a mock chromadb.Client that handles in-memory collection ops."""
    collection = MagicMock()
    collection.count.return_value = 5
    collection.query.return_value = {
        "documents": [["Mock retention playbook entry about strategy"]],
        "metadatas": [[{}]],
    }

    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client, collection


def _make_embedder_mock():
    """Return a mock SentenceTransformer whose encode() returns a list-able ndarray."""
    embedder = MagicMock()
    # encode() must return something with .tolist() for single calls and indexing for multi
    encoded = MagicMock()
    encoded.tolist.return_value = [[0.1] * 384]
    # Also support [0] indexing used in retrieve()
    embedder.encode.return_value = encoded
    return embedder


# ---------------------------------------------------------------------------
# Fixture: mock OpenAI client
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_openai(monkeypatch):
    """Patch the OpenAI singleton and chat completions."""
    mock_client = MagicMock()

    def side_effect(*args, **kwargs):
        messages = kwargs.get("messages", args[0] if args else [])
        # Determine intent from both system and user messages
        all_content = " ".join(m.get("content", "") for m in messages).lower()

        # Order matters: most specific checks first
        if "analyse the engagement health" in all_content:
            content = json.dumps({
                "summary": "Customer shows declining engagement with reduced logins.",
                "health_status": "declining",
                "key_signals": ["Login frequency dropped to 1.5/week", "High support burden"],
            })
        elif "calculate the churn risk score" in all_content:
            content = json.dumps({
                "risk_score": 80.0,
                "risk_level": "critical",
                "key_drivers": ["Low login frequency", "High support tickets", "Negative NPS"],
                "explanation": "Customer shows critical churn signals across all RFM dimensions.",
            })
        elif "generate a personalised retention offer" in all_content:
            content = json.dumps({
                "subject_line": "A personal note from your Customer Success team, ENT-7842",
                "message_body": (
                    "Hi there,\n\nAs an enterprise customer with 18 months of subscription, "
                    "your account represents significant value to us.\n\n"
                    "We noticed your login frequency has dropped to 1.5 times per week and "
                    "you have 7 open support tickets. We want to resolve this personally.\n\n"
                    "Your dedicated CSM would like to schedule an account review."
                ),
                "offer_details": "20% loyalty discount for the next 3 months plus dedicated priority support",
                "discount_percentage": 20.0,
                "call_to_action": "Schedule a call with your dedicated CSM",
            })
        elif "evaluation criteria" in all_content or "evaluate the personalised" in all_content:
            content = json.dumps({
                "passed": True,
                "score": 82.0,
                "feedback": "Offer is personalised and within discount threshold.",
                "issues": [],
            })
        else:
            content = json.dumps({"result": "mock response"})

        return _make_openai_response(content)

    mock_client.chat.completions.create.side_effect = side_effect

    # Patch the lazy singleton
    import agents.base_agents as base_agents
    monkeypatch.setattr(base_agents, "_openai_client", mock_client)
    return mock_client


@pytest.fixture
def mock_chromadb(monkeypatch):
    """Patch chromadb.Client and SentenceTransformer."""
    chroma_client, collection = _make_chroma_mock()
    embedder = _make_embedder_mock()

    import agents.base_agents as base_agents
    monkeypatch.setattr("chromadb.Client", lambda: chroma_client)

    # Patch SentenceTransformer
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", lambda model: embedder)

    return chroma_client, collection, embedder


# ---------------------------------------------------------------------------
# Test 1: Routing threshold logic
# ---------------------------------------------------------------------------
class TestRoutingThresholds:
    """Test that churn routing rules map correctly."""

    def _get_route(self, risk_score: float, plan_tier: str, feature_usage_score: float) -> str:
        from agents.retention_strategy_routing_agent import _apply_routing_rules
        return _apply_routing_rules(risk_score, plan_tier, feature_usage_score)

    def test_enterprise_high_risk_gets_account_review_call(self):
        route = self._get_route(risk_score=80.0, plan_tier="enterprise", feature_usage_score=70)
        assert route == "Account Review Call"

    def test_enterprise_high_risk_case_insensitive(self):
        route = self._get_route(risk_score=75.0, plan_tier="Enterprise", feature_usage_score=50)
        assert route == "Account Review Call"

    def test_non_enterprise_high_risk_gets_discount_offer(self):
        route = self._get_route(risk_score=76.0, plan_tier="pro", feature_usage_score=60)
        assert route == "Discount Offer"

    def test_pro_high_risk_boundary_75_gets_discount(self):
        route = self._get_route(risk_score=75.0, plan_tier="pro", feature_usage_score=50)
        assert route == "Discount Offer"

    def test_medium_risk_low_feature_usage_gets_feature_education(self):
        route = self._get_route(risk_score=60.0, plan_tier="pro", feature_usage_score=30)
        assert route == "Feature Education"

    def test_medium_risk_low_feature_boundary(self):
        route = self._get_route(risk_score=50.0, plan_tier="starter", feature_usage_score=39)
        assert route == "Feature Education"

    def test_medium_risk_adequate_feature_usage_gets_proactive_engagement(self):
        route = self._get_route(risk_score=65.0, plan_tier="pro", feature_usage_score=50)
        assert route == "Proactive Engagement"

    def test_medium_risk_boundary_50_gets_proactive_engagement(self):
        route = self._get_route(risk_score=50.0, plan_tier="pro", feature_usage_score=40)
        assert route == "Proactive Engagement"

    def test_low_risk_gets_accept_churn(self):
        route = self._get_route(risk_score=30.0, plan_tier="free", feature_usage_score=10)
        assert route == "Accept Churn"

    def test_boundary_just_below_50_gets_accept_churn(self):
        route = self._get_route(risk_score=49.9, plan_tier="pro", feature_usage_score=20)
        assert route == "Accept Churn"

    def test_zero_risk_gets_accept_churn(self):
        route = self._get_route(risk_score=0.0, plan_tier="enterprise", feature_usage_score=100)
        assert route == "Accept Churn"

    def test_exactly_75_enterprise_gets_account_review(self):
        route = self._get_route(risk_score=75.0, plan_tier="enterprise", feature_usage_score=80)
        assert route == "Account Review Call"

    def test_free_tier_high_risk_gets_discount_not_account_review(self):
        route = self._get_route(risk_score=90.0, plan_tier="free", feature_usage_score=5)
        assert route == "Discount Offer"


# ---------------------------------------------------------------------------
# Test 2: Discount cap enforcement
# ---------------------------------------------------------------------------
class TestDiscountCapEnforcement:
    """Test that discount > 30% is caught by the evaluator."""

    def test_evaluator_fails_discount_over_30_percent(self):
        from agents.retention_evaluation_agent import RetentionEvaluationAgent
        from models.schemas import ChurnState, CustomerInput, PersonalisedOffer

        agent = RetentionEvaluationAgent()
        customer = CustomerInput(
            customer_id="TEST-001",
            plan_tier="pro",
            monthly_revenue=500.0,
            login_frequency_per_week=1.0,
            feature_usage_score=30.0,
            support_tickets_last_30_days=2,
            nps_score=10,
            months_subscribed=6,
        )
        # Offer with illegal 40% discount
        offer = PersonalisedOffer(
            subject_line="Special offer for you",
            message_body="We would like to offer you a significant discount.",
            offer_details="40% off for 3 months",
            discount_percentage=40.0,
            call_to_action="Reply to accept",
        )
        state = ChurnState(customer_input=customer, personalised_offer=offer)

        result_state = agent.run(state)

        assert result_state.evaluation_result is not None
        assert result_state.evaluation_result.passed is False
        assert result_state.passed is False
        assert "40" in result_state.evaluation_result.feedback or "30" in result_state.evaluation_result.feedback

    def test_evaluator_does_not_fail_30_percent_exactly(self, mock_openai):
        from agents.retention_evaluation_agent import RetentionEvaluationAgent
        from models.schemas import ChurnState, CustomerInput, PersonalisedOffer

        agent = RetentionEvaluationAgent()
        customer = CustomerInput(
            customer_id="TEST-002",
            plan_tier="pro",
            monthly_revenue=500.0,
            login_frequency_per_week=2.0,
            feature_usage_score=45.0,
            support_tickets_last_30_days=1,
            nps_score=20,
            months_subscribed=12,
        )
        offer = PersonalisedOffer(
            subject_line="Loyalty discount for you",
            message_body="As a valued Pro customer with 12 months tenure, we want to offer you a loyalty discount.",
            offer_details="30% off for 2 months",
            discount_percentage=30.0,
            call_to_action="Accept offer",
        )
        state = ChurnState(customer_input=customer, personalised_offer=offer)

        # Mock the LLM evaluation to pass
        result_state = agent.run(state)

        # Hard discount check should not trigger (30% is exactly the limit)
        # The LLM mock returns passed=True
        assert result_state.evaluation_result is not None
        # Should not be an automatic fail
        assert "AUTOMATIC FAIL" not in (result_state.evaluation_result.feedback or "")

    def test_personalised_offer_agent_caps_discount_at_30(self, mock_openai, mock_chromadb, monkeypatch):
        """PersonalisedOfferAgent should cap any discount returned by the LLM at 30%."""
        from agents.personalised_offer_agent import PersonalisedOfferAgent
        from models.schemas import ChurnState, CustomerInput, ChurnRiskResult, RetentionStrategy

        # Override the mock to return a 50% discount (clear side_effect first)
        import agents.base_agents as base_agents
        mock_client = base_agents._openai_client
        over_limit_content = json.dumps({
            "subject_line": "Big discount offer",
            "message_body": "We're offering you a huge discount.",
            "offer_details": "50% off",
            "discount_percentage": 50.0,
            "call_to_action": "Claim now",
        })
        mock_client.chat.completions.create.side_effect = None
        mock_client.chat.completions.create.return_value = _make_openai_response(over_limit_content)

        agent = PersonalisedOfferAgent()
        # Inject mocked chromadb
        chroma_client, _, embedder = mock_chromadb
        agent._chroma_client = chroma_client
        agent._collection = chroma_client.get_or_create_collection("retention_playbook")
        agent._embedder = embedder

        customer = CustomerInput(
            customer_id="TEST-003",
            plan_tier="pro",
            monthly_revenue=300.0,
            login_frequency_per_week=0.5,
            feature_usage_score=15.0,
            support_tickets_last_30_days=5,
            nps_score=-20,
            months_subscribed=4,
        )
        state = ChurnState(
            customer_input=customer,
            churn_risk=ChurnRiskResult(
                risk_score=80.0,
                risk_level="critical",
                key_drivers=["Low logins"],
                explanation="High risk.",
            ),
            retention_strategy=RetentionStrategy(
                strategy_tier="Discount Offer",
                rationale="High risk customer.",
            ),
        )

        result_state = agent.run(state)

        assert result_state.personalised_offer is not None
        assert result_state.personalised_offer.discount_percentage <= 30.0


# ---------------------------------------------------------------------------
# Test 3: End-to-end pipeline (fully mocked)
# ---------------------------------------------------------------------------
class TestEndToEndPipeline:
    """Test the full pipeline runs without errors."""

    def _make_customer_state(self):
        from models.schemas import ChurnState, CustomerInput
        customer = CustomerInput(
            customer_id="E2E-001",
            plan_tier="enterprise",
            monthly_revenue=4500.0,
            login_frequency_per_week=1.5,
            feature_usage_score=72.0,
            support_tickets_last_30_days=7,
            nps_score=-15,
            months_subscribed=18,
        )
        return ChurnState(customer_input=customer)

    def test_pipeline_runs_end_to_end(self, mock_openai, mock_chromadb):
        """Full pipeline executes without exception and produces a report."""
        from agents.orchestrator import ChurnOrchestrator
        from models.schemas import ChurnState

        orchestrator = ChurnOrchestrator()

        # Inject mocked chromadb into the offer agent
        chroma_client, collection, embedder = mock_chromadb
        offer_agent = orchestrator._agents["personalised_offer"]
        offer_agent._chroma_client = chroma_client
        offer_agent._collection = collection
        offer_agent._embedder = embedder

        state = self._make_customer_state()
        result = orchestrator.run(state)

        assert result.report is not None
        assert result.report.customer_id == "E2E-001"
        assert result.engagement_health is not None
        assert result.churn_risk is not None
        assert result.retention_strategy is not None
        assert result.personalised_offer is not None
        assert result.evaluation_result is not None

    def test_pipeline_produces_correct_strategy_for_enterprise_high_risk(self, mock_openai, mock_chromadb):
        """Enterprise high-risk customer should route to Account Review Call."""
        from agents.orchestrator import ChurnOrchestrator

        orchestrator = ChurnOrchestrator()
        chroma_client, collection, embedder = mock_chromadb
        offer_agent = orchestrator._agents["personalised_offer"]
        offer_agent._chroma_client = chroma_client
        offer_agent._collection = collection
        offer_agent._embedder = embedder

        state = self._make_customer_state()
        result = orchestrator.run(state)

        # ENT customer with risk_score=80 should get Account Review Call
        assert result.retention_strategy is not None
        assert result.retention_strategy.strategy_tier == "Account Review Call"

    def test_pipeline_report_has_all_fields(self, mock_openai, mock_chromadb):
        """CustomerReport should have all required fields populated."""
        from agents.orchestrator import ChurnOrchestrator

        orchestrator = ChurnOrchestrator()
        chroma_client, collection, embedder = mock_chromadb
        offer_agent = orchestrator._agents["personalised_offer"]
        offer_agent._chroma_client = chroma_client
        offer_agent._collection = collection
        offer_agent._embedder = embedder

        state = self._make_customer_state()
        result = orchestrator.run(state)

        report = result.report
        assert report.engagement_health is not None
        assert report.churn_risk is not None
        assert report.retention_strategy is not None
        assert report.personalised_offer is not None
        assert report.evaluation_result is not None
        assert report.evaluation_iterations >= 1


# ---------------------------------------------------------------------------
# Test 4: Evaluation loop terminates within 3 iterations
# ---------------------------------------------------------------------------
class TestEvaluationLoop:
    """Test the evaluator-optimizer loop terminates within max 3 iterations."""

    def test_loop_terminates_when_evaluation_passes_immediately(self, mock_openai, mock_chromadb):
        """If evaluation passes on first attempt, loop should stop at iteration 1."""
        from agents.orchestrator import ChurnOrchestrator
        from models.schemas import CustomerInput, ChurnState

        orchestrator = ChurnOrchestrator()
        chroma_client, collection, embedder = mock_chromadb
        offer_agent = orchestrator._agents["personalised_offer"]
        offer_agent._chroma_client = chroma_client
        offer_agent._collection = collection
        offer_agent._embedder = embedder

        customer = CustomerInput(
            customer_id="LOOP-001",
            plan_tier="pro",
            monthly_revenue=299.0,
            login_frequency_per_week=2.0,
            feature_usage_score=40.0,
            support_tickets_last_30_days=1,
            nps_score=5,
            months_subscribed=6,
        )
        state = ChurnState(customer_input=customer)
        result = orchestrator.run(state)

        # Mock returns passed=True, so should pass on first attempt
        assert result.iteration <= 3
        assert result.passed is True

    def test_loop_terminates_after_max_3_iterations_on_persistent_failure(self, mock_chromadb, monkeypatch):
        """Loop must not exceed 3 iterations even if evaluation always fails."""
        import agents.base_agents as base_agents

        mock_client = MagicMock()

        def always_fail_eval(*args, **kwargs):
            messages = kwargs.get("messages", [])
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = m.get("content", "")
                    break

            if "Analyse the engagement" in last_user:
                content = json.dumps({
                    "summary": "Customer shows declining engagement.",
                    "health_status": "declining",
                    "key_signals": ["Low logins"],
                })
            elif "Calculate the churn risk" in last_user:
                content = json.dumps({
                    "risk_score": 60.0,
                    "risk_level": "high",
                    "key_drivers": ["Low logins", "Low feature usage"],
                    "explanation": "Medium-high risk.",
                })
            elif "Generate a personalised retention offer" in last_user or "generate a personalised" in last_user.lower():
                content = json.dumps({
                    "subject_line": "Generic offer",
                    "message_body": "Dear valued customer, we noticed you haven't been using our platform.",
                    "offer_details": "10% discount",
                    "discount_percentage": 10.0,
                    "call_to_action": "Contact us",
                })
            else:
                # Evaluation always fails
                content = json.dumps({
                    "passed": False,
                    "score": 40.0,
                    "feedback": "Message is too generic. Lacks personalisation.",
                    "issues": ["Generic language detected", "Missing specific customer references"],
                })

            return _make_openai_response(content)

        mock_client.chat.completions.create.side_effect = always_fail_eval
        monkeypatch.setattr(base_agents, "_openai_client", mock_client)

        from agents.orchestrator import ChurnOrchestrator
        from models.schemas import CustomerInput, ChurnState

        orchestrator = ChurnOrchestrator()
        chroma_client, collection, embedder = mock_chromadb
        offer_agent = orchestrator._agents["personalised_offer"]
        offer_agent._chroma_client = chroma_client
        offer_agent._collection = collection
        offer_agent._embedder = embedder

        customer = CustomerInput(
            customer_id="LOOP-002",
            plan_tier="starter",
            monthly_revenue=99.0,
            login_frequency_per_week=1.0,
            feature_usage_score=25.0,
            support_tickets_last_30_days=3,
            nps_score=-5,
            months_subscribed=4,
        )
        state = ChurnState(customer_input=customer)
        result = orchestrator.run(state)

        # Must not exceed 3 evaluation iterations
        assert result.iteration <= 3, f"Expected <= 3 iterations, got {result.iteration}"

    def test_loop_terminates_on_second_attempt(self, mock_chromadb, monkeypatch):
        """Loop should stop on the second attempt when evaluation passes."""
        import agents.base_agents as base_agents

        call_counts = {"eval_calls": 0}
        mock_client = MagicMock()

        def pass_on_second_eval(*args, **kwargs):
            messages = kwargs.get("messages", [])
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = m.get("content", "")
                    break

            if "Analyse the engagement" in last_user:
                content = json.dumps({
                    "summary": "Customer shows some engagement decline.",
                    "health_status": "declining",
                    "key_signals": ["Reduced logins"],
                })
            elif "Calculate the churn risk" in last_user:
                content = json.dumps({
                    "risk_score": 55.0,
                    "risk_level": "high",
                    "key_drivers": ["Reduced logins"],
                    "explanation": "Moderate risk.",
                })
            elif "Generate a personalised" in last_user or "generate a personalised" in last_user.lower():
                content = json.dumps({
                    "subject_line": "Personalised offer for you",
                    "message_body": "Hi [Customer], we see you joined 3 months ago on the Pro plan.",
                    "offer_details": "15% off next month",
                    "discount_percentage": 15.0,
                    "call_to_action": "Click here to activate",
                })
            else:
                # Fail first eval, pass second
                call_counts["eval_calls"] += 1
                if call_counts["eval_calls"] == 1:
                    content = json.dumps({
                        "passed": False,
                        "score": 55.0,
                        "feedback": "Improve personalisation.",
                        "issues": ["Too generic"],
                    })
                else:
                    content = json.dumps({
                        "passed": True,
                        "score": 85.0,
                        "feedback": "Good personalisation.",
                        "issues": [],
                    })

            return _make_openai_response(content)

        mock_client.chat.completions.create.side_effect = pass_on_second_eval
        monkeypatch.setattr(base_agents, "_openai_client", mock_client)

        from agents.orchestrator import ChurnOrchestrator
        from models.schemas import CustomerInput, ChurnState

        orchestrator = ChurnOrchestrator()
        chroma_client, collection, embedder = mock_chromadb
        offer_agent = orchestrator._agents["personalised_offer"]
        offer_agent._chroma_client = chroma_client
        offer_agent._collection = collection
        offer_agent._embedder = embedder

        customer = CustomerInput(
            customer_id="LOOP-003",
            plan_tier="pro",
            monthly_revenue=199.0,
            login_frequency_per_week=1.5,
            feature_usage_score=35.0,
            support_tickets_last_30_days=2,
            nps_score=0,
            months_subscribed=3,
        )
        state = ChurnState(customer_input=customer)
        result = orchestrator.run(state)

        assert result.iteration == 2
        assert result.passed is True
