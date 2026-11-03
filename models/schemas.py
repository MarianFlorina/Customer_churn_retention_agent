from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    customer_id: str
    plan_tier: str  # free, starter, pro, enterprise
    monthly_revenue: float
    login_frequency_per_week: float
    feature_usage_score: float  # 0-100
    support_tickets_last_30_days: int
    nps_score: int  # -100 to 100
    months_subscribed: int


class EngagementHealth(BaseModel):
    summary: str
    health_status: str  # healthy, declining, critical
    key_signals: list[str] = Field(default_factory=list)


class ChurnRiskResult(BaseModel):
    risk_score: float  # 0-100
    risk_level: str  # low, medium, high, critical
    key_drivers: list[str] = Field(default_factory=list)
    explanation: str


class RetentionStrategy(BaseModel):
    strategy_tier: str  # Proactive Engagement / Discount Offer / Feature Education / Account Review Call / Accept Churn
    rationale: str


class PersonalisedOffer(BaseModel):
    subject_line: str
    message_body: str
    offer_details: str
    discount_percentage: Optional[float] = None
    call_to_action: str


class EvaluationResult(BaseModel):
    passed: bool
    score: float  # 0-100
    feedback: str
    issues: list[str] = Field(default_factory=list)


class CustomerReport(BaseModel):
    customer_id: str
    customer_input: CustomerInput
    engagement_health: Optional[EngagementHealth] = None
    churn_risk: Optional[ChurnRiskResult] = None
    retention_strategy: Optional[RetentionStrategy] = None
    personalised_offer: Optional[PersonalisedOffer] = None
    evaluation_result: Optional[EvaluationResult] = None
    evaluation_iterations: int = 0


class ChurnState(BaseModel):
    # Core AgentState fields
    messages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    iteration: int = 0
    passed: bool = False

    # Domain fields
    customer_input: Optional[CustomerInput] = None
    engagement_health: Optional[EngagementHealth] = None
    churn_risk: Optional[ChurnRiskResult] = None
    retention_strategy: Optional[RetentionStrategy] = None
    personalised_offer: Optional[PersonalisedOffer] = None
    evaluation_result: Optional[EvaluationResult] = None
    evaluation_feedback: str = ""
    report: Optional[CustomerReport] = None
