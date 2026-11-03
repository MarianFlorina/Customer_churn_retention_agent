# Project 14: Multi-Agent Customer Churn Prediction and Retention Advisory System

**Domain:** Customer Support | **Difficulty:** Intermediate  
**Pattern:** Orchestrator-Workers + Evaluator-Optimizer Loop + State Management

## Overview

A production-grade multi-agent system that predicts customer churn risk and generates personalised retention strategies. The system processes customer behavioural signals through a pipeline of specialised AI agents, each contributing to a final personalised retention offer that is validated by an evaluation loop.

## Agent Architecture

```
User Input (CustomerInput)
       ↓
ChurnOrchestrator (manages ChurnState)
       ↓
EngagementAnalysisAgent   → engagement_health: EngagementHealth
       ↓
ChurnRiskScoringAgent     → churn_risk: ChurnRiskResult (0-100 score)
       ↓
RetentionStrategyRoutingAgent → retention_strategy: RetentionStrategy
       ↓
┌──────────────────────────────────────────┐
│  Evaluator-Optimizer Loop (max 3 iters)  │
│  PersonalisedOfferAgent → offer          │
│  RetentionEvaluationAgent → pass/fail    │
└──────────────────────────────────────────┘
       ↓
CustomerReport → JSON + HTML
```

## Retention Strategy Routing Rules

| Condition | Strategy |
|-----------|----------|
| risk ≥ 75 AND plan == enterprise | Account Review Call |
| risk ≥ 75 | Discount Offer |
| risk 50–74 AND feature_usage < 40 | Feature Education |
| risk 50–74 | Proactive Engagement |
| risk < 50 | Accept Churn |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

```bash
# Run full pipeline for all sample customers
python main.py

# Launch Streamlit UI
streamlit run app.py

# Run tests
pytest tests/ -v
```

## Sample Customers

| ID | Plan | MRR | Profile |
|----|------|-----|---------|
| ENT-7842 | enterprise | $4,500 | High-value at-risk power user |
| FREE-3391 | free | $0 | Low-engagement ghost account |
| PRO-1156 | pro | $299 | Recently onboarded, struggling |

## Tech Stack

- **LLM:** OpenAI GPT-4o
- **RAG:** ChromaDB + sentence-transformers (all-MiniLM-L6-v2)
- **Validation:** Pydantic v2
- **Resilience:** tenacity (3 retries, exponential backoff)
- **UI:** Streamlit
- **Testing:** pytest + pytest-mock
