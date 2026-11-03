"""
Base agent classes for the Multi-Agent Customer Churn Retention System.

All agents follow the pattern: run(state: ChurnState) -> ChurnState
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy OpenAI singleton
# ---------------------------------------------------------------------------
_openai_client = None


def _get_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


# ---------------------------------------------------------------------------
# 1. DirectPromptAgent
# ---------------------------------------------------------------------------
class DirectPromptAgent:
    """Raw LLM call wrapper — no system prompt injection."""

    MODEL = "gpt-4o"

    def __init__(self, name: str = "DirectPromptAgent"):
        self.name = name

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=20))
    def _chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        client = _get_client()
        response = client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def run(self, state: Any) -> Any:
        user_content = state.messages[-1]["content"] if state.messages else ""
        reply = self._chat([{"role": "user", "content": user_content}])
        state.messages.append({"role": "assistant", "content": reply})
        return state


# ---------------------------------------------------------------------------
# 2. AugmentedPromptAgent
# ---------------------------------------------------------------------------
class AugmentedPromptAgent(DirectPromptAgent):
    """Injects a persona/system prompt before the user message."""

    def __init__(self, name: str = "AugmentedPromptAgent", system_prompt: str = ""):
        super().__init__(name=name)
        self.system_prompt = system_prompt

    def _build_messages(self, user_content: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def run(self, state: Any) -> Any:
        user_content = state.messages[-1]["content"] if state.messages else ""
        messages = self._build_messages(user_content)
        reply = self._chat(messages)
        state.messages.append({"role": "assistant", "content": reply})
        return state


# ---------------------------------------------------------------------------
# 3. KnowledgeAugmentedPromptAgent
# ---------------------------------------------------------------------------
class KnowledgeAugmentedPromptAgent(AugmentedPromptAgent):
    """Domain-constrained with static knowledge injected into the system prompt."""

    def __init__(
        self,
        name: str = "KnowledgeAugmentedPromptAgent",
        system_prompt: str = "",
        knowledge: str = "",
    ):
        super().__init__(name=name, system_prompt=system_prompt)
        self.knowledge = knowledge

    def _build_messages(self, user_content: str) -> list[dict[str, str]]:
        system = self.system_prompt
        if self.knowledge:
            system = f"{system}\n\n## Domain Knowledge\n{self.knowledge}"
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        return messages


# ---------------------------------------------------------------------------
# 4. RAGKnowledgePromptAgent
# ---------------------------------------------------------------------------
class RAGKnowledgePromptAgent(AugmentedPromptAgent):
    """Retrieval-augmented generation using ChromaDB in-memory + sentence-transformers."""

    EMBED_MODEL = "all-MiniLM-L6-v2"
    TOP_K = 5

    def __init__(
        self,
        name: str = "RAGKnowledgePromptAgent",
        system_prompt: str = "",
        collection_name: str = "knowledge_base",
    ):
        super().__init__(name=name, system_prompt=system_prompt)
        self.collection_name = collection_name
        self._embedder: Optional[SentenceTransformer] = None
        self._chroma_client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.EMBED_MODEL)
        return self._embedder

    def _get_collection(self) -> chromadb.Collection:
        if self._chroma_client is None:
            self._chroma_client = chromadb.Client()
            self._collection = self._chroma_client.get_or_create_collection(
                self.collection_name
            )
        return self._collection  # type: ignore[return-value]

    def index_documents(self, documents: list[dict[str, Any]]) -> None:
        """Index a list of {id, text, metadata} dicts into ChromaDB."""
        collection = self._get_collection()
        embedder = self._get_embedder()
        ids = [str(doc["id"]) for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        embeddings = embedder.encode(texts).tolist()
        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def retrieve(self, query: str) -> str:
        """Return top-K relevant passages as a single string."""
        collection = self._get_collection()
        embedder = self._get_embedder()
        query_embedding = embedder.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(self.TOP_K, collection.count()),
        )
        docs = results.get("documents", [[]])[0]
        return "\n\n---\n\n".join(docs)

    def _build_messages(self, user_content: str) -> list[dict[str, str]]:
        context = self.retrieve(user_content)
        system = self.system_prompt
        if context:
            system = f"{system}\n\n## Relevant Retention Knowledge\n{context}"
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        return messages


# ---------------------------------------------------------------------------
# 5. EvaluationAgent
# ---------------------------------------------------------------------------
class EvaluationAgent(AugmentedPromptAgent):
    """Validates output against criteria; returns pass/fail + feedback."""

    EVAL_SYSTEM = (
        "You are a strict quality evaluator. Evaluate the provided content against "
        "the given criteria. Respond ONLY with a JSON object with keys: "
        "'passed' (bool), 'score' (0-100 float), 'feedback' (str), 'issues' (list of str)."
    )

    def __init__(self, name: str = "EvaluationAgent", criteria: str = ""):
        super().__init__(name=name, system_prompt=self.EVAL_SYSTEM)
        self.criteria = criteria

    def evaluate(self, content: str) -> dict[str, Any]:
        prompt = f"## Evaluation Criteria\n{self.criteria}\n\n## Content to Evaluate\n{content}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        raw = self._chat(messages, response_format={"type": "json_object"})
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"passed": False, "score": 0.0, "feedback": raw, "issues": []}

    def run(self, state: Any) -> Any:
        content = state.messages[-1]["content"] if state.messages else ""
        result = self.evaluate(content)
        state.passed = bool(result.get("passed", False))
        state.messages.append({"role": "assistant", "content": json.dumps(result)})
        return state


# ---------------------------------------------------------------------------
# 6. RoutingAgent
# ---------------------------------------------------------------------------
class RoutingAgent(AugmentedPromptAgent):
    """Classifies input and selects a route from a predefined list."""

    def __init__(
        self,
        name: str = "RoutingAgent",
        system_prompt: str = "",
        routes: Optional[list[str]] = None,
    ):
        super().__init__(name=name, system_prompt=system_prompt)
        self.routes = routes or []

    def classify(self, user_content: str) -> str:
        routes_str = ", ".join(self.routes)
        prompt = (
            f"Given the following input, classify it into exactly one of these routes: "
            f"[{routes_str}].\n\nRespond with ONLY the route name, nothing else.\n\n"
            f"Input:\n{user_content}"
        )
        messages = self._build_messages(prompt)
        reply = self._chat(messages)
        reply_clean = reply.strip().strip('"').strip("'")
        # Match to nearest known route
        for route in self.routes:
            if route.lower() in reply_clean.lower():
                return route
        return self.routes[0] if self.routes else reply_clean

    def run(self, state: Any) -> Any:
        user_content = state.messages[-1]["content"] if state.messages else ""
        route = self.classify(user_content)
        state.messages.append({"role": "assistant", "content": route})
        state.metadata["route"] = route
        return state


# ---------------------------------------------------------------------------
# 7. ActionPlanningAgent
# ---------------------------------------------------------------------------
class ActionPlanningAgent(AugmentedPromptAgent):
    """Decomposes goals into executable steps."""

    PLANNING_SUFFIX = (
        "\n\nDecompose the user's goal into a numbered list of clear, executable steps. "
        "Return ONLY the numbered list."
    )

    def __init__(self, name: str = "ActionPlanningAgent", system_prompt: str = ""):
        augmented = system_prompt + self.PLANNING_SUFFIX
        super().__init__(name=name, system_prompt=augmented)

    def plan(self, goal: str) -> list[str]:
        messages = self._build_messages(goal)
        raw = self._chat(messages)
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        steps = [line for line in lines if line[0].isdigit()]
        return steps if steps else lines

    def run(self, state: Any) -> Any:
        goal = state.messages[-1]["content"] if state.messages else ""
        steps = self.plan(goal)
        plan_text = "\n".join(steps)
        state.messages.append({"role": "assistant", "content": plan_text})
        state.metadata["plan"] = steps
        return state


# ---------------------------------------------------------------------------
# MultiAgentOrchestrator (ABC)
# ---------------------------------------------------------------------------
class MultiAgentOrchestrator(ABC):
    """Abstract base for multi-agent orchestrators."""

    def __init__(self, name: str = "MultiAgentOrchestrator"):
        self.name = name
        self._agents: dict[str, Any] = {}

    def register_agent(self, key: str, agent: Any) -> None:
        self._agents[key] = agent

    @abstractmethod
    def route(self, state: Any) -> str:
        """Return the key of the next agent to run."""

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=20))
    def _chat(self, messages: list[dict[str, str]], model: str = "gpt-4o", **kwargs: Any) -> str:
        client = _get_client()
        response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        return response.choices[0].message.content or ""

    @abstractmethod
    def run(self, state: Any) -> Any:
        """Execute the full orchestration pipeline."""
