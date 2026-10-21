"""Decision-model (JEV) client — the Ch2 style-substitution seam.

``select_design_patterns`` samples ``brief.architecture_style`` from the A+B LLM call
(entropy channel 2 in docs/SELECTION_VARIABILITY_ANALYSIS.md §2.2). This client
replaces that prose sample with one JEV ``choice`` question over the 35
``STYLE_AFFINITY`` keys plus ``none``, behind the default-off ``decisions``
config switch.

Contract (mirrors ReasoningClient):
- Silent per-call degradation: any failure yields ``None`` (the LLM-extracted
  style stands), never an exception to the caller.
- Cancellation is not a failure: ``asyncio.CancelledError`` propagates.
- Exact-match LRU cache on the description bytes is the only determinism:
  hosted JEV flips labels on identical bytes (2/20, askdona 2026-10-01).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from src.style_affinity import STYLE_AFFINITY

if TYPE_CHECKING:
    from src.config import DecisionsConfig

logger = logging.getLogger(__name__)

#: Question id under which the style choice is requested and answered.
STYLE_QUESTION_ID = "architecture_style"

#: ``none`` criterion key: no listed style applies.
_NONE_KEY = "none"

#: LRU cap of the exact-match description cache (bounded memory over a long-lived server).
_CACHE_MAX_ENTRIES = 256

#: Fixed criterion id → one-line gloss of when the style applies, in STYLE_AFFINITY
#: insertion order plus ``none`` last. Written from the style names' plain meaning;
#: if the guide later grows canonical style descriptions (§10.1), derive from there instead.
_STYLE_GLOSSES: dict[str, str] = {
    "event-driven": "Components communicate primarily through asynchronous events and message flows",
    "broker": "Independent components interact only through a central message broker",
    "enterprise-service-bus": "Heterogeneous services integrate through a central enterprise service bus",
    "event-sourcing": "State is derived by replaying an append-only log of domain events",
    "command-query-responsibility-segregation": "Command and query paths use separate models optimized for writes and reads",
    "saga": "A long-running transaction is coordinated as a sequence of local steps with compensations",
    "microservices": "The system is composed of small independently deployable services",
    "service-oriented-architecture": "Business capabilities are exposed as interoperable services over standard protocols",
    "hexagonal": "Domain logic sits at the core behind ports, with adapters for all external concerns",
    "layered-monolith": "A single deployable is organized in strict horizontal layers",
    "modular-monolith": "A single deployable is organized in well-separated vertical modules",
    "monolithic": "The whole system is built, deployed, and scaled as one unit",
    "pipe-and-filter": "Data flows through a chain of independent processing stages",
    "model-view-controller": "UI concerns split into model, view, and controller responsibilities",
    "presentation-abstraction-control": "Each UI unit pairs presentation, abstraction, and control cooperating agents",
    "microkernel-plugin": "A minimal core is extended by plug-ins for most capabilities",
    "reflection-architecture": "The system inspects and adapts its own structure and behavior at runtime",
    "reactive-architecture": "The design is built around asynchronous message-driven responsiveness and backpressure",
    "actor-based": "Concurrent computation is carried by isolated actors exchanging messages",
    "half-sync-half-async": "Synchronous and asynchronous processing layers are bridged by queues",
    "task-control-architecture": "Hierarchical task controllers coordinate concurrent robotic or real-time behaviors",
    "serverless": "Code runs as provider-managed functions scaled to zero when idle",
    "edge-computing": "Computation is pushed to nodes near the data source or user",
    "space-based": "Scalability comes from distributed in-memory processing units and spaces",
    "data-mesh": "Analytical data is owned as products by decentralized domain teams",
    "lambda-architecture": "Batch and speed layers are combined to serve complete and fresh views",
    "kappa-architecture": "A single streaming pipeline serves both real-time and reprocessed views",
    "api-gateway": "Clients reach backend services through a single gateway entry point",
    "backend-for-frontend": "Each frontend gets a dedicated backend tailored to its needs",
    "service-mesh": "Service-to-service traffic is managed by a dedicated infrastructure layer",
    "rule-based-system": "Behavior is driven by a set of declarative if-then rules",
    "aiml-centric": "AI/ML models are the primary value-bearing components of the system",
    "blockchain-based": "Trust and state rely on a distributed ledger with consensus",
    "hybrid-cloud": "Workloads span on-premises infrastructure and public cloud",
    "multi-cloud": "Workloads span more than one public cloud provider",
}


def build_style_criteria() -> dict[str, str]:
    """Criteria map for the style ``choice`` question: STYLE_AFFINITY keys + ``none``.

    The affinity table is the single source of truth for keys; this function's
    gloss dict must cover every key (checked eagerly at request time).
    """
    criteria = dict(_STYLE_GLOSSES)
    criteria[_NONE_KEY] = "No listed architecture style applies"
    missing = [key for key in STYLE_AFFINITY if key not in criteria]
    if missing:
        raise ValueError(f"Style glosses missing STYLE_AFFINITY keys: {missing}")
    return {key: criteria[key] for key in [*list(STYLE_AFFINITY), _NONE_KEY]}


@dataclass(frozen=True)
class StyleDecision:
    """One resolved style choice: an affinity key, or None when ``none`` won."""

    style: str | None
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)


class DecisionModelClient:
    """JEV choice-question client with cache-first, degrade-on-failure semantics."""

    def __init__(self, config: DecisionsConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._config = config
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._cache: OrderedDict[bytes, StyleDecision] = OrderedDict()

    def enabled(self) -> bool:
        """True when the master switch is on AND the API key env var resolves non-empty."""
        return self._config.enabled and bool(os.environ.get(self._config.api_key_env))

    async def choose_style(self, description: str) -> StyleDecision | None:
        """Resolve the style choice for ``description``; None degrades to the LLM style."""
        key = description.encode("utf-8")
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            logger.info(
                "Decision-model style choice",
                extra={
                    "model": self._config.model,
                    "style": cached.style if cached.style is not None else _NONE_KEY,
                    "confidence": cached.confidence,
                    "cached": True,
                },
            )
            return cached
        if not self.enabled():
            return None
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                decision = await self._request(description)
        except asyncio.CancelledError:
            raise
        except (TimeoutError, httpx.HTTPError) as exc:
            logger.warning("Decision-model style choice degraded: %s", exc)
            return None
        if decision is None:
            return None
        self._cache[key] = decision
        if len(self._cache) > _CACHE_MAX_ENTRIES:
            self._cache.popitem(last=False)
        logger.info(
            "Decision-model style choice",
            extra={
                "model": self._config.model,
                "style": decision.style if decision.style is not None else _NONE_KEY,
                "confidence": decision.confidence,
                "cached": False,
            },
        )
        return decision

    async def close(self) -> None:
        """Close the lazily-created HTTP client, if any."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request(self, description: str) -> StyleDecision | None:
        criteria = build_style_criteria()
        body = {
            "model": self._config.model,
            "state": description,
            "questions": {
                STYLE_QUESTION_ID: {
                    "type": "choice",
                    "instructions": (
                        "Which architecture style best characterizes the system described "
                        "in the state? Pick 'none' when no listed style applies."
                    ),
                    "criteria": criteria,
                }
            },
        }
        api_key = os.environ.get(self._config.api_key_env, "")
        if self._client is None:
            self._client = httpx.AsyncClient(transport=self._transport, timeout=self._config.timeout_seconds)
        response = await self._client.post(
            f"{self._config.base_url.rstrip('/')}/systemone",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
        )
        if response.status_code != 200:
            logger.warning(
                "Decision-model style choice degraded: HTTP %s from %s",
                response.status_code,
                self._config.base_url,
            )
            return None
        try:
            payload: Any = response.json()
        except ValueError as exc:
            logger.warning("Decision-model style choice degraded: invalid JSON: %s", exc)
            return None
        return _parse_choice(payload, set(criteria))


def _entry_key(entry: dict[str, Any]) -> str | None:
    name = entry.get("name", entry.get("id"))
    return name if isinstance(name, str) else None


def _parse_choice(payload: Any, allowed: set[str]) -> StyleDecision | None:
    """Normalize the map- or list-shaped ``answers`` payload into one StyleDecision.

    Map shape (opencode docs): ``{"answers": {<id>: {type, choice, probabilities, confidence}}}``.
    List shape (jevper SDK): ``{"answers": [{"name"/"id": <id>, ...}]}``.
    Anything unexpected logs a warning and degrades (None).
    """
    if not isinstance(payload, dict):
        logger.warning("Decision-model style choice degraded: payload is not an object")
        return None
    answers = payload.get("answers")
    entry: Any = None
    if isinstance(answers, dict):
        entry = answers.get(STYLE_QUESTION_ID)
    elif isinstance(answers, list):
        for item in answers:
            if isinstance(item, dict) and _entry_key(item) == STYLE_QUESTION_ID:
                entry = item
                break
    if not isinstance(entry, dict):
        logger.warning("Decision-model style choice degraded: no %r answer", STYLE_QUESTION_ID)
        return None
    choice = entry.get("choice")
    confidence = entry.get("confidence")
    probabilities = entry.get("probabilities")
    if not isinstance(choice, str) or choice not in allowed:
        logger.warning("Decision-model style choice degraded: bad choice %r", choice)
        return None
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        logger.warning("Decision-model style choice degraded: bad confidence %r", confidence)
        return None
    if not isinstance(probabilities, dict) or not all(
        isinstance(label, str) and isinstance(prob, (int, float)) and not isinstance(prob, bool)
        for label, prob in probabilities.items()
    ):
        logger.warning("Decision-model style choice degraded: bad probabilities")
        return None
    probs = {label: float(prob) for label, prob in probabilities.items()}
    return StyleDecision(
        style=None if choice == _NONE_KEY else choice,
        confidence=float(confidence),
        probabilities=probs,
    )
