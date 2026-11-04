"""AnyJev Tacit decision client — the Ch2 style, J1 rerank, J4 weights, J2 category and J5 gate seams.

The AnyJev gateway (``python -m anyjev.serve``) answers ``POST /v1/decide`` with one
decision per request (``{"state", "question", "options", "kind"}``) or a batch
(``{"items": [...]}`` → ``{"decisions": [...]}``). Every decision comes back as
``{answer, index, probs, margin, route}``: ``probs`` is the next-token distribution
renormalised over the option labels only, ``route`` is ``"one_forward"`` or ``"cot"``
(escalated to the model's own thinking). ``GET /v1/stats`` reports escalation share.

Wire facts pinned from ``anyjev==0.3.0`` (``anyjev/serve.py``, ``anyjev/tacit.py``):
kinds are ``choice`` / ``yes_no`` (alias ``noul``) / ``score``; a ``score`` question
takes its ordered level labels as ``options`` (lowest first, 2-10 levels) and answers
with the winning level *label*, so the expected value is computed here. ``choice`` and
``score`` are hard-capped at 26 options (``anyjev/question.py`` ``MAX_OPTIONS``) and
``yes_no`` is exempt (its labels are fixed) — which is why the 36-way Ch2 style choice
is served as TWO stages here (category → shortlist) instead of one question.

Contract (mirrors a well-behaved seam client):
- Silent per-call degradation: any transport, status or shape failure logs a warning
  and yields ``None`` — the caller's non-model path stands, no tool call fails.
- Cancellation is not a failure: ``asyncio.CancelledError`` propagates.
- Exact-match LRU cache over (kind, question, options, state) is the only
  determinism; one forward is deterministic on the same bytes, and the cache keeps
  a repeated part from paying the round trip twice.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Sequence, get_args

import httpx

from src.decisions.seam_prompts import (
    CATEGORY_GLOSSES,
    CATEGORY_SLUGS,
    DECISION_STATE_MAX_CHARS,
    GATE_STATE_MAX_CHARS,
    STYLE_NONE_KEY,
    WEIGHT_LEVELS,
    axis_question,
    category_question,
    gate_question,
    style_question,
    styles_for_category,
)
from src.schemas.enums import QUALITY_AXES, PatternCategory

if TYPE_CHECKING:
    from src.config import DecisionSeamConfig

logger = logging.getLogger(__name__)

#: LRU cap of the exact-match decision cache (bounded memory over a long-lived server).
_CACHE_MAX_ENTRIES = 256


def category_options() -> list[str]:
    """The nine ``PatternCategory`` values in enum order, and a coverage check.

    The gloss table is the single source of the option text; a category that gains a
    value without a gloss (or loses one) fails loudly here rather than silently
    asking the model about a slug nobody defined.
    """
    values = list(get_args(PatternCategory))
    missing = [value for value in values if value not in CATEGORY_GLOSSES]
    extra = [value for value in CATEGORY_GLOSSES if value not in values]
    if missing or extra:
        raise ValueError(f"category glosses out of sync with PatternCategory: missing={missing} extra={extra}")
    if tuple(values) != CATEGORY_SLUGS:
        raise ValueError("seam_prompts.CATEGORY_SLUGS is out of sync with PatternCategory")
    return values


def gate_state(phase: str, payload: dict[str, Any]) -> str:
    """Compact gate question state: the phase plus the payload's scalar fields.

    Emits EVERY scalar key of the payload (no subset, no reordering), capped at
    ``seam_prompts.GATE_STATE_MAX_CHARS`` — the shape the trainer must reproduce.
    """
    scalars = [
        f"{key}: {value}"
        for key, value in payload.items()
        if isinstance(value, (str, int, float)) and not isinstance(value, bool)
    ]
    return "\n".join([f"phase: {phase}", *scalars])[:GATE_STATE_MAX_CHARS]


def reasoning_gate(
    client: TacitClient | None,
    seam: DecisionSeamConfig,
) -> Callable[[str, dict[str, Any]], Awaitable[bool | None]] | None:
    """The J5 gate closure over ``client``; ``None`` when the seam is not wired.

    Fail-open by construction: ``TacitClient.gate`` returns ``None`` on any
    transport/shape failure, and ``None`` here means "reasoning proceeds". Shared by the
    server and the Stage-0 harness so an arm cannot record a gate it never wired.
    """
    if client is None:
        return None

    async def gate(phase: str, payload: dict[str, Any]) -> bool | None:
        probability = await client.gate(gate_state(phase, payload))
        if probability is None:
            return None
        return probability >= seam.confidence_floor

    return gate


@dataclass(frozen=True)
class TacitDecision:
    """One AnyJev decision: the winning option, its full distribution, and the route."""

    answer: str | None
    probabilities: dict[str, float] = field(default_factory=dict)
    route: str = "one_forward"


@dataclass(frozen=True)
class StyleDecision:
    """One resolved two-stage style choice: a style key, or ``None`` when ``none`` won."""

    style: str | None
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)


class TacitClient:
    """AnyJev ``/v1/decide`` client: cache-first, degrade-on-failure semantics."""

    def __init__(self, config: DecisionSeamConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._config = config
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._cache: OrderedDict[tuple[str, str, tuple[str, ...], str], TacitDecision] = OrderedDict()

    def enabled(self) -> bool:
        """True when the seam switch is on and its credential (if it wants one) resolves."""
        if not self._config.enabled:
            return False
        if self._config.api_key_env is None:
            return True
        return bool(os.environ.get(self._config.api_key_env))

    async def close(self) -> None:
        """Close the lazily-created HTTP client, if any."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- the three seams ---------------------------------------------------------

    async def choose_style(self, description: str) -> StyleDecision | None:
        """Ch2 substitution: category → shortlist; ``None`` degrades to the LLM style.

        Stage 1 narrows the 36-way style question to one category's shortlist (the
        largest is 16 styles + ``none`` = 17 ≤ the AnyJev 26-option cap); stage 2 picks
        within it. Both stages must clear ``confidence_floor``; ``none`` winning, an
        off-shortlist answer, or any transport/shape failure returns ``None`` so the
        pipeline keeps the LLM-sampled style.
        """
        state = description[:DECISION_STATE_MAX_CHARS]
        category, stage1 = await self.choose_category(state)
        if category is None or stage1 < self._config.confidence_floor:
            return None
        styles = styles_for_category(category)
        options = [*styles, STYLE_NONE_KEY]
        decision = await self.decide(state, style_question(styles), options, kind="choice")
        if decision is None or decision.answer is None or decision.answer not in options:
            return None
        confidence = decision.probabilities.get(decision.answer, 0.0)
        if confidence < self._config.confidence_floor:
            return None
        style = None if decision.answer == STYLE_NONE_KEY else decision.answer
        logger.info(
            "Decision-model style choice",
            extra={
                "model": self._config.model,
                "style": style or STYLE_NONE_KEY,
                "confidence": confidence,
                "category": category,
            },
        )
        return StyleDecision(
            style=style,
            confidence=confidence,
            probabilities={key: value for key, value in decision.probabilities.items() if key in options},
        )

    async def choose_category(self, state: str) -> tuple[str | None, float]:
        """J2: soft 9-way category choice → ``(slug, confidence)``; ``(None, 0.0)`` degrades."""
        options = category_options()
        decision = await self.decide(state, category_question(), options, kind="choice")
        if decision is None or decision.answer is None or decision.answer not in CATEGORY_GLOSSES:
            return None, 0.0
        confidence = decision.probabilities.get(decision.answer, 0.0)
        return decision.answer, confidence

    async def score_axes(self, state: str, axes: Sequence[str] = QUALITY_AXES) -> dict[str, float] | None:
        """J4: one ``score`` question per axis, batched; expected values in [0, 1]."""
        vectors = await self.score_axes_many([state], axes=axes)
        return vectors[0]

    async def score_axes_many(
        self, states: Sequence[str], axes: Sequence[str] = QUALITY_AXES
    ) -> list[dict[str, float] | None]:
        """J4 for several parts in ONE gateway round trip; per part ``None`` when unusable.

        ``None`` means a partially- or non-resolved vector: filling the gap with zeros
        would silently reweight the part, so the caller sends that part down the LLM path.
        """
        items = [
            {
                "state": state,
                "question": axis_question(axis),
                "options": list(WEIGHT_LEVELS),
                "kind": "score",
            }
            for state in states
            for axis in axes
        ]
        decisions = await self.decide_batch(items)
        width = len(axes)
        return [_vector_from(decisions[start : start + width], axes) for start in range(0, len(items), width)]

    async def gate(self, state: str) -> float | None:
        """J5: p(deliberation needed) from a ``yes_no`` question; ``None`` degrades."""
        decision = await self.decide(state, gate_question(), ("Yes", "No"), kind="yes_no")
        if decision is None:
            return None
        probability = decision.probabilities.get("Yes")
        return None if probability is None else float(probability)

    # -- generic wire ------------------------------------------------------------

    async def decide(
        self,
        state: str,
        question: str,
        options: Sequence[str],
        *,
        kind: str = "choice",
    ) -> TacitDecision | None:
        """One decision; ``None`` on any failure (degrade, never raise)."""
        item = {"state": state, "question": question, "options": list(options), "kind": kind}
        key = _cache_key(item)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        if not self.enabled():
            return None
        raw = await self._post({"state": state, "question": question, "options": list(options), "kind": kind})
        if raw is None:
            return None
        decision = _parse_decision(raw)
        if decision is None:
            return None
        self._store(key, decision)
        return decision

    async def decide_batch(self, items: Sequence[dict[str, Any]]) -> list[TacitDecision | None]:
        """Batch decision (one round trip); missing/unusable entries come back as ``None``."""
        results: list[TacitDecision | None] = [None] * len(items)
        missing: list[int] = []
        keys: list[tuple[str, str, tuple[str, ...], str]] = []
        for index, item in enumerate(items):
            key = _cache_key(item)
            keys.append(key)
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                results[index] = cached
            else:
                missing.append(index)
        if not missing or not self.enabled():
            return results
        raw = await self._post({"items": [dict(items[index]) for index in missing]})
        if raw is None:
            return results
        if not isinstance(raw, list) or len(raw) != len(missing):
            logger.warning("Tacit batch decision degraded: expected %d decisions", len(missing))
            return results
        for index, entry in zip(missing, raw, strict=True):
            decision = _parse_decision(entry)
            if decision is not None:
                results[index] = decision
                self._store(keys[index], decision)
        return results

    def _store(self, key: tuple[str, str, tuple[str, ...], str], decision: TacitDecision) -> None:
        self._cache[key] = decision
        if len(self._cache) > _CACHE_MAX_ENTRIES:
            self._cache.popitem(last=False)

    async def _post(self, body: dict[str, Any]) -> Any:
        """POST the body, returning the parsed JSON payload (``None`` on any failure)."""
        headers: dict[str, str] = {}
        if self._config.api_key_env is not None:
            headers["Authorization"] = f"Bearer {os.environ.get(self._config.api_key_env, '')}"
        if self._client is None:
            self._client = httpx.AsyncClient(transport=self._transport, timeout=self._config.timeout_seconds)
        url = f"{self._config.base_url.rstrip('/')}/v1/decide"
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                response = await self._client.post(url, headers=headers, json=body)
        except asyncio.CancelledError:
            raise
        except (TimeoutError, httpx.HTTPError) as exc:
            logger.warning("Tacit decision degraded: %s", exc)
            return None
        if response.status_code != 200:
            logger.warning("Tacit decision degraded: HTTP %s from %s", response.status_code, url)
            return None
        try:
            payload: Any = response.json()
        except ValueError as exc:
            logger.warning("Tacit decision degraded: invalid JSON: %s", exc)
            return None
        if isinstance(payload, dict) and "items" in body:
            decisions = payload.get("decisions")
            return decisions if isinstance(decisions, list) else None
        return payload


def _cache_key(item: dict[str, Any]) -> tuple[str, str, tuple[str, ...], str]:
    options = item.get("options")
    labels = tuple(str(option) for option in options) if isinstance(options, list) else ()
    return (
        str(item.get("kind", "choice")),
        str(item.get("question", "")),
        labels,
        str(item.get("state", "")),
    )


def _parse_decision(payload: Any) -> TacitDecision | None:
    """Validate one ``{answer, index, probs, margin, route}`` object; ``None`` degrades."""
    if not isinstance(payload, dict):
        logger.warning("Tacit decision degraded: payload is not an object")
        return None
    answer = payload.get("answer")
    probabilities = payload.get("probs")
    route = payload.get("route", "one_forward")
    if answer is not None and not isinstance(answer, str):
        logger.warning("Tacit decision degraded: bad answer %r", answer)
        return None
    if not isinstance(probabilities, dict) or not all(
        isinstance(label, str) and isinstance(prob, (int, float)) and not isinstance(prob, bool)
        for label, prob in probabilities.items()
    ):
        logger.warning("Tacit decision degraded: bad probs")
        return None
    if not isinstance(route, str):
        logger.warning("Tacit decision degraded: bad route %r", route)
        return None
    return TacitDecision(
        answer=answer,
        probabilities={label: float(prob) for label, prob in probabilities.items()},
        route=route,
    )


def _expected_level(decision: TacitDecision) -> float | None:
    """Expected level of a ``score`` decision, mapped from ``WEIGHT_LEVELS`` to [0, 1]."""
    levels = WEIGHT_LEVELS
    top = len(levels) - 1
    total = 0.0
    seen = 0.0
    for index, level in enumerate(levels):
        probability = decision.probabilities.get(level)
        if probability is None:
            continue
        total += index * probability
        seen += probability
    if seen <= 0.0:
        return None
    return total / seen / top


def _vector_from(decisions: Sequence[TacitDecision | None], axes: Sequence[str]) -> dict[str, float] | None:
    """One per-axis weight vector from ``len(axes)`` score decisions, or ``None``."""
    if len(decisions) != len(axes):
        return None
    scored: dict[str, float] = {}
    for axis, decision in zip(axes, decisions, strict=True):
        if decision is None:
            return None
        value = _expected_level(decision)
        if value is None:
            return None
        scored[axis] = value
    return scored
