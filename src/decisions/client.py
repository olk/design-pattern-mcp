"""Decision-model client — the Ch2 style-substitution seam (CLM ``/v1/systemone``).

``select_design_patterns`` samples ``brief.architecture_style`` from the A+B LLM call
(entropy channel 2 in docs/SELECTION_VARIABILITY_ANALYSIS.md §2.2). This client
replaces that prose sample with one ``choice`` question over the 35
``STYLE_VOCABULARY`` styles plus ``none``, behind the default-off
``decisions.style`` seam (``decisions.style.provider = "clm"``).

The wire is the TypeSafe-compatible ``POST /v1/systemone`` shape, served locally by
``clm-serve`` (the CLM-8B heads stack) and still reachable for hosted judging by
pointing ``base_url`` + ``api_key_env`` at any provider that speaks it.

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

from src.decisions.seam_prompts import (
    STYLE_GLOSSES,
    STYLE_INSTRUCTIONS,
    STYLE_NONE_GLOSS,
    STYLE_NONE_KEY,
    STYLE_VOCABULARY,
)

if TYPE_CHECKING:
    from src.config import DecisionSeamConfig

logger = logging.getLogger(__name__)

#: Question id under which the style choice is requested and answered.
STYLE_QUESTION_ID = "architecture_style"

#: ``none`` criterion key: no listed style applies (shared with the CLM exporter).
_NONE_KEY = STYLE_NONE_KEY

#: LRU cap of the exact-match description cache (bounded memory over a long-lived server).
_CACHE_MAX_ENTRIES = 256

#: Criterion id → one-line gloss, sourced from the shared prompt vocabulary so the
#: CLM fine-tune trains on exactly the criteria text this seam serves. If the guide
#: later grows canonical style descriptions (§10.1), derive from there instead.
_STYLE_GLOSSES: dict[str, str] = STYLE_GLOSSES


def build_style_criteria() -> dict[str, str]:
    """Criteria map for the style ``choice`` question, in served order + ``none`` last.

    Keyed and ordered by :data:`STYLE_VOCABULARY` — the answer space the seam may name,
    which is also the order the audited corpus was exported in — NOT by
    ``src.style_affinity.STYLE_AFFINITY``: that table is the style→category PRIOR, a
    different concern, and reading its order here coupled the served bytes to it. CLM
    embeds each criterion independently and has no option cap, so the whole 36-entry map
    is one question (the AnyJev shard exists for a 26-option wire this client never uses).
    """
    criteria = dict(_STYLE_GLOSSES)
    criteria[_NONE_KEY] = STYLE_NONE_GLOSS
    ordered = [*STYLE_VOCABULARY, _NONE_KEY]
    missing = [key for key in ordered if key not in criteria]
    if missing:
        raise ValueError(f"Style glosses missing STYLE_VOCABULARY keys: {missing}")
    return {key: criteria[key] for key in ordered}


@dataclass(frozen=True)
class StyleDecision:
    """One resolved style choice: a vocabulary style, or None when ``none`` won."""

    style: str | None
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)


class DecisionModelClient:
    """CLM ``/v1/systemone`` choice-question client: cache-first, degrade-on-failure."""

    def __init__(self, config: DecisionSeamConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._config = config
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._cache: OrderedDict[bytes, StyleDecision] = OrderedDict()

    def enabled(self) -> bool:
        """True when the seam switch is on and its credential (if it wants one) resolves."""
        if not self._config.enabled:
            return False
        if self._config.api_key_env is None:
            return True
        return bool(os.environ.get(self._config.api_key_env))

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
                    "instructions": STYLE_INSTRUCTIONS,
                    "criteria": criteria,
                }
            },
        }
        headers: dict[str, str] = {}
        if self._config.api_key_env is not None:
            headers["Authorization"] = f"Bearer {os.environ.get(self._config.api_key_env, '')}"
        if self._client is None:
            self._client = httpx.AsyncClient(transport=self._transport, timeout=self._config.timeout_seconds)
        response = await self._client.post(
            f"{self._config.base_url.rstrip('/')}/systemone",
            headers=headers,
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
