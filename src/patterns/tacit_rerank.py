"""AnyJev Tacit rerank leg (§6 step 4, J1) — opt-in provider over ``POST /v1/decide``.

Two-stage retrieve-then-rerank, shaped by AnyJev's wire: the fused pool from hybrid
retrieval is the stage-1 shortlist (already deterministic-ranked); this leg is stage 2 —
one independent ``yes_no`` relevance question per unique document. ``yes_no`` is exempt
from the gateway's 26-option cap (its labels are fixed ``Yes``/``No``), so the pool size
is unconstrained; each document is scored by ``p(Yes)`` and the scores are merged back
onto the original indices.

Scores are absolute per-document probabilities, not candidate-set-relative like a softmax
over the whole pool — so they are comparable across pools and decision parts, at the price
of one forward per candidate instead of one cached-embedding request.

Degradation invariant (docs/JEV_DECISION_MODEL_ANALYSIS_PLAN.md §7): retrieval never
fails because the Tacit stack is down — any transport, status or shape failure logs a
warning and re-issues the identical call against the TEI provider.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import httpx

from src.decisions.seam_prompts import RERANK_RELEVANCE_QUESTION, rerank_state

if TYPE_CHECKING:
    from src.patterns.safe_tei_rerank import RerankerProtocol

logger = logging.getLogger(__name__)


class _DecideResponseError(ValueError):
    """The ``/v1/decide`` batch answer could not be interpreted; degrades to TEI."""


class TacitReranker:
    """AnyJev ``/v1/decide`` per-candidate reranker with the TEI provider as fallback."""

    def __init__(
        self,
        base_url: str,
        timeout: float,
        fallback: RerankerProtocol,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("Tacit reranker base_url is empty — cannot reach the anyjev gateway")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._fallback = fallback
        self._transport = transport

    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        """Score ``documents`` against ``query`` best-first; TEI answers when Tacit cannot.

        Output guarantees are TEIReranker's: indices in range, at most ``top_n`` pairs,
        sorted score-descending (ties keep the original order). Identical document texts
        are sent once and restored afterwards, so the paired-down list never changes which
        index wins.
        """
        if not documents or top_n < 1:
            return []
        unique: list[str] = []
        index_of: dict[str, list[int]] = {}
        for index, document in enumerate(documents):
            slots = index_of.get(document)
            if slots is None:
                index_of[document] = [index]
                unique.append(document)
            else:
                slots.append(index)
        try:
            scored = await self._request(query, unique)
        except asyncio.CancelledError:
            raise
        except (TimeoutError, httpx.HTTPError, _DecideResponseError) as exc:
            logger.warning("Tacit rerank degraded to TEI: %s", exc)
            return await self._fallback.rerank(query, documents, top_n)
        pairs: list[tuple[int, float]] = []
        for document, probability in zip(unique, scored, strict=True):
            for index in index_of.get(document, ()):
                pairs.append((index, probability))
        if not pairs:
            logger.warning("Tacit rerank degraded to TEI: no scored document matched the pool")
            return await self._fallback.rerank(query, documents, top_n)
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        return pairs[:top_n]

    async def _request(self, query: str, documents: list[str]) -> list[float]:
        """One batched ``POST /v1/decide``; ``p(Yes)`` per document, in input order."""
        items = [
            {
                "state": rerank_state(query, document),
                "question": RERANK_RELEVANCE_QUESTION,
                "options": ["Yes", "No"],
                "kind": "yes_no",
            }
            for document in documents
        ]
        async with httpx.AsyncClient(transport=self._transport, timeout=self._timeout) as client:
            async with asyncio.timeout(self._timeout):
                response = await client.post(f"{self._base_url}/v1/decide", json={"items": items})
        if response.status_code != 200:
            raise _DecideResponseError(f"HTTP {response.status_code} from {self._base_url}/v1/decide")
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise _DecideResponseError(f"invalid JSON: {exc}") from exc
        decisions = payload.get("decisions") if isinstance(payload, dict) else None
        if not isinstance(decisions, list) or len(decisions) != len(documents):
            found = len(decisions) if isinstance(decisions, list) else "none"
            raise _DecideResponseError(f"expected {len(documents)} decisions, got {found}")
        scored: list[float] = []
        for entry in decisions:
            probs = entry.get("probs") if isinstance(entry, dict) else None
            probability = probs.get("Yes") if isinstance(probs, dict) else None
            if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                raise _DecideResponseError(f"bad p(Yes) {probability!r}")
            scored.append(float(probability))
        return scored
