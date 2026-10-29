"""CLM-8B rank leg (§6 step 4, J1) — opt-in provider over ``clm-serve POST /v1/rank``.

The CLM container already hosts the 8B encoder for the style seam; its ``/v1/rank``
endpoint answers a list-completion question (``context`` + ``question`` + ``answers``)
with a softmax over the candidates, so the same weights can rerank the fused pool.
The endpoint has no candidate cap (each answer is embedded independently), and it is
deliberately *not* the TEI cross-encoder: it is a second scorer, opt-in through
``reranker.provider = "clm"``.

Degradation invariant (docs/JEV_DECISION_MODEL_ANALYSIS_PLAN.md §7): retrieval never
fails because the CLM stack is down — any transport, status or shape failure logs a
warning and re-issues the identical call against the TEI provider.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from src.patterns.safe_tei_rerank import RerankerProtocol

logger = logging.getLogger(__name__)


class _RankResponseError(ValueError):
    """The ``/v1/rank`` answer could not be interpreted; degrades to the TEI provider."""


class CLMReranker:
    """``clm-serve`` ``/v1/rank`` reranker with the TEI provider as its fallback."""

    def __init__(
        self,
        base_url: str,
        timeout: float,
        fallback: RerankerProtocol,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("CLM reranker base_url is empty — cannot reach clm-serve")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._fallback = fallback
        self._transport = transport

    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        """Score ``documents`` against ``query`` best-first; TEI answers when CLM cannot.

        Output guarantees are TEIReranker's: indices in range, at most ``top_n`` pairs,
        sorted score-descending. Identical document texts are sent once and restored
        afterwards, so the paired-down list never changes which index wins.
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
            ranked = await self._request(query, unique)
        except asyncio.CancelledError:
            raise
        except (TimeoutError, httpx.HTTPError) as exc:
            logger.warning("CLM rerank degraded to TEI: %s", exc)
            return await self._fallback.rerank(query, documents, top_n)
        except _RankResponseError as exc:
            logger.warning("CLM rerank degraded to TEI: %s", exc)
            return await self._fallback.rerank(query, documents, top_n)
        pairs: list[tuple[int, float]] = []
        for candidate, probability in ranked:
            for index in index_of.get(candidate, ()):
                pairs.append((index, probability))
        if not pairs:
            logger.warning("CLM rerank degraded to TEI: no ranked candidate matched the documents")
            return await self._fallback.rerank(query, documents, top_n)
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        return pairs[:top_n]

    async def _request(self, query: str, documents: list[str]) -> list[tuple[str, float]]:
        """POST ``/v1/rank``; returns ``(candidate, probability)`` pairs in server order."""
        body = {"context": "", "question": query, "answers": documents}
        async with httpx.AsyncClient(transport=self._transport, timeout=self._timeout) as client:
            async with asyncio.timeout(self._timeout):
                response = await client.post(f"{self._base_url}/rank", json=body)
        if response.status_code != 200:
            raise _RankResponseError(f"HTTP {response.status_code} from {self._base_url}/rank")
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise _RankResponseError(f"invalid JSON: {exc}") from exc
        ranked = payload.get("ranked") if isinstance(payload, dict) else None
        if not isinstance(ranked, list) or not ranked:
            raise _RankResponseError("payload carries no 'ranked' entries")
        pairs: list[tuple[str, float]] = []
        for entry in ranked:
            if not isinstance(entry, dict):
                raise _RankResponseError("ranked entry is not an object")
            candidate = entry.get("candidate")
            probability = entry.get("prob")
            if not isinstance(candidate, str):
                raise _RankResponseError(f"bad ranked candidate {candidate!r}")
            if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                raise _RankResponseError(f"bad ranked prob {probability!r}")
            pairs.append((candidate, float(probability)))
        return pairs
