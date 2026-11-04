"""AnyJev Tacit rerank leg contract: per-candidate ``yes_no`` batch, dedupe, cap, TEI fallback.

The Tacit reranker is an opt-in second scorer, so its degradation path is part of the
contract: every failure must re-issue the *identical* call against the TEI provider,
which keeps retrieval independent of the Tacit stack being up.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from src.config import RerankerConfig
from src.decisions.seam_prompts import RERANK_RELEVANCE_QUESTION, rerank_state
from src.errors import ERROR_RETRIEVAL_UNAVAILABLE, DesignPatternError
from src.patterns.safe_tei_rerank import TEIReranker, build_reranker
from src.patterns.tacit_rerank import TacitReranker

BASE_URL = "http://tacit.test:8100"
QUERY = "Wrap external payment providers."
DOCUMENTS = ["Adapter", "Observer", "Memento"]


class _RecordingFallback:
    """Stand-in for the TEI reranker: records the call and answers best-first by index."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        self.calls.append((query, list(documents), top_n))
        return [(index, 1.0 - index / 10) for index in range(min(top_n, len(documents)))]


def _decisions(probabilities: Sequence[float]) -> dict[str, Any]:
    return {
        "decisions": [
            {"answer": "Yes", "index": 0, "probs": {"Yes": value, "No": 1.0 - value}, "route": "one_forward"}
            for value in probabilities
        ]
    }


def _client(
    calls: list[httpx.Request],
    fallback: _RecordingFallback,
    *,
    status: int = 200,
    payload: Any = None,
    timeout: float = 30.0,
) -> TacitReranker:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json=payload)

    return TacitReranker(BASE_URL, timeout, fallback=fallback, transport=httpx.MockTransport(handler))


async def test_rerank_scores_documents_by_yes_probability_best_first() -> None:
    calls: list[httpx.Request] = []
    reranker = _client(calls, _RecordingFallback(), payload=_decisions([0.2, 0.9, 0.5]))
    pairs = await reranker.rerank(QUERY, DOCUMENTS, top_n=2)
    assert pairs == [(1, 0.9), (2, 0.5)]
    assert calls[0].url == httpx.URL(f"{BASE_URL}/v1/decide")
    assert json.loads(calls[0].content.decode("utf-8")) == {
        "items": [
            {
                "state": rerank_state(QUERY, document),
                "question": RERANK_RELEVANCE_QUESTION,
                "options": ["Yes", "No"],
                "kind": "yes_no",
            }
            for document in DOCUMENTS
        ]
    }


async def test_duplicate_documents_are_sent_once_and_restored() -> None:
    """Identical texts share one forward; every original index keeps the score."""
    calls: list[httpx.Request] = []
    reranker = _client(calls, _RecordingFallback(), payload=_decisions([0.9, 0.1]))
    pairs = await reranker.rerank("question", ["Adapter", "Observer", "Adapter"], top_n=5)
    assert pairs == [(0, 0.9), (2, 0.9), (1, 0.1)]
    sent = json.loads(calls[0].content.decode("utf-8"))["items"]
    assert [item["state"] for item in sent] == [rerank_state("question", text) for text in ("Adapter", "Observer")]


async def test_ties_keep_the_pool_order() -> None:
    calls: list[httpx.Request] = []
    reranker = _client(calls, _RecordingFallback(), payload=_decisions([0.4, 0.4, 0.4]))
    assert await reranker.rerank(QUERY, DOCUMENTS, top_n=3) == [(0, 0.4), (1, 0.4), (2, 0.4)]


@pytest.mark.parametrize("status", [400, 401, 422, 500, 502])
async def test_error_status_degrades_to_tei(status: int) -> None:
    calls: list[httpx.Request] = []
    fallback = _RecordingFallback()
    reranker = _client(calls, fallback, status=status)
    pairs = await reranker.rerank(QUERY, DOCUMENTS, top_n=2)
    assert pairs == [(0, 1.0), (1, 0.9)]
    assert fallback.calls == [(QUERY, DOCUMENTS, 2)]


@pytest.mark.parametrize(
    "payload",
    [
        {"decisions": []},
        {"decisions": "nope"},
        {"other": []},
        {"decisions": [{"probs": {"Yes": 0.5}}]},
        {"decisions": [{"probs": {"No": 1.0}}]},
        {"decisions": [{"probs": {"Yes": "high"}}]},
        {"decisions": [{"probs": {"Yes": True}}]},
        {"decisions": ["nope"]},
        {"decisions": [{"probs": {"Yes": 0.5}}, {"probs": {"Yes": 0.5}}]},
    ],
)
async def test_malformed_payload_degrades_to_tei(payload: Any) -> None:
    calls: list[httpx.Request] = []
    fallback = _RecordingFallback()
    reranker = _client(calls, fallback, payload=payload)
    pairs = await reranker.rerank(QUERY, DOCUMENTS, top_n=1)
    assert pairs == [(0, 1.0)]
    assert fallback.calls == [(QUERY, DOCUMENTS, 1)]


async def test_invalid_json_degrades_to_tei() -> None:
    calls: list[httpx.Request] = []
    fallback = _RecordingFallback()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=b"not json")

    reranker = TacitReranker(BASE_URL, 5.0, fallback=fallback, transport=httpx.MockTransport(handler))
    assert await reranker.rerank(QUERY, DOCUMENTS, top_n=1) == [(0, 1.0)]
    assert fallback.calls == [(QUERY, DOCUMENTS, 1)]


async def test_timeout_degrades_to_tei() -> None:
    fallback = _RecordingFallback()

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return httpx.Response(200, json=_decisions([1.0]))

    reranker = TacitReranker(BASE_URL, 0.01, fallback=fallback, transport=httpx.MockTransport(handler))
    assert await reranker.rerank(QUERY, DOCUMENTS, top_n=1) == [(0, 1.0)]
    assert fallback.calls == [(QUERY, DOCUMENTS, 1)]


async def test_connect_error_degrades_to_tei() -> None:
    fallback = _RecordingFallback()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")

    reranker = TacitReranker(BASE_URL, 5.0, fallback=fallback, transport=httpx.MockTransport(handler))
    assert await reranker.rerank(QUERY, DOCUMENTS, top_n=3) == [(0, 1.0), (1, 0.9), (2, 0.8)]
    assert fallback.calls == [(QUERY, DOCUMENTS, 3)]


async def test_cancelled_propagates_without_fallback() -> None:
    fallback = _RecordingFallback()

    async def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    reranker = TacitReranker(BASE_URL, 5.0, fallback=fallback, transport=httpx.MockTransport(handler))
    with pytest.raises(asyncio.CancelledError):
        await reranker.rerank(QUERY, DOCUMENTS, top_n=1)
    assert fallback.calls == []


async def test_empty_pool_or_bad_cap_never_calls_the_wire() -> None:
    calls: list[httpx.Request] = []
    reranker = _client(calls, _RecordingFallback(), payload=_decisions([1.0]))
    assert await reranker.rerank(QUERY, [], top_n=3) == []
    assert await reranker.rerank(QUERY, DOCUMENTS, top_n=0) == []
    assert calls == []


async def test_rerank_state_is_capped_and_carries_the_document() -> None:
    """One item's state is the statement plus the document, capped for the 8k context."""
    assert rerank_state("decide X", "pattern text") == "decide X\n\nPattern: pattern text"
    capped = rerank_state("s", "d" * 10_000)
    assert len(capped) == 6000


def test_empty_base_url_is_refused() -> None:
    with pytest.raises(ValueError):
        TacitReranker("   ", 30.0, fallback=_RecordingFallback())


def test_build_reranker_switches_on_provider() -> None:
    tei_config = RerankerConfig.model_validate({})
    assert isinstance(build_reranker(tei_config), TEIReranker)
    tacit_config = RerankerConfig.model_validate({"provider": "tacit"})
    assert isinstance(build_reranker(tacit_config), TacitReranker)


def test_build_reranker_refuses_tacit_without_a_base_url() -> None:
    config = RerankerConfig.model_validate({"provider": "tacit", "tacit_base_url": "  "})
    with pytest.raises(DesignPatternError) as excinfo:
        build_reranker(config)
    assert excinfo.value.code == ERROR_RETRIEVAL_UNAVAILABLE


def test_build_reranker_still_refuses_empty_tei_base_url_under_tacit() -> None:
    """The TEI fallback is mandatory even when Tacit serves the leg."""
    config = RerankerConfig.model_validate({"provider": "tacit", "config": {"base_url": " "}})
    with pytest.raises(DesignPatternError):
        build_reranker(config)


def test_reranker_provider_rejects_the_retired_clm_value() -> None:
    with pytest.raises(ValueError):
        RerankerConfig.model_validate({"provider": "clm"})
