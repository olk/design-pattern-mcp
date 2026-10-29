"""CLM ``/v1/rank`` reranker contract: ranks, dedupe, cap, and TEI fallback.

The CLM reranker is an opt-in second scorer, so its degradation path is part of the
contract: every failure must re-issue the *identical* call against the TEI provider,
which keeps retrieval independent of the CLM stack being up.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from src.config import RerankerConfig
from src.errors import ERROR_RETRIEVAL_UNAVAILABLE, DesignPatternError
from src.patterns.clm_rerank import CLMReranker
from src.patterns.safe_tei_rerank import TEIReranker, build_reranker

BASE_URL = "http://clm.test:8700/v1"
DOCUMENTS = ["Adapter", "Observer", "Memento"]


class _RecordingFallback:
    """Stand-in for the TEI reranker: records the call and answers best-first by index."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        self.calls.append((query, list(documents), top_n))
        return [(index, 1.0 - index / 10) for index in range(min(top_n, len(documents)))]


def _client(
    calls: list[httpx.Request],
    fallback: _RecordingFallback,
    *,
    status: int = 200,
    payload: Any = None,
    timeout: float = 30.0,
) -> CLMReranker:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json=payload)

    return CLMReranker(BASE_URL, timeout, fallback=fallback, transport=httpx.MockTransport(handler))


async def test_rank_maps_candidates_back_to_indices_best_first() -> None:
    calls: list[httpx.Request] = []
    payload = {
        "ranked": [
            {"rank": 1, "candidate": "Memento", "prob": 0.7},
            {"rank": 2, "candidate": "Adapter", "prob": 0.2},
            {"rank": 3, "candidate": "Observer", "prob": 0.1},
        ]
    }
    reranker = _client(calls, _RecordingFallback(), payload=payload)
    pairs = await reranker.rerank("Wrap external payment providers.", DOCUMENTS, top_n=2)
    assert pairs == [(2, 0.7), (0, 0.2)]
    assert calls[0].url == httpx.URL(f"{BASE_URL}/rank")
    body = json.loads(calls[0].content.decode("utf-8"))
    assert body == {
        "context": "",
        "question": "Wrap external payment providers.",
        "answers": DOCUMENTS,
    }


async def test_duplicate_documents_are_sent_once_and_restored() -> None:
    """Identical texts share one server answer; every original index keeps the score."""
    calls: list[httpx.Request] = []
    payload = {"ranked": [{"rank": 1, "candidate": "Adapter", "prob": 0.9}]}
    reranker = _client(calls, _RecordingFallback(), payload=payload)
    pairs = await reranker.rerank("question", ["Adapter", "Observer", "Adapter"], top_n=5)
    assert pairs == [(0, 0.9), (2, 0.9)]
    assert json.loads(calls[0].content.decode("utf-8"))["answers"] == ["Adapter", "Observer"]


async def test_unmatched_candidates_fall_back_to_tei() -> None:
    calls: list[httpx.Request] = []
    fallback = _RecordingFallback()
    payload = {"ranked": [{"rank": 1, "candidate": "NotInThePool", "prob": 0.9}]}
    reranker = _client(calls, fallback, payload=payload)
    pairs = await reranker.rerank("question", DOCUMENTS, top_n=3)
    assert pairs == [(0, 1.0), (1, 0.9), (2, 0.8)]
    assert fallback.calls == [("question", DOCUMENTS, 3)]


@pytest.mark.parametrize("status", [401, 422, 500, 502])
async def test_error_status_degrades_to_tei(status: int) -> None:
    calls: list[httpx.Request] = []
    fallback = _RecordingFallback()
    reranker = _client(calls, fallback, status=status)
    pairs = await reranker.rerank("question", DOCUMENTS, top_n=2)
    assert pairs == [(0, 1.0), (1, 0.9)]
    assert fallback.calls == [("question", DOCUMENTS, 2)]


@pytest.mark.parametrize(
    "payload",
    [
        {"ranked": []},
        {"ranked": "nope"},
        {"other": []},
        {"ranked": [{"candidate": "Adapter"}]},
        {"ranked": [{"candidate": 7, "prob": 0.5}]},
        {"ranked": [{"candidate": "Adapter", "prob": "high"}]},
        {"ranked": ["Adapter"]},
    ],
)
async def test_malformed_payload_degrades_to_tei(payload: Any) -> None:
    calls: list[httpx.Request] = []
    fallback = _RecordingFallback()
    reranker = _client(calls, fallback, payload=payload)
    pairs = await reranker.rerank("question", DOCUMENTS, top_n=1)
    assert pairs == [(0, 1.0)]
    assert fallback.calls == [("question", DOCUMENTS, 1)]


async def test_timeout_degrades_to_tei() -> None:
    fallback = _RecordingFallback()

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return httpx.Response(200, json={"ranked": [{"candidate": "Adapter", "prob": 1.0}]})

    reranker = CLMReranker(BASE_URL, 0.01, fallback=fallback, transport=httpx.MockTransport(handler))
    pairs = await reranker.rerank("question", DOCUMENTS, top_n=1)
    assert pairs == [(0, 1.0)]
    assert fallback.calls == [("question", DOCUMENTS, 1)]


async def test_cancelled_propagates_without_fallback() -> None:
    fallback = _RecordingFallback()

    async def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    reranker = CLMReranker(BASE_URL, 5.0, fallback=fallback, transport=httpx.MockTransport(handler))
    with pytest.raises(asyncio.CancelledError):
        await reranker.rerank("question", DOCUMENTS, top_n=1)
    assert fallback.calls == []


async def test_empty_pool_or_bad_cap_never_calls_the_wire() -> None:
    calls: list[httpx.Request] = []
    reranker = _client(calls, _RecordingFallback(), payload={"ranked": []})
    assert await reranker.rerank("question", [], top_n=3) == []
    assert await reranker.rerank("question", DOCUMENTS, top_n=0) == []
    assert calls == []


def test_empty_clm_base_url_is_refused() -> None:
    with pytest.raises(ValueError):
        CLMReranker("   ", 30.0, fallback=_RecordingFallback())


def test_build_reranker_switches_on_provider() -> None:
    tei_config = RerankerConfig.model_validate({})
    assert isinstance(build_reranker(tei_config), TEIReranker)
    clm_config = RerankerConfig.model_validate({"provider": "clm"})
    assert isinstance(build_reranker(clm_config), CLMReranker)


def test_build_reranker_refuses_clm_without_a_base_url() -> None:
    config = RerankerConfig.model_validate({"provider": "clm", "clm_base_url": "  "})
    with pytest.raises(DesignPatternError) as excinfo:
        build_reranker(config)
    assert excinfo.value.code == ERROR_RETRIEVAL_UNAVAILABLE


def test_build_reranker_still_refuses_empty_tei_base_url_under_clm() -> None:
    """The TEI fallback is mandatory even when CLM serves the leg."""
    config = RerankerConfig.model_validate({"provider": "clm", "config": {"base_url": " "}})
    with pytest.raises(DesignPatternError):
        build_reranker(config)
