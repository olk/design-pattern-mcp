"""Decision-model (JEV) client contract: cache, dual-shape parse, degrade-on-failure.

All transport is mocked (httpx.MockTransport) — no network. The map-shape happy
path asserts against the committed live fixture when present
(tests/unit/fixtures/jev_choice_response.json, Verification step 4); a synthetic
list-shape variant proves the gateway-tolerant parser.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.config import DecisionsConfig
from src.decisions.client import DecisionModelClient, StyleDecision, build_style_criteria
from src.style_affinity import STYLE_AFFINITY

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "jev_choice_response.json"
DESCRIPTION = "Event-driven IoT ingestion with Kafka and worker pool"


def _enabled_config(**overrides: Any) -> DecisionsConfig:
    payload: dict[str, Any] = {"enabled": True}
    payload.update(overrides)
    return DecisionsConfig.model_validate(payload)


def _map_answer(choice: str, confidence: float) -> dict[str, Any]:
    return {
        "answers": {
            "architecture_style": {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: confidence, "microservices": 1.0 - confidence},
                "confidence": confidence,
            }
        }
    }


def _mock_client(
    payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    calls: list[httpx.Request],
    status: int = 200,
) -> DecisionModelClient:
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json=payload)

    return DecisionModelClient(_enabled_config(), transport=httpx.MockTransport(handler))


async def test_happy_path_returns_decision_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[httpx.Request] = []
    client = _mock_client(_map_answer("event-driven", 0.9), monkeypatch, calls)
    first = await client.choose_style(DESCRIPTION)
    second = await client.choose_style(DESCRIPTION)
    assert isinstance(first, StyleDecision)
    assert first.style == "event-driven"
    assert first.confidence == pytest.approx(0.9)
    assert second is first
    assert len(calls) == 1
    body = json.loads(calls[0].content.decode("utf-8"))
    assert body["model"] == "jev-1.13-free"
    assert body["state"] == DESCRIPTION
    criteria = body["questions"]["architecture_style"]["criteria"]
    assert set(criteria) == set(STYLE_AFFINITY) | {"none"}
    assert criteria["none"] == "No listed architecture style applies"
    await client.close()


async def test_none_wins_returns_none_style_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[httpx.Request] = []
    client = _mock_client(_map_answer("none", 0.8), monkeypatch, calls)
    first = await client.choose_style(DESCRIPTION)
    second = await client.choose_style(DESCRIPTION)
    assert isinstance(first, StyleDecision)
    assert first.style is None
    assert second is first
    assert len(calls) == 1
    await client.close()


async def test_style_choice_logs_decision_on_fresh_and_cached_paths(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Both choose_style paths log one record with model/style/confidence/cache flag."""
    calls: list[httpx.Request] = []
    client = _mock_client(_map_answer("event-driven", 0.9), monkeypatch, calls)
    with caplog.at_level(logging.INFO, logger="src.decisions.client"):
        first = await client.choose_style(DESCRIPTION)
        second = await client.choose_style(DESCRIPTION)
    assert second is first
    records = [r for r in caplog.records if r.getMessage() == "Decision-model style choice"]
    assert len(records) == 2
    assert records[0].model == "jev-1.13-free"  # type: ignore[attr-defined]
    assert records[0].style == "event-driven"  # type: ignore[attr-defined]
    assert records[0].confidence == pytest.approx(0.9)  # type: ignore[attr-defined]
    assert records[0].cached is False  # type: ignore[attr-defined]
    assert records[1].cached is True  # type: ignore[attr-defined]
    assert records[1].style == "event-driven"  # type: ignore[attr-defined]
    await client.close()


async def test_none_wins_logs_none_key(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A 'none' choice logs the 'none' key (style is None) so the outcome is visible."""
    calls: list[httpx.Request] = []
    client = _mock_client(_map_answer("none", 0.8), monkeypatch, calls)
    with caplog.at_level(logging.INFO, logger="src.decisions.client"):
        decision = await client.choose_style(DESCRIPTION)
    assert isinstance(decision, StyleDecision)
    assert decision.style is None
    record = next(r for r in caplog.records if r.getMessage() == "Decision-model style choice")
    assert record.style == "none"  # type: ignore[attr-defined]
    assert record.cached is False  # type: ignore[attr-defined]
    assert record.confidence == pytest.approx(0.8)  # type: ignore[attr-defined]
    await client.close()


@pytest.mark.parametrize(
    "payload",
    [
        _map_answer("not-a-style", 0.9),
        {"answers": {"architecture_style": {"type": "choice", "choice": "event-driven"}}},
        {
            "answers": {
                "architecture_style": {
                    "type": "choice",
                    "choice": "event-driven",
                    "confidence": "high",
                    "probabilities": {},
                }
            }
        },
        {"answers": {"architecture_style": {"type": "choice", "choice": "event-driven", "confidence": 0.9}}},
        {"answers": {}},
        {"nope": []},
        [],
    ],
)
async def test_malformed_answer_degrades_without_cache(payload: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[httpx.Request] = []
    client = _mock_client(payload, monkeypatch, calls)
    assert await client.choose_style(DESCRIPTION) is None
    assert await client.choose_style(DESCRIPTION) is None
    assert len(calls) == 2
    await client.close()


async def test_list_shape_variant_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[httpx.Request] = []
    payload = {
        "answers": [
            {
                "name": "architecture_style",
                "type": "choice",
                "choice": "microservices",
                "probabilities": {"microservices": 0.75, "none": 0.25},
                "confidence": 0.75,
            }
        ]
    }
    client = _mock_client(payload, monkeypatch, calls)
    decision = await client.choose_style(DESCRIPTION)
    assert isinstance(decision, StyleDecision)
    assert decision.style == "microservices"
    assert decision.confidence == pytest.approx(0.75)
    await client.close()


async def test_live_fixture_parses_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    if not FIXTURE_PATH.exists():
        pytest.skip("live fixture not captured yet (Verification step 4)")
    calls: list[httpx.Request] = []
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    client = _mock_client(payload, monkeypatch, calls)
    decision = await client.choose_style(DESCRIPTION)
    assert isinstance(decision, StyleDecision)
    assert decision.style is None or decision.style in STYLE_AFFINITY
    await client.close()


async def test_connect_error_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")

    client = DecisionModelClient(_enabled_config(), transport=httpx.MockTransport(handler))
    assert await client.choose_style(DESCRIPTION) is None
    await client.close()


async def test_timeout_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return httpx.Response(200, json=_map_answer("event-driven", 0.9))

    client = DecisionModelClient(_enabled_config(timeout_seconds=0.01), transport=httpx.MockTransport(handler))
    assert await client.choose_style(DESCRIPTION) is None
    await client.close()


async def test_cancelled_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")

    async def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    client = DecisionModelClient(_enabled_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(asyncio.CancelledError):
        await client.choose_style(DESCRIPTION)
    await client.close()


def test_enabled_requires_switch_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    assert DecisionModelClient(DecisionsConfig.model_validate({})).enabled() is False
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    assert DecisionModelClient(DecisionsConfig.model_validate({})).enabled() is False
    monkeypatch.delenv("OPENCODE_API_KEY")
    assert DecisionModelClient(_enabled_config()).enabled() is False
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    assert DecisionModelClient(_enabled_config()).enabled() is True


def test_criteria_covers_every_affinity_key_in_order() -> None:
    criteria = build_style_criteria()
    assert list(criteria) == [*list(STYLE_AFFINITY), "none"]
    assert all(isinstance(gloss, str) and gloss for gloss in criteria.values())
