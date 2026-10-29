"""CLM ``/v1/systemone`` style-seam contract: cache, dual-shape parse, degrade-on-failure.

All transport is mocked (httpx.MockTransport) — no network. The map-shape happy path
asserts against a committed fixture of the documented ``clm-serve`` answer shape
(``tests/unit/fixtures/clm_systemone_choice_response.json``); a synthetic list-shape
variant proves the gateway-tolerant parser.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.config import DecisionSeamConfig
from src.decisions.client import DecisionModelClient, StyleDecision, build_style_criteria
from src.style_affinity import STYLE_AFFINITY

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "clm_systemone_choice_response.json"
DESCRIPTION = "Event-driven IoT ingestion with Kafka and worker pool"
BASE_URL = "http://clm.test:8700/v1"


def _enabled_config(**overrides: Any) -> DecisionSeamConfig:
    payload: dict[str, Any] = {"enabled": True, "base_url": BASE_URL, "model": "clm-latest"}
    payload.update(overrides)
    return DecisionSeamConfig.model_validate(payload)


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
    calls: list[httpx.Request],
    *,
    status: int = 200,
    **overrides: Any,
) -> DecisionModelClient:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json=payload)

    return DecisionModelClient(_enabled_config(**overrides), transport=httpx.MockTransport(handler))


async def test_happy_path_returns_decision_and_caches() -> None:
    calls: list[httpx.Request] = []
    client = _mock_client(_map_answer("event-driven", 0.9), calls)
    first = await client.choose_style(DESCRIPTION)
    second = await client.choose_style(DESCRIPTION)
    assert isinstance(first, StyleDecision)
    assert first.style == "event-driven"
    assert first.confidence == pytest.approx(0.9)
    assert second is first
    assert len(calls) == 1
    body = json.loads(calls[0].content.decode("utf-8"))
    assert body["model"] == "clm-latest"
    assert body["state"] == DESCRIPTION
    criteria = body["questions"]["architecture_style"]["criteria"]
    assert set(criteria) == set(STYLE_AFFINITY) | {"none"}
    assert criteria["none"] == "No listed architecture style applies"
    await client.close()


async def test_local_seam_sends_no_authorization_header() -> None:
    """A clm-serve seam without ``api_key_env`` talks to the sidecar with no credential."""
    calls: list[httpx.Request] = []
    client = _mock_client(_map_answer("event-driven", 0.9), calls)
    await client.choose_style(DESCRIPTION)
    assert calls[0].url == httpx.URL(f"{BASE_URL}/systemone")
    assert "authorization" not in calls[0].headers
    await client.close()


async def test_credential_env_sends_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[httpx.Request] = []
    monkeypatch.setenv("JUDGE_API_KEY", "secret-key")
    client = _mock_client(_map_answer("event-driven", 0.9), calls, api_key_env="JUDGE_API_KEY")
    await client.choose_style(DESCRIPTION)
    assert calls[0].headers["authorization"] == "Bearer secret-key"
    await client.close()


async def test_none_wins_returns_none_style_and_caches() -> None:
    calls: list[httpx.Request] = []
    client = _mock_client(_map_answer("none", 0.8), calls)
    first = await client.choose_style(DESCRIPTION)
    second = await client.choose_style(DESCRIPTION)
    assert isinstance(first, StyleDecision)
    assert first.style is None
    assert second is first
    assert len(calls) == 1
    await client.close()


async def test_style_choice_logs_decision_on_fresh_and_cached_paths(caplog: pytest.LogCaptureFixture) -> None:
    """Both choose_style paths log one record with model/style/confidence/cache flag."""
    calls: list[httpx.Request] = []
    client = _mock_client(_map_answer("event-driven", 0.9), calls)
    with caplog.at_level(logging.INFO, logger="src.decisions.client"):
        first = await client.choose_style(DESCRIPTION)
        second = await client.choose_style(DESCRIPTION)
    assert second is first
    records = [r for r in caplog.records if r.getMessage() == "Decision-model style choice"]
    assert len(records) == 2
    assert records[0].model == "clm-latest"  # type: ignore[attr-defined]
    assert records[0].style == "event-driven"  # type: ignore[attr-defined]
    assert records[0].confidence == pytest.approx(0.9)  # type: ignore[attr-defined]
    assert records[0].cached is False  # type: ignore[attr-defined]
    assert records[1].cached is True  # type: ignore[attr-defined]
    assert records[1].style == "event-driven"  # type: ignore[attr-defined]
    await client.close()


async def test_none_wins_logs_none_key(caplog: pytest.LogCaptureFixture) -> None:
    """A 'none' choice logs the 'none' key (style is None) so the outcome is visible."""
    calls: list[httpx.Request] = []
    client = _mock_client(_map_answer("none", 0.8), calls)
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
async def test_malformed_answer_degrades_without_cache(payload: Any) -> None:
    calls: list[httpx.Request] = []
    client = _mock_client(payload, calls)
    assert await client.choose_style(DESCRIPTION) is None
    assert await client.choose_style(DESCRIPTION) is None
    assert len(calls) == 2
    await client.close()


async def test_list_shape_variant_parses() -> None:
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
    client = _mock_client(payload, calls)
    decision = await client.choose_style(DESCRIPTION)
    assert isinstance(decision, StyleDecision)
    assert decision.style == "microservices"
    assert decision.confidence == pytest.approx(0.75)
    await client.close()


async def test_committed_fixture_parses() -> None:
    calls: list[httpx.Request] = []
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    client = _mock_client(payload, calls)
    decision = await client.choose_style(DESCRIPTION)
    assert isinstance(decision, StyleDecision)
    assert decision.style == "event-driven"
    assert decision.confidence == pytest.approx(0.87)
    await client.close()


async def test_unprocessable_status_degrades() -> None:
    """A 422 (clm-serve rejects an unparseable question) degrades to the LLM style."""
    calls: list[httpx.Request] = []
    client = _mock_client(_map_answer("event-driven", 0.9), calls, status=422)
    assert await client.choose_style(DESCRIPTION) is None
    await client.close()


async def test_connect_error_degrades() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")

    client = DecisionModelClient(_enabled_config(), transport=httpx.MockTransport(handler))
    assert await client.choose_style(DESCRIPTION) is None
    await client.close()


async def test_timeout_degrades() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return httpx.Response(200, json=_map_answer("event-driven", 0.9))

    client = _enabled_config(timeout_seconds=0.01)
    assert await DecisionModelClient(client, transport=httpx.MockTransport(handler)).choose_style(DESCRIPTION) is None


async def test_cancelled_propagates() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    client = DecisionModelClient(_enabled_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(asyncio.CancelledError):
        await client.choose_style(DESCRIPTION)
    await client.close()


def test_enabled_requires_switch_and_resolvable_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """A seam is live only with its switch on and (when it declares one) a set key."""
    assert DecisionModelClient(DecisionSeamConfig.model_validate({})).enabled() is False
    # No api_key_env: the local sidecar needs no credential.
    assert DecisionModelClient(_enabled_config()).enabled() is True
    # api_key_env declares a credential: enabled only while that env var resolves.
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    assert DecisionModelClient(_enabled_config(api_key_env="JUDGE_API_KEY")).enabled() is False
    monkeypatch.setenv("JUDGE_API_KEY", "secret-key")
    assert DecisionModelClient(_enabled_config(api_key_env="JUDGE_API_KEY")).enabled() is True
    # An empty placeholder is "no credential", never "look up ''".
    assert _enabled_config(api_key_env="").api_key_env is None


async def test_disabled_client_never_calls_the_wire() -> None:
    calls: list[httpx.Request] = []
    client = _mock_client(_map_answer("event-driven", 0.9), calls, enabled=False)
    assert await client.choose_style(DESCRIPTION) is None
    assert calls == []
    await client.close()


def test_criteria_covers_every_affinity_key_in_order() -> None:
    criteria = build_style_criteria()
    assert list(criteria) == [*list(STYLE_AFFINITY), "none"]
    assert all(isinstance(gloss, str) and gloss for gloss in criteria.values())
