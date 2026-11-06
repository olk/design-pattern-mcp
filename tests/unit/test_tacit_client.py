"""AnyJev Tacit client contract: wire shape, batch, score EV, category, gate, degrade.

The wire is pinned to the ``anyjev==0.3.0`` gateway (``anyjev/serve.py`` +
``anyjev/tacit.py``): ``POST /v1/decide`` takes ``{state, question, options, kind}``
and answers ``{answer, index, probs, margin, route}``; a batch takes ``{items: [...]}``
and answers ``{decisions: [...]}``. All transport is mocked — no network.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from src.config import DecisionSeamConfig
from src.decisions.seam_prompts import (
    DECISION_STATE_MAX_CHARS,
    STYLE_GLOSSES,
    STYLE_INSTRUCTIONS,
    STYLE_NONE_GLOSS,
    STYLE_NONE_KEY,
    STYLE_VOCABULARY,
    style_question,
    style_shards,
)
from src.decisions.tacit import (
    CATEGORY_GLOSSES,
    WEIGHT_LEVELS,
    StyleDecision,
    TacitClient,
    TacitDecision,
    category_options,
)
from src.schemas.enums import QUALITY_AXES, PatternCategory
from src.style_affinity import STYLE_AFFINITY

BASE_URL = "http://tacit.test:8100"


def _config(**overrides: Any) -> DecisionSeamConfig:
    payload: dict[str, Any] = {
        "enabled": True,
        "base_url": BASE_URL,
        "model": "tacit-4b",
        "timeout_seconds": 5.0,
    }
    payload.update(overrides)
    return DecisionSeamConfig.model_validate(payload)


def _answer(item: dict[str, Any], winner: int = 0) -> dict[str, Any]:
    """One gateway decision for ``item``: all mass on option ``winner``."""
    options = item["options"]
    probs = {str(option): 0.0 for option in options}
    probs[str(options[winner])] = 1.0
    return {"answer": str(options[winner]), "index": winner, "probs": probs, "margin": 4.0, "route": "one_forward"}


def _client(
    calls: list[httpx.Request],
    *,
    status: int = 200,
    mutate: Any = None,
    **overrides: Any,
) -> TacitClient:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = json.loads(request.content.decode("utf-8"))
        items = body.get("items", [body])
        decisions = [_answer(item) for item in items]
        payload: Any = {"decisions": decisions} if "items" in body else decisions[0]
        if mutate is not None:
            payload = mutate(payload)
        return httpx.Response(status, json=payload)

    return TacitClient(_config(**overrides), transport=httpx.MockTransport(handler))


async def test_single_decision_wire_and_cache() -> None:
    calls: list[httpx.Request] = []
    client = _client(calls)
    options = ["persistence-state", "behavior-selection"]
    first = await client.decide("We cache computed order views.", "Which concern?", options)
    second = await client.decide("We cache computed order views.", "Which concern?", options)
    assert isinstance(first, TacitDecision)
    assert first.answer == "persistence-state"
    assert first.route == "one_forward"
    assert first.probabilities == {"persistence-state": 1.0, "behavior-selection": 0.0}
    assert second is first
    assert len(calls) == 1
    assert calls[0].url == httpx.URL(f"{BASE_URL}/v1/decide")
    body = json.loads(calls[0].content.decode("utf-8"))
    assert body == {
        "state": "We cache computed order views.",
        "question": "Which concern?",
        "options": options,
        "kind": "choice",
    }
    await client.close()


async def test_batch_round_trip_maps_decisions_in_order() -> None:
    calls: list[httpx.Request] = []
    client = _client(calls)
    items = [
        {"state": "s1", "question": "q1", "options": ["a", "b"], "kind": "choice"},
        {"state": "s2", "question": "q2", "options": ["a", "b"], "kind": "choice"},
    ]
    decisions = await client.decide_batch(items)
    assert [decision.answer if decision else None for decision in decisions] == ["a", "a"]
    assert len(calls) == 1
    assert "items" in json.loads(calls[0].content.decode("utf-8"))
    await client.close()


async def test_batch_only_sends_cache_misses() -> None:
    """A cached item is served from the LRU and never re-sent in the batch."""
    calls: list[httpx.Request] = []
    client = _client(calls)
    warm = {"state": "s1", "question": "q1", "options": ["a", "b"], "kind": "choice"}
    await client.decide_batch([warm])
    decisions = await client.decide_batch(
        [warm, {"state": "s2", "question": "q2", "options": ["a", "b"], "kind": "choice"}]
    )
    assert len(decisions) == 2 and all(decision is not None for decision in decisions)
    assert len(calls) == 2
    sent = json.loads(calls[1].content.decode("utf-8"))["items"]
    assert [item["state"] for item in sent] == ["s2"]
    await client.close()


async def test_score_axes_maps_levels_to_unit_interval() -> None:
    """8 axes in one round trip; a level-2 answer is 2/4 = 0.5 on the [0, 1] scale."""

    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        decisions = []
        for item in payload["decisions"]:
            probs = dict.fromkeys(WEIGHT_LEVELS, 0.0)
            probs[WEIGHT_LEVELS[2]] = 1.0
            decisions.append(
                {"answer": WEIGHT_LEVELS[2], "index": 2, "probs": probs, "margin": 3.0, "route": "one_forward"}
            )
        return {"decisions": decisions}

    calls: list[httpx.Request] = []
    client = _client(calls, mutate=mutate)
    vector = await client.score_axes("System: cache views\nDecision part: invalidation")
    assert vector is not None
    assert set(vector) == set(QUALITY_AXES)
    assert all(value == pytest.approx(0.5) for value in vector.values())
    assert len(calls) == 1
    body = json.loads(calls[0].content.decode("utf-8"))
    assert len(body["items"]) == len(QUALITY_AXES)
    assert body["items"][0]["kind"] == "score"
    assert body["items"][0]["options"] == list(WEIGHT_LEVELS)
    await client.close()


async def test_score_axes_many_returns_one_vector_per_part() -> None:
    calls: list[httpx.Request] = []
    client = _client(calls)
    vectors = await client.score_axes_many(["state-a", "state-b"])
    assert len(vectors) == 2
    assert vectors[0] is not None and vectors[1] is not None
    assert vectors[0] is not vectors[1]
    sent = json.loads(calls[0].content.decode("utf-8"))["items"]
    assert len(sent) == 2 * len(QUALITY_AXES)
    await client.close()


async def test_score_axes_degrades_when_any_axis_is_missing() -> None:
    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        payload["decisions"][3] = {"answer": "high", "probs": {}, "route": "one_forward"}
        return payload

    calls: list[httpx.Request] = []
    client = _client(calls, mutate=mutate)
    assert await client.score_axes("state") is None
    await client.close()


async def test_choose_category_returns_slug_and_confidence() -> None:
    calls: list[httpx.Request] = []
    client = _client(calls)
    category, confidence = await client.choose_category("state")
    assert category == "domain-modeling"
    assert confidence == pytest.approx(1.0)
    body = json.loads(calls[0].content.decode("utf-8"))
    assert body["options"] == category_options()
    assert body["kind"] == "choice"
    assert set(body["options"]) == set(PatternCategory.__args__)  # type: ignore[attr-defined]
    assert body["options"] == list(CATEGORY_GLOSSES)
    await client.close()


async def test_choose_category_rejects_an_off_taxonomy_answer() -> None:
    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        return {"answer": "not-a-category", "probs": {"not-a-category": 1.0}, "route": "one_forward"}

    calls: list[httpx.Request] = []
    client = _client(calls, mutate=mutate)
    # Narrowing is only ever a legal taxonomy move: an off-taxonomy slug is no signal.
    assert await client.choose_category("state") == (None, 0.0)
    await client.close()


async def test_gate_returns_yes_probability() -> None:
    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        return {"answer": "Yes", "probs": {"Yes": 0.82, "No": 0.18}, "route": "one_forward"}

    calls: list[httpx.Request] = []
    client = _client(calls, mutate=mutate)
    probability = await client.gate("phase: analyze")
    assert probability == pytest.approx(0.82)
    body = json.loads(calls[0].content.decode("utf-8"))
    assert body["kind"] == "yes_no"
    assert body["options"] == ["Yes", "No"]
    await client.close()


async def test_gate_without_yes_key_degrades() -> None:
    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        return {"answer": "Yes", "probs": {"No": 1.0}, "route": "one_forward"}

    calls: list[httpx.Request] = []
    client = _client(calls, mutate=mutate)
    assert await client.gate("phase: analyze") is None
    await client.close()


@pytest.mark.parametrize("status", [400, 422, 500, 502])
async def test_unprocessable_status_degrades(status: int) -> None:
    calls: list[httpx.Request] = []
    client = _client(calls, status=status)
    assert await client.decide("state", "question", ["a", "b"]) is None
    assert await client.score_axes("state") is None
    await client.close()


@pytest.mark.parametrize(
    "payload",
    [
        "not-an-object",
        {"answer": 3, "probs": {"a": 1.0}},
        {"answer": "a", "probs": "nope"},
        {"answer": "a", "probs": {"a": "high"}},
        {"answer": "a", "probs": {"a": 1.0}, "route": 7},
    ],
)
async def test_malformed_payload_degrades(payload: Any) -> None:
    calls: list[httpx.Request] = []
    client = _client(calls, mutate=lambda _payload: payload)
    assert await client.decide("state", "question", ["a", "b"]) is None
    assert await client.decide("state", "question", ["a", "b"]) is None
    assert len(calls) == 2  # a degraded answer is never cached
    await client.close()


async def test_batch_shape_mismatch_degrades() -> None:
    calls: list[httpx.Request] = []
    client = _client(calls, mutate=lambda payload: {"decisions": payload["decisions"][:1]})
    decisions = await client.decide_batch(
        [
            {"state": "s1", "question": "q", "options": ["a", "b"], "kind": "choice"},
            {"state": "s2", "question": "q", "options": ["a", "b"], "kind": "choice"},
        ]
    )
    assert decisions == [None, None]
    await client.close()


async def test_timeout_degrades() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return httpx.Response(200, json={"answer": "a", "probs": {"a": 1.0}, "route": "one_forward"})

    client = TacitClient(_config(timeout_seconds=0.01), transport=httpx.MockTransport(handler))
    assert await client.decide("state", "question", ["a", "b"]) is None
    await client.close()


async def test_connect_error_degrades() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")

    client = TacitClient(_config(), transport=httpx.MockTransport(handler))
    assert await client.choose_category("state") == (None, 0.0)
    await client.close()


async def test_cancelled_propagates() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    client = TacitClient(_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(asyncio.CancelledError):
        await client.decide("state", "question", ["a", "b"])
    await client.close()


async def test_disabled_client_never_calls_the_wire() -> None:
    calls: list[httpx.Request] = []
    client = _client(calls, enabled=False)
    assert await client.decide("state", "question", ["a", "b"]) is None
    assert await client.score_axes("state") is None
    assert calls == []
    await client.close()


def test_enabled_requires_switch_and_resolvable_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    assert TacitClient(DecisionSeamConfig.model_validate({})).enabled() is False
    assert TacitClient(_config()).enabled() is True
    monkeypatch.delenv("TACIT_API_KEY", raising=False)
    assert TacitClient(_config(api_key_env="TACIT_API_KEY")).enabled() is False
    monkeypatch.setenv("TACIT_API_KEY", "k")
    assert TacitClient(_config(api_key_env="TACIT_API_KEY")).enabled() is True


def test_category_glosses_cover_the_taxonomy() -> None:
    assert category_options() == list(PatternCategory.__args__)  # type: ignore[attr-defined]
    assert set(CATEGORY_GLOSSES) == set(category_options())
    assert all(gloss for gloss in CATEGORY_GLOSSES.values())


def test_style_vocabulary_matches_the_prior_and_shards_are_legal_and_balanced() -> None:
    """The served vocabulary is exactly the styles §10.1 knows, split into legal shards.

    The affinity table stays in ``src/style_affinity.py`` as the pipeline PRIOR; the prompt
    module carries only the flat vocabulary, because reading the prior backwards as a filter
    is what made 16 of 22 labelled holdout rows unanswerable.
    """
    assert set(STYLE_VOCABULARY) == set(STYLE_GLOSSES) == set(STYLE_AFFINITY)
    shards = style_shards()
    assert sum(len(shard) for shard in shards) == len(STYLE_VOCABULARY)
    assert {style for shard in shards for style in shard} == set(STYLE_VOCABULARY)
    # Equal width: a narrow shard's softmax concentrates more mass, so an uneven split would
    # win cross-shard comparisons for free and bias the floor the calibration fits.
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    for shard in shards:
        assert len(shard) + 1 <= 26
        assert all(style in STYLE_GLOSSES for style in shard)


def test_style_question_carries_the_instructions_and_every_gloss() -> None:
    styles = style_shards()[0]
    question = style_question(styles)
    assert question.startswith(STYLE_INSTRUCTIONS)
    assert f"{STYLE_NONE_KEY} = {STYLE_NONE_GLOSS}" in question
    for style in styles:
        assert f"{style} = {STYLE_GLOSSES[style]}" in question


# -- Ch2 style: the 35-way question, sharded under the AnyJev 26-option cap ------------


def _sharded_client(
    calls: list[httpx.Request],
    *,
    style_index: int = 0,
    style_confidence: float = 0.9,
    style_answer: str | None = None,
    fail_on_call: int | None = None,
    fail_every_call: bool = False,
    **overrides: Any,
) -> TacitClient:
    """A gateway that answers every style shard question the same way."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if fail_every_call or (fail_on_call is not None and len(calls) == fail_on_call):
            raise httpx.ConnectError("unreachable")
        body = json.loads(request.content.decode("utf-8"))
        options = [str(option) for option in body["options"]]
        winner = style_answer if style_answer is not None else options[style_index]
        rest = (1.0 - style_confidence) / (len(options) - 1)
        probs = dict.fromkeys(options, rest)
        probs[winner] = style_confidence
        return httpx.Response(200, json={"answer": winner, "probs": probs, "route": "one_forward"})

    return TacitClient(_config(**overrides), transport=httpx.MockTransport(handler))


async def test_choose_style_asks_every_shard_and_keeps_the_best_answer() -> None:
    calls: list[httpx.Request] = []
    client = _sharded_client(calls, style_index=1)
    description = "A payments platform with adapters around every provider."
    first = await client.choose_style(description)
    second = await client.choose_style(description)
    shards = style_shards()
    assert isinstance(first, StyleDecision)
    assert first.style == shards[0][1]
    assert first.confidence == pytest.approx(0.9)
    assert set(first.probabilities) == {*shards[0], STYLE_NONE_KEY}
    assert second == first
    # One question per shard, and none for the cached repeat (the client's exact-match
    # decision cache covers the shard questions too).
    assert len(calls) == len(shards)
    for call, styles in zip(calls, shards, strict=True):
        body = json.loads(call.content.decode("utf-8"))
        assert body["state"] == description
        assert body["question"] == style_question(styles)
        assert body["options"] == [*styles, STYLE_NONE_KEY]
        assert body["kind"] == "choice"
    await client.close()


async def test_choose_style_offers_the_whole_vocabulary_across_shards() -> None:
    """The regression that matters: no affinity filter may narrow the style answer space.

    Deriving the shortlist from a stage-1 category left most styles unanswerable — 16 of 22
    labelled holdout rows named a style their scenario's category could never offer.
    """
    calls: list[httpx.Request] = []
    client = _sharded_client(calls)
    await client.choose_style("some description")
    offered: set[str] = set()
    for call in calls:
        offered |= set(json.loads(call.content.decode("utf-8"))["options"])
    assert offered == {*STYLE_VOCABULARY, STYLE_NONE_KEY}
    await client.close()


async def test_choose_style_degrades_when_no_shard_clears_the_floor() -> None:
    calls: list[httpx.Request] = []
    client = _sharded_client(calls, style_confidence=0.4)
    assert await client.choose_style("some description") is None
    assert len(calls) == len(style_shards())
    await client.close()


async def test_choose_style_abstains_when_none_wins() -> None:
    calls: list[httpx.Request] = []
    client = _sharded_client(calls, style_index=-1)
    assert await client.choose_style("a bare algorithm inside one service") is None
    await client.close()


async def test_choose_style_skips_a_shard_that_answers_off_offer() -> None:
    calls: list[httpx.Request] = []
    client = _sharded_client(calls, style_answer="not-a-served-style")
    assert await client.choose_style("some description") is None
    assert len(calls) == len(style_shards())
    await client.close()


async def test_choose_style_survives_one_shard_failing() -> None:
    """One dead shard must not kill the seam: the other shard still answers."""
    calls: list[httpx.Request] = []
    client = _sharded_client(calls, fail_on_call=1)
    decision = await client.choose_style("some description")
    assert isinstance(decision, StyleDecision)
    assert len(calls) == len(style_shards())
    await client.close()


async def test_choose_style_degrades_when_every_shard_fails() -> None:
    calls: list[httpx.Request] = []
    client = _sharded_client(calls, fail_every_call=True)
    assert await client.choose_style("some description") is None
    assert len(calls) == len(style_shards())
    await client.close()


async def test_choose_style_caps_the_state_like_serving() -> None:
    calls: list[httpx.Request] = []
    client = _sharded_client(calls)
    description = "x" * (DECISION_STATE_MAX_CHARS + 500)
    await client.choose_style(description)
    assert len(calls) == len(style_shards())
    for call in calls:
        assert json.loads(call.content.decode("utf-8"))["state"] == description[:DECISION_STATE_MAX_CHARS]
    await client.close()


async def test_choose_style_disabled_never_calls_the_wire() -> None:
    calls: list[httpx.Request] = []
    client = _sharded_client(calls, enabled=False)
    assert await client.choose_style("some description") is None
    assert calls == []
    await client.close()
