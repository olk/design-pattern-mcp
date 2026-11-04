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
    CATEGORY_SLUGS,
    DECISION_STATE_MAX_CHARS,
    STYLE_CATEGORY_AFFINITY,
    STYLE_GLOSSES,
    STYLE_INSTRUCTIONS,
    STYLE_NONE_GLOSS,
    STYLE_NONE_KEY,
    category_question,
    style_question,
    styles_for_category,
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


def test_style_shortlists_match_the_affinity_table_and_fit_the_option_cap() -> None:
    """``seam_prompts``' literal table must mirror ``src/style_affinity.py``, and every
    stage-2 shortlist plus ``none`` must stay inside AnyJev's 26-option choice cap."""
    assert set(STYLE_CATEGORY_AFFINITY) == set(STYLE_GLOSSES) == set(STYLE_AFFINITY)
    for style, categories in STYLE_CATEGORY_AFFINITY.items():
        assert set(categories) == set(STYLE_AFFINITY[style]), style
    for category in CATEGORY_SLUGS:
        styles = styles_for_category(category)
        assert len(styles) + 1 <= 26, category
        assert all(style in STYLE_GLOSSES for style in styles)


def test_styles_for_category_rejects_an_unknown_category() -> None:
    with pytest.raises(ValueError, match="no style has affinity"):
        styles_for_category("not-a-category")


def test_style_question_carries_the_instructions_and_every_gloss() -> None:
    styles = styles_for_category("domain-modeling")
    question = style_question(styles)
    assert question.startswith(STYLE_INSTRUCTIONS)
    assert f"{STYLE_NONE_KEY} = {STYLE_NONE_GLOSS}" in question
    for style in styles:
        assert f"{style} = {STYLE_GLOSSES[style]}" in question


# -- Ch2 style: two stages over the AnyJev 26-option cap -------------------------------


def _two_stage_client(
    calls: list[httpx.Request],
    *,
    category: str = "domain-modeling",
    category_confidence: float = 0.9,
    style_index: int = 0,
    style_confidence: float = 0.9,
    style_answer: str | None = None,
    fail_on_call: int | None = None,
    **overrides: Any,
) -> TacitClient:
    """A gateway that answers the category question, then the shortlist question."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if fail_on_call is not None and len(calls) == fail_on_call:
            raise httpx.ConnectError("unreachable")
        body = json.loads(request.content.decode("utf-8"))
        options = [str(option) for option in body["options"]]
        if body["question"] == category_question():
            confidence = category_confidence
            winner = category
        else:
            confidence = style_confidence
            winner = style_answer if style_answer is not None else options[style_index]
        rest = (1.0 - confidence) / (len(options) - 1)
        probs = dict.fromkeys(options, rest)
        probs[winner] = confidence
        return httpx.Response(
            200, json={"answer": winner, "probs": probs, "route": "one_forward"}
        )

    return TacitClient(_config(**overrides), transport=httpx.MockTransport(handler))


async def test_choose_style_runs_category_then_shortlist() -> None:
    calls: list[httpx.Request] = []
    client = _two_stage_client(calls, style_index=1)
    first = await client.choose_style("A payments platform with adapters around every provider.")
    second = await client.choose_style("A payments platform with adapters around every provider.")
    styles = styles_for_category("domain-modeling")
    assert isinstance(first, StyleDecision)
    assert first.style == styles[1]
    assert first.confidence == pytest.approx(0.9)
    assert set(first.probabilities) == {*styles, STYLE_NONE_KEY}
    assert second == first
    # Two wire calls for the first description, none for the cached repeat (both
    # questions are served from the client's exact-match decision cache).
    assert len(calls) == 2
    stage1, stage2 = (json.loads(call.content.decode("utf-8")) for call in calls)
    assert stage1 == {
        "state": "A payments platform with adapters around every provider.",
        "question": category_question(),
        "options": category_options(),
        "kind": "choice",
    }
    assert stage2["question"] == style_question(styles)
    assert stage2["options"] == [*styles, STYLE_NONE_KEY]
    assert stage2["kind"] == "choice"
    await client.close()


async def test_choose_style_degrades_when_stage_one_stays_below_the_floor() -> None:
    calls: list[httpx.Request] = []
    client = _two_stage_client(calls, category_confidence=0.4)
    assert await client.choose_style("some description") is None
    assert len(calls) == 1  # the shortlist question is never asked
    await client.close()


async def test_choose_style_degrades_when_stage_two_stays_below_the_floor() -> None:
    calls: list[httpx.Request] = []
    client = _two_stage_client(calls, style_confidence=0.4)
    assert await client.choose_style("some description") is None
    assert len(calls) == 2
    await client.close()


async def test_choose_style_none_winning_returns_a_none_style_with_its_confidence() -> None:
    calls: list[httpx.Request] = []
    client = _two_stage_client(calls, style_index=-1)
    decision = await client.choose_style("a bare algorithm inside one service")
    assert isinstance(decision, StyleDecision)
    assert decision.style is None
    assert decision.confidence == pytest.approx(0.9)
    await client.close()


async def test_choose_style_rejects_an_off_shortlist_answer() -> None:
    calls: list[httpx.Request] = []
    client = _two_stage_client(calls, style_answer="microservices")
    # ``microservices`` has no affinity with domain-modeling, so stage 2 never offers it.
    assert await client.choose_style("some description") is None
    await client.close()


@pytest.mark.parametrize("stage", [1, 2])
async def test_choose_style_transport_failure_degrades(stage: int) -> None:
    calls: list[httpx.Request] = []
    client = _two_stage_client(calls, fail_on_call=stage)
    assert await client.choose_style("some description") is None
    assert len(calls) == stage
    await client.close()


async def test_choose_style_caps_the_state_like_serving() -> None:
    calls: list[httpx.Request] = []
    client = _two_stage_client(calls)
    description = "x" * (DECISION_STATE_MAX_CHARS + 500)
    await client.choose_style(description)
    assert len(calls) == 2
    for call in calls:
        assert json.loads(call.content.decode("utf-8"))["state"] == description[:DECISION_STATE_MAX_CHARS]
    await client.close()


async def test_choose_style_shortlists_stay_inside_the_option_cap() -> None:
    """Every stage-2 question is a legal AnyJev choice: shortlist + 'none' <= 26."""
    for category in CATEGORY_SLUGS:
        calls: list[httpx.Request] = []
        client = _two_stage_client(calls, category=category)
        decision = await client.choose_style("some description")
        assert decision is not None, category
        options = json.loads(calls[1].content.decode("utf-8"))["options"]
        assert len(options) <= 26, category
        assert options[-1] == STYLE_NONE_KEY
        assert options[:-1] == list(styles_for_category(category))
        await client.close()


async def test_choose_style_disabled_never_calls_the_wire() -> None:
    calls: list[httpx.Request] = []
    client = _two_stage_client(calls, enabled=False)
    assert await client.choose_style("some description") is None
    assert calls == []
    await client.close()
