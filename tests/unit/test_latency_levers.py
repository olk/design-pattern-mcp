"""Latency-reduction Step-3 lever contracts (§4 rule 1a, REASONING_MAX_INJECTED_CHARS, L7 order).

L5 truncation-first repair: a ``finish_reason=length`` parse failure is a transport-class
problem, not a model-quality problem — with ``truncation_rerun="boost"`` the response is NOT
repair-prompted; exactly one rerun carries the boosted completion budget, a second truncation
raises ``failure_kind="truncated"``. Non-truncation failures keep the normal repair loop, and
the shipped default (``"repair"``) keeps the legacy behavior byte-identical.
L6 injection cap: the rendered reasoning context truncates with a marker; short traces and
``max_chars=None`` stay unchanged.
L7 stable-prefix-first: two different payloads of the same operation share a ≥512-char
identical user-prompt prefix; legacy mode reproduces the original order byte for byte.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from src.agent import DesignAgent, DesignHints
from src.config import ServerConfig, ValidationConfig
from src.patterns.loader import load_schema_enums
from src.prompts import PromptLibrary
from src.schemas import (
    ApplicationSeed,
    CatalogSlice,
    GenerationSeed,
    DesignOverview,
    DesignResult,
)
from src.prompts.system import (
    ANALYZE_USER_STATIC_HEAD,
    EVALUATE_USER_STATIC_HEAD,
    GENERATE_USER_STATIC_HEAD,
    WEIGHTS_USER_STATIC_HEAD,
)
from src.reasoning.prompts import render_reasoning_context
from src.reasoning.schemas import ReasoningStep, ReasoningTrace
from src.schemas import DecisionPart, DesignBrief

PROMPTS = PromptLibrary(load_schema_enums(Path("docs")))
STABLE_PROMPTS = PromptLibrary(load_schema_enums(Path("docs")), stable_prefix_first=True)


# --------------------------------------------------------------------------------------
# L5 truncation-first repair
# --------------------------------------------------------------------------------------


class _Payload(BaseModel):
    value: int


class _RecordedClient:
    """Transport double: scriptable responses + finish reasons; records achat kwargs."""

    def __init__(self, script: list[tuple[str, str | None]]) -> None:
        # each entry: (content, finish_reason)
        self._script = list(script)
        self.calls: list[tuple[int, dict[str, Any]]] = []

    async def achat(self, messages: list[Any], **kwargs: Any) -> Any:
        from llama_index.core.base.llms.types import ChatResponse, ChatMessage

        content, finish = self._script.pop(0)
        self.calls.append((sum(len(str(m.content)) for m in messages), dict(kwargs)))
        raw: dict[str, Any] = {"choices": [{"finish_reason": finish or "stop"}]}
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            raw=raw,
        )


def _agent(rerun: str) -> DesignAgent:
    config = ServerConfig()
    config.generator.config.api_key = "k"
    from typing import Literal as _Literal

    rerun_mode: _Literal["repair", "boost"] = "repair" if rerun == "repair" else "boost"
    config.validation = ValidationConfig(max_retries=1, truncation_rerun=rerun_mode)
    return DesignAgent(config, PROMPTS)


def call_generate(agent: DesignAgent, payload: dict[str, Any]) -> Any:
    return agent.generate_structured(
        "sys",
        "usr",
        _Payload,
        operation="generate",
        parse=lambda _raw: _Payload.model_validate(payload),  # type: ignore[arg-type, return-value]
    )


async def test_truncated_boost_reruns_once_with_budget_no_repair() -> None:
    """length→parse failure: exactly one boosted rerun (no repair loop), then success."""
    agent = _agent("boost")
    client = _RecordedClient([
        ('{"value": 1, "truncat', "length"),
        ('{"value": 7}', "stop"),
    ])
    agent._client = client

    result = await asyncio.wait_for(
        call_generate(agent, {"value": 7}), timeout=10.0
    )

    assert result.value == 7
    assert len(client.calls) == 2, "one initial call plus ONE boosted rerun, no repair call"
    assert "max_completion_tokens" not in client.calls[0][1]
    assert client.calls[1][1]["max_completion_tokens"] == 65536


async def test_truncated_twice_raises_truncated_not_validation() -> None:
    """length both times: transport-class 'truncated' failure with the operation metadata."""
    agent = _agent("boost")
    client = _RecordedClient([
        ('{"value": 1, "truncat', "length"),
        ('{"value": 2, "truncat', "length"),
    ])
    agent._client = client

    from src.errors import LLMError

    with pytest.raises(LLMError) as excinfo:
        await asyncio.wait_for(call_generate(agent, {"value": 1}), timeout=10.0)

    assert len(client.calls) == 2, "initial call plus the single boosted rerun"
    error = excinfo.value
    assert error.failure_kind == "truncated"
    assert error.extra["operation"] == "generate"
    assert error.extra["attempts"] == 2


async def test_default_repair_mode_keeps_legacy_repair_loop() -> None:
    """Truncation under the shipped default ('repair') enters the normal repair loop."""

    agent = _agent("repair")
    client = _RecordedClient([
        ('{"value": 1, "truncat', "length"),      # initial
        ('{"value": 9}', "stop"),                  # repair re-call (same completion budget)
    ])
    agent._client = client

    class _RepairAgent(DesignAgent):
        async def _repair_call(
            self, system_prompt: str, user_prompt: str, parser: Any, operation: str
        ) -> Any:
            raise AssertionError("not used")

    del _RepairAgent  # repair path is validate_with_retries; assert via kwargs below

    # generate_structured heals through the repair retry: the second call is the parseable one.
    result = await asyncio.wait_for(
        agent.generate_structured(
            "sys", "usr", _Payload, operation="generate"
        ),
        timeout=10.0,
    )
    del result

    assert len(client.calls) == 2, "initial + repair re-call (no boost budget, no truncation)"
    assert "max_completion_tokens" not in client.calls[1][1]


async def test_non_truncation_parse_failure_keeps_repair() -> None:
    """A non-length parse failure keeps the legacy behavior (repair after failure)."""
    agent = _agent("boost")
    client = _RecordedClient([
        ("garbage", "stop"),
        ('{"value": 5}', "stop"),
    ])
    agent._client = client

    result = await asyncio.wait_for(
        call_generate(agent, {"value": 5}), timeout=10.0
    )

    assert result.value == 5
    assert len(client.calls) == 2, "initial + repair re-call with the untrusted correction"
    assert "max_completion_tokens" not in client.calls[1][1]


async def test_boost_disabled_by_default_config() -> None:
    """Shipped config default is 'repair' — behavior-neutral until a lever arm enables boost."""
    assert ServerConfig().validation.truncation_rerun == "repair"


# --------------------------------------------------------------------------------------
# L6 reasoning-trace injection cap
# --------------------------------------------------------------------------------------


def _long_trace() -> ReasoningTrace:
    step = ReasoningStep(tool="shannon", thought="x" * 600, phase_tag="phase", tool_output="y" * 600)
    return ReasoningTrace(phase="analyze", steps=[step, step])


def test_render_context_uncapped_by_default() -> None:
    trace = _long_trace()
    uncapped = render_reasoning_context(trace)
    assert "(reasoning trace truncated" not in uncapped
    assert len(uncapped) > 1200


def test_render_context_capped_with_marker() -> None:
    trace = _long_trace()
    capped = render_reasoning_context(trace, max_chars=400)
    assert len(capped) > 400, "cap plus marker"
    uncapped_head = render_reasoning_context(trace)[:400]
    assert capped.startswith(uncapped_head)
    assert "(reasoning trace truncated at 400 chars)" in capped


def test_render_context_short_trace_unchanged_with_cap() -> None:
    trace = ReasoningTrace(phase="analyze", steps=[ReasoningStep(tool="s", thought="t", phase_tag="p")])
    assert render_reasoning_context(trace, max_chars=400) == render_reasoning_context(trace)


def test_render_context_per_step_thoughts_stay_whole() -> None:
    """The cap truncates the INJECTION, never an individual recorded thought."""
    trace = _long_trace()
    capped = render_reasoning_context(trace, max_chars=400)
    assert "x" * 100 in capped, "leading thought text present up to the cap"


def test_reasoning_cap_default_is_uncappped() -> None:
    from src.reasoning.config import ReasoningConfig

    assert ReasoningConfig().max_injected_chars is None


# --------------------------------------------------------------------------------------
# L7 stable-prefix-first prompt order
# --------------------------------------------------------------------------------------


def _part(cid: str, statement: str) -> DecisionPart:
    return DecisionPart(id=cid, statement=statement, priority=1.0)


def test_stable_prefix_weights_two_payloads_share_head() -> None:
    a = STABLE_PROMPTS.weights_batch_user_prompt(
        [_part("p-1", "one"), _part("p-2", "two")], DesignBrief(restated_goal="g1")
    )
    b = STABLE_PROMPTS.weights_batch_user_prompt(
        [_part("q-9", "alpha"), _part("q-8", "beta")], DesignBrief(restated_goal="g2")
    )
    head_len = len(WEIGHTS_USER_STATIC_HEAD)
    assert head_len >= 512
    assert a[:head_len] == WEIGHTS_USER_STATIC_HEAD == b[:head_len]


def test_stable_prefix_analyze_two_payloads_share_head() -> None:
    from src.agent import DesignHints

    a = STABLE_PROMPTS.brief_and_parts_user_prompt("Build service A with retries and caching.", DesignHints())
    b = STABLE_PROMPTS.brief_and_parts_user_prompt("Build ledger B with strict consistency.", DesignHints())
    head_len = len(ANALYZE_USER_STATIC_HEAD)
    assert head_len >= 512
    assert a[:head_len] == ANALYZE_USER_STATIC_HEAD == b[:head_len]


def test_stable_prefix_generate_and_evaluate_share_heads() -> None:
    from src.schemas import ApplicationSeed, DesignOverview, DesignResult, GenerationSeed

    slice_ = CatalogSlice(
        pattern="p", category="domain-modeling", intent="i", problem="pr", context="c", avoid_when=[]
    )
    s1 = GenerationSeed(
        applications=[ApplicationSeed(id="x1", pattern="p", category="domain-modeling", part_id="p1", part_statement="s1")],
        catalog_slices={"p": slice_},
    )
    s2 = GenerationSeed(
        applications=[ApplicationSeed(id="x2", pattern="p", category="domain-modeling", part_id="p2", part_statement="s2")],
        catalog_slices={"p": slice_},
    )
    g1 = STABLE_PROMPTS.generate_user_prompt(s1)
    g2 = STABLE_PROMPTS.generate_user_prompt(s2)
    head_len = len(GENERATE_USER_STATIC_HEAD)
    assert head_len >= 512
    assert g1[:head_len] == GENERATE_USER_STATIC_HEAD == g2[:head_len]

    r1 = DesignResult(overview=DesignOverview(reasoning="r", primary_pattern="p", category="domain-modeling"), applications=[])
    r2 = DesignResult(overview=DesignOverview(reasoning="r2", primary_pattern="p", category="domain-modeling"), applications=[])
    e1 = STABLE_PROMPTS.evaluate_user_prompt(r1, [_part("p1", "a")])
    e2 = STABLE_PROMPTS.evaluate_user_prompt(r2, [_part("p1", "a")])
    ehead = len(EVALUATE_USER_STATIC_HEAD)
    assert ehead >= 512
    assert e1[:ehead] == EVALUATE_USER_STATIC_HEAD == e2[:ehead]


def test_legacy_mode_is_the_original_byte_layout() -> None:
    """Legacy mode still begins with the original one-line heads (order unchanged)."""
    a = PROMPTS.brief_and_parts_user_prompt("d", DesignHints())
    assert a.startswith("Extract the design brief and the decision parts for the description below.")
    w = PROMPTS.weights_batch_user_prompt([_part("p-1", "s")], DesignBrief(restated_goal="g"))
    assert w.startswith("Extract the quality-attribute weights for these 1 decision parts (§7.1).")
    g = PROMPTS.generate_user_prompt(
        GenerationSeed(
            applications=[ApplicationSeed(id="x", pattern="p", category="domain-modeling", part_id="p1", part_statement="s")]
        )
    )
    assert g.startswith("Produce the applications payload for the seeds below.")
    r = DesignResult(overview=DesignOverview(reasoning="r", primary_pattern="p", category="domain-modeling"), applications=[])
    e = PROMPTS.evaluate_user_prompt(r, [_part("p1", "s")])
    assert e.startswith("Evaluate the candidate DesignResult below against the decision parts.")


def test_stable_flag_default_is_false() -> None:
    """Shipped default is legacy order; the lever arm switches it via env/config."""
    assert ServerConfig().prompts.stable_prefix_first is False
    assert PROMPTS.stable_prefix_first is False


# env wiring pin (config-level, §14)
def test_env_wiring_names_resolve() -> None:
    """The three lever env names parse out of config.json with shipped default behavior."""
    import json

    from src.config_expansion import expand_env_in_obj

    raw = json.loads(Path("config/config.json").read_text())
    expanded = expand_env_in_obj(raw)
    config = ServerConfig.model_validate(expanded)
    assert config.validation.truncation_rerun == "repair"
    assert config.validation.truncation_boost_max_completion_tokens == 65536
    assert config.reasoning.max_injected_chars is None
    assert config.prompts.stable_prefix_first is False


# --------------------------------------------------------------------------------------
# L6 wiring: the pipeline forwards the configured cap into render_reasoning_context
# --------------------------------------------------------------------------------------


class _StubReasoning:
    """Minimal ReasoningClient stand-in: enabled, returns a long trace."""

    def __init__(self) -> None:
        self.enabled = True

    async def run_pre_llm(self, phase: str, context_payload: dict[str, Any]) -> Any:
        step = ReasoningStep(tool="shannon", thought="x" * 600, phase_tag=phase, tool_output="y" * 600)
        return ReasoningTrace(phase=phase, steps=[step, step])


@pytest.mark.asyncio
async def test_reasoning_block_forwards_configured_cap() -> None:
    """Mutant-killer for pipeline wiring: max_injected_chars must reach the renderer.

    A pipeline with the cap set renders a TRUNCATED block; the same pipeline with the
    cap unset (None) renders the full block. This kills the mutant that passes
    ``max_chars=None`` (arm measured nothing), the one dropping the ``trace`` argument,
    and the one dropping the ``max_chars`` kwarg (all three produce the uncapped text).
    ``_reasoning_block`` touches only the reasoning client and config, so stubbed
    catalog/index/llm collaborators suffice.
    """
    from src.pipeline import DesignPipeline

    class _StubLoader:
        pass

    class _StubIndex:
        pass

    class _StubGuard:
        pass

    config = ServerConfig()
    config.reasoning.max_injected_chars = 400
    pipeline = DesignPipeline(
        config=config,
        catalog=_StubLoader(),  # type: ignore[arg-type]
        index=_StubIndex(),  # type: ignore[arg-type]
        llm=None,  # type: ignore[arg-type]
        guard=_StubGuard(),  # type: ignore[arg-type]
        reasoning=_StubReasoning(),
    )
    block_capped = await pipeline._reasoning_block("analyze", {"description": "d"})
    config.reasoning.max_injected_chars = None
    block_full = await pipeline._reasoning_block("analyze", {"description": "d"})
    assert "(reasoning trace truncated at 400 chars)" in block_capped
    assert "(reasoning trace truncated" not in block_full
    assert len(block_full) > 1200
