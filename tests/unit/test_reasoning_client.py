"""Reasoning-layer contract tests: the config vocabulary vs the MCP wire names.

Two name spaces meet in this module and they are not interchangeable:

* **config key** — the ``ReasoningTool`` literal (``"shannon"`` / ``"code"``), the
  vocabulary a strategy's ``tools`` list holds and pydantic validates at load time;
* **wire name** — ``ToolContract.name`` (``"shannonthinking"`` / ``"code-reasoning"``),
  the tool name the npm packages actually expose.

``TOOL_CONTRACTS`` is keyed by the former. Keying it by the latter made every
strategy lookup raise ``KeyError('code')``, which the silent-degradation contract
turned into an empty reasoning trace — the failure was only visible as
``"Reasoning loop for phase weights degraded: 'code'"`` in the logs.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest
from fastmcp.client.logging import LogMessage
from pydantic import ValidationError

import src.reasoning.client as client_module
from src.reasoning.client import (
    FORWARDED_LOGGER,
    BannerParseNoiseFilter,
    ReasoningClient,
    _subprocess_env,
    contract_name,
)
from src.reasoning.config import ReasoningConfig, ReasoningTool
from src.reasoning.prompts import REASONING_STEP_SYSTEM_PROMPT
from src.reasoning.schemas import (
    CODE_TOOL,
    SHANNON_TOOL,
    TOOL_CONTRACTS,
    ReasoningTrace,
    ThoughtDraft,
    ToolContract,
)


class _StubAgent:
    """Draft generator seam: one fixed draft per call, no LLM."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(self, *_args: object, **_kwargs: object) -> ThoughtDraft:
        self.calls += 1
        return ThoughtDraft(thought=f"stub thought {self.calls}", phase_tag="model")


def _client() -> tuple[ReasoningClient, _StubAgent]:
    config = ReasoningConfig(enabled=True, quiet_stderr=True)
    agent = _StubAgent()
    return ReasoningClient(config, agent), agent  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Vocabulary: every config key resolves, and it is not a wire name
# --------------------------------------------------------------------------------------


def test_every_reasoning_tool_literal_resolves_to_a_contract() -> None:
    for key in get_args(ReasoningTool):
        assert key in TOOL_CONTRACTS, f"config key {key!r} has no contract"
        contract = TOOL_CONTRACTS[key]
        assert contract.name != key, (
            f"{key!r} is both a config key and a wire name — the two vocabularies "
            "must not collide, or a lookup cannot tell them apart"
        )


def test_contract_name_maps_config_keys_to_wire_names() -> None:
    assert contract_name("shannon") == SHANNON_TOOL.name == "shannonthinking"
    assert contract_name("code") == CODE_TOOL.name == "code-reasoning"


def test_every_shipped_strategy_tool_resolves() -> None:
    """The production default config is the exact payload that broke."""
    config = ReasoningConfig()
    assert config.per_phase, "no per-phase strategies configured"
    for phase, strategy in config.per_phase.items():
        for tool_key in strategy.tools:
            assert tool_key in TOOL_CONTRACTS, f"phase {phase!r} lists unknown tool {tool_key!r}"


# --------------------------------------------------------------------------------------
# Behavior: the lookup must not degrade the reasoning loop
# --------------------------------------------------------------------------------------


def test_step_prompt_names_every_draft_field() -> None:
    """The draft prompt and the ``ThoughtDraft`` parser are one contract: the model
    can only produce the fields it is told about, so a schema field missing from
    the prompt is a guaranteed validation failure on a real provider."""
    prompt = REASONING_STEP_SYSTEM_PROMPT
    for field_name in ThoughtDraft.model_fields:
        assert f'"{field_name}"' in prompt, f"ThoughtDraft.{field_name} not requested by the prompt"


class TestGenerateTraceContractResolution:
    def test_trace_runs_steps_for_the_weights_phase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``weights`` ships both scratchpads, so its 3 thoughts alternate."""
        client, agent = _client()

        async def _fake_call_tool(
            contract: object, draft: ThoughtDraft, number: int, total: int
        ) -> str:
            return f"{contract.name}:{number}"

        monkeypatch.setattr(client, "_call_tool", _fake_call_tool)

        trace = asyncio.run(client._generate_trace("weights", client._config.get_strategy("weights"), {}))

        assert isinstance(trace, ReasoningTrace)
        assert trace.steps, f"no steps ran: reason={trace.degradation_reason!r}"
        assert trace.degradation_reason != "'code'"
        assert {step.tool for step in trace.steps} == {CODE_TOOL.name, SHANNON_TOOL.name}
        assert agent.calls >= 1


    def test_run_pre_llm_does_not_degrade_on_the_contract_lookup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _agent = _client()

        async def _fake_call_tool(*_args: object, **_kwargs: object) -> str:
            return "ok"

        monkeypatch.setattr(client, "_call_tool", _fake_call_tool)

        trace = asyncio.run(client.run_pre_llm("weights", {"part": "p1"}))
        assert trace.steps, f"degraded: {trace.degradation_reason!r}"

    def test_alternating_phase_uses_both_wire_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``analyze`` is ``tools=["shannon", "code"]``: steps alternate per tool."""

        client, _agent = _client()

        async def _fake_call_tool(
            contract: ToolContract, draft: ThoughtDraft, number: int, total: int
        ) -> str:
            return "ok"

        monkeypatch.setattr(client, "_call_tool", _fake_call_tool)
        strategy = client._config.get_strategy("analyze")
        trace = asyncio.run(client._generate_trace("analyze", strategy, {}))

        assert trace.steps
        tools = {step.tool for step in trace.steps}
        assert tools <= {SHANNON_TOOL.name, CODE_TOOL.name}
        assert len(tools) == min(2, len(strategy.tools)), (
            "a multi-tool strategy must reach both servers, not just the first"
        )


# --------------------------------------------------------------------------------------
# Per-phase controls: enabled flag, default-merge, active tools
# --------------------------------------------------------------------------------------


class TestPerPhaseControls:
    def test_shipped_defaults_enable_every_phase_with_both_tools(self) -> None:
        config = ReasoningConfig()

        assert set(config.per_phase) == {"analyze", "weights", "generate", "evaluate"}
        assert {phase: strategy.pre_llm_thoughts for phase, strategy in config.per_phase.items()} == {
            "analyze": 2,
            "weights": 2,
            "generate": 2,
            "evaluate": 2,
        }
        for phase, strategy in config.per_phase.items():
            assert strategy.enabled is True, phase
            assert strategy.tools == ["shannon", "code"], phase

    def test_partial_override_keeps_agenda_and_directive(self) -> None:
        """A phase override names only what it changes; the rest stays shipped.

        Replacing the whole strategy would silently drop the agenda/directive the
        thought prompt is built from, so the override must merge.
        """
        defaults = ReasoningConfig().get_strategy("weights")

        config = ReasoningConfig(per_phase={"weights": {"enabled": False, "pre_llm_thoughts": 1}})

        strategy = config.get_strategy("weights")
        assert strategy.enabled is False
        assert strategy.pre_llm_thoughts == 1
        assert strategy.step_agenda == defaults.step_agenda
        assert strategy.system_directive == defaults.system_directive
        assert strategy.tools == defaults.tools

    def test_partial_override_may_also_narrow_the_tools(self) -> None:
        config = ReasoningConfig(per_phase={"evaluate": {"tools": ["code"]}})

        assert config.get_strategy("evaluate").tools == ["code"]

    def test_unknown_nested_phase_key_still_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningConfig(per_phase={"weights": {"tools": ["shannon"], "bogus": 1}})

    def test_active_tools_is_first_seen_order_deduplicated(self) -> None:
        config = ReasoningConfig(
            per_phase={
                "analyze": {"tools": ["code"]},
                "weights": {"tools": ["shannon", "code"]},
                "generate": {"tools": ["code", "shannon"]},
            }
        )

        assert config.active_tools() == ["code", "shannon"]

    def test_active_tools_skips_disabled_phases(self) -> None:
        config = ReasoningConfig(
            per_phase={
                "analyze": {"enabled": False, "tools": ["shannon"]},
                "weights": {"tools": ["code"]},
            }
        )

        assert config.active_tools() == ["code"]

    def test_disabled_phase_runs_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, agent = _client()
        client._config.per_phase["weights"] = client._config.get_strategy("weights").model_copy(
            update={"enabled": False}
        )

        async def _explode(*_args: object, **_kwargs: object) -> str:
            raise AssertionError("a disabled phase must not reach a reasoning tool")

        monkeypatch.setattr(client, "_call_tool", _explode)

        trace = asyncio.run(client.run_pre_llm("weights", {"part": "p1"}))

        assert trace.phase == "weights"
        assert trace.steps == []
        assert trace.degraded is False
        assert trace.degradation_reason is None
        assert client._trace_cache == {}, "a disabled phase must not consume a cache slot"
        assert agent.calls == 0, "a disabled phase must not call the draft generator"

    def test_global_switch_still_degrades(self) -> None:
        config = ReasoningConfig(enabled=False)
        client, agent = ReasoningClient(config, _StubAgent()), _StubAgent()  # type: ignore[arg-type]

        trace = asyncio.run(client.run_pre_llm("weights", {"part": "p1"}))

        assert trace.degraded is True
        assert trace.degradation_reason == "reasoning disabled"
        assert agent.calls == 0

    def test_health_check_probes_only_active_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Narrowing a phase's tools must not force the unused server to be installed."""
        probes: list[str] = []

        class _FakeClient:
            def __init__(self, transport: object, **_kwargs: object) -> None:
                self._transport = transport

            async def __aenter__(self) -> _FakeClient:
                return self

            async def __aexit__(self, *_exc: object) -> bool:
                return False

            async def list_tools(self) -> list[SimpleNamespace]:
                return [
                    SimpleNamespace(name=SHANNON_TOOL.name),
                    SimpleNamespace(name=CODE_TOOL.name),
                ]

        def _record_cmd(name: str, _cmd: object) -> tuple[str, list[str]]:
            probes.append(name)
            return ("node", ["index.js"])

        monkeypatch.setattr(client_module, "Client", _FakeClient)
        monkeypatch.setattr(client_module, "_resolve_cmd", _record_cmd)
        config = ReasoningConfig(
            per_phase={
                "analyze": {"enabled": False, "tools": ["shannon"]},
                "weights": {"tools": ["code"]},
            }
        )
        client = ReasoningClient(config, _StubAgent())  # type: ignore[arg-type]

        statuses = asyncio.run(client.health_check())

        assert probes == [CODE_TOOL.name]
        assert statuses == {CODE_TOOL.name: "ok"}


# --------------------------------------------------------------------------------------
# Cancellation: a cancelled call is not a failure, it must not be degraded or swallowed
# --------------------------------------------------------------------------------------


class _CancelAwareAgent:
    """Blocks on the first draft call (a cancellation can land there), then answers."""

    def __init__(self) -> None:
        self.first_call_entered = asyncio.Event()
        self.calls = 0

    async def generate_structured(self, *_args: object, **_kwargs: object) -> ThoughtDraft:
        self.calls += 1
        if self.calls == 1:
            self.first_call_entered.set()
            await asyncio.Event().wait()
        return ThoughtDraft(thought=f"stub thought {self.calls}", phase_tag="model")


def _cancel_aware_client() -> tuple[ReasoningClient, _CancelAwareAgent]:
    agent = _CancelAwareAgent()
    config = ReasoningConfig(enabled=True, quiet_stderr=True)
    return ReasoningClient(config, agent), agent  # type: ignore[arg-type]


async def _ok_call_tool(*_args: object, **_kwargs: object) -> str:
    return "ok"


async def test_cancelled_loop_propagates_and_releases_the_inflight_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_pre_llm`` used to read an unbound ``trace`` in its ``finally``.

    A BaseException (asyncio cancellation — job cancel, client disconnect) never
    reached the ``except Exception`` branch, so the ``finally`` raised
    ``UnboundLocalError: cannot access local variable 'trace'``. The pipeline's
    degradation contract swallowed that as an "unexpected reasoning failure": the
    cancelled phase continued without a trace, and a co-waiting caller held a
    future that could never resolve.
    """
    client, agent = _cancel_aware_client()
    monkeypatch.setattr(client, "_call_tool", _ok_call_tool)

    task = asyncio.create_task(client.run_pre_llm("analyze", {"part": "p1"}))
    await agent.first_call_entered.wait()
    assert client._inflight, "the running loop must own an in-flight future"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client._inflight == {}, "a cancelled loop must not leave a dangling future"
    assert client._trace_cache == {}, "a cancelled loop must not cache a result"


async def test_co_waiting_caller_rejoins_after_a_cancelled_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled leader leaves no result: the waiter re-runs the loop itself."""
    client, agent = _cancel_aware_client()
    monkeypatch.setattr(client, "_call_tool", _ok_call_tool)
    payload = {"part": "p1"}

    leader = asyncio.create_task(client.run_pre_llm("analyze", payload))
    await agent.first_call_entered.wait()
    follower = asyncio.create_task(client.run_pre_llm("analyze", payload))
    await asyncio.sleep(0)  # the follower parks on asyncio.shield(leader's future)

    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader

    trace = await asyncio.wait_for(follower, timeout=5)
    assert trace.steps, f"the waiter did not run the loop: {trace.degradation_reason!r}"
    assert agent.calls >= 2, "the waiter must generate its own drafts"


async def test_co_waiting_caller_cancellation_is_not_absorbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the *waiter* re-raises; it must not take over the shared loop."""
    client, agent = _cancel_aware_client()
    monkeypatch.setattr(client, "_call_tool", _ok_call_tool)
    payload = {"part": "p1"}

    leader = asyncio.create_task(client.run_pre_llm("analyze", payload))
    await agent.first_call_entered.wait()
    follower = asyncio.create_task(client.run_pre_llm("analyze", payload))
    await asyncio.sleep(0)

    follower.cancel()
    with pytest.raises(asyncio.CancelledError):
        await follower
    assert agent.calls == 1, "a cancelled waiter must not start its own loop"

    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader


# --------------------------------------------------------------------------------------
# Subprocess hygiene: protocol-pure stdout, banner-noise suppression
# --------------------------------------------------------------------------------------


class TestConsoleShim:
    """The npm servers print banners on stdout — the MCP stdio protocol stream."""

    def test_shipped_shim_exists_next_to_the_module(self) -> None:
        assert client_module.CONSOLE_SHIM_CANDIDATES, "no console shim candidate configured"
        for path in client_module.CONSOLE_SHIM_CANDIDATES:
            assert Path(path).is_file(), f"console shim missing: {path}"

    def test_shim_is_preloaded_when_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        shim = tmp_path / "quiet-console.cjs"
        shim.write_text("// shim\n", encoding="utf-8")
        monkeypatch.setattr(client_module, "CONSOLE_SHIM_CANDIDATES", (str(shim),))
        monkeypatch.delenv("NODE_OPTIONS", raising=False)

        assert _subprocess_env()["NODE_OPTIONS"] == f"--require {shim}"

    def test_existing_node_options_are_preserved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        shim = tmp_path / "quiet-console.cjs"
        shim.write_text("// shim\n", encoding="utf-8")
        monkeypatch.setattr(client_module, "CONSOLE_SHIM_CANDIDATES", (str(shim),))
        monkeypatch.setenv("NODE_OPTIONS", "--max-old-space-size=512")

        assert _subprocess_env()["NODE_OPTIONS"] == f"--max-old-space-size=512 --require {shim}"

    def test_missing_shim_leaves_node_options_untouched(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            client_module, "CONSOLE_SHIM_CANDIDATES", (str(tmp_path / "absent.cjs"),)
        )
        monkeypatch.delenv("NODE_OPTIONS", raising=False)

        assert "NODE_OPTIONS" not in _subprocess_env()


class TestBannerParseNoiseFilter:
    def test_forwarded_banner_noise_is_filtered(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        _client()  # every client installs the filter on the forwarded-message logger
        caplog.set_level(logging.INFO, logger=FORWARDED_LOGGER)
        forwarded = logging.getLogger(FORWARDED_LOGGER)

        forwarded.info(
            "Received NOTICE from server (code-reasoning): "
            "{'message': '🚀 Code-Reasoning MCP Server ready.'}"
        )
        forwarded.info(
            "Received WARNING from server (shannonthinking): tool call timed out"
        )

        assert [record.getMessage() for record in caplog.records] == [
            "Received WARNING from server (shannonthinking): tool call timed out"
        ]

    def test_filter_installation_is_idempotent(self) -> None:
        _client()
        _client()

        installed = [
            filter_
            for filter_ in logging.getLogger(FORWARDED_LOGGER).filters
            if isinstance(filter_, BannerParseNoiseFilter)
        ]
        assert len(installed) == 1


# --------------------------------------------------------------------------------------
# Forwarded log levels: the servers' info chatter is DEBUG, not INFO
# --------------------------------------------------------------------------------------


def _notification(level: str, data: object = "\n💭 ", logger_name: str = "code-reasoning") -> LogMessage:
    return LogMessage(level=level, logger=logger_name, data=data)


class TestForwardedLogLevels:
    """code-reasoning sends its "💭 …" thinking notes as info-level notifications.

    At the server's default INFO log level they are noise; the handler passed to every
    ``Client`` emits them at DEBUG instead (report 2026-09-24).
    """

    def test_info_notification_is_invisible_at_info_level(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger=FORWARDED_LOGGER)

        asyncio.run(client_module.reasoning_log_handler(_notification("info")))

        assert caplog.records == []

    def test_info_notification_is_a_debug_record(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG, logger=FORWARDED_LOGGER)

        asyncio.run(client_module.reasoning_log_handler(_notification("info")))

        assert [(record.levelname, record.name) for record in caplog.records] == [
            ("DEBUG", FORWARDED_LOGGER)
        ]
        # fastmcp's text is preserved, so existing greps keep working.
        assert caplog.records[0].getMessage() == (
            "Received INFO from server (code-reasoning): \n💭 "
        )

    def test_notice_is_demoted_too(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG, logger=FORWARDED_LOGGER)

        asyncio.run(
            client_module.reasoning_log_handler(
                _notification("notice", {"message": "thinking step finished"})
            )
        )

        assert [record.levelno for record in caplog.records] == [logging.DEBUG]

    def test_warning_and_error_keep_their_levels(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger=FORWARDED_LOGGER)

        asyncio.run(client_module.reasoning_log_handler(_notification("warning", "slow")))
        asyncio.run(client_module.reasoning_log_handler(_notification("error", "boom")))

        assert [record.levelname for record in caplog.records] == ["WARNING", "ERROR"]

    def test_unknown_level_falls_back_to_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Defensive: ``LogMessage`` types the level as a Literal, a rogue peer may not."""
        caplog.set_level(logging.INFO, logger=FORWARDED_LOGGER)
        rogue = SimpleNamespace(level="emergency-ish", logger="code-reasoning", data="?")

        asyncio.run(client_module.reasoning_log_handler(rogue))  # type: ignore[arg-type]

        assert [record.levelname for record in caplog.records] == ["INFO"]

    def test_forwarded_logger_is_governed_by_the_app_level(self) -> None:
        """The emission logger must not live under ``fastmcp``.

        That ancestor is pinned at INFO with its own handlers and ``propagate=False``,
        so a DEBUG record there is unreachable at *any* ``logging_level`` — the pin
        wins over the app's root level and the chatter would silently never appear.
        """
        forwarded = logging.getLogger(FORWARDED_LOGGER)
        root = logging.getLogger()
        previous = root.level
        try:
            assert not forwarded.name.startswith("fastmcp.")
            root.setLevel(logging.DEBUG)
            assert forwarded.getEffectiveLevel() == logging.DEBUG
            root.setLevel(logging.INFO)
            assert forwarded.getEffectiveLevel() == logging.INFO
        finally:
            root.setLevel(previous)

    async def test_call_tool_forwards_the_demoting_handler_and_pools_the_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dropped ``log_handler`` kwarg is silent — pin the wiring at the pool's single construction site."""
        recorded: list[dict[str, object]] = []
        instances: list[object] = []

        class _FakeClient:
            constructed = 0

            def __init__(self, transport: object, **kwargs: object) -> None:
                _FakeClient.constructed += 1
                self.transport = transport
                self.closed = False
                recorded.append(kwargs)
                instances.append(self)

            async def __aenter__(self) -> _FakeClient:
                return self

            async def __aexit__(self, *_exc: object) -> bool:
                return False

            async def call_tool(self, name: str, payload: object) -> object:
                return object()

            async def close(self) -> None:
                self.closed = True

        monkeypatch.setattr(client_module, "Client", _FakeClient)
        monkeypatch.setattr(client_module, "_resolve_cmd", lambda name, cmd: ("node", ["index.js"]))
        client, _agent = _client()

        await client._call_tool(CODE_TOOL, ThoughtDraft(thought="stub", phase_tag="model"), 1, 1)
        await client._call_tool(CODE_TOOL, ThoughtDraft(thought="stub 2", phase_tag="model"), 2, 1)

        assert recorded == [{"log_handler": client_module.reasoning_log_handler}], (
            "the single pooled construction must carry the demoting handler"
        )
        assert _FakeClient.constructed == 1, "one persistent client per tool, reused across steps"
        await client.close()
        fake = instances[0]
        assert isinstance(fake, _FakeClient)
        assert fake.closed, "close() shuts down the pooled client"

    async def test_call_tool_respawns_client_after_transport_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Any call_tool failure drops the pooled client; the next call builds a fresh one."""
        fail_first = True

        class _FakeClient:
            constructed = 0

            def __init__(self, transport: object, **_kwargs: object) -> None:
                _FakeClient.constructed += 1
                self.closed = False

            async def __aenter__(self) -> _FakeClient:
                return self

            async def __aexit__(self, *_exc: object) -> bool:
                return False

            async def call_tool(self, name: str, payload: object) -> object:
                if fail_first and _FakeClient.constructed == 1:
                    raise ConnectionError("stdio pipe broke")
                return object()

            async def close(self) -> None:
                self.closed = True

        monkeypatch.setattr(client_module, "Client", _FakeClient)
        monkeypatch.setattr(client_module, "_resolve_cmd", lambda name, cmd: ("node", ["index.js"]))
        client, _agent = _client()

        with pytest.raises(ConnectionError):
            await client._call_tool(CODE_TOOL, ThoughtDraft(thought="stub", phase_tag="model"), 1, 1)
        assert client._clients == {}, "failed client must be dropped, not left pooled"
        text = await client._call_tool(CODE_TOOL, ThoughtDraft(thought="stub 2", phase_tag="model"), 2, 1)
        assert text == ""
        assert _FakeClient.constructed == 2, "second call respawns a fresh client"

    async def test_call_tool_spawn_timeout_bounds_client_connect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wedged spawn consumes ``spawn_timeout_seconds`` and fails the call."""

        class _WedgedClient:
            def __init__(self, transport: object, **_kwargs: object) -> None:
                pass

            async def __aenter__(self) -> _WedgedClient:
                await asyncio.sleep(30)
                raise AssertionError("unreachable")

            async def close(self) -> None:
                pass

        monkeypatch.setattr(client_module, "Client", _WedgedClient)
        monkeypatch.setattr(client_module, "_resolve_cmd", lambda name, cmd: ("node", ["index.js"]))
        config = ReasoningConfig(enabled=True, quiet_stderr=True, spawn_timeout_seconds=0.05)
        client = ReasoningClient(config, _StubAgent())  # type: ignore[arg-type]

        with pytest.raises(asyncio.TimeoutError):
            await client._call_tool(SHANNON_TOOL, ThoughtDraft(thought="stub", phase_tag="proof"), 1, 1)
        assert client._clients == {}, "wedged spawn must not leave a pooled client"


# --------------------------------------------------------------------------------------
# J5 gate: a decision-model veto on one call's deliberation
# --------------------------------------------------------------------------------------


class TestReasoningGate:
    """The gate is fail-open and cache-neutral: only an explicit ``False`` skips."""

    def test_false_yields_an_empty_non_degraded_trace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, agent = _client()
        seen: list[tuple[str, dict[str, object]]] = []

        async def _gate(phase: str, payload: dict[str, object]) -> bool | None:
            seen.append((phase, payload))
            return False

        client._gate = _gate

        async def _explode(*_args: object, **_kwargs: object) -> str:
            raise AssertionError("a gated-out call must not reach a reasoning tool")

        monkeypatch.setattr(client, "_call_tool", _explode)

        trace = asyncio.run(client.run_pre_llm("weights", {"part": "p1"}))

        assert trace.phase == "weights"
        assert trace.steps == []
        assert trace.degraded is False
        assert trace.degradation_reason is None
        assert client._trace_cache == {}, "a gated-out call must not consume a cache slot"
        assert client._inflight == {}
        assert agent.calls == 0
        assert seen == [("weights", {"part": "p1"})]

    def test_true_proceeds_to_the_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, agent = _client()

        async def _gate(_phase: str, _payload: dict[str, object]) -> bool | None:
            return True

        client._gate = _gate

        async def _fake_call_tool(*_args: object, **_kwargs: object) -> str:
            return "ok"

        monkeypatch.setattr(client, "_call_tool", _fake_call_tool)

        trace = asyncio.run(client.run_pre_llm("weights", {"part": "p1"}))
        assert trace.steps, f"degraded: {trace.degradation_reason!r}"
        assert agent.calls >= 1

    def test_none_proceeds_to_the_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``None`` is the degradation signal (TacitClient down): reasoning still runs."""
        client, _agent = _client()

        async def _gate(_phase: str, _payload: dict[str, object]) -> bool | None:
            return None

        client._gate = _gate

        async def _fake_call_tool(*_args: object, **_kwargs: object) -> str:
            return "ok"

        monkeypatch.setattr(client, "_call_tool", _fake_call_tool)

        trace = asyncio.run(client.run_pre_llm("weights", {"part": "p1"}))
        assert trace.steps

    def test_gate_exception_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A broken gate closure must never fail the tool call: reasoning proceeds."""
        client, _agent = _client()

        async def _gate(_phase: str, _payload: dict[str, object]) -> bool | None:
            raise RuntimeError("boom")

        client._gate = _gate

        async def _fake_call_tool(*_args: object, **_kwargs: object) -> str:
            return "ok"

        monkeypatch.setattr(client, "_call_tool", _fake_call_tool)

        trace = asyncio.run(client.run_pre_llm("weights", {"part": "p1"}))
        assert trace.steps, f"degraded: {trace.degradation_reason!r}"

    def test_disabled_phase_never_asks_the_gate(self) -> None:
        client, _agent = _client()
        client._config.per_phase["weights"] = client._config.get_strategy("weights").model_copy(
            update={"enabled": False}
        )
        asked: list[str] = []

        async def _gate(phase: str, _payload: dict[str, object]) -> bool | None:
            asked.append(phase)
            return False

        client._gate = _gate
        asyncio.run(client.run_pre_llm("weights", {"part": "p1"}))
        assert asked == []
