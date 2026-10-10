"""End-to-end FastMCP surface: tools, prompts, resources and the as-tools transforms.

The server is driven in memory (``Client(server._mcp)`` inside the lifespan, which is what
registers the tools) with the repository configuration, the offline retrieval index and a scripted
model — so the smoke proves the protocol wiring, not a model's opinion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastmcp import Client

from src.schemas import DesignEvaluation, RawBriefAndParts, RawDecisionPart, RawDesignBrief
from src.server import DesignPatternServer
from tests.unit.conftest import DOCS_DIR
from tests.unit.fakes import ScriptedLLM

DESCRIPTION = (
    "Component GeoEnricher calls a Redis geo-lookup synchronously while enriching sensor events; "
    "enrichment must stay within the p99 latency budget and Redis outages must not stall the pipeline."
)

NATIVE_TOOLS = {
    "select_design_patterns",
    "analyze_design",
    "generate_design",
    "evaluate_design",
    "list_design_patterns",
    "get_design_pattern",
    "submit_design_job",
    "get_design_status",
    "cancel_design",
}
TRANSFORM_TOOLS = {"list_prompts", "get_prompt", "list_resources", "read_resource"}
PROMPTS = {
    "select_design_patterns_workflow",
    "explore_pattern_catalog",
    "evaluate_my_design",
    "compare_pattern_candidates",
}


@pytest.fixture
def scripted_llm(monkeypatch: pytest.MonkeyPatch) -> ScriptedLLM:
    """Patch the server's LLM factory so the pipeline never calls a provider."""
    llm = ScriptedLLM(
        brief_and_parts_result=RawBriefAndParts(
            design_brief=RawDesignBrief(
                restated_goal="harden the enrichment stage",
                components=[{"name": "GeoEnricher", "type": "worker", "responsibilities": ["enrich events"]}],
                constraints=["p99 < 50 ms per event"],
            ),
            decision_parts=[
                RawDecisionPart(
                    id="enrichment-resilience",
                    statement="bound the synchronous Redis lookup so enrichment survives an outage",
                    component_refs=["GeoEnricher"],
                    scope="task-lifetime",
                    category_hypothesis="concurrency-resilience",
                    priority=1.0,
                )
            ],
        ),
        weights={"reliability": 1.0, "performance": 0.8, "simplicity": 0.6, "testability": 0.5,
                 "maintainability": 0.4, "scalability": 0.3, "extensibility": 0.2, "security": 0.2},
        evaluation=DesignEvaluation(score=88.0),
    )
    monkeypatch.setattr("src.server.build_llm_client", lambda config: llm)
    return llm


async def test_protocol_surface(config_path: str, scripted_llm: ScriptedLLM) -> None:
    """Native tools, prompt/resource tools, prompts, resources and a real tool call."""
    server = DesignPatternServer(config_path, offline=True)
    async with server._mcp.lifespan(), Client(server._mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        assert set(tools) >= NATIVE_TOOLS
        assert set(tools) >= TRANSFORM_TOOLS
        native = set(tools) - TRANSFORM_TOOLS
        assert len(native) >= len(NATIVE_TOOLS)
        # the PromptsAsTools annotation shim (#3459) must be visible to clients
        assert tools["list_prompts"].annotations is not None
        assert tools["get_prompt"].annotations is not None
        assert tools["list_resources"].annotations is not None
        assert tools["select_design_patterns"].annotations.read_only_hint is False

        prompts = await client.list_prompts()
        assert {prompt.name for prompt in prompts} >= PROMPTS
        rendered = await client.get_prompt(
            "select_design_patterns_workflow", {"description": DESCRIPTION}
        )
        assert "select_design_patterns" in str(rendered)

        resources = await client.list_resources()
        assert any(str(resource.uri) == "design-patterns://variants" for resource in resources)
        listing = await client.call_tool("list_resources", {})
        assert "design-patterns://variants" in str(listing.data)

        payload = await client.read_resource("design-patterns://variants")
        parsed = json.loads(str(payload[0].text))
        assert "typescript" in parsed and "timeout" in parsed["typescript"]

        per_pattern = await client.read_resource("design-patterns://variants/timeout")
        timeout_slice = json.loads(str(per_pattern[0].text))
        assert timeout_slice["pattern"] == "timeout"
        assert "typescript" in timeout_slice["languages"]

        result = await client.call_tool("select_design_patterns", {"description": DESCRIPTION, "language": "typescript"})
        data = result.data
        assert data["overview"]["primary_pattern"]
        assert data["overview"]["languages"] == ["typescript"]
        assert data["applications"]
        assert all(application["language_binding"]["language"] == "typescript" for application in data["applications"])

        from jsonschema import Draft7Validator

        schema = json.loads((DOCS_DIR / "result-schema.json").read_text(encoding="utf-8"))
        Draft7Validator(schema).validate(data)


async def test_job_trio_lifecycle(config_path: str, scripted_llm: ScriptedLLM) -> None:
    """The job trio: submit returns a job id, polling reaches a terminal state, cancel is honest."""
    server = DesignPatternServer(config_path, offline=True)
    async with server._mcp.lifespan(), Client(server._mcp) as client:
        submitted = await client.call_tool("submit_design_job", {"description": DESCRIPTION})
        job_id = submitted.data["job_id"]
        assert submitted.data["status"] == "pending"

        status = None
        for _ in range(50):
            polled = await client.call_tool("get_design_status", {"job_id": job_id})
            status = polled.data
            if status["status"] in {"completed", "failed", "cancelled"}:
                break
            import asyncio

            await asyncio.sleep(0.05)
        assert status is not None
        assert status["status"] == "completed"
        assert status["result"]["applications"]
        assert status["result"]["overview"]["primary_pattern"]

        cancel = await client.call_tool("cancel_design", {"job_id": job_id})
        assert cancel.data["status"] == "completed"
        assert cancel.data["cancelled"] is False


async def test_short_description_is_a_structured_error(config_path: str, scripted_llm: ScriptedLLM) -> None:
    """E1: the tool maps the intake failure to a ToolError carrying ERR_001."""
    from fastmcp.exceptions import ToolError

    server = DesignPatternServer(config_path, offline=True)
    async with server._mcp.lifespan(), Client(server._mcp) as client:
        with pytest.raises(ToolError, match="ERR_001"):
            await client.call_tool("select_design_patterns", {"description": "too short"})


async def test_unknown_job_is_a_structured_error(config_path: str, scripted_llm: ScriptedLLM) -> None:
    """An unknown job id raises the documented ERR_404 tool error."""
    from fastmcp.exceptions import ToolError

    server = DesignPatternServer(config_path, offline=True)
    async with server._mcp.lifespan(), Client(server._mcp) as client:
        with pytest.raises(ToolError, match="ERR_404"):
            await client.call_tool("get_design_status", {"job_id": "does-not-exist"})


async def test_phase_chain_over_mcp(config_path: str, scripted_llm: ScriptedLLM) -> None:
    """analyze_design → generate_design → evaluate_design wired through the protocol."""
    server = DesignPatternServer(config_path, offline=True)
    async with server._mcp.lifespan(), Client(server._mcp) as client:
        analyzed = await client.call_tool(
            "analyze_design", {"description": DESCRIPTION, "language": "typescript"}
        )
        analysis = analyzed.data
        assert analysis["selected_patterns"]
        assert analysis["selections"]
        part_id, part_selection = next(iter(analysis["selections"].items()))
        assert part_id in {part["id"] for part in analysis["decision_parts"]}
        assert part_selection["primary"]
        assert analysis["effective_language"] == "typescript"

        generated = await client.call_tool("generate_design", {"analysis": analysis})
        design = generated.data
        assert design["applications"]
        assert design["overview"]["primary_pattern"]
        assert design["overview"]["languages"] == ["typescript"]
        assert all(
            application["language_binding"]["language"] == "typescript"
            for application in design["applications"]
        )

        evaluated = await client.call_tool(
            "evaluate_design", {"design_json": design, "requirements": DESCRIPTION}
        )
        evaluation = evaluated.data
        assert 0.0 <= evaluation["score"] <= 100.0
        assert evaluation["summary"]["overall_score"] == evaluation["score"]


async def test_catalog_tools_over_mcp(config_path: str, scripted_llm: ScriptedLLM) -> None:
    """list_design_patterns covers the full catalog; get_design_pattern resolves and rejects."""
    from fastmcp.exceptions import ToolError

    server = DesignPatternServer(config_path, offline=True)
    async with server._mcp.lifespan(), Client(server._mcp) as client:
        listing = await client.call_tool("list_design_patterns", {})
        entries = listing.data
        assert len(entries) == 438
        assert all(
            {"name", "category", "description"} <= set(entry) and entry["name"] for entry in entries
        )

        empty = await client.call_tool("list_design_patterns", {"category": "nope"})
        assert empty.data == []

        fetched = await client.call_tool("get_design_pattern", {"name": "singleton"})
        record = fetched.data
        assert record["name"] == "singleton"
        assert "typescript" in record["languages"]
        assert record["participants"]

        with pytest.raises(ToolError, match="ERR_013"):
            await client.call_tool("get_design_pattern", {"name": "no-such-pattern"})



# --------------------------------------------------------------------------------------
# Whole-run deadline (tasks.deadline_seconds -> ERR_014)
# --------------------------------------------------------------------------------------


class _NeverReturningLLM(ScriptedLLM):
    """Pipeline double whose first LLM call never settles — the hang the deadline must end."""

    def __init__(self) -> None:
        super().__init__(evaluation=DesignEvaluation(score=88.0))
        self.cancelled = False

    async def extract_brief_and_parts(
        self, description: str, hints: object, *, reasoning_context: str = ""
    ) -> RawBriefAndParts:
        import asyncio

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")


@pytest.fixture
def short_deadline_config(tmp_path, config_path: str) -> str:
    """A copy of the shipped config with a short whole-run deadline.

    Two seconds, not milliseconds: the deadline must fire *while the pipeline is in flight*
    (the assertions below witness the cancelled LLM coroutine), so it has to outlast the
    pipeline's own startup and catalog/retrieval work. At 50 ms a loaded machine spent the whole
    budget on startup, ERR_014 fired before the first LLM call was entered, and the test failed
    on ``llm.cancelled`` while the ERR_014 path itself was correct.
    """
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    raw["tasks"]["deadline_seconds"] = 2.0
    raw["tasks"]["heartbeat_enabled"] = False
    target = tmp_path / "config.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    return str(target)


async def test_sync_tool_deadline_is_a_structured_tool_error(
    short_deadline_config: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """select_design_patterns ends with ERR_014 instead of hanging on the provider coroutine."""
    from fastmcp.exceptions import ToolError

    llm = _NeverReturningLLM()
    monkeypatch.setattr("src.server.build_llm_client", lambda config: llm)
    server = DesignPatternServer(short_deadline_config, offline=True)

    async with server._mcp.lifespan(), Client(server._mcp) as client:
        with pytest.raises(ToolError, match="ERR_014"):
            await client.call_tool("select_design_patterns", {"description": DESCRIPTION})

    assert llm.cancelled is True, "the deadline must cancel the in-flight pipeline coroutine"


async def test_job_deadline_settles_the_job_as_failed(
    short_deadline_config: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The job runner lets ERR_014 reach its existing set_failed path."""
    import asyncio

    llm = _NeverReturningLLM()
    monkeypatch.setattr("src.server.build_llm_client", lambda config: llm)
    server = DesignPatternServer(short_deadline_config, offline=True)

    async with server._mcp.lifespan(), Client(server._mcp) as client:
        submitted = await client.call_tool("submit_design_job", {"description": DESCRIPTION})
        job_id = submitted.data["job_id"]

        status = None
        # budget: 200 polls at 50 ms covers the 2 s deadline (short_deadline_config) plus slack
        for _ in range(200):
            polled = await client.call_tool("get_design_status", {"job_id": job_id})
            status = polled.data
            if status["status"] in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.05)

    assert status is not None
    assert status["status"] == "failed"
    assert "ERR_014" in status["error"]


async def test_lifespan_starts_with_reasoning_disabled(
    tmp_path, config_path: str, scripted_llm: ScriptedLLM
) -> None:
    """Disabling the reasoning layer must not break startup (no health check, no AttributeError)."""
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    raw["reasoning"]["enabled"] = False
    target = tmp_path / "config-no-reasoning.json"
    target.write_text(json.dumps(raw), encoding="utf-8")

    server = DesignPatternServer(str(target), offline=True)
    async with server._mcp.lifespan(), Client(server._mcp) as client:
        result = await client.call_tool("select_design_patterns", {"description": DESCRIPTION})

    assert result.data["applications"]
    assert server._reasoning_client is None
    assert server._reasoning_health == {}
