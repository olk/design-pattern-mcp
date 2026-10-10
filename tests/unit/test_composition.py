"""§9 — composition, §11 loop integration and the outbound result contract.

The strongest check here is end-to-end: a full pipeline run with scripted model calls must produce
a payload that validates against ``docs/result-schema.json`` (the schema the tool returns).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from jsonschema import Draft7Validator

from src.patterns.loader import CATALOG_DOC_FILE
from src.config import ServerConfig
from src.tools.design import pipeline_result_to_output
from src.pipeline import compose_selection
from src.schemas import (
    AnnotatedCandidate,
    DesignBrief,
    DesignEvaluation,
    PartSelection,
    RawBriefAndParts,
    RawDecisionPart,
    RawDesignBrief,
)
from tests.unit.conftest import DOCS_DIR
from tests.unit.fakes import ScriptedLLM, StubHit, StubIndex

APPLICATION_ID_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


def _candidate(name: str, fit: float, *, analysis: float = 80.0, category: str = "concurrency-resilience") -> AnnotatedCandidate:
    return AnnotatedCandidate(
        name=name,
        category=category,  # type: ignore[arg-type]
        retrieval_score=0.03,
        analysis_score=analysis,
        fusion_score_normalized=50.0,
        blended=fit,
        fit_score=fit,
    )


def _selection(
    part: RawDecisionPart, candidates: list[AnnotatedCandidate], catalog, primary: str, supporting=()
) -> PartSelection:
    from src.schemas import DecisionPart

    decision_part = DecisionPart(
        id=part.id,
        statement=part.statement,
        component_refs=list(part.component_refs),
        scope=part.scope or "module",  # type: ignore[arg-type]
        category_hypothesis=part.category_hypothesis,  # type: ignore[arg-type]
        priority=part.priority,
    )
    return PartSelection(
        part=decision_part,
        candidates=candidates,
        primary=primary,
        primary_category=next(c.category for c in candidates if c.name == primary),
        supporting=list(supporting) if supporting else [],
        alternatives=[],
    )


def _raw_part(part_id: str, *, priority: float, component: str, category: str, statement: str) -> RawDecisionPart:
    return RawDecisionPart(
        id=part_id,
        statement=statement,
        component_refs=[component],
        scope="module",
        category_hypothesis=category,
        priority=priority,
    )


def test_dominant_part_election_and_application_ids(catalog) -> None:
    """§9 rules 1–3: dominance = priority × fit/100, ids follow ``{part}-{pattern}``."""
    from src.schemas import SupportingChoice

    dominant_part = _raw_part(
        "enrichment-resilience", priority=1.0, component="GeoEnricher",
        category="concurrency-resilience", statement="bound the lookup",
    )
    minor_part = _raw_part(
        "parser-tolerance", priority=0.5, component="JsonParser",
        category="interaction-boundary", statement="tolerate unknown fields",
    )
    selections = {
        "enrichment-resilience": _selection(
            dominant_part,
            [_candidate("timeout", 93.5), _candidate("retry-with-backoff", 87.3)],
            catalog,
            "timeout",
            supporting=[SupportingChoice(pattern="retry-with-backoff", role="absorb transient failures (enrichment-resilience)", fit_score=87.3)],
        ),
        "parser-tolerance": _selection(
            minor_part,
            [_candidate("tolerant-reader", 70.0, category="interaction-boundary")],
            catalog,
            "tolerant-reader",
        ),
    }
    composed = compose_selection(
        selections=selections,
        brief=DesignBrief(restated_goal="goal", constraints=["at-least-once"]),
        catalog=catalog,
        config=ServerConfig(),
    )
    assert composed.dominant_part_id == "enrichment-resilience"
    assert composed.primary == "timeout"
    assert composed.category == "concurrency-resilience"
    ids = [application.id for application in composed.applications]
    assert ids == [
        "enrichment-resilience-timeout",
        "parser-tolerance-tolerant-reader",
        "enrichment-resilience--retry-with-backoff",
    ]
    assert all(APPLICATION_ID_RE.match(application_id) for application_id in ids)
    assert [pattern.pattern for pattern in composed.supporting] == ["retry-with-backoff"]


def test_supporting_union_dedupes_and_caps(catalog) -> None:
    """§9 rule 2/4: the union dedupes by name and the pattern stays one entry."""
    from src.schemas import SupportingChoice

    part_a = _raw_part("a-part", priority=1.0, component="A", category="behavior-selection", statement="s")
    part_b = _raw_part("b-part", priority=0.9, component="B", category="behavior-selection", statement="s")
    shared = SupportingChoice(pattern="dependency-injection", role="wire collaborators (a-part)", fit_score=60.0)
    selections = {
        "a-part": _selection(part_a, [_candidate("strategy", 80.0, category="behavior-selection")], catalog, "strategy", supporting=[shared]),
        "b-part": _selection(part_b, [_candidate("command", 75.0, category="behavior-selection")], catalog, "command", supporting=[shared]),
    }
    composed = compose_selection(
        selections=selections,
        brief=DesignBrief(restated_goal="goal"),
        catalog=catalog,
        config=ServerConfig(),
    )
    assert [pattern.pattern for pattern in composed.supporting] == ["dependency-injection"]
    injection_applications = [
        application for application in composed.applications if application.pattern == "dependency-injection"
    ]
    # §9 rule 4: two parts selecting the same pattern yield two applications (different problems)
    # while the pattern appears exactly once in overview.supporting_patterns.
    assert len(injection_applications) == 2


def test_interaction_seeds_from_brief_flows(catalog) -> None:
    """§9 rule 5: a flow between components of different applications seeds an interaction."""
    part_a = _raw_part("enrich", priority=1.0, component="GeoEnricher", category="concurrency-resilience", statement="s")
    part_b = _raw_part("sink", priority=0.8, component="InfluxSink", category="persistence-state", statement="s")
    brief = DesignBrief(
        restated_goal="goal",
        flows=[{"source": "GeoEnricher", "target": "InfluxSink", "kind": "event-stream"}],  # type: ignore[list-item]
    )
    selections = {
        "enrich": _selection(part_a, [_candidate("timeout", 90.0)], catalog, "timeout"),
        "sink": _selection(part_b, [_candidate("repository", 80.0, category="persistence-state")], catalog, "repository"),
    }
    composed = compose_selection(selections=selections, brief=brief, catalog=catalog, config=ServerConfig())
    assert [seed.type for seed in composed.interaction_seeds] == ["event-stream"]
    assert composed.interaction_seeds[0].source == "enrich-timeout"
    assert composed.interaction_seeds[0].target == "sink-repository"


async def test_end_to_end_result_validates_against_the_result_schema(catalog) -> None:
    """§11/§12: a full run returns a payload the outbound schema accepts, with one language echo."""
    raw_parts = [
        RawDecisionPart(
            id="enrichment-resilience",
            statement="bound the synchronous Redis lookup so enrichment survives an outage",
            component_refs=["GeoEnricher"],
            scope="task-lifetime",
            category_hypothesis="concurrency-resilience",
            constraints=["retry-with-backoff absorbs transient Redis failures"],
            priority=1.0,
        ),
        RawDecisionPart(
            id="sink-idempotency",
            statement="duplicate deliveries must not double-write the sinks",
            component_refs=["InfluxSink"],
            scope="message-flow",
            category_hypothesis="workflow-messaging",
            priority=0.8,
        ),
    ]
    brief = RawBriefAndParts(
        design_brief=RawDesignBrief(
            restated_goal="harden the event pipeline",
            architecture_style="event-driven",
            constraints=["at-least-once delivery", "p99 < 50 ms per event"],
            components=[
                {"name": "GeoEnricher", "type": "worker", "responsibilities": ["enrich"], "technologies": ["TypeScript"]},
                {"name": "InfluxSink", "type": "sink", "responsibilities": ["write"], "technologies": ["TypeScript 5"]},
            ],
            flows=[{"source": "GeoEnricher", "target": "InfluxSink", "kind": "event-stream"}],
            negative_signals=["no reflection"],
        ),
        decision_parts=raw_parts,
    )
    llm = ScriptedLLM(
        brief_and_parts_result=brief,
        weights={
            "reliability": 1.0, "performance": 0.8, "simplicity": 0.6, "testability": 0.5,
            "maintainability": 0.4, "scalability": 0.3, "extensibility": 0.2, "security": 0.2,
        },
        evaluation=DesignEvaluation(score=72.0),
    )
    index = StubIndex(
        by_statement={
            "bound the synchronous Redis lookup so enrichment survives an outage": [
                StubHit("timeout", 0.0317, 1, 1),
                StubHit("retry-with-backoff", 0.0300, 2, 2),
                StubHit("circuit-breaker", 0.0283, 3, 3),
            ],
            "duplicate deliveries must not double-write the sinks": [
                StubHit("idempotent-receiver", 0.0310, 1, 1),
            ],
        }
    )
    from src.pipeline import DesignPipeline

    pipeline = DesignPipeline(
        config=ServerConfig(), catalog=catalog, index=index, llm=llm  # type: ignore[arg-type]
    )
    result = pipeline_result_to_output(await pipeline.run_design(
        description=(
            "Component GeoEnricher calls a Redis geo-lookup synchronously while enriching sensor "
            "events; InfluxSink writes every enriched event and must not double-write on retries."
        ),
        language="typescript",
        # §8.4 layer 1: a verbatim variant_id in focus binds that realization exactly.
        focus="bind variant 'abort-signal-timeout' for the enrichment step",
    ))

    schema = json.loads((DOCS_DIR / "result-schema.json").read_text(encoding="utf-8"))
    Draft7Validator.check_schema(schema)
    Draft7Validator(schema).validate(result)
    assert result["overview"]["languages"] == ["typescript"]
    assert result["overview"]["primary_pattern"] == "timeout"
    assert result["overview"]["supporting_patterns"]
    assert all("language_binding" in application for application in result["applications"])
    binding = result["applications"][0]["language_binding"]
    assert binding["language"] == "typescript"
    assert binding["support"] in {"native", "idiomatic", "library", "manual"}
    assert binding["bound_variant"] == "abort-signal-timeout"
    assert "from" in result["applications"][0]["interactions"][0] if result["applications"][0]["interactions"] else True


async def test_generic_mode_reports_generic_bindings(catalog) -> None:
    """§8.4 request echo: an absent language hint yields generic bindings and one generic echo."""
    brief = RawBriefAndParts(
        design_brief=RawDesignBrief(restated_goal="goal"),
        decision_parts=[
            RawDecisionPart(
                id="lookup", statement="bound the lookup", scope="module",
                category_hypothesis="concurrency-resilience", priority=1.0,
            )
        ],
    )
    llm = ScriptedLLM(brief_and_parts_result=brief, weights={"reliability": 1.0})
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])
    from src.pipeline import DesignPipeline

    result = pipeline_result_to_output(await DesignPipeline(
        config=ServerConfig(), catalog=catalog, index=index, llm=llm  # type: ignore[arg-type]
    ).run_design(description="bound the lookup with a timeout so it cannot hang"))
    assert result["overview"]["languages"] == ["generic"]
    assert result["applications"][0]["language_binding"]["language"] == "generic"


def test_out_of_scope_terms_come_from_the_catalog_document() -> None:
    """§5 rule 5: the anti-scope term set is parsed from the Background section, not hardcoded."""
    from src.patterns.loader import parse_out_of_scope_terms

    terms = parse_out_of_scope_terms(Path(DOCS_DIR) / CATALOG_DOC_FILE)
    assert "layered" in terms and "service discovery" in terms
    assert "fleet-level resiliency" in terms


async def test_reasoning_trace_discloses_statement_and_full_weight_vector(catalog) -> None:
    """§6.3 disclosure: the sampled decomposition question and the full 8-axis vector are client-visible."""
    from src.pipeline import DesignPipeline
    from src.schemas.enums import QUALITY_AXES

    llm = ScriptedLLM(
        brief_and_parts_result=RawBriefAndParts(
            design_brief=RawDesignBrief(restated_goal="publish domain events"),
            decision_parts=[
                RawDecisionPart(
                    id="event-publishing",
                    statement="publish domain events to the broker",
                    component_refs=[],
                    scope="message-flow",
                    category_hypothesis="workflow-messaging",
                    priority=1.0,
                )
            ],
        ),
        weights={"reliability": 1.0},
    )
    pipeline = DesignPipeline(
        config=ServerConfig(),
        catalog=catalog,
        index=StubIndex(hits=[StubHit(name="publish-subscribe-channel", retrieval_score=5.0)]),
        llm=llm,
    )
    result = pipeline_result_to_output(
        await pipeline.run_design(description="An EventPublisher publishes domain events to the broker.")
    )
    reasoning = result["overview"]["reasoning"]
    assert "statement: publish domain events to the broker" in reasoning
    match = re.search(r"^  weights vector: (.+)$", reasoning, re.MULTILINE)
    assert match is not None
    vector = dict(pair.split("=") for pair in match.group(1).split())
    assert set(vector) == set(QUALITY_AXES)
    assert float(vector["reliability"]) == pytest.approx(0.7375, abs=1e-3)
    for axis in QUALITY_AXES:
        if axis != "reliability":
            assert float(vector[axis]) == pytest.approx(0.0375, abs=1e-3)


def test_reasoning_trace_marks_a_missing_weight_vector(catalog) -> None:
    """A selection built without a vector discloses 'none', not a fake all-zero vector."""
    part = _raw_part("explicit-choice", priority=1.0, component="Gateway",
                     category="behavior-selection", statement="pick explicitly")
    selection = _selection(part, [_candidate("strategy", 90.0, category="behavior-selection")],
                           catalog, "strategy")
    composed = compose_selection(
        selections={selection.part.id: selection},
        brief=DesignBrief(restated_goal="g"),
        catalog=catalog,
        config=ServerConfig(),
    )
    assert "  weights vector: none" in composed.reasoning
    assert "statement: pick explicitly" in composed.reasoning
