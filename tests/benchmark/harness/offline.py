"""Offline wiring: scripted Stage-C hits + scripted LLM over the real catalog (plan §9).

Offline mode is the *plumbing and metric-integrity* runner: it is deterministic, needs no
network and finishes in well under a minute. It is **not** a quality instrument — its retrieval
hits are read from the scenario's own label, so its hit rate is a fixture property, not a
measurement of the pipeline (``README.md`` "Offline mode"). What it does prove, end to end and
through the real pipeline:

- every stage of ``DesignPipeline.run_design`` executes against a real 486-record catalog
  (Stage A/B repair, weights, vetoes, scoring, §8 selection, GENERATE, EVALUATE, the G–H loop);
- every metric the harness reports is computed from real ``AnalysisResult``/``PipelineResult``
  objects rather than synthetic fixtures (the formulas themselves are pinned by ``selfcheck``);
- the two-arm compare workflow runs on two complete run directories.

The script mirrors the deployment contract: the scripted ``LLMClientProtocol`` answers the
pipeline's four call kinds, and the scripted index answers ``retrieve_many`` with the label pool
(acceptable primaries first, then acceptable supporting patterns) so the real §6 vetoes and
§7/§8 scoring decide the outcome. ``--flip`` replaces that pool with decoys outside the label —
the deterministic miss the compare dry run needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import ServerConfig
from src.design_normalization import truncate
from src.patterns.loader import PatternLoader
from src.pipeline import DesignPipeline
from src.schemas import (
    RawBriefAndParts,
    RawBriefComponent,
    RawDecisionPart,
    RawDesignBrief,
)
from src.schemas.enums import QUALITY_AXES
from tests.unit.fakes import ScriptedLLM, StubHit, StubIndex

from .config import Scenario
from .probes import LLMTimingProxy, RetrievalIndexTimingProxy, StageRecorder

#: Decoys scripted by ``--flip``: enough to form a pool, all outside the scenario's label.
DECOY_COUNT = 3

#: Descending retrieval blends so the pool order is the label order (ties are impossible).
_RETRIEVAL_STEP = 0.02


@dataclass(frozen=True)
class OfflineScript:
    """One scenario's scripted seam state plus the provenance of the scripted pool."""

    llm: ScriptedLLM
    index: StubIndex
    pool: tuple[str, ...]
    scope: str | None
    flip: bool
    pool_source: str

    def describe(self) -> dict[str, Any]:
        """Compact provenance block for the run record."""
        return {
            "pool_source": self.pool_source,
            "pool": list(self.pool),
            "scope": self.scope,
            "flip": self.flip,
        }


def build_script(scenario: Scenario, catalog: PatternLoader, *, flip: bool) -> OfflineScript:
    """Build the scripted seams for ``scenario`` (label pool, or decoys when flipped).

    The pool keeps the label order and skips any acceptable pattern the request language marks
    ``discouraged`` (the §6 step-5 language veto would remove it anyway, so scripting it would
    only add noise). The part scope is chosen to be the one no pool member lists as unsuitable.
    """
    pool, pool_source = _select_pool(scenario, catalog, flip=flip)
    scope = _choose_scope(pool, catalog)
    hits = [
        StubHit(name=name, retrieval_score=round(1.0 - _RETRIEVAL_STEP * index, 6), fused_rank=index + 1, rerank_rank=index + 1)
        for index, name in enumerate(pool)
    ]
    component = f"{scenario.family} component"
    raw = RawBriefAndParts(
        design_brief=RawDesignBrief(
            restated_goal=truncate(scenario.description, 300),
            constraints=[],
            components=[
                RawBriefComponent(
                    name=component,
                    type="service",
                    responsibilities=[scenario.decision_statement],
                )
            ],
            flows=[],
            negative_signals=[],
        ),
        decision_parts=[
            RawDecisionPart(
                id=f"{scenario.scenario_id}-decision",
                statement=scenario.decision_statement,
                component_refs=[component],
                scope=scope,
                priority=1.0,
            )
        ],
    )
    return OfflineScript(
        llm=ScriptedLLM(
            brief_and_parts_result=raw,
            weights=dict.fromkeys(QUALITY_AXES, 1.0),
        ),
        index=StubIndex(hits=hits),
        pool=tuple(pool),
        scope=scope,
        flip=flip,
        pool_source=pool_source,
    )


def build_pipeline(
    script: OfflineScript,
    *,
    config: ServerConfig,
    catalog: PatternLoader,
    recorder: StageRecorder,
) -> DesignPipeline:
    """A real :class:`DesignPipeline` over the scripted seams (generation/evaluation are scripted).

    ``ScriptedLLM`` answers GENERATE from the seed the pipeline passes and EVALUATE with the
    default 100.0 score, so the G–H loop converges on its first attempt exactly like a clean
    production run.
    """
    return DesignPipeline(
        config=config,
        catalog=catalog,
        index=RetrievalIndexTimingProxy(script.index, recorder),  # type: ignore[arg-type]
        llm=LLMTimingProxy(script.llm, recorder),  # type: ignore[arg-type]
        reasoning=None,
    )


def _select_pool(
    scenario: Scenario, catalog: PatternLoader, *, flip: bool
) -> tuple[list[str], str]:
    """The scripted hit pool: the scenario's labels, or decoys when ``flip``."""
    language = _effective_language(scenario, catalog)
    label_pool = [
        name for name in dict.fromkeys(scenario.acceptable_primary + scenario.acceptable_supporting)
        if not _vetoed_in_language(catalog, name, language)
    ]
    if not flip:
        if scenario.primary not in label_pool:
            raise ValueError(
                f"scenario {scenario.scenario_id!r}: the expected primary {scenario.primary!r} is "
                f"discouraged in language {language!r}; pick another language for this scenario"
            )
        return label_pool, "label"
    return _decoy_pool(scenario, catalog), "flip-decoys"


def _decoy_pool(scenario: Scenario, catalog: PatternLoader) -> list[str]:
    """Deterministic decoys outside the label (preferring another category): a guaranteed miss."""
    acceptable = set(scenario.acceptable)
    seed = int.from_bytes(scenario.scenario_id.encode("utf-8"), "big") % 997
    others = [name for name in catalog.order if name not in acceptable]
    foreign = [name for name in others if catalog.records[name].category != scenario.category]
    rotated = foreign[seed % len(foreign):] + foreign[: seed % len(foreign)] if foreign else []
    ordered = rotated + others
    pool: list[str] = []
    for name in ordered:
        if name not in pool:
            pool.append(name)
        if len(pool) == DECOY_COUNT:
            break
    if not pool:
        raise ValueError(f"scenario {scenario.scenario_id!r}: catalog holds no decoy pattern")
    return pool


def _effective_language(scenario: Scenario, catalog: PatternLoader) -> str:
    """The effective language the pipeline will resolve for ``scenario`` (generic when unset)."""
    if scenario.language is None or scenario.language not in catalog.enums.languages:
        return "generic"
    return scenario.language


def _vetoed_in_language(catalog: PatternLoader, name: str, language: str) -> bool:
    """True when the §6 step-5 language veto would drop ``name`` (entry support ``discouraged``)."""
    entry = catalog.records[name].entry(language)
    return entry is not None and entry.support == "discouraged"


def _choose_scope(pool: list[str], catalog: PatternLoader) -> str | None:
    """The scope no pool member lists as unsuitable (keeps the label pool §6-eligible)."""
    if not pool:
        return None
    scored: list[tuple[int, str]] = []
    for scope in catalog.enums.scopes:
        conflicts = sum(1 for name in pool if scope in catalog.records[name].unsuitable_scopes)
        scored.append((conflicts, scope))
    scored.sort()
    return scored[0][1]
