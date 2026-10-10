"""Deterministic test doubles for the injected seams (LLM, retrieval index, hits).

These are *test* doubles only: production code never imports them. ``ScriptedLLM`` records every
call so a test can assert what the pipeline asked for, and replies with per-call scripts, which is
what makes the deterministic core testable without a model.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.errors import ERROR_LLM_PROVIDER, DesignPatternError
from src.agent import DesignHints
from src.patterns.retriever import PartQuery
from src.schemas import (
    DesignBrief,
    DecisionPart,
    DesignResult,
    EvaluationFinding,
    DesignEvaluation,
    GenerationSeed,
    RawBriefAndParts,
    RawGeneration,
    RequirementWeights,
)


@dataclass(frozen=True)
class StubHit:
    """A retrieval hit with exactly the fields the pipeline reads from ``RetrievalHit``."""

    name: str
    retrieval_score: float
    fused_rank: int = 1
    rerank_rank: int = 1


@dataclass
class StubIndex:
    """Retrieval index double: returns a scripted hit list (optionally per part statement)."""

    hits: list[StubHit] = field(default_factory=list)
    by_statement: dict[str, list[StubHit]] = field(default_factory=dict)
    raise_error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def retrieve(
        self,
        *,
        statement: str,
        constraints: Sequence[str],
        scope: str,
        exclude: Collection[str] = (),
    ) -> list[StubHit]:
        """Return the scripted hits for this statement, honouring the sticky exclusions."""
        return (
            await self.retrieve_many(
                queries=[PartQuery(statement, constraints, scope, exclude, None)]
            )
        )[0]

    async def retrieve_many(
        self,
        queries: Sequence[PartQuery],
    ) -> list[list[StubHit]]:
        """Return scripted hits per query, honouring the sticky exclusions."""
        for query in queries:
            self.calls.append(
                {
                    "statement": query.statement,
                    "constraints": list(query.constraints),
                    "scope": query.scope,
                }
            )
        if self.raise_error is not None:
            raise self.raise_error
        return [
            [hit for hit in self.by_statement.get(query.statement, self.hits) if hit.name not in set(query.exclude)]
            for query in queries
        ]


@dataclass
class ScriptedLLM:
    """Scripted ``LLMClientProtocol`` implementation recording every call it receives."""

    brief_and_parts_result: RawBriefAndParts | None = None
    weights: dict[str, float] = field(default_factory=dict)
    #: Per-batch scripted vectors (one dict per part id of that batch), E4 retry order.
    weights_batch_sequence: list[dict[str, dict[str, float]]] = field(default_factory=list)

    generation: RawGeneration = field(default_factory=RawGeneration)
    generation_sequence: list[RawGeneration] = field(default_factory=list)
    evaluation: DesignEvaluation = field(default_factory=lambda: DesignEvaluation(score=100.0))
    evaluation_sequence: list[DesignEvaluation] = field(default_factory=list)
    fail_brief: bool = False
    fail_weights: bool = False
    fail_generation: bool = False
    #: When set, GENERATE answers the seed the pipeline actually passed (the usual case).
    generate_from_seed: bool = True
    calls: list[str] = field(default_factory=list)
    #: 1-based index of the GENERATE call that must raise a transport failure (None = all succeed).
    fail_on_call: int | None = None

    async def extract_brief_and_parts(
        self, description: str, hints: DesignHints, *, reasoning_context: str = ""
    ) -> RawBriefAndParts:
        """Return the scripted brief+parts (or raise the documented LLM failure)."""
        self.calls.append("extract_brief_and_parts")
        if self.fail_brief:
            raise DesignPatternError(ERROR_LLM_PROVIDER, "scripted brief failure")
        if self.brief_and_parts_result is None:
            raise DesignPatternError(ERROR_LLM_PROVIDER, "no scripted brief")
        return self.brief_and_parts_result

    async def extract_weights_batch(
        self, parts: Sequence[DecisionPart], brief: DesignBrief, *, reasoning_context: str = ""
    ) -> dict[str, RequirementWeights]:
        """Answer ONE call for the whole part list (all-zero when nothing is scripted)."""
        self.calls.append(f"extract_weights_batch:{','.join(part.id for part in parts)}")
        if self.fail_weights:
            raise DesignPatternError(ERROR_LLM_PROVIDER, "scripted weights failure")
        if self.weights_batch_sequence:
            scripted = self.weights_batch_sequence.pop(0)
            return {part.id: RequirementWeights(**scripted[part.id]) for part in parts}
        return {part.id: RequirementWeights(**self.weights) for part in parts}

    async def generate(self, seed: GenerationSeed, *, reasoning_context: str = "") -> RawGeneration:
        """Return the next scripted generation payload (or answer the seed that was passed)."""
        self.calls.append("generate")
        if self.fail_generation:
            raise DesignPatternError(ERROR_LLM_PROVIDER, "scripted generation failure")
        if self.generation_sequence:
            return self.generation_sequence.pop(0)
        if self.generate_from_seed:
            return generation_for_seeds(seed)
        return self.generation

    async def evaluate(
        self, result: DesignResult, parts: Sequence[DecisionPart], *, reasoning_context: str = ""
    ) -> DesignEvaluation:
        """Return the next scripted evaluation."""
        self.calls.append("evaluate")
        if self.evaluation_sequence:
            return self.evaluation_sequence.pop(0)
        return self.evaluation


def generation_for_seeds(seed: GenerationSeed) -> RawGeneration:
    """A valid GENERATE payload keyed by the real seed ids."""
    from src.schemas import RawApplication, RawParticipant

    return RawGeneration(
        applications=[
            RawApplication(
                id=application.id,
                pattern=application.pattern,
                name=f"{application.pattern} for {application.part_id}",
                problem=application.part_statement,
                solution=f"realize {application.pattern} in the described component",
                participants=[
                    RawParticipant(
                        role="client",
                        maps_to=(application.component_refs[0] if application.component_refs else "Component"),
                        responsibility="drives the collaboration",
                    )
                ],
                interactions=[],
                implementation_steps=["wire the participants"],
                test_notes="assert the decision at the seam",
            )
            for application in seed.applications
        ]
    )


def evaluation_with(pattern: str, *, score: float = 30.0) -> DesignEvaluation:
    """An evaluation naming a selection defect for ``pattern`` (drives the reselection rule)."""
    return DesignEvaluation(
        score=score,
        findings=[
            EvaluationFinding(
                kind="selection",
                detail=f"{pattern} does not own the decision",
                pattern=pattern,
            )
        ],
    )
