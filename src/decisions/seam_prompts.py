"""The decision-model prompt vocabulary, in one stdlib-only module.

Both sides of the training/serving boundary read these strings:

* the server, through :mod:`src.decisions.tacit` (the J4/J2/J5 seams);
* the Tacit image build, which copies this one file into the trainer so the student
  is trained on exactly the prompts it will serve.

Nothing here imports third-party packages: the trainer image has torch, not pydantic.
"""

from __future__ import annotations

#: Ordered weight levels for one quality axis (``score`` question, lowest first).
#: The expected value is ``Σ i·p_i / (len-1)``, i.e. a five-level axis maps to [0, 1].
WEIGHT_LEVELS: tuple[str, ...] = ("none", "low", "moderate", "high", "critical")

#: The ``PatternCategory`` taxonomy, in enum order (``src.schemas.enums``).
CATEGORY_SLUGS: tuple[str, ...] = (
    "domain-modeling",
    "construction-composition",
    "behavior-selection",
    "interaction-boundary",
    "workflow-messaging",
    "persistence-state",
    "concurrency-resilience",
    "security-verification",
    "specialized-components",
)

#: Fixed one-line gloss per category (the category ``choice`` question shows the slug
#: list itself; the glosses are what makes a slug decidable at all).
CATEGORY_GLOSSES: dict[str, str] = {
    "domain-modeling": (
        "Modeling domain concepts, responsibilities and invariants (entities, aggregates, value objects)"
    ),
    "construction-composition": (
        "How objects and modules are constructed and wired together (factories, injection, composition)"
    ),
    "behavior-selection": "Selecting behavior at runtime (strategy, state, dispatch)",
    "interaction-boundary": "Interfaces and boundaries between components (facades, adapters, DTOs)",
    "workflow-messaging": "Event and message flow, coordination, pipelines (observer, mediator, pipes)",
    "persistence-state": "Storage, serialization and state lifecycle (repository, unit of work, memento)",
    "concurrency-resilience": "Concurrency control and failure containment (locks, circuit breaker, timeout, retry)",
    "security-verification": "Protection and validation of inputs and access (guard clauses, validation)",
    "specialized-components": "Domain-specific components without a general home",
}


def category_question() -> str:
    """The J2 category ``choice`` question, every gloss inline in enum order."""
    glosses = "; ".join(f"{slug} = {CATEGORY_GLOSSES[slug]}" for slug in CATEGORY_SLUGS)
    return (
        "Which design-pattern category best describes the concern of this decision part? "
        f"Options are category slugs: {glosses}"
    )


def axis_question(axis: str) -> str:
    """The J4 weight question for one quality axis."""
    return f"How important is {axis} quality for this decision part?"


def gate_question() -> str:
    """The J5 gate ``yes_no`` question."""
    return "Would deliberation improve this step, or is the answer already determined?"
