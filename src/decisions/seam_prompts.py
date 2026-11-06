"""The decision-model prompt vocabulary, in one stdlib-only module.

Both sides of the training/serving boundary read these strings:

* the server, through :mod:`src.decisions.tacit` (the Ch2 style, J1 rerank, J2
  category, J4 weights and J5 gate seams — every seam is served by the AnyJev
  Tacit-4B gateway);
* the Tacit image build, which copies this one file into the trainer so the student
  is trained on exactly the prompts it will serve.

Nothing here imports third-party packages: the trainer image has torch, not pydantic.
"""

from __future__ import annotations

from collections.abc import Sequence

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

#: Fixed one-line gloss per architecture style (the Ch2 style ``choice`` question
#: shows these as the criterion texts; server and Tacit trainer share them so the
#: student is trained on exactly the prompt it will serve).
STYLE_GLOSSES: dict[str, str] = {
    "event-driven": "Components communicate primarily through asynchronous events and message flows",
    "broker": "Independent components interact only through a central message broker",
    "enterprise-service-bus": "Heterogeneous services integrate through a central enterprise service bus",
    "event-sourcing": "State is derived by replaying an append-only log of domain events",
    "command-query-responsibility-segregation": "Command and query paths use separate models optimized for writes and reads",
    "saga": "A long-running transaction is coordinated as a sequence of local steps with compensations",
    "microservices": "The system is composed of small independently deployable services",
    "service-oriented-architecture": "Business capabilities are exposed as interoperable services over standard protocols",
    "hexagonal": "Domain logic sits at the core behind ports, with adapters for all external concerns",
    "layered-monolith": "A single deployable is organized in strict horizontal layers",
    "modular-monolith": "A single deployable is organized in well-separated vertical modules",
    "monolithic": "The whole system is built, deployed, and scaled as one unit",
    "pipe-and-filter": "Data flows through a chain of independent processing stages",
    "model-view-controller": "UI concerns split into model, view, and controller responsibilities",
    "presentation-abstraction-control": "Each UI unit pairs presentation, abstraction, and control cooperating agents",
    "microkernel-plugin": "A minimal core is extended by plug-ins for most capabilities",
    "reflection-architecture": "The system inspects and adapts its own structure and behavior at runtime",
    "reactive-architecture": "The design is built around asynchronous message-driven responsiveness and backpressure",
    "actor-based": "Concurrent computation is carried by isolated actors exchanging messages",
    "half-sync-half-async": "Synchronous and asynchronous processing layers are bridged by queues",
    "task-control-architecture": "Hierarchical task controllers coordinate concurrent robotic or real-time behaviors",
    "serverless": "Code runs as provider-managed functions scaled to zero when idle",
    "edge-computing": "Computation is pushed to nodes near the data source or user",
    "space-based": "Scalability comes from distributed in-memory processing units and spaces",
    "data-mesh": "Analytical data is owned as products by decentralized domain teams",
    "lambda-architecture": "Batch and speed layers are combined to serve complete and fresh views",
    "kappa-architecture": "A single streaming pipeline serves both real-time and reprocessed views",
    "api-gateway": "Clients reach backend services through a single gateway entry point",
    "backend-for-frontend": "Each frontend gets a dedicated backend tailored to its needs",
    "service-mesh": "Service-to-service traffic is managed by a dedicated infrastructure layer",
    "rule-based-system": "Behavior is driven by a set of declarative if-then rules",
    "aiml-centric": "AI/ML models are the primary value-bearing components of the system",
    "blockchain-based": "Trust and state rely on a distributed ledger with consensus",
    "hybrid-cloud": "Workloads span on-premises infrastructure and public cloud",
    "multi-cloud": "Workloads span more than one public cloud provider",
}

#: ``none`` criterion key: no listed style applies. Last in the served criteria map.
STYLE_NONE_KEY = "none"

#: The ``none`` criterion text, shared so train and serve never drift the abstention candidate.
STYLE_NONE_GLOSS = "No listed architecture style applies"

#: The Ch2 style ``choice`` instructions, verbatim on both sides of the boundary.
STYLE_INSTRUCTIONS = (
    "Which architecture style best characterizes the system described "
    "in the state? Pick 'none' when no listed style applies."
)

#: The Ch2 style answer space: every architecture style the seam may name, in the order the
#: implementation guide lists them (§10.1). Deliberately a flat vocabulary: ``src/style_affinity.py``
#: owns the style→category PRIOR, and this module (copied standalone into the trainer image, so it
#: cannot import the server package) must not restate that prior — reading it backwards as a
#: filter is exactly what made 16 of 22 labelled holdout rows unanswerable. A unit test pins this
#: set to that table's keys so a style can neither appear here nor vanish unnoticed.
STYLE_VOCABULARY: tuple[str, ...] = (
    "event-driven",
    "broker",
    "enterprise-service-bus",
    "event-sourcing",
    "command-query-responsibility-segregation",
    "saga",
    "microservices",
    "service-oriented-architecture",
    "hexagonal",
    "layered-monolith",
    "modular-monolith",
    "monolithic",
    "pipe-and-filter",
    "model-view-controller",
    "presentation-abstraction-control",
    "microkernel-plugin",
    "reflection-architecture",
    "reactive-architecture",
    "actor-based",
    "half-sync-half-async",
    "task-control-architecture",
    "serverless",
    "edge-computing",
    "space-based",
    "data-mesh",
    "lambda-architecture",
    "kappa-architecture",
    "api-gateway",
    "backend-for-frontend",
    "service-mesh",
    "rule-based-system",
    "aiml-centric",
    "blockchain-based",
    "hybrid-cloud",
    "multi-cloud",
)

#: AnyJev's letter readout caps ``choice``/``score`` at 26 options (``anyjev``
#: ``question.py`` ``MAX_OPTIONS``; the gateway answers HTTP 400 beyond). A shard keeps a
#: slot free for ``none``, so one question carries at most this many styles.
_STYLE_OPTION_CAP = 26
_STYLE_SHARD_CAP = _STYLE_OPTION_CAP - 1


def style_shards() -> tuple[tuple[str, ...], ...]:
    """The Ch2 style shortlists: the whole vocabulary split into BALANCED shards.

    The question is 35-way and AnyJev takes at most 25 styles plus ``none`` per ask, so
    serving asks one question per shard and keeps the most confident named style across them
    (``src/decisions/tacit.py::choose_style``). Every shard is a legal question on its own and
    their union IS :data:`STYLE_VOCABULARY`, so no style is ever unanswerable.

    Shards are EQUAL WIDTH by construction (18 + 17 today, not 25 + 10): a softmax over 10
    options concentrates more mass than one over 25, so an uneven split would let the narrow
    shard win cross-shard comparisons for free and bias the very floor the calibration fits.

    Deriving the shards from category affinity was a defect, not a shortcut: the §10.1 table
    lists the styles that *favour* a pattern category, not every style a system in that
    category may have. It crams ``security-verification`` into three cloud styles, so a
    security scenario answered ``layered-monolith`` could not be expressed at all — and
    16 of 22 labelled holdout rows became unwinnable before the model was consulted.
    """
    total = len(STYLE_VOCABULARY)
    count = -(-total // _STYLE_SHARD_CAP)  # ceil: shards needed at the widest legal width
    width = -(-total // count)  # ceil: equalising them keeps the softmaxes comparable
    shards = tuple(STYLE_VOCABULARY[start : start + width] for start in range(0, total, width))
    assert all(len(shard) + 1 <= _STYLE_OPTION_CAP for shard in shards), shards  # noqa: S101
    return shards


def style_question(styles: Sequence[str]) -> str:
    """The Ch2 stage-2 style ``choice`` question: instructions plus inline glosses."""
    glosses = "; ".join(f"{style} = {STYLE_GLOSSES[style]}" for style in styles)
    return f"{STYLE_INSTRUCTIONS} Options: {glosses}; {STYLE_NONE_KEY} = {STYLE_NONE_GLOSS}"


# J1 rerank ---------------------------------------------------------------------

#: Fixed per-candidate relevance question; the candidate document rides in the state.
RERANK_RELEVANCE_QUESTION = (
    "Does the pattern described in the state address the decision part in the state? Answer for the pattern as a whole."
)

#: Cap on one rerank item's state (part statement + document text). The sidecar serves
#: ``--max-model-len 8192``; 6000 chars ≈ 1.5-2k tokens leaves room for the question,
#: option labels and chat template.
RERANK_STATE_MAX_CHARS = 6000


def rerank_state(statement: str, document: str) -> str:
    """One rerank item's state: the part statement, then the pattern document."""
    return f"{statement}\n\nPattern: {document}"[:RERANK_STATE_MAX_CHARS]


#: The pipeline phases that carry a reasoning strategy and can hit the J5 gate
#: (src/reasoning/config.py::_default_per_phase; payloads at src/pipeline.py:1680-1808).
GATE_PHASES: tuple[str, ...] = ("analyze", "weights", "generate", "evaluate")

#: Scalar payload keys each phase puts into gate_state, in the order the pipeline
#: builds them (src/pipeline.py:1680-1808). ``tacit.gate_state`` emits all of a
#: phase's keys, so a question missing one is a state the server never sends; the
#: trainer rejects those rather than training on them.
GATE_PHASE_PAYLOAD_KEYS: dict[str, tuple[str, ...]] = {
    "analyze": ("description",),
    "weights": ("description", "restated_goal", "decision_parts"),
    "generate": ("restated_goal", "applications"),
    "evaluate": ("primary_pattern", "applications", "parts"),
}

#: Cap on the gate-state text (the gate only needs the gist). Shared so the trainer
#: truncates exactly where ``src.decisions.tacit.gate_state`` does.
GATE_STATE_MAX_CHARS = 1500

#: Cap on the decision-part state text (the seams only need the gist). Shared so the
#: trainer truncates exactly where ``src.pipeline._decision_part_state`` does.
DECISION_STATE_MAX_CHARS = 1500


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
