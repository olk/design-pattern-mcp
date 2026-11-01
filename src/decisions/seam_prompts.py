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

#: Fixed one-line gloss per architecture style (the Ch2 style ``choice`` question
#: shows these as the criterion texts; server and CLM exporter share them so the
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
