"""The decision-model prompt vocabulary for the Ch2 style seam, in one stdlib-only module.

Both sides of the training/serving boundary read these strings:

* the server, through :mod:`src.decisions.client` (the Ch2 style seam's served criteria map);
* the linear style head's image build, which copies this one file into the trainer so the class
  order it learns is exactly the vocabulary the seam serves; the label derivation
  (`derive_style_labels.py`) reads the same style vocabulary.

Nothing here imports third-party packages: the trainer loads this file by path with stdlib only.
"""

from __future__ import annotations

#: Fixed one-line gloss per architecture style (the Ch2 style ``choice`` question
#: shows these as the criterion texts; server and linear head trainer share them so the
#: head is fitted on exactly the criteria it will serve).
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
    "monolithic": (
        "The whole system is built, deployed, and scaled as one unit; "
        "also the answer for one deployable with no other style signal"
    ),
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
#: Carries the labeling rubric's default rule (``derive_style_labels.py``): a bare
#: library/algorithm/code-shape decision with no system topology is ``none``, exactly as the
#: gold was drawn — the served question must ask what the label answers.
STYLE_NONE_GLOSS = (
    "No listed architecture style applies, or the state describes only a library, "
    "algorithm, or code-shape decision with no system topology"
)

#: The Ch2 style ``choice`` instructions, verbatim on both sides of the boundary.
STYLE_INSTRUCTIONS = (
    "Which architecture style best characterizes the system described "
    "in the state? Pick 'none' when no listed style applies."
)

#: The Ch2 style answer space: every architecture style the seam may name, in the order the
#: implementation guide lists them (§10.1). Deliberately a flat vocabulary: ``src/style_affinity.py``
#: owns the style→category PRIOR, which the linear head learns from the label derivation
#: instead of restating here — reading the affinity table backwards as a shortlist filter
#: is exactly what made 16 of 22 labelled holdout rows unanswerable. A unit test pins this
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
