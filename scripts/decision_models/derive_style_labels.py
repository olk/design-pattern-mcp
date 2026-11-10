#!/usr/bin/env python
"""Derive style gold labels from the benchmark scenarios (review input, not labels).

Reads ``tests/benchmark/scenarios/full.json`` and writes ``qa/typed/style.jsonl``:
one row per scenario, in the same ``{"id", "state", "type", "instructions",
"criteria", "label", "source"}`` shape as ``qa/typed/choice.jsonl``. The state is
the raw scenario description bytes (that is what the Ch2 seam receives), and the
criteria/instructions are the seam's own prompt vocabulary from
``src/decisions/seam_prompts.py``. Labels come from ``STYLE_LABELS`` below, which
is the reviewable artifact: every key is a reviewed ``scenario_id``, every value
a criterion key plus a rationale quoting the scenario's own evidence.

Split integrity mirrors the corpus: ``source.split`` is the scenario's own split,
so the train-only exporter rule keeps holdout families out of the weights by
construction.

Labeling rubric (the only judgment in this file): the dominant architectural
style of the system *under decision* — the one the brief's architecture_style
should resolve to. A scenario whose system is one deployable with no named style
signals labels ``monolithic``; a bare library/algorithm/code-shape decision with
no system topology visible labels ``none``. Rationales quote the evidence that
selects the label over its nearest alternative.

Rubric calibration: applied blind to the four seed scenarios carrying human
``hints.architecture_style`` (tests/benchmark/scenarios/seed.json — disjoint from
the corpus, so not part of STYLE_LABELS), the rubric reproduces all four:
wm-order-router → event-driven (brokered event stream, per-kind handlers),
cr-billing-circuit-breaker → microservices (orchestrator + separately failing
card processor), cc-module-wiring → hexagonal (domain services + adapters +
infrastructure, entry point wiring), sc-dashboard-mvvm → model-view-controller
(view state derived from the model, template binds without logic). 4/4 match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.decisions import seam_prompts  # noqa: E402

#: scenario_id -> (label, rationale). Every train-split scenario must appear here;
#: holdout scenarios appear too (labeled, but never exported to training weights).
#: Corpus labels are agent-curated, human expert review pending (same status as the
#: choice.jsonl labels — see qa/typed/README.md). Status is machine-readable:
#: ``qa/typed/review.json`` records per-scenario review (reviewer, labels_sha256,
#: rationale_ok) and ``tacit_calibrate.py`` refuses to recommend a confidence floor
#: until the ledger covers this file byte-for-byte — the floor is the seam's
#: enablement, so unreviewed labels cannot be calibrated into production.
STYLE_LABELS: dict[str, tuple[str, str]] = {
    # -- behavior-selection ---------------------------------------------------
    "algorithms-traversal-variation-pricing-rules": (
        "pipe-and-filter",
        "A Checkout Pipeline whose Pricing Engine is one stage selecting pricing rules",
    ),
    "algorithms-traversal-variation-route-traversal": (
        "none",
        "A bare route-storage/traversal decision inside one service; no system topology is described",
    ),
    "dispatch-pair-interactions": (
        "none",
        "A game-simulation collision dispatch decision; no deployable system shape is described",
    ),
    "dispatch-plugin-extensions": (
        "microkernel-plugin",
        "A minimal document-analysis core extended by independently installed dialect and operation plugins",
    ),
    "extension-compile-time-matrix": (
        "none",
        "A single C++ storage-class policy decision; no system topology is described",
    ),
    "extension-module-host": (
        "microkernel-plugin",
        "A fleet server acquiring tracing backends from separately packaged modules discovered at start-up",
    ),
    "functional-composition-order-pipeline": (
        "pipe-and-filter",
        "An order workflow of named stages (profile -> region -> quote -> reservation) composed in sequence",
    ),
    "functional-composition-request-validation": (
        "none",
        "A field-validation combinator decision inside one request validator; no system shape is described",
    ),
    # -- concurrency-resilience ------------------------------------------------
    "concurrency-resilience-order-mailbox": (
        "half-sync-half-async",
        "HTTP/scheduler threads submit commands that a worker consumes through a shared mailbox queue",
    ),
    "concurrency-resilience-registry-capacity": (
        "monolithic",
        "One inventory platform process with a shared in-memory registry; no distribution or broker is named",
    ),
    "coordination-primitives-bounded-payment-handoff": (
        "broker",
        "Merchant webhooks and settlement workers interact only through a bounded hand-off channel",
    ),
    "coordination-primitives-lazy-schema-cache": (
        "monolithic",
        "One query service with a shared in-memory schema registry; no distributed shape is named",
    ),
    "concurrency-resilience-fanout-deadline": (
        "microservices",
        "A page request fanning out concurrently to independently deployed Catalog/Pricing/Reviews/Recommendation services",
    ),
    "concurrency-resilience-multi-dependency-pools": (
        "microservices",
        "A fulfillment service calling independent Inventory/Shipping/Address dependencies that fail separately",
    ),
    "concurrency-resilience-hot-config": (
        "monolithic",
        "One edge gateway process with 48 reader threads on a shared routing snapshot; the gateway is a single deployable",
    ),
    "concurrency-resilience-payload-handles": (
        "pipe-and-filter",
        "A telemetry pipeline of framing -> checksum -> compression stages passing a buffer by value",
    ),
    "queue-drain-shutdown": (
        "half-sync-half-async",
        "Producers enqueue media batches and three blocked workers drain them; shutdown is coordinated through the queue",
    ),
    "request-cancel-propagation": (
        "microservices",
        "A dashboard handler calling independent profile/billing/recommendation services whose failures cancel separately",
    ),
    # -- construction-composition ----------------------------------------------
    "composition-encapsulation-checkout-entry": (
        "layered-monolith",
        "HTTP and CLI entry points over one shared application layer (validator, pricing, inventory, gateway, ledger)",
    ),
    "composition-encapsulation-provider-boundary": (
        "hexagonal",
        "Billing business code behind a PaymentClient port, with the vendor SDK translated at an adapter",
    ),
    "creation-config-growing-client-options": (
        "none",
        "A library constructor-shape decision for a telemetry client; no system topology is described",
    ),
    "creation-config-matched-database-products": (
        "none",
        "A provider-set creation decision inside one reporting deployment; no multi-component shape is named",
    ),
    "dependency-wiring-api-handler-collaborators": (
        "layered-monolith",
        "One checkout API whose controller receives repository/gateway/clock/mailer selected once at startup",
    ),
    "dependency-wiring-job-runner-lifecycle": (
        "microkernel-plugin",
        "A reusable job engine whose lifecycle hosts customer-supplied job operations as plug-ins",
    ),
    "functional-composition-inbound-normalization": (
        "pipe-and-filter",
        "A shared sequence of pure normalization changes applied in declared order across adapters",
    ),
    "functional-composition-request-stages": (
        "pipe-and-filter",
        "A checkout request decomposed into named replaceable stages passing output to the next",
    ),
    "method-structure-buffer-assignment": (
        "none",
        "A C++ assignment-operator guarantee for one document index class; no system is described",
    ),
    "method-structure-quote-decomposition": (
        "none",
        "A method-decomposition decision inside one shipping quote service; no system shape is described",
    ),
    # -- domain-modeling ---------------------------------------------------------
    "domain-logic-shipment-import": (
        "monolithic",
        "One back-office import flow (controller -> staging -> live table) with no distribution named",
    ),
    "domain-logic-supplier-update": (
        "monolithic",
        "One web endpoint with ~10 straight-line controller steps in a one-screen application; no indirection, so no horizontal layers",
    ),
    "functional-data-modeling-expression-evaluation": (
        "none",
        "A compiler node-representation decision; no system topology is described",
    ),
    "functional-data-modeling-order-lifecycle": (
        "monolithic",
        "One order service whose lifecycle states are the whole system under decision; no distributed shape is named",
    ),
    "model-clarity-ledger-contract-checks": (
        "monolithic",
        "One treasury platform with account/posting components; invariants are in-process, no services are named",
    ),
    "model-clarity-reproducible-pricing": (
        "monolithic",
        "One grocer pricing service deriving totals from explicit inputs; no distribution is named",
    ),
    "tactical-customer-identity": (
        "monolithic",
        "One billing platform with a customer directory; identity is in-process, no services are named",
    ),
    "tactical-inventory-consistency": (
        "monolithic",
        "One warehouse system with reservation commands over items; the referenced shipments are data, not services",
    ),
    "time-booking-intervals": (
        "monolithic",
        "One room-booking service with a reservation repository; no distributed shape is named",
    ),
    "time-measurement-normalization": (
        "monolithic",
        "One fleet telemetry service with ingestion/aggregation/alerting components; no services are named",
    ),
    "values-correctness-distinct-identifiers": (
        "layered-monolith",
        "Domain APIs over an infrastructure boundary (repositories, cache, audit) inside one billing/shipment system",
    ),
    "values-correctness-upload-lifecycle": (
        "monolithic",
        "One document pipeline with a staged upload session object; no distributed shape is named",
    ),
    "enforcement-api-policy": (
        "layered-monolith",
        "One travel API whose controllers sit behind authentication and authorization filters plus a policy service",
    ),
    "enforcement-owned-secret-vault": (
        "layered-monolith",
        "Billing/reporting/jobs behind one owning KeyStore component; a layered boundary, not separate services",
    ),
    "identity-multi-channel-normalization": (
        "api-gateway",
        "Browser/mobile/service requests normalized to one acting-user representation at the gateway trust boundary",
    ),
    "identity-pluggable-credential-verification": (
        "hexagonal",
        "A deployment-selected verifier registry behind a boundary the controller and domain must not change — ports and adapters",
    ),
    "regression-boundary-independent-mobile-api": (
        "microservices",
        "A mobile app and ClaimsProvider service built and released by separate teams over a versioned API",
    ),
    "regression-boundary-legacy-reporting": (
        "monolithic",
        "One nightly reporting job (engine split into parser/calculation/render) producing PDFs; no services named",
    ),
    "security-pipeline-api-gateway": (
        "api-gateway",
        "A gateway security component verifying credentials for every protected route before controller dispatch",
    ),
    "security-pipeline-event-ingress": (
        "event-driven",
        "Regional shipping events validated at one ingress boundary before reaching parcel domain handlers",
    ),
    "security-verification-payment-release": (
        "monolithic",
        (
            "One HarborPay system (console + workflow + adapter + ledger) with no deployment or messaging signals: "
            "the create -> approve -> release handoffs are human authorization gates that leave the payment pending, "
            "not compensating local transactions, so 'saga' over-reads the domain; per the rubric this is one deployable, "
            "not a bare library decision ('none')."
        ),
    ),
    "security-verification-unmatched-api-access": (
        "api-gateway",
        "A gateway fronting orders/invoices/profiles APIs that rejects any request without an explicit policy grant",
    ),
    "test-doubles-device-protocol": (
        "none",
        "A test-double command-sequence decision for one credential client; no system topology is described",
    ),
    "test-doubles-policy-outcomes": (
        "none",
        "A test-responder decision for one policy service; the classifier is a remote dependency, not a style signal",
    ),
    "security-verification-admin-ui": (
        "none",
        "Selenium page-object wrappers around console screens: test-code shape, no M/V/C roles, no topology",
    ),
    "security-verification-focused-user-setup": (
        "none",
        "A test-setup API decision for one lockout policy; no system topology is described",
    ),
    "testability-seams-certificate-clock": (
        "none",
        "A legacy-class clock-control decision for tests; no system topology is described",
    ),
    "testability-seams-checkout-screen": (
        "model-view-controller",
        "Payment validation and navigation rules moved out of the framework screen into testable components",
    ),
    "validation-error-contracts-payment-authorization": (
        "pipe-and-filter",
        "A payment request flowing through parse -> policy -> reserve stages with propagated failure categories",
    ),
    "validation-error-contracts-registry-lookup": (
        "monolithic",
        "One feature registry resolving names against an in-memory map plus overlay; no distributed shape is named",
    ),
    "client-composition-permissions": (
        "none",
        "A React enhancer-factory decision for client components; no deployable system shape is described",
    ),
    "client-composition-tabs": (
        "none",
        "A tab-state ownership decision inside one design system; no system topology is described",
    ),
    "language-implementation-expression-operators": (
        "microkernel-plugin",
        "A spreadsheet engine core whose operator set grows through plug-in registrations",
    ),
    "language-implementation-handwritten-grammar": (
        "pipe-and-filter",
        "Policy text flowing through lexer -> parser -> evaluator stages",
    ),
    "crowd-proximity-lookup": (
        "none",
        "A spatial-index decision inside one crowd simulation tick; no system topology is described",
    ),
    "runtime-actor-advance": (
        "none",
        "An actor-update contract inside one game frame driver; no system topology is described",
    ),
    "checkout-state-transitions": (
        "model-view-controller",
        "A checkout screen deriving one immutable snapshot from explicit gesture transitions",
    ),
    "local-search-field-state": (
        "none",
        "Component-local state ownership on a search page; no model, view, or controller roles, no topology",
    ),
    "workflow-messaging-module-event-wiring": (
        "broker",
        "Console modules loaded/unloaded independently, coupled only through a shared notification hub",
    ),
    "workflow-messaging-silent-inventory-flow": (
        "event-driven",
        "A warehouse scheduler publishing hourly snapshots that a reporting consumer imports; silence is the failure",
    ),
    "messaging-delivery-order-commands": (
        "broker",
        "A storefront API and ten pricing workers coupled only through an order command queue with ack semantics",
    ),
    "workflow-messaging-operational-route-changes": (
        "broker",
        "Collectors and regional sinks coupled through a telemetry gateway with reloadable routing rules",
    ),
    "message-contracts-async-quotes": (
        "event-driven",
        "A merchant console and pricing service exchanging correlated async requests/replies through a broker connection",
    ),
    "message-contracts-order-facts": (
        "event-driven",
        "An order service publishing one-way facts to independently deployed billing/search/loyalty/audit consumers",
    ),
    "message-endpoints-durable-order-projection": (
        "event-driven",
        "A projection resuming an order-events topic subscription after downtime; the broker is the coupling",
    ),
    "message-endpoints-inventory-request": (
        "broker",
        "Checkout and inventory coupled through a request-reply broker topic with correlation and reply addresses",
    ),
    "remote-contracts-audit-traversal": (
        "monolithic",
        "One service's paged export endpoint with an opaque continuation; no gateway fronting backend services",
    ),
    "remote-contracts-order-submission": (
        "monolithic",
        "One coarse SubmitOrder operation inside the Merchandise service: a facade on a single deployable, not a gateway",
    ),
    "server-presentation-multichannel-invoice": (
        "model-view-controller",
        "Invoice content produced once and rendered for HTML/PDF/email channels — content/view separation",
    ),
    "server-presentation-page-ownership": (
        "model-view-controller",
        "Each page/form owning input parsing, domain invocation, and response selection — per-page controllers",
    ),
    "translation-checkout-mapping": (
        "hexagonal",
        "Checkout domain types behind one bidirectional mapping owner facing the Shop API DTOs",
    ),
    "translation-erp-procurement": (
        "hexagonal",
        "ProcurementCore behind its own translation boundary facing the AcmeERP SOAP SDK",
    ),
    # -- persistence-state -------------------------------------------------------
    "change-tracking-claim-editor": (
        "monolithic",
        "One claims portal (two instances of the same deployable) with optimistic revisions; no services are named",
    ),
    "change-tracking-customer-session": (
        "monolithic",
        "One web request session sharing in-memory customer/address/subscription instances across services in-process",
    ),
    "data-access-gateways-customer-rows": (
        "layered-monolith",
        "Transaction scripts over a shared row/table-gateway data layer inside one billing workshop",
    ),
    "data-access-gateways-order-lines": (
        "layered-monolith",
        "Console services over one OrderLinesData component owning every SQL operation for the table",
    ),
    "domain-to-store-catalog-record": (
        "layered-monolith",
        "HTTP handlers and presentation over one compact persistence component keeping SQL out of the edge",
    ),
    "domain-to-store-order-bundle": (
        "monolithic",
        "One order service persisting order graphs as a unit; no distributed shape is named",
    ),
    "persistence-state-central-request-enforcement": (
        "layered-monolith",
        "Controllers and repository helpers under one mandatory framework/supertype acquisition boundary",
    ),
    "persistence-state-order-workspace": (
        "monolithic",
        "One orders application with a shared concurrency guard over the order cluster; access channels are clients",
    ),
    "queries-metadata-external-field-declarations": (
        "layered-monolith",
        "Domain classes over one generic MappingEngine driven by external field declarations",
    ),
    "queries-metadata-invoice-collection-access": (
        "hexagonal",
        "BillingService over an InvoiceStore collection port with replaceable Postgres/in-memory adapters",
    ),
    "relational-state-hierarchy": (
        "monolithic",
        "One workforce registry persisting a worker hierarchy; no distributed shape is named",
    ),
    "relational-state-identity": (
        "monolithic",
        "One shipping persistence layer with stable database handles; services are in-process callers",
    ),
    "lifecycle-archive-channel": (
        "monolithic",
        "One attachment service lazily acquiring an archive gateway session; the gateway is a resource, not a service",
    ),
    "lifecycle-catalog-cache": (
        "layered-monolith",
        "ProductService over repository plus Redis cache with PostgreSQL authoritative — one layered read path",
    ),
    # -- security-verification (train families only; holdout families below) -----
    "audit-sensitive-admin-evidence": (
        "event-sourcing",
        "Privileged activity as an attributable verifiable event sequence validated independently of the writer",
    ),
    "audit-sensitive-payment-boundary": (
        "microservices",
        "Checkout, Funds and Ledger are separately deployable services behind synchronous calls; audit events are a side channel",
    ),
    "capabilities-isolation-plugin-control": (
        "microkernel-plugin",
        "An extension runtime whose supervisor revokes an untrusted plugin's access without touching the core",
    ),
    "capabilities-isolation-session-logout": (
        "monolithic",
        "One account server invalidating a session handler's access in-process; the control service is a component",
    ),
    "compile-time-access-factory-construction": (
        "none",
        "A C++ constructor-capability decision for one firmware class; no system topology is described",
    ),
    "compile-time-access-iterator-core": (
        "none",
        "A C++ iterator-adapter access decision for one graph database; no system shape is described",
    ),
    # -- style-coverage extension (23 Ch2 criteria x 3, appended 2026-11) ------
    # actor-based
    "actor-based-dispatch-extensibility-manual-ops": (
        "actor-based",
        "twelve machines running thousands of lightweight isolated processes that own private state plus a mailbox and exchange immutable envelopes by address, with no shared memory and no direct calls",
    ),
    "actor-based-algorithms-parcel-fare-variants": (
        "actor-based",
        "a delivery mesh where every district coordinator and every courier unit holds private queue and routing state and nothing is shared, all coordination being immutable records sent between units",
    ),
    "actor-based-functional-composition-order-admission": (
        "actor-based",
        "six machines of small private-state processes with inboxes and a supervision tree restarting crashed processes from snapshots, placement location transparent and no remote method calls at all",
    ),
    # aiml-centric
    "aiml-centric-temporal-scoring-authority-window": (
        "aiml-centric",
        "a screening service built around a fleet of versioned predictors pulled from a predictor registry onto accelerator-backed pods, with a training orchestrator publishing new weight bundles as the revenue path",
    ),
    "aiml-centric-values-checkpoint-identifier-confusion": (
        "aiml-centric",
        "a personalization platform whose revenue path is serving versioned ranking checkpoints, with a feature ledger for point-in-time frames and a scheduler publishing weight bundles to a versioned registry",
    ),
    "aiml-centric-model-clarity-basket-quote-reproducibility": (
        "aiml-centric",
        "a personalized-offers platform where request revenue flows through learned predictors served by an autoscaled inference fleet behind a gateway, with canary rollout and drift-triggered rollback",
    ),
    # backend-for-frontend
    "backend-for-frontend-concurrent-objects-mobile-batch-queue": (
        "backend-for-frontend",
        "all mobile calls go only to MobileEdge and all web calls only to WebEdge, two services exposing different shapes over the same inventory, pricing and order services",
    ),
    "backend-for-frontend-dependency-failure-mobile-edge-starvation": (
        "backend-for-frontend",
        "ShopSphere Mobile calls only MobileEdge and the web SPA calls only WebEdge, each backend with its own aggregation code over shared catalog, pricing, inventory and recommender services",
    ),
    "backend-for-frontend-coordination-checkout-surge-mobile-edge": (
        "backend-for-frontend",
        "native Android and iOS apps reach only ShopMobileEdge while the browser app reaches only ShopWebEdge, two independently deployable gateways aggregating the same downstream services differently",
    ),
    # blockchain-based
    "blockchain-based-message-provenance-reply-matching": (
        "blockchain-based",
        "nine organizations each run a replica node holding a full hash-chained append-only copy, and an update settles only after more than two thirds of the nodes sign votes in numbered rounds",
    ),
    "blockchain-based-server-side-presentation-provenance-shell": (
        "blockchain-based",
        "four retailers and three certifiers each operate a peer holding a tamper-evident append-only log, with endorsements collected before an ordering cluster sequences batches and peers validate them by quorum",
    ),
    "blockchain-based-remote-contracts-custody-history-walk": (
        "blockchain-based",
        "twelve validator peers plus a five-sequencer ordering cluster, where a transfer settles only once matching execution results are collected from a quorum of peers onto the shared log",
    ),
    # command-query-responsibility-segregation
    "command-query-responsibility-segregation-tactical-domain-modelling-cart-credit": (
        "command-query-responsibility-segregation",
        "update handling and display handling split into separate deployables with separate stores, a writer owning every change against normalized tables and a reader served only from a denormalized store",
    ),
    "command-query-responsibility-segregation-model-clarity-hold-invariants": (
        "command-query-responsibility-segregation",
        "submission intents applied to a normalized transactional ledger that is the sole authority for changes, while a projector maintains denormalized seat-map and order-history views from committed entries",
    ),
    "command-query-responsibility-segregation-time-contract-amendment-versions": (
        "command-query-responsibility-segregation",
        "two isolated paths with two stores: amendments appended to a normalized ledger that emits accepted events, and a view builder maintaining a denormalized snapshot store for the portal",
    ),
    # data-mesh
    "data-mesh-translation-ledger-intake": (
        "data-mesh",
        "independent crews with separate repositories, databases and stream namespaces each own and serve their own shared tables, while a thin platform crew supplies storage, a schema catalog and identities but writes no business tables",
    ),
    "data-mesh-change-tracking-registry-overwrite": (
        "data-mesh",
        "three squads each running their own account, database, bucket and stream, publishing shares into a shared catalog index and serving reads from their own bucket with no central warehouse team",
    ),
    "data-mesh-domain-to-store-mapping-shipment-plan": (
        "data-mesh",
        "checkout, inventory and risk run as autonomous squads with their own databases, versioned extracts and its own read API, joined through a policy gateway rather than a central warehouse",
    ),
    # edge-computing
    "edge-computing-time-store-price-history": (
        "edge-computing",
        "each shop's cabinet server authorizes sales, applies promotions and reprints receipts locally while the wide-area link is down for days, with the central catalog only synchronizing",
    ),
    "edge-computing-values-vineyard-ids-units": (
        "edge-computing",
        "every vineyard shed runs its own hub computer wired to probes and valves, issuing spray and valve commands within seconds while the cellular link to the cloud is down",
    ),
    "edge-computing-tactical-domain-model-signal-plan-ownership": (
        "edge-computing",
        "each junction's roadside cabinet runs sensor fusion and phase scheduling and drives the lamps within a 40 ms budget, independent of the hub and across multi-day link loss",
    ),
    # enterprise-service-bus
    "enterprise-service-bus-dependency-wiring-shared-ops-packaging": (
        "enterprise-service-bus",
        "five incompatible applications on a mainframe over message queues, Java over HTTPS, Python over gRPC, SAP over RFC and a SaaS email API, integrating only through one separately deployed hub that translates schemas and bridges protocols",
    ),
    "enterprise-service-bus-task-lifetimes-centralflow-drain": (
        "enterprise-service-bus",
        "SOAP/XML on an application server, REST/JSON and EDI flat files over SFTP are all picked up by protocol adapters and forwarded as one canonical envelope into a centrally operated mediation tier",
    ),
    "enterprise-service-bus-creation-vendor-trio": (
        "enterprise-service-bus",
        "three surrounding systems speak RFC, REST and TCP SQL, and every exchange passes through one shared clustered hub that normalizes payloads between them",
    ),
    # hybrid-cloud
    "hybrid-cloud-translation-plant-usage-unification": (
        "hybrid-cloud",
        "usage facts originate in an on-premises plant computer room from fixed-width historian files and a lab system, and converge with a third-party cloud feed into one intake",
    ),
    "hybrid-cloud-change-tracking-plant-reconciler-duplicates": (
        "hybrid-cloud",
        "plant scanning writes to a database on the next rack in the factory basement for sub-millisecond acknowledgement while the catalog service runs in a provider-managed cluster in another region over a wide-area link",
    ),
    "hybrid-cloud-data-access-gateways-parts-stock": (
        "hybrid-cloud",
        "the parts-stock cluster stays in a company-operated data centre for export-control reasons while the customer-facing storefront runs on managed cloud functions and a managed gateway",
    ),
    # kappa-architecture
    "kappa-architecture-resource-lifecycle-and-caching-cityflow-enrichment": (
        "kappa-architecture",
        "an ordered durable log with 90-day retention is the only source of truth and one long-running job is the only computation over it, so a rebuild replays the same log instead of a separate offline path",
    ),
    "kappa-architecture-audit-cityflow-payout-trail": (
        "kappa-architecture",
        "a retained ordered topic holds every raw event for 400 days and a single streaming job is the only computation over that history, serving both live views and replays",
    ),
    "kappa-architecture-capabilities-isolation-tenant-ledger-expiry": (
        "kappa-architecture",
        "ingestion appends to an immutable durable ordered store and one long-lived engine reads it from the latest offset while a replay coordinator restarts that same job to recompute",
    ),
    # lambda-architecture
    "lambda-architecture-queries-driver-aggregates-access": (
        "lambda-architecture",
        "nightly full-history rebuilders rescan the log from day zero into canonical aggregates while a parallel live job maintains deltas, so batch and speed paths serve the same platform",
    ),
    "lambda-architecture-long-running-change-merchant-curation": (
        "lambda-architecture",
        "a nightly job rereads the whole archive from day zero and overwrites the totals table while a streaming deployment upserts per-minute increments over the same events",
    ),
    "lambda-architecture-relational-structure-video-collections": (
        "lambda-architecture",
        "a nightly rebuild cluster scans the full immutable archive to recompute authoritative totals while a live path derives increments, and both land in the serving store",
    ),
    # modular-monolith
    "modular-monolith-dependency-failure-shared-workers": (
        "modular-monolith",
        "one release artifact per host behind a load balancer, with checkout, stock, loyalty and messaging areas owning their tables and reachable only through each area's public facade",
    ),
    "modular-monolith-concurrent-io-freshcart-pickqueue": (
        "modular-monolith",
        "one service image launched as one OS process, with ordering, stock and fulfilment slices owning their own tables and crossing only through explicit public functions",
    ),
    "modular-monolith-coordination-parcel-intake-handoff": (
        "modular-monolith",
        "built and released as one executable on a single deployment, with scan, route, billing and audit packages strictly isolated behind public interfaces",
    ),
    # multi-cloud
    "multi-cloud-long-running-change-voyage-checkout": (
        "multi-cloud",
        "US planners are served from one public cloud and EU planners from a second, both writing the same ledger in the first provider's database through a replica in the second",
    ),
    "multi-cloud-queries-cross-border-claims": (
        "multi-cloud",
        "ingress runs as managed functions in one public cloud, scoring as functions in a second, and settlement on containers in the first, all sharing one logical store split across both providers",
    ),
    "multi-cloud-relational-structure-mapping-film-performer-pairs": (
        "multi-cloud",
        "ingest runs on a cluster in one public cloud and the portal on a cluster in another, over a primary database in the first and a read replica in the second",
    ),
    # presentation-abstraction-control
    "presentation-abstraction-control-message-ephemeral-scoring-route": (
        "presentation-abstraction-control",
        "a fleet console built as a tree of self-contained interactive units, each split into a view, a store and a liaison that alone talks to other units",
    ),
    "presentation-abstraction-control-remote-contracts-strip-refresh": (
        "presentation-abstraction-control",
        "every console pane is a closed trio of presentation, abstraction and control, with the abstraction owning the boundary to the shared holdings",
    ),
    "presentation-abstraction-control-server-side-presentation-harbor-shell": (
        "presentation-abstraction-control",
        "the operations board is a hierarchy of UI units, each owning three cooperating parts for rendering, state and coordination of traffic",
    ),
    # reactive-architecture
    "reactive-architecture-capabilities-isolation-tenant-window-cutoff": (
        "reactive-architecture",
        "routing feeds single-threaded event loops that never block, emitting derived alerts downstream while consumers advertise demand and producers stall when buffers fill",
    ),
    "reactive-architecture-compile-time-access-control-batch-assembly": (
        "reactive-architecture",
        "ingress pushes ticks into bounded non-blocking queues feeding workers pinned to their own event loops, with downstream stages returning demand credits",
    ),
    "reactive-architecture-audit-payout-trail-integrity": (
        "reactive-architecture",
        "a topology of stateless stages fans out over bounded event channels in which each consumer advertises demand and producers stall when buffers fill",
    ),
    # reflection-architecture
    "reflection-architecture-queries-live-sensor-catalog": (
        "reflection-architecture",
        "a long-lived service that imports a newly added dataclass module without restarting, reads its annotations and column tags, and refreshes a central type catalog at runtime",
    ),
    "reflection-architecture-relational-structure-mapping-blueprint-spec-pairing": (
        "reflection-architecture",
        "an in-memory graph of which detectors are running and how the router connects them is inspected and rewired through an admin API while the fabric keeps running without restarts",
    ),
    "reflection-architecture-long-running-change-permit-hub-collision": (
        "reflection-architecture",
        "a session coordinator tracks open edit sessions and routes each request to the current service implementations without restarting the hub",
    ),
    # rule-based-system
    "rule-based-system-functional-composition-claim-math-separation": (
        "rule-based-system",
        "adjudication is driven by analyst-authored policy bundles listing patterns over a fact store plus the assertions and retractions to perform when they hold",
    ),
    "rule-based-system-functional-data-modelling-underwriting-verdict": (
        "rule-based-system",
        "several hundred analyst-authored clauses are hot-swapped from a repository and matched against a shared working memory, driving asserts, retracts and updates",
    ),
    "rule-based-system-method-clause-assembly": (
        "rule-based-system",
        "a matcher loads about 240 analyst-authored clauses that reload without restart and pushes matches onto a salience-ordered agenda for consequence execution",
    ),
    # saga
    "saga-message-contracts-trip-pairing": (
        "saga",
        "one planning service drives flight, hotel, payment and car holds against four external providers it does not operate, reversing earlier holds through their cancel endpoints when a later step is refused",
    ),
    "saga-message-endpoint-holiday-replies": (
        "saga",
        "one managing service drives external seat, room and card providers in sequence and reverses the earlier holds when a later stage declines or times out, with outcomes persisted per step",
    ),
    "saga-remote-contracts-trip-coordinator-vendor-access": (
        "saga",
        "one container coordinates seat, room, payment and ground-dispatch parties it does not operate, cancelling holds and refunding charges through their APIs when a later party refuses",
    ),
    # serverless
    "serverless-method-nightly-digest-query-assembly": (
        "serverless",
        "handlers are launched by the vendor only when messages arrive, billed per invocation and per millisecond, and nothing stays running in quiet periods",
    ),
    "serverless-domain-logic-seat-hold-flow": (
        "serverless",
        "three isolated on-demand handlers started only when an event or call arrives, billed for the milliseconds used and discarded afterwards",
    ),
    "serverless-functional-composition-receipt-amount": (
        "serverless",
        "a managed front door invokes one function, object writes fire a second and a backlog queue fires a third, each created per event with no always-on host",
    ),
    # service-mesh
    "service-mesh-task-lifetimes-ingest-overload": (
        "service-mesh",
        "every pod runs the service container plus a companion network container that all inbound and outbound calls pass through, enforcing mutual TLS, timeouts and retries pushed from a control plane",
    ),
    "service-mesh-dependency-wiring-and-discovery-order-processor": (
        "service-mesh",
        "application code never dials a peer directly, calling localhost where a co-located traffic agent injects mutual TLS, retries and load balancing",
    ),
    "service-mesh-composition-pricing-client-options": (
        "service-mesh",
        "each pod pairs a business container with a co-located traffic agent injected at deploy time that intercepts outbound localhost calls",
    ),
    # service-oriented-architecture
    "service-oriented-architecture-long-running-change-policy-edit-collision": (
        "service-oriented-architecture",
        "independently deployable units on different stacks each publish a contract to a central registry and call each other only over standard protocol, with no shared database",
    ),
    "service-oriented-architecture-queries-claims-persistence-centralization": (
        "service-oriented-architecture",
        "separately versioned units publish contracts and talk only through them, with a central directory holding endpoint addresses so callers in three languages bind without custom adapters",
    ),
    "service-oriented-architecture-relational-structure-mapping-invoice-money-columns": (
        "service-oriented-architecture",
        "four separately deployed processes on two stacks publish contracts and exchange messages over standard protocol through a central endpoint directory",
    ),
    # space-based
    "space-based-message-venue-catalog-replication": (
        "space-based",
        "identical cells each hold a heap-resident partition of inventory and stay coherent through a peer replication backbone, with no central database on the hot path",
    ),
    "space-based-remote-contracts-peer-transport-abstraction": (
        "space-based",
        "identical replicas bundle request handling with a RAM-resident partition of live state, authoritative in process and flushed asynchronously, with capacity added by starting more cells",
    ),
    "space-based-server-side-presentation-seat-portal-shell": (
        "space-based",
        "edge gateways hash each event to one of twelve cells that hold their slice of seat state in local memory and complete most checkouts without touching the centre",
    ),
    # task-control-architecture
    "task-control-architecture-task-lifetimes-rover-abort-propagation": (
        "task-control-architecture",
        "a fleet supervisor assigns missions that each rover's supervisor expands into concurrent navigation, scanning and lift activities, each driving lower sequencers in a nested control tree",
    ),
    "task-control-architecture-creation-configuration-traverse-order": (
        "task-control-architecture",
        "a unit supervisor splits a job into locomotion and manipulation activities ticked at 20 Hz in isolated partitions, emitting setpoint streams to servo loops",
    ),
    "task-control-architecture-dependency-wiring-onboard-helpers": (
        "task-control-architecture",
        "each robot hosts three simultaneous loops where a mission sequencer steps a pick job and can preempt or suspend the lower navigation and manipulation loops",
    ),
    # -- style-label gap 2026-11 (reviewed: two blind annotators + adjudication) (reviewed, two blind annotators + adjudication) --
    "actor-based-activate-command-handler-outline": (
        "actor-based",
        "The handler runs inside a per-line behaviour with private state, a mailbox, one command at a time and a reply address. Nearest alternative is monolithic, but the deployment is a cluster of per-line units restarted by supervision, not one deployable serving everything.",
    ),
    "actor-based-authorisation-archive-composable-read": (
        "actor-based",
        "Issuer links are isolated stateful units with private hold state, mailboxes, one request at a time and reply addresses; the console is an adjunct to that fleet. Nearest alternative is none: the ask looks like a library decision inside one reporting module, but that module serves a switch whose authorisations are carried by isolated per-issuer units with supervision, so the system under decision has a runtime shape.",
    ),
    "actor-based-bid-package-single-round-trip-entry": (
        "actor-based",
        "Every part is an isolated stateful process with a registered name, a mailbox, one message at a time and a supervisor. Nearest alternative is microservices: the boxes here are units inside one runtime with names, not independently released deployables, and no service boundary is claimed.",
    ),
    "actor-based-billing-account-optional-contract-handle": (
        "actor-based",
        "The unit of computation is a single isolated stateful activation with a FIFO message stream, addressed by account key, one request at a time, restarted or relocated by the runtime \u2014 'nothing outside it reads its fields'. Nearest alternative is microservices, but no independently released deployable is described; the isolation is a per-entity runtime boundary inside one cluster.",
    ),
    "actor-based-catalogue-shard-index-capacity-release": (
        "actor-based",
        "Twenty-four isolated units, each owning a private index, fed by an inbox channel, one message at a time, addressed by shard identity, restarted and reloaded by a monitor goroutine \u2014 no shard reads another's slice. Nearest alternative is space-based: the catalogue is sharded in memory, but there is no shared space or replicated grid, and each shard is reached only through its own inbox.",
    ),
    "actor-based-circuit-provisioning-operation-boundary": (
        "actor-based",
        "Circuits are isolated stateful processes with mailboxes, addressed by name, one command at a time, supervised and restartable; the request process is itself such a unit. Nearest alternative is event-driven, where steps would travel as published records and no caller would wait for one outcome; here a single response inside a window is required.",
    ),
    "actor-based-claim-submission-stable-write-identity": (
        "actor-based",
        "Claim files are isolated stateful entities, addressed by reference, one message at a time, relocated and restarted by the cluster runtime, with private reserve state. Nearest alternative is event-driven: there the submission would be a published record that subscribers deduplicate from their own offsets, not a directed request whose answer depends on the entity's private record.",
    ),
    "actor-based-clause-grammar-position-cached-rules": (
        "actor-based",
        "One isolated stateful worker per document, addressed by handle, one call at a time, restarted with state rebuilt from storage, private parse tables and cache. Nearest alternative is pipe-and-filter, which would stage the document through transformers; here the unit is a single stateful worker whose cache must persist across calls.",
    ),
    "actor-based-cold-chain-excursion-notice-schedule": (
        "actor-based",
        "Each lane is an isolated stateful process addressed by name, with a mailbox, strictly one message at a time, and a supervisor that restarts it \u2014 'no other process, on that node or the others, can read those fields'. Nearest alternative is event-driven, which would carry readings as records several stateless consumers observe; here the consuming unit keeps private mutable state and cannot be replicated by a second reader.",
    ),
    "actor-based-enrolment-credential-principal-normalisation": (
        "actor-based",
        "Attempts are isolated units with private nonce and rate state, mailboxes, strict one message at a time, and no shared memory with verifier units they address by name. Nearest alternative is microservices, but nothing is independently released: it is one Python runtime whose units are created by a local supervisor.",
    ),
    "actor-based-ledger-posting-five-stage-chain": (
        "none",
        "Adjudicated to none: the described system is one nine-hundred-line routine inside a single coroutine with no deployment, no runtime topology and no inter-component boundary. The staged pipeline of resident coroutines is the redesign the decision asks for, and clarification 4 forbids labelling the requested change. Recorded finding: this row's authoring brief asked for actor-based, so the row does not evidence its planned style.",
    ),
    "actor-based-price-book-leaf-amendment-path": (
        "actor-based",
        "The price book is private state of an Akka Typed entity, addressed by account, one message at a time, with readers never handed the tree. Nearest alternative is microservices: a per-account entity in a cluster resembles a service, but these entities are not released or operated independently \u2014 the cluster runtime creates and relocates them from one program image.",
    ),
    "actor-based-rebate-rule-tree-single-traversal": (
        "actor-based",
        "Workers are isolated units holding a private tree, addressed by partner, one member at a time, restarted by a supervisor; no other worker can reach that tree. Nearest alternative is none, since a deployed fleet with mailboxes and restarts is visible; if a reader doubts the fleet, event-driven is the next temptation \u2014 but there is no shared stream of records here, only directed requests.",
    ),
    "actor-based-reservation-row-as-live-process": (
        "actor-based",
        "Each reservation is an isolated stateful process, registered by booking reference, handling one message at a time, restarted by its supervisor and rebuilding from durable state \u2014 'exactly one live process'. Nearest alternative is monolithic, which would ignore the per-record concurrency carrier; here the mailbox and per-unit identity do the work.",
    ),
    "actor-based-section-worker-shared-loop-shell": (
        "actor-based",
        "Sections are isolated stateful workers, addressed individually, one message at a time, restarted after a panic, with no shared mutable state between them. Nearest alternative is reactive-architecture, which would be about demand signalling and backpressure; nothing here bounds queues or propagates demand, the mailboxes are simply per-unit serialisation.",
    ),
    "actor-based-store-lane-offline-settlement-backlog": (
        "actor-based",
        "Till lanes and the store unit are isolated stateful mailboxes, addressed individually, one message at a time, restarted by a supervisor, with no shared memory between them. Nearest alternative is broker, which the head office does run \u2014 but the question is not the central routing tier, it is how one consumer's position survives an outage.",
    ),
    "actor-based-terminal-instruction-payload-kind": (
        "actor-based",
        "Each equipment unit is an isolated stateful entity with a private instruction queue and position, addressed by equipment id, one instruction at a time, supervised by the runtime. Nearest alternative is broker \u2014 no central component owns the wiring here; the units talk to the planner directly over an unreliable link.",
    ),
    "actor-based-till-basket-payment-path-composition": (
        "actor-based",
        "Baskets and payment terminals are isolated stateful entities with mailboxes, addressed by reference, one message at a time, restarted by a supervisor \u2014 'the basket never inspects it'. Nearest alternative is event-driven: there the payment step would be a published record with independent consumers rather than a directed request whose reply returns to the asking entity's address.",
    ),
    "actor-based-till-lane-intake-capacity-policy": (
        "actor-based",
        "Lanes and downstream entities are isolated stateful units with mailboxes, one message at a time, addressed by lane identity, no shared mutable totals. Nearest alternative is reactive-architecture: backpressure appears as a failure mode here, but the system is not built around demand signalling end to end \u2014 the decision is capacity and overflow policy on one intake.",
    ),
    "actor-based-underwriting-case-domain-boundary": (
        "actor-based",
        "Each risk is an isolated stateful worker, addressed by reference, one message at a time, supervised, with private exposure and referral state. Nearest alternative is layered-monolith \u2014 the engine could be read as one deployable in layers, but the text names no layer ordering and its carrier is a fleet of per-risk units.",
    ),
    "aiml-centric-bid-score-flush-boundary": (
        "aiml-centric",
        "Style evidence: 'a graph model and a gradient-boosted model are retrained nightly on confirmed enforcement cases, approved by a trust-and-safety reviewer, and registered with the graph snapshot they were built on'. Nearest alternative style event-driven: bids arrive as a stream, but scoring them with registered models is what the platform sells; the transport is not the mechanism under decision.",
    ),
    "aiml-centric-claim-finalisation-fact": (
        "aiml-centric",
        "Style evidence: two models 'retrained monthly on adjudicated outcomes, evaluated on a locked holdout, and registered with the feature definitions they were trained against', with a feature store shared by training and serving and a drift monitor. Nearest alternative style event-driven: reactions exist, but the carrier is not the mechanism under decision; the system's value is the prediction that starts the chain.",
    ),
    "aiml-centric-container-calibration-lock-scope": (
        "aiml-centric",
        "Style evidence: 'A gradient-boosted model, retrained weekly on carrier-reported actuals, is registered with its feature definitions and served by a Java service that recomputes features from the same definitions'. Nearest alternative style layered-monolith: one service with caches, but what it sells is the served prediction, and the decision is the lock's scope, not a layer stack.",
    ),
    "aiml-centric-degradation-expression-representation": (
        "aiml-centric",
        "Style evidence: 'three models ... are retrained weekly on work-order outcomes, validated against an engineer-reviewed holdout, and registered with the feature-code hash they were fitted on' plus serving that 'recomputes features from the same code'. Nearest alternative style rule-based-system: expressions look declarative, but there is no rule base matched over facts by an engine with an agenda; they parameterise features for registered models.",
    ),
    "aiml-centric-demand-snapshot-sharing": (
        "aiml-centric",
        "Style evidence: the value-bearing component is a fitted model with a 'training' history, a 'registered' version and an alias 'that serving resolves', plus a 'drift monitor' comparing predictions with outcomes. Nearest alternative style none: nine consumers, a serving tier and a sidecar are real topology, but the decision is about sharing one data structure, and the platform's product is still a prediction.",
    ),
    "aiml-centric-fee-schedule-native-resource": (
        "aiml-centric",
        "Style evidence: 'the gradient-boosted overpayment scorer registered by version' and 'feature computation for the anomaly model', alongside fee pricing per claim. Nearest alternative style hexagonal: there is a clear boundary around an external concern, but the boundary here is one native resource behind a class, and the platform's value is the model-scored adjudication.",
    ),
    "aiml-centric-fraud-case-collection-interface": (
        "aiml-centric",
        "Style evidence: 'three models ... are retrained nightly, evaluated against the incumbent on a labelled holdout, and registered in MLflow by version', with a serving tier resolving versions per segment. Nearest alternative style monolithic: scoring and case management are separate deployables, but the decision is about the boundary to the prediction store, and the platform's value is the trained scorer.",
    ),
    "aiml-centric-home-screen-conversation-entry": (
        "backend-for-frontend",
        "Both blind annotators labelled this backend-for-frontend where the authoring intent was aiml-centric",
    ),
    "aiml-centric-irrigation-request-config-plumbing": (
        "aiml-centric",
        "Style evidence: models 'retrained weekly on farmer-confirmed irrigation events, validated on a farm-disjoint holdout, registered with their feature definitions, and served by a Go inference service'. Nearest alternative style pipe-and-filter: stages form a chain, but they are per-request serving steps over registered models rather than reorderable stream transformations; the value is the prediction.",
    ),
    "aiml-centric-listing-batch-worker-shutdown": (
        "aiml-centric",
        "Style evidence: models 'retrained nightly on closed-listing outcomes, evaluated on a chronological holdout, and registered with the tokeniser and image-preprocessing versions they were trained against'. Nearest alternative style kappa-architecture: a single processing tier exists, but the platform's product is the trained score, and the decision is about terminating consumers rather than about reprocessing semantics.",
    ),
    "aiml-centric-maintenance-record-chain": (
        "aiml-centric",
        "Style evidence: 'a survival model and a gradient-boosted classifier are retrained quarterly on inspection outcomes, validated on a fleet-disjoint holdout, and registered with the data snapshot and code commit that produced them'. Nearest alternative style blockchain-based: the chain of authenticators invites that reading, but there is one writer, no consensus and no shared ledger \u2014 the verifier only recomputes links.",
    ),
    "aiml-centric-model-applicability-dates": (
        "aiml-centric",
        "Style evidence: models 'retrained per tax year', 'validated against the prior year's production model, and registered with the tax year, the feature-set version and the jurisdiction pack they were fitted on', with serving resolving a version per tax year and jurisdiction. Nearest alternative style rule-based-system: jurisdiction packs sound statutory, but no authored clause base is evaluated; the served artifact is a trained model.",
    ),
    "aiml-centric-partner-access-switch-off": (
        "aiml-centric",
        "Style evidence: 'A model flags likely-critical results ... it is retrained weekly on clinician-confirmed outcomes, validated on a hospital-disjoint holdout, and registered with version metadata; serving returns a flag with the model version and a confidence band'. Nearest alternative style api-gateway: partners call one published endpoint, but the text's decision is about withdrawing one partner's authority, and the value-bearing component is the served model.",
    ),
    "aiml-centric-provisioning-response-baseline": (
        "aiml-centric",
        "Style evidence: two models 'retrained nightly on closed orders, validated on a region-disjoint holdout, and registered with the inventory snapshot they were trained against', served behind a TypeScript API. Nearest alternative style microservices: three provisioning systems call it, but it is one provider API, and the value-bearing component is the trained failure prediction.",
    ),
    "aiml-centric-replenishment-cache-locking": (
        "aiml-centric",
        "Style evidence: 'a gradient-boosted model with per-category configuration is retrained weekly, validated on a store-disjoint holdout, and registered; serving recomputes features from versioned definitions and returns a forecast distribution'. Nearest alternative style space-based: an in-memory cache sits on the hot path, but scalability comes from model serving, and no shared space with a partitioned processor set is described.",
    ),
    "aiml-centric-scheduling-endpoint-checkpoint": (
        "aiml-centric",
        "Style evidence: 'a gradient-boosted model is retrained fortnightly on attended and missed outcomes, validated on a clinic-disjoint holdout, registered with its feature definitions, and served by a Python inference service'. Nearest alternative style api-gateway: one access point with cross-cutting policy exists, but the decision is where identity is established across 14 model endpoints, and the product is the served prediction.",
    ),
    "aiml-centric-scoring-run-tables": (
        "aiml-centric",
        "Style evidence: models are 'retrained', 'registered', and 'promoted through staging and production' with a consistency check, while serving 'scores them with a model resolved from the registry by version'. Nearest alternative style layered-monolith: persistence has a layer, but the system's core is the model lifecycle and its predictions; the table question is a consequence of the prediction volume.",
    ),
    "aiml-centric-telematics-value-types": (
        "aiml-centric",
        "Style evidence: 'a gradient-boosted model is retrained monthly on insurer-observed claims, evaluated on a holdout ... and registered with its feature-definition hash; serving is a Go service that recomputes features from the same definitions and writes a risk score with the model version'. Nearest alternative style none: the value-type decision is a code shape, but a training pipeline and a versioned serving path are described, so the model is the product.",
    ),
    "api-gateway-basket-allocation-concept": (
        "api-gateway",
        "Three client types - couriers, store consoles and partner grocers - reach the picking, substitution and delivery services only through one authenticated endpoint that validates payloads and applies quotas.",
    ),
    "api-gateway-billing-mediation-standin": (
        "api-gateway",
        "Two partner-facing interfaces terminate at one address that authenticates them, applies per-partner quotas and routes into the rating, dispute and settlement services.",
    ),
    "api-gateway-billing-plan-history": (
        "api-gateway",
        "Subscribers and partners reach the subscription service only through one published endpoint that authenticates them and applies a plan-aware quota.",
    ),
    "api-gateway-contact-centre-queue-walk": (
        "api-gateway",
        "Agent desktops authenticate at one endpoint that applies a per-queue quota and routes into the interaction service.",
    ),
    "api-gateway-dispatch-record-metadata": (
        "api-gateway",
        "Every technician visit reaches the visit service through one published endpoint that authenticates the technician and applies a per-region quota before forwarding.",
    ),
    "api-gateway-dispatcher-tenant-counters": (
        "api-gateway",
        "Eight thousand technician apps address one authorised, quota-limited entry point that routes into the job, parts and scheduling services, none of which is reachable from outside.",
    ),
    "api-gateway-enrolment-boundary-restriction": (
        "api-gateway",
        "Only one endpoint is published to the clearing house, holding the certificate and the partner quota, while eligibility, seat allocation, fee assessment and document intake stay unreachable and are reached by route from that endpoint.",
    ),
    "api-gateway-enrolment-record-mapping": (
        "api-gateway",
        "Three callers - portal, registrar console and clearing feed - address one authenticated, quota-limited endpoint that routes into the enrolment service and nothing else is published.",
    ),
    "api-gateway-freight-claim-audit": (
        "api-gateway",
        "Broker submissions reach the claim service only through one published endpoint that authenticates the broker and applies a per-broker quota.",
    ),
    "api-gateway-hvac-mapper-hierarchy": (
        "api-gateway",
        "Building controllers and operator dashboards authenticate at one endpoint that applies a per-building quota and routes into the device service, which is not published itself.",
    ),
    "api-gateway-hvac-setpoint-test": (
        "api-gateway",
        "Building controllers and operator consoles reach the setpoint service only through one published endpoint that authenticates the controller and applies a per-building quota.",
    ),
    "api-gateway-identity-consent-revocation": (
        "api-gateway",
        "Banks and employers authenticate at one published endpoint that applies a per-organisation quota and routes into the document, biometric and consent services.",
    ),
    "api-gateway-identity-lock-policy": (
        "api-gateway",
        "Every mobile upload reaches the document service through one published endpoint that performs authentication, per-tenant quota and a concurrency limit, which is the entry structure the style names.",
    ),
    "api-gateway-itinerary-adapter-family": (
        "api-gateway",
        "Three client channels - agents, consumer web and two partners - reach search, hold and ticketing services only through one address that authenticates each channel and applies its quota before routing.",
    ),
    "api-gateway-lab-analyzer-integration": (
        "api-gateway",
        "Every hospital system is served by one published endpoint that authenticates it, applies its quota and routes into the order, result and notification services.",
    ),
    "api-gateway-loyalty-consumer-drain": (
        "api-gateway",
        "Retail partners authenticate at one endpoint that applies per-partner quotas and routes into the ledger service whose queue the consumers drain; nothing else is published to partners.",
    ),
    "api-gateway-loyalty-reconcile-probe": (
        "api-gateway",
        "Partner settlement files and internal finance queries all arrive at one published endpoint that authenticates them, applies quotas and routes into the reconciliation service.",
    ),
    "api-gateway-manifest-heterogeneous-cache": (
        "api-gateway",
        "Forwarders, carriers and brokers each authenticate at one published endpoint that applies their quota and routes into the manifest service, which holds every uploaded document.",
    ),
    "api-gateway-matching-request-outline": (
        "api-gateway",
        "Every driver app authenticates at one endpoint that applies a per-region quota and routes into the availability service, and nothing else is published to drivers.",
    ),
    "api-gateway-origination-reply-channel": (
        "api-gateway",
        "Twenty-two broker systems authenticate at one endpoint that applies their quota and routes into the underwriting service, and no broker reaches a party integration directly.",
    ),
    "api-gateway-pos-terminal-channel-endpoint": (
        "api-gateway",
        "Every store terminal authenticates at one published endpoint that enforces its store quota and routes into the sale, stock and receipt services, none of which terminals can address.",
    ),
    "api-gateway-robotics-unit-records": (
        "api-gateway",
        "Every operator action reaches the unit, task and telemetry services through one authenticated endpoint that applies a per-site limit before routing, and no backend is otherwise published.",
    ),
    "api-gateway-roster-context-ownership": (
        "api-gateway",
        "Operations users from every airline authenticate at one endpoint that applies a per-airline quota and routes into the roster service, which is not published itself.",
    ),
    "api-gateway-subscriber-session-edges": (
        "api-gateway",
        "The mobile app, the web portal and the support console all pass through one published endpoint that authenticates the subscriber and applies a per-tenant quota and concurrency limit before routing into the billing service.",
    ),
    "api-gateway-telematics-rejection-cases": (
        "api-gateway",
        "The whole device fleet authenticates at one endpoint that applies a per-device quota and routes decoded frames into the position, diagnostic and maintenance services, which are unpublished.",
    ),
    "api-gateway-term-record-consistency": (
        "api-gateway",
        "The student portal, registrar console and waitlist worker all address one signed, quota-limited entry point that fronts the backend pool, with no client reaching a backend directly.",
    ),
    "api-gateway-transit-fare-contract": (
        "api-gateway",
        "Three client populations reach the fare, entitlement and settlement services only through one authenticating, quota-limited address.",
    ),
    "api-gateway-trial-visit-intake": (
        "api-gateway",
        "Every site coordinator action reaches the visit service through one authenticated endpoint that applies a per-site quota and routes, and no backend is published to coordinators.",
    ),
    "backend-for-frontend-auction-bid-client-backend": (
        "backend-for-frontend",
        "The bidder web app, the mobile app and the auctioneer console each have their own dedicated backend with client-specific payloads.",
    ),
    "backend-for-frontend-auction-partner-client-backend": (
        "backend-for-frontend",
        "The bidder web, mobile and partner API frontends each have their own dedicated backend with client-specific payloads.",
    ),
    "backend-for-frontend-banking-console-client-backend": (
        "backend-for-frontend",
        "The branch console, mobile onboarding and partner API each have their own dedicated backend with client-specific shaping.",
    ),
    "backend-for-frontend-billing-portal-client-backend": (
        "backend-for-frontend",
        "The self-service portal, call centre and partner API each have their own dedicated backend with client-specific payloads.",
    ),
    "backend-for-frontend-claims-app-client-backend": (
        "backend-for-frontend",
        "The adjuster console, member app and broker portal each have their own dedicated backend with client-specific payloads.",
    ),
    "backend-for-frontend-coldchain-partner-client-backend": (
        "backend-for-frontend",
        "The shipper portal, carrier integration and mobile scan frontends each have their own dedicated backend with their own payload conventions.",
    ),
    "backend-for-frontend-diagnostics-tool-client-backend": (
        "backend-for-frontend",
        "The workshop tablet, technician console and fleet API each have their own dedicated backend with their own payload conventions.",
    ),
    "backend-for-frontend-forecast-map-client-backend": (
        "backend-for-frontend",
        "The map widget, the alerting console and partner utilities each have their own dedicated backend shaped to their own payloads and released alongside them.",
    ),
    "backend-for-frontend-freight-portal-client-backend": (
        "backend-for-frontend",
        "The shipper portal, carrier integration and customs console frontends each have their own dedicated backend with client-specific payload shaping.",
    ),
    "backend-for-frontend-hotel-web-client-backend": (
        "backend-for-frontend",
        "The guest web, front desk and partner extranet frontends each have a dedicated backend with client-specific payload shaping.",
    ),
    "backend-for-frontend-hvac-desk-client-backend": (
        "backend-for-frontend",
        "Each of the three frontends has its own dedicated backend with its own payload shape and release cadence.",
    ),
    "backend-for-frontend-identity-mobile-client-backend": (
        "backend-for-frontend",
        "The document-capture app, reviewer console and partner API each have their own dedicated backend with their own payload conventions.",
    ),
    "backend-for-frontend-imagery-map-client-backend": (
        "backend-for-frontend",
        "The map viewer, calibration console and ingest control each have their own dedicated backend with client-specific responses.",
    ),
    "backend-for-frontend-procurement-portal-client-backend": (
        "backend-for-frontend",
        "The supplier portal, buyer console and mobile approval app each have their own dedicated backend with client-specific shaping.",
    ),
    "backend-for-frontend-settlement-desk-client-backend": (
        "backend-for-frontend",
        "The trader console, risk dashboard and regulator feed each have their own dedicated backend with client-specific payloads.",
    ),
    "backend-for-frontend-tax-app-client-backend": (
        "backend-for-frontend",
        "The web filing, mobile capture and agent console frontends each have their own dedicated backend with client-specific shaping.",
    ),
    "backend-for-frontend-transit-desk-client-backend": (
        "backend-for-frontend",
        "The operations desk, mobile app and gate terminal each have their own dedicated backend with client-specific payloads.",
    ),
    "backend-for-frontend-transit-tap-client-backend": (
        "backend-for-frontend",
        "Three frontends - app, gate terminal, operations desk - each have their own dedicated backend shaped to their payload and error needs and released with their client, not one shared entry for all clients.",
    ),
    "backend-for-frontend-transit-terminal-client-backend": (
        "backend-for-frontend",
        "The gate terminal, mobile app and rail operator each have their own dedicated backend with client-specific payloads and release cadence.",
    ),
    "blockchain-based-attestation-batch-compaction": (
        "blockchain-based",
        "Style evidence: \"the customer's own bank and two attesting banks endorse an attestation, and personal data stays in private collections visible only to the banks named in it.\" Nearest alternative style is `none`: the onboarding service is deployed and imports across a network boundary, so this is not a bare code-shape row.",
    ),
    "blockchain-based-batch-advice-per-item-proposal": (
        "blockchain-based",
        "Style evidence: \"nine validators, one per member, and an advice is final only when the issuer and the acquirer endorse it.\" Nearest alternative style is `event-driven`: the parts are individually addressed ledger submissions, not published records several consumers react to.",
    ),
    "blockchain-based-certificate-commit-callback": (
        "blockchain-based",
        "Style evidence: \"A joint venture of one contractor, forty subcontractors and a settlement bank runs payment certificates on a Fabric channel: the contractor's peer and the bank's peer must both endorse a certificate, and the committed entry is the only thing the bank pays against.\" Nearest alternative style is `broker`: there is no central message broker owning the wiring, only the peer's own event stream.",
    ),
    "blockchain-based-clinic-day-slot-rules": (
        "blockchain-based",
        "Style evidence: \"fourteen validator nodes, one per member group, hold identical replicas and no member can revise a committed appointment.\" Nearest alternative style is `modular-monolith`: the module here is a way of organising rules inside one Rust service, while the system's load-bearing structure is the shared multi-party record.",
    ),
    "blockchain-based-consignment-exclusive-claim": (
        "blockchain-based",
        "Style evidence: \"each forwarder runs a validator and holds the same replica, and no forwarder's operator can amend a released bill.\" Nearest alternative style is `monolithic`: the claim's reach is across organisations, which is what the shared multi-party record provides.",
    ),
    "blockchain-based-credential-format-compile-select": (
        "blockchain-based",
        "Style evidence: \"A national digital-identity scheme: eleven banks, a government registry and four wallet vendors anchor credential schemas and revocation entries on a Hyperledger Indy network, so a verifier can check an issuer's public definition without asking that issuer whether it still exists.\" Nearest alternative style is `none`: the description names a deployed verifier service with a revocation data path, and the decision sits inside it.",
    ),
    "blockchain-based-dispensing-contract-assembly": (
        "blockchain-based",
        "Style evidence: \"five participants must endorse a dispensation before it is final, and no participant's operator can alter an entry once committed.\" Nearest alternative style is `microservices`: the assembly and implementation are packages inside each pharmacy's estate, not independently scalable network services.",
    ),
    "blockchain-based-driver-record-cluster-write": (
        "blockchain-based",
        "Style evidence: \"Thirty taxi cooperatives and four ride aggregators share trip assignment and fare settlement records on a Fabric channel;\" Nearest alternative style is `microservices`: these are members of one shared settlement network, not services of one product released separately.",
    ),
    "blockchain-based-fare-recompute-from-arguments": (
        "blockchain-based",
        "Style evidence: \"Twelve airlines, three online travel agencies and a clearing bank share ticket and passenger-record state on a Fabric channel, with the airline and the clearing bank endorsing any fare change.\" Nearest alternative style is `monolithic`: the rebooking service is one deployable per airline, and the style rests on the shared record that makes the fare final.",
    ),
    "blockchain-based-flexibility-unit-row-key": (
        "blockchain-based",
        "Style evidence: \"each operator's node holds the replica and a commitment is final when the operator and the aggregator endorse it.\" Nearest alternative style is `kappa-architecture`: the clearing window is a batch step inside one service, not a streaming path over reprocessed history.",
    ),
    "blockchain-based-inventory-access-no-rule-refusal": (
        "blockchain-based",
        "Style evidence: \"Each group runs validators, the onchain permissioning contract holds the account allowlist, and a commitment is final when the group and one agency address endorse it.\" Nearest alternative style is `rule-based-system`: the grants are not evaluated by a separate rules engine deciding behaviour; the question is the safe default of a check inside the access path.",
    ),
    "blockchain-based-meter-ingest-per-member-pool": (
        "blockchain-based",
        "Style evidence: \"Six grid operators meter settlement data on a shared network where each operator holds a replica and an endorsement from three of the six is required before a reading becomes final.\" Nearest alternative style is `microservices`: the ingest application is one deployable whose interior partitioning is the question, not a landscape of independently released services.",
    ),
    "blockchain-based-moderation-case-variants": (
        "blockchain-based",
        "Style evidence: \"Eight social platforms and a regulator share hashed signals about known illegal material on a permissioned network so each platform can act on a match without any of them exchanging the content itself.\" Nearest alternative style is `monolithic`: the workbench is one deployable per platform, so the style comes from the shared multi-party record rather than from the interior of that deployable.",
    ),
    "blockchain-based-recall-sweep-single-recursion": (
        "blockchain-based",
        "Style evidence: \"Nine hundred pharmacies, three wholesalers and a medicines regulator track batch movements on a shared network where a batch's history is authoritative only after the manufacturer and two wholesalers endorse it.\" Nearest alternative style is `lambda-architecture`: the recursion is one service's traversal of committed shared state, not a batch and speed split over the same data.",
    ),
    "blockchain-based-returns-desk-exemplar-copy": (
        "blockchain-based",
        "Style evidence: \"Forty independent retailers and one shared claims processor settle return-fraud checks against a common Besu network: twelve validator nodes, one per member, run QBFT so every member holds the same committed sales and returns record, and no member's operator can rewrite a settled entry.\" Nearest alternative style is `event-sourcing`: nothing here derives current state by replaying records; the committed shared ledger is the truth every member reads.",
    ),
    "blockchain-based-roaming-test-object-collaborator": (
        "blockchain-based",
        "Style evidence: \"A roaming settlement venture: thirty charge-point operators and twelve mobility providers settle charging sessions on a permissioned Ethereum network;\" Nearest alternative style is `microservices`: the settlement service is one Python deployable, so the style rests on the shared record the settlement posts to.",
    ),
    "blockchain-based-shipment-excursion-field-correspondence": (
        "blockchain-based",
        "Style evidence: \"The providers' integration service is Kotlin and persists a local projection of each shipment's excursion summary in Postgres, because the regulator's auditors query that projection directly and cannot be given network credentials.\" Nearest alternative style is `layered-monolith`: the service is one deployable, and the only layering mentioned is persistence code inside it.",
    ),
    "blockchain-based-tasking-order-operation-payload": (
        "blockchain-based",
        "Style evidence: \"Four satellite operators, twelve ground stations and two insurers coordinate imaging tasking and delivery on a Fabric channel;\" Nearest alternative style is `event-driven`: these messages are addressed to exactly one receiver and settle a record, not published for whoever reacts.",
    ),
    "blockchain-based-transplant-peer-capability-contract": (
        "blockchain-based",
        "Style evidence: \"Seven transplant centres and three organ-procurement organisations run a shared allocation network on Hyperledger Fabric.\" Nearest alternative style is `event-sourcing`: streaming committed events out of the peer is a transport convenience here, not the mechanism that makes the record authoritative.",
    ),
    "broker-billing-column-correspondence": (
        "broker",
        "Call records pass through one central hub with rating, invoicing and dispute programs subscribing independently.",
    ),
    "broker-booking-aggregate-creation": (
        "broker",
        "All booking events pass through one central hub whose three programs subscribe independently.",
    ),
    "broker-booking-hub-resource-lifetime": (
        "broker",
        "All itinerary changes pass through one central hub that three programs subscribe to independently.",
    ),
    "broker-catalogue-consumer-shutdown": (
        "broker",
        "Encode jobs are published into one central hub and three programs subscribe with worker pools, so every consumer reads the hub rather than being called.",
    ),
    "broker-claims-ordered-batch": (
        "broker",
        "Every claim document is published into one central hub that the adjudication, imaging and audit programs subscribe to, with no program calling another.",
    ),
    "broker-claims-subject-representation": (
        "broker",
        "All claim submissions pass through one central hub whose fraud, payment and audit programs subscribe independently.",
    ),
    "broker-diagnostics-acquire-release": (
        "broker",
        "The platform's work is distributed through one hub that every consumer subscribes to, with producers never calling consumers.",
    ),
    "broker-dispensing-policy-language": (
        "broker",
        "Prescription events pass through one central hub whose three programs subscribe independently.",
    ),
    "broker-freight-distinct-identifiers": (
        "broker",
        "Declarations are published into one central hub with duty, document and risk programs subscribing independently.",
    ),
    "broker-grid-flow-silence": (
        "broker",
        "All substation frames pass through one central hub whose three programs subscribe independently, so silence at the hub is observable to a supervisor.",
    ),
    "broker-hvac-standalone-check": (
        "broker",
        "Frames and commands travel through one central hub that the setpoint, alarm and reporting programs each subscribe to independently.",
    ),
    "broker-identity-hub-adapter": (
        "broker",
        "All events travel through one central hub with independent subscribers, and no service calls another directly.",
    ),
    "broker-maintenance-foreign-key": (
        "broker",
        "Machine events are published into one central hub that the planning, warranty and parts programs each subscribe to.",
    ),
    "broker-provisioning-message-shape": (
        "broker",
        "Change requests pass through one central hub that element, inventory and billing programs subscribe to independently.",
    ),
    "broker-sensor-hub-facade": (
        "broker",
        "Sensor batches travel through one central hub whose three programs subscribe independently, so no program calls another.",
    ),
    "broker-tax-form-fragments": (
        "broker",
        "Return submissions are published into one central hub whose validation, archival and refund programs subscribe independently.",
    ),
    "broker-ticketing-ast-nodes": (
        "broker",
        "Sale events travel through one central hub with seating, pricing and access programs subscribing independently.",
    ),
    "broker-till-shared-price-index": (
        "broker",
        "Sales and price changes are published into one central hub that the pricing, stock and loyalty programs subscribe to independently.",
    ),
    "broker-transit-validity-span": (
        "broker",
        "Tap events travel through one central hub whose fare, entitlement and settlement programs subscribe independently.",
    ),
    "command-query-responsibility-segregation-forecast-cluster-lock": (
        "command-query-responsibility-segregation",
        "Forecast runs are written through one model while the map layer, alerting feed and verification report are served from separate projections.",
    ),
    "command-query-responsibility-segregation-forecast-partial-availability": (
        "command-query-responsibility-segregation",
        "Forecast runs are written through one model while the map layer, alerting feed and verification report are read projections from separate paths.",
    ),
    "command-query-responsibility-segregation-freight-typed-groups": (
        "command-query-responsibility-segregation",
        "Shipments are written through one model while the manifest view, duty estimate and broker feed are served from separate read projections.",
    ),
    "command-query-responsibility-segregation-inventory-dual-approval": (
        "command-query-responsibility-segregation",
        "Adjustments are written through one model while the bin map and reorder list are read projections from a separate path.",
    ),
    "command-query-responsibility-segregation-loyalty-counter-stripes": (
        "command-query-responsibility-segregation",
        "Movements are written through one model while member statements and tier reports are served from separate read projections.",
    ),
    "command-query-responsibility-segregation-loyalty-inmemory-store": (
        "command-query-responsibility-segregation",
        "Movements are written through one model while statements and tier reports are served from separate read projections.",
    ),
    "command-query-responsibility-segregation-maintenance-impl-hiding": (
        "command-query-responsibility-segregation",
        "Work orders are written through one model while the planner board, parts forecast and compliance extract are read projections.",
    ),
    "command-query-responsibility-segregation-portitinerary-creation": (
        "command-query-responsibility-segregation",
        "Movements are written through one model while the yard map and customs extract read separate projections.",
    ),
    "command-query-responsibility-segregation-provisioning-test-structure": (
        "command-query-responsibility-segregation",
        "Change requests go through one write model while the circuit map, SLA report and billing extract are served from separate read projections.",
    ),
    "command-query-responsibility-segregation-review-nested-update": (
        "command-query-responsibility-segregation",
        "Matter changes go through one write model while the review board, deadline list and billing extract are read projections.",
    ),
    "command-query-responsibility-segregation-robot-transform-chain": (
        "command-query-responsibility-segregation",
        "Assignments go through one write model while the floor map, throughput report and maintenance queue are read projections.",
    ),
    "command-query-responsibility-segregation-roster-rule-specialisation": (
        "command-query-responsibility-segregation",
        "Rosters are written through one model while the crew app, payroll preview and regulator return are read projections.",
    ),
    "command-query-responsibility-segregation-scheduling-revocable-read": (
        "command-query-responsibility-segregation",
        "Appointment commands are handled by one write model while the board, reminder list and utilisation report are served from separate read projections, with no read path touching the write model.",
    ),
    "command-query-responsibility-segregation-settlement-write-batch": (
        "command-query-responsibility-segregation",
        "Trade writes are handled by one model and the blotter, statement and extract are served from separate read projections rather than from the write model.",
    ),
    "command-query-responsibility-segregation-transit-narrow-view": (
        "command-query-responsibility-segregation",
        "Fare and tap writes are handled by one model while the gate display, revenue report and fraud list read separate projections.",
    ),
    "command-query-responsibility-segregation-waste-lock-ownership": (
        "command-query-responsibility-segregation",
        "Collection writes go to one model while the depot board and recycling report are served from separate read projections.",
    ),
    "data-mesh-acquisition-order-model-separation": (
        "data-mesh",
        "Resources are owned team by team - acquisition requests, passes, cloud forecasts and downlink manifests 'each belong to the team that operates that resource', 'published with a schema and a freshness target', 'rebuilt by that team's own Airflow jobs' - and the centre runs only 'the storage, the catalog and the Trino endpoints, and none of the scheduling logic'. The nearest alternative is pipe-and-filter, since a tasking chain runs from request to downlink, but no chain of independent transforming stages is described; the decision is a model/storage separation inside one team's service.",
    ),
    "data-mesh-appliance-protocol-two-roles": (
        "data-mesh",
        "Step teams own their analytical output - identity outcomes, document results, eligibility decisions and channel events 'each belong to the team that operates that step', 'published with a schema and a quality expectation', 'rebuilt by that team's own jobs' - over a platform that 'builds no dataset'. The nearest alternative style is microservices, because an orchestrator and an appliance exchange requests, but the appliance is a third-party system rather than an independently released service of this product, and no fleet of owned deployables is described.",
    ),
    "data-mesh-availability-index-atomic-swap": (
        "data-mesh",
        "Team ownership of analytical output is explicit - rate plans, availability, reservation events and channel bookings 'each belong to the team that operates them', are 'published with a schema and a quality expectation', 'rebuilt by that team's own jobs' - while 'a platform group runs warehouses, the catalog and the sharing grants, and builds no models'. That is domain-owned analytical output over a shared plane, not a single central pipeline. The nearest alternative is space-based, since an in-memory index serves reads, but the index is one service's local cache, not a distributed memory tier that provides the scalability.",
    ),
    "data-mesh-claim-event-commit-atomicity": (
        "data-mesh",
        "The estate is federated by intake stage: registrations, checklists, referrals and assignments are each published, rebuilt and shape-controlled by the team that runs that stage, and 'a platform group operates the object store, the catalogue and the query endpoints, and writes no logic'. The tempting alternative is event-driven, because a Kafka topic and three consumers are named, but the carrier is not the decision: the question is how one service couples its own commit to its own announcement.",
    ),
    "data-mesh-console-region-fragment-assembly": (
        "data-mesh",
        "The estate is federated: fallout, circuit inventory, appointments and activation events 'each belong to the team that runs that part of the flow', are 'published with a schema and a freshness target', 'rebuilt by that team's own jobs', while the platform group supplies the plane 'and no transformations'. Product teams also own their own console fragments, which reinforces the same decentralised ownership. The nearest alternative style is model-view-controller, since a server-rendered console is described, but no controller-per-request-flow with a shared model is laid out; the load-bearing question is how one page is composed from separately owned pieces.",
    ),
    "data-mesh-container-event-wire-translation": (
        "data-mesh",
        "Ownership is federated by team and dataset - berth planning, yard moves, gate transactions and customs release 'each publish their own datasets with a schema and a freshness target, and are rebuilt by the owning team's jobs' - and the centre 'running storage, the catalog and the Trino endpoints but none of the logic' is a platform plane, not a pipeline. The tempting alternative is event-driven, since Kafka carries the events, but the decision is not about the carrier; it is about who owns translation between a team's domain types and a shared partner format.",
    ),
    "data-mesh-deductible-schedule-independence": (
        "data-mesh",
        "Datasets are owned per team - 'risk appetite, claims history, broker performance and exposure accumulation each publish a dataset with a schema and a freshness target and are rebuilt by the owning team's jobs' - while the platform group 'runs the metastore, the cluster policies and the catalog grants, and writes no transformation'. Several teams own their analytical output end to end, and the centre supplies only the plane. The nearest alternative is modular-monolith, since one Kotlin service is described, but nothing shows vertical modules inside one deployment; the decision is a class-local refactor in a service that fronts a federated estate.",
    ),
    "data-mesh-fare-combination-walk-shape": (
        "data-mesh",
        "The estate shows several domain teams owning their own analytical output - fare filings, inventory snapshots, ancillary products - each with 'a schema plus a freshness target' and its 'own jobs that materialise its Iceberg tables', while 'the platform plane supplies object storage, the catalog, the Trino endpoints and a shared Spark pool' and builds nothing. The nearest alternative style is pipe-and-filter, because the fare walk branches leg by leg, but no chain of independent stages transforming a stream is described; the decision concerns one search kernel inside one team-owned service.",
    ),
    "data-mesh-fleet-release-persistence-interface": (
        "data-mesh",
        "Ownership sits with the operations teams per dataset - container master, movements, demurrage, dwell 'each belong to the operations team responsible for that part of the fleet', are 'published with a schema and a quality expectation', 'rebuilt by that team's own Spark jobs' - while the platform group 'models nothing'. That is domain ownership of analytical output with a self-serve plane. The nearest alternative style is layered-monolith, since one JVM booking service with persistence is described, but no horizontal layer ordering is stated and the estate's defining structure is who owns which dataset.",
    ),
    "data-mesh-price-and-lead-time-as-of-date": (
        "data-mesh",
        "Dataset ownership is federated - batch records, lead times, price schedules and depot stock 'each belong to the team that operates that part of the supply chain', 'published with a schema and a quality expectation', 'rebuilt by that team's own jobs' - over a platform that supplies only storage, catalog and query endpoints. The nearest alternative style is layered-monolith, since a Rust service with a Postgres store is described, but no layer ordering is stated; the system's load-bearing shape is which team owns which analytical dataset.",
    ),
    "data-mesh-rate-lock-row-mapping": (
        "data-mesh",
        "Team ownership again decides the shape: funnel events, valuations, rate-lock history and underwriting decisions 'each belong to the team that operates that stage', are 'published with a schema and a freshness target', 'rebuilt by that team's own jobs', while a platform group supplies 'storage, the Trino fleet and the access policies'. The nearest alternative style is monolithic, since one origination service is named, but the decision concerns a mapping inside a service that writes into a federated analytical estate; no single-unit build/deploy claim is made for the estate.",
    ),
    "data-mesh-replenishment-step-sequencing": (
        "data-mesh",
        "Ownership is per team and per dataset: demand signals, lead times, stock positions and the promotion calendar 'each belong to the team that understands them', are 'published with a schema and a freshness target', 'rebuilt by that team's own dbt or .NET jobs', while the platform group runs storage, metastore and query endpoints and 'is deliberately not where any team's logic lives'. That is domain ownership of analytical output with a self-serve plane, not one central pipeline. The tempting alternative is pipe-and-filter because five ordered steps are named, but the steps are not independent streaming stages and the estate's shape is decided by team ownership.",
    ),
    "data-mesh-reply-matching-shared-queues": (
        "data-mesh",
        "Analytical output is owned by the operations team for each part of the flow - 'each belong to the operations team that owns that part of the flow', 'published with a schema and a freshness target', 'rebuilt by that team's own Spark jobs' - while the platform group runs catalog, query engine and compute 'and does not touch the transformations'. That is team ownership with a self-serve plane. The nearest alternative is microservices, since Spring Boot services on ECS are named, but the text does not claim they are released or scaled on their own; the load-bearing structure is the federated ownership of the datasets, with one routing estate beside them.",
    ),
    "data-mesh-scratch-buffer-lease-return": (
        "data-mesh",
        "Each team owns its own analytical output - fingerprints, review outcomes, retention decisions and regulator extracts 'each belong to the team that operates that stage', 'published with a schema and a freshness target', 'rebuilt by that team's own jobs' - while the platform group owns storage, metastore and query endpoints 'and no team's logic'. The nearest alternative is pipe-and-filter, since a batch passes through hashing and upload steps, but those steps share one rented buffer rather than forming a chain of stages that transform a stream.",
    ),
    "data-mesh-session-settlement-version-validate": (
        "data-mesh",
        "Ownership is per team and per analytical dataset: session telemetry, tariff schedules, asset health and settlement are 'each published with a schema and a freshness target and rebuilt by the owning team's own dbt or Flink jobs', while 'a platform group runs object storage, the catalog, the Trino fleet and the Kafka backbone, and owns none of the transformations'. Domain teams own their outputs end to end over a shared plane. The nearest alternative is saga, because a multi-step transaction across parties is described, but no compensating actions or external participants are in scope; the question is how one service validates a long read.",
    ),
    "data-mesh-solver-run-deferred-result": (
        "data-mesh",
        "Ownership is per team and per dataset - pairings, legality findings, standby lists and disruption events 'each belong to the team that operates that part of the roster', with a schema and a freshness target and 'that team's own jobs' - while the platform group runs storage, metastore and query endpoints and no roster logic. The nearest alternative style is modular-monolith, since one .NET application is described, but nothing shows vertical modules inside it; the estate's structure is the federated ownership, and the decision is about one call's lifetime.",
    ),
    "data-mesh-supplier-feed-grammar-functions": (
        "data-mesh",
        "Ownership is explicitly federated by team: catalogue, depot stock, freight commitments and backorder ageing are each published, rebuilt and shape-controlled by their own team, and 'the shared platform group keeps the object store, the catalogue service and the SQL endpoints running and never writes a transformation'. Four teams own their analytical output with a self-serve plane beneath them. The nearest alternative style is monolithic, since one Rust ingest binary exists, but it is a single component inside a federated estate, not the whole system's build unit.",
    ),
    "data-mesh-tile-buffer-shared-until-write": (
        "data-mesh",
        "Ownership is per team and per dataset - raw downlink scenes, calibration coefficients, cloud masks and derived indices 'each belong to the team that produces them', are 'published with a schema and a quality expectation', 'rebuilt by that team's own Spark jobs' - and the centre runs 'the object store, the catalog, the Trino endpoints and the GPU pool, and none of the algorithms'. Several domain teams own analytical output with a self-service platform beneath. The nearest alternative is pipe-and-filter, since an ingest service feeds several derived products, but the products are sibling consumers of one buffer, not a chain of stages.",
    ),
    "data-mesh-work-order-durable-hops": (
        "data-mesh",
        "Four operational areas own their own analytical output - 'each team publishes its own tables with a schema and a freshness target to the catalogue and rebuilds them in its own jobs' - while the platform group keeps storage, catalogue and SQL endpoints healthy 'and writes no transformation'. The nearest alternative style is enterprise-service-bus, since a central bridge mediates between heterogeneous depot systems, but this bridge translates nothing and owns no routing policy; it is a hop-by-hop transport discipline, and the analytical estate is what the teams own.",
    ),
    "edge-computing-cab-unit-incident-fact-publication": (
        "edge-computing",
        "Detection and reaction run on the vehicle gateway in each cab and on the depot gateway, because coverage is patchy and the record must exist locally before any uplink \u2014 the decision is taken at the vehicle, not in a central service. Nearest alternative is event-driven: the facts are indeed published, but the carrier is a unit-local queue with in-process reactors, and the load-bearing structure is a unit per cab that decides and records without the centre.",
    ),
    "edge-computing-campus-portal-region-composition": (
        "edge-computing",
        "The page is assembled on a node in each campus closet, close to the students and far from the central student system, because a round trip to the centre misses the latency budget. Nearest alternative is model-view-controller, and a single controller per page would be plausible \u2014 but the load-bearing mechanism here is region composition across reusable pieces, and the placement is what makes it necessary.",
    ),
    "edge-computing-charging-feeder-claim-before-session": (
        "edge-computing",
        "Session admission is decided on the operator-owned site controller with the central system unreachable for minutes at a time, and the local claim is what makes that decision sound. Nearest alternative is monolithic: each site runs one controller program, but admission is decided on nine hundred autonomous controllers rather than in the centre, which is the placement being described.",
    ),
    "edge-computing-custodian-instruction-type-tables": (
        "edge-computing",
        "The validation runs on a box at each custodian, against that custodian's own local data, before a nightly file drop to the central engine. Nearest alternative is hybrid-cloud, but no estate split is the subject; placement of the check beside the custodian's data is.",
    ),
    "edge-computing-forecourt-authorisation-step-chain": (
        "edge-computing",
        "The authorisation decision is taken on the forecourt controller when the link is down \u2014 'the box, not the acquirer, holds the floor limit' \u2014 and results are forwarded and reconciled afterwards. Nearest alternative is monolithic, but the described system is a fleet of site controllers acting locally, not one deployable serving all sites.",
    ),
    "edge-computing-forecourt-batch-position-marking": (
        "edge-computing",
        "The batch boundary is decided at each forecourt controller, on equipment the processor does not schedule, and results are forwarded to the central settle path once complete. Nearest alternative is broker, but nothing routes through a central component that owns the wiring; site controllers push over their own links.",
    ),
    "edge-computing-kiosk-device-handle-scope": (
        "edge-computing",
        "Document and face matching is decided on the post-office kiosk itself, against its own hardware, with only the outcome forwarded over a shared 4G modem. Nearest alternative is none: a lifetime question about one value, but the matching decision runs on hardware at two hundred counters, which is the placement.",
    ),
    "edge-computing-lab-gateway-instrument-contract-handle": (
        "edge-computing",
        "The deciding work sits on boxes in each hospital laboratory, on hardware the pathology network does not schedule, and results are forwarded only after a local validation decision \u2014 'validated results must leave the box within thirty seconds, buffering locally'. Nearest alternative is serverless, but these modules are long-lived containers on hospital-owned machines, not provider-managed functions scaled to zero.",
    ),
    "edge-computing-machine-footprint-single-unit-of-work": (
        "edge-computing",
        "The deciding and recording happen on a gateway in the plant's control room, installed and operated by the plant, with the corporate system reached only over a thin satellite link and receiving one forwarded bundle per change. Nearest alternative is hybrid-cloud, but no split between estates is the subject; placement near the machines is, and the upstream system is one remote consumer.",
    ),
    "edge-computing-office-box-ledger-lazy-scan": (
        "edge-computing",
        "The return is computed on each accounting office's own PC, against an archive that may not leave the premises, rather than at a central service. Nearest alternative is monolithic: one program runs per office, but the regulator-forbidden round trip is what puts the computation there, and that constraint is the placement.",
    ),
    "edge-computing-saleroom-lot-record-single-document": (
        "edge-computing",
        "Bid acceptance and lot closure are decided at the saleroom console, on the house's own equipment, and only the result is forwarded \u2014 'committed to the console's local store before anything is forwarded'. Nearest alternative is monolithic, if a console is read as one deployable; the eight salerooms holding their own lot state through a live sale are the placement.",
    ),
    "edge-computing-site-controller-absent-session": (
        "edge-computing",
        "The lookup serves the attendants' tablet from a box in the car park while the operator's cloud is unreachable, so the answer is produced next to the equipment. Nearest alternative is none: a signature question inside one lookup, but the answer must be produced without the centre, which is the placement.",
    ),
    "edge-computing-site-session-history-page-token": (
        "edge-computing",
        "Session history lives on a controller at each car park, on the operator's own hardware, and is served from there rather than from a central store \u2014 'the site must show ... their recent sessions even when the uplink is saturated'. Nearest alternative is none, if the reader sees only an API design question; the per-site serving of local data is the placement.",
    ),
    "edge-computing-site-tablet-dose-entry-primitive": (
        "model-view-controller",
        "Adjudicated to model-view-controller: one behaviour (widget state plus event handling) is shared by two presentations that each render their own markup, which is a model/controller split with a view per shell. presentation-abstraction-control would need a hierarchy of cooperating presentation, abstraction and control agents per UI unit, and only the two shells are distinct presentations here. Recorded as a rubric tension: the authoring brief asked for edge-computing and neither annotator found placement near the data source load-bearing, so this row does not evidence its planned style.",
    ),
    "edge-computing-station-ingest-dependency-seam": (
        "edge-computing",
        "Quality control runs on a box beside the instruments at each site and observations are forwarded only after local acceptance \u2014 'applies local quality control ... holding them locally when the satellite link is down'. Nearest alternative is monolithic: one daemon per station, but the acceptance decision at two hundred and forty sites is what the row describes.",
    ),
    "edge-computing-ward-hub-snapshot-state": (
        "edge-computing",
        "Alert decisions are made on a hub box in each ward and only a nightly summary is forwarded, so the deciding state and its readers sit together beside the patients. Nearest alternative is event-driven, because readers consume published updates \u2014 but the publisher is one in-process state holder and the readers are local modules, not independent subscribers with their own offsets.",
    ),
    "edge-computing-zone-plan-commit-exchange": (
        "edge-computing",
        "Setpoints are decided on a plant-room box every thirty seconds with the cloud treated as a convenience and a locally held plan as the live state \u2014 'the zone must keep its previous plan'. Nearest alternative is none: the commit path is a class-design question, but the values it commits are decided in each plant room without the centre, and that local decision is the system's purpose.",
    ),
    "enterprise-service-bus-banking-onboarding-audit": (
        "enterprise-service-bus",
        "Every integration between fourteen native systems passes through one mediation tier that translates protocols and formats and routes by content, and no pair of systems is wired directly.",
    ),
    "enterprise-service-bus-billing-record-filtering": (
        "enterprise-service-bus",
        "Vendor switches, the rating engine and the billing ERP are integrated only through one mediation tier that translates each vendor format.",
    ),
    "enterprise-service-bus-card-payment-view-ownership": (
        "command-query-responsibility-segregation",
        "Both blind annotators labelled this command-query-responsibility-segregation where the authoring intent was enterprise-service-bus",
    ),
    "enterprise-service-bus-catalogue-deadline-carriage": (
        "enterprise-service-bus",
        "Five heterogeneous partners and systems meet at one mediation tier that terminates each protocol and routes entitlement checks.",
    ),
    "enterprise-service-bus-charging-queue-dispatch": (
        "enterprise-service-bus",
        "The five networks, payment providers and the grid operator are joined only through one mediation tier that terminates each protocol.",
    ),
    "enterprise-service-bus-claims-frame-stepping": (
        "enterprise-service-bus",
        "The platform's external systems integrate only through one mediation tier that terminates each protocol.",
    ),
    "enterprise-service-bus-claims-sync-async-bridge": (
        "enterprise-service-bus",
        "The clearing houses, imaging and fraud systems integrate only through one mediation tier that terminates each native protocol and translates claim formats.",
    ),
    "enterprise-service-bus-diagnostics-narrow-tool-view": (
        "enterprise-service-bus",
        "Every diagnostic protocol and every back-office system is integrated through one mediation tier that terminates the native protocol and routes by VIN.",
    ),
    "enterprise-service-bus-enrolment-generated-structure": (
        "enterprise-service-bus",
        "The student-record system, fees engine and reporting gateway integrate only through one mediation tier that translates each protocol.",
    ),
    "enterprise-service-bus-hospital-result-translation": (
        "enterprise-service-bus",
        "Eleven hospital systems on six protocols and formats integrate only through one mediation backbone that terminates, translates and routes, with no pair wired directly.",
    ),
    "enterprise-service-bus-inventory-role-context": (
        "enterprise-service-bus",
        "The platform's five native systems are integrated only through one mediation tier that terminates each protocol and routes by item id.",
    ),
    "enterprise-service-bus-moderation-edit-claim": (
        "enterprise-service-bus",
        "The platform's feeds, classifier and regulator gateway integrate only through one mediation tier that translates and routes.",
    ),
    "enterprise-service-bus-moderation-table-module": (
        "enterprise-service-bus",
        "The five feeds, the classifier and the reporting gateway integrate only through one mediation tier that translates and routes.",
    ),
    "enterprise-service-bus-provisioning-boundary-restriction": (
        "enterprise-service-bus",
        "Forty heterogeneous network elements and three inventory systems integrate only through one mediation tier that translates each native protocol and routes by circuit.",
    ),
    "enterprise-service-bus-review-lazy-associations": (
        "enterprise-service-bus",
        "Documents, billing, court gateways and retention are joined only through one mediation tier that terminates each native protocol.",
    ),
    "enterprise-service-bus-review-tree-focus": (
        "enterprise-service-bus",
        "Every integration between the document system, citator and billing passes through one mediation tier that terminates each protocol.",
    ),
    "enterprise-service-bus-sensor-prototype-templates": (
        "enterprise-service-bus",
        "Every sensor vendor and both external systems are integrated through one mediation tier that terminates each vendor payload.",
    ),
    "enterprise-service-bus-telematics-store-family": (
        "enterprise-service-bus",
        "Telematics, maintenance and two insurance systems integrate only through one mediation tier that translates each protocol and routes by fleet.",
    ),
    "event-driven-catalogue-now-or-never": (
        "event-driven",
        "The workers react to published transcode events at their own offset rather than being called by the producer.",
    ),
    "event-driven-claims-timebound-authority": (
        "event-driven",
        "Fraud, payment and audit each consume the same published claim events independently with their own offsets, so the record is the coordination medium.",
    ),
    "event-driven-diagnostics-typed-groups": (
        "event-driven",
        "Diagnostic events are published once and each program consumes them at its own offset, never by direct call.",
    ),
    "event-driven-dispatch-restricted-publish": (
        "event-driven",
        "Producers publish onto topics consumed independently by three programs, so the flow is asynchronous publication with no direct consumer calls.",
    ),
    "event-driven-dispensing-host-extensions": (
        "event-driven",
        "Prescriptions are published once onto topics and each program consumes them at its own offset.",
    ),
    "event-driven-forecast-optional-capability": (
        "event-driven",
        "Forecast runs are published once onto topics and the three programs consume them independently at their own offsets.",
    ),
    "event-driven-freight-capacity-isolation": (
        "event-driven",
        "Events are published onto topics and each program consumes them at its own offset without direct invocation.",
    ),
    "event-driven-hvac-sensitive-frames": (
        "event-driven",
        "Every frame and command is published onto a topic once and three programs consume it independently at their own offsets, with no program calling another.",
    ),
    "event-driven-imagery-consumer-category": (
        "event-driven",
        "Every scene is published once onto a topic and each consumer reacts at its own offset without the producer knowing it.",
    ),
    "event-driven-imagery-consumer-enhancement": (
        "event-driven",
        "Each program reacts to published scene events at its own offset rather than being invoked by the producer.",
    ),
    "event-driven-maintenance-event-loop": (
        "event-driven",
        "Programs receive machine events from published topics at their own offsets, never by being invoked by the producer.",
    ),
    "event-driven-origination-contract-unit": (
        "event-driven",
        "Internal consumers read published underwriting events at their own offsets, so the platform is coordinated by published record.",
    ),
    "event-driven-provisioning-chain-owner": (
        "event-driven",
        "The three programs each consume published change events at their own offset, so coordination happens through the event record.",
    ),
    "event-driven-ride-position-flow": (
        "event-driven",
        "Positions are published once onto a partitioned stream and three programs consume them independently with their own offsets, so coordination is by published records rather than by direct calls.",
    ),
    "event-driven-sensor-state-snapshot": (
        "event-driven",
        "Each program consumes published sensor events at its own offset, so the platform is coordinated by the event record rather than by calls.",
    ),
    "event-driven-telematics-row-mapping": (
        "event-driven",
        "Events are published once and consumed independently by each program at its own offset.",
    ),
    "event-driven-transit-concurrent-writes": (
        "event-driven",
        "Each program reads the same published tap events at its own offset and acts independently, which is coordination through the event record rather than direct invocation.",
    ),
    "event-sourcing-diagnostic-serialized-payload": (
        "event-sourcing",
        "The workshop view and fault statistics are projections replayed from the appended diagnostic facts, which are the only authoritative record.",
    ),
    "event-sourcing-dispatch-contract-replay": (
        "event-sourcing",
        "The route view, labour report and promise feed are all replays of the appended dispatch facts, and the contract work happens on the API over them.",
    ),
    "event-sourcing-fleet-consistency-boundary": (
        "event-sourcing",
        "The map and maintenance views are rebuilt by replaying telematics facts, and the write side is being restructured around that record.",
    ),
    "event-sourcing-maintenance-rule-language": (
        "event-sourcing",
        "Warranty and planning views are projections replayed from the appended maintenance facts, which remain the authoritative source.",
    ),
    "event-sourcing-port-ingress-validation": (
        "event-sourcing",
        "The yard map and demurrage ledger are rebuilt by replaying the appended movement facts, with the log the authoritative record.",
    ),
    "event-sourcing-review-document-envelope": (
        "event-sourcing",
        "Matter state exists only as an appended history of domain facts, with every served view rebuilt from that history and nothing else holding authoritative state.",
    ),
    "event-sourcing-roster-history-mapping": (
        "event-sourcing",
        "Compliance and payroll views are rebuilt by replaying appended rostering facts, so derived state comes from the record rather than from mutable rows.",
    ),
    "event-sourcing-schedule-projection-cache": (
        "event-sourcing",
        "Every screen is served from projections rebuilt by replaying appended appointment facts, which are the only authoritative state.",
    ),
    "event-sourcing-sensor-fact-routing": (
        "event-sourcing",
        "Yield and irrigation views are rebuilt by replaying the appended field facts, so the log carries every fact the projections need.",
    ),
    "event-sourcing-settlement-temporal-versions": (
        "event-sourcing",
        "Reportable state is derived by replaying the appended settlement facts, and each report must reflect the instruction as it stood on its date.",
    ),
    "event-sourcing-subscription-swap-handles": (
        "event-sourcing",
        "The invoice run and revenue report are projections replayed from appended subscription facts, and nothing else holds authoritative account state.",
    ),
    "event-sourcing-telematics-bulk-transfer": (
        "event-sourcing",
        "The fleet map and forecast are projections replayed from the appended telematics log, with the log authoritative for every derived figure.",
    ),
    "event-sourcing-ticket-history-lazy": (
        "event-sourcing",
        "Seat availability and settlement figures come from projections replayed over the appended ticketing history, not from mutable current-state rows.",
    ),
    "event-sourcing-till-claim-check": (
        "event-sourcing",
        "The shift report and stock ledger are rebuilt by replaying the store's appended sale facts, which are the authoritative record.",
    ),
    "event-sourcing-trade-projection-transform": (
        "event-sourcing",
        "The blotter, statement and extract are all projections replayed from the appended trade log, and any of them can be rebuilt from it.",
    ),
    "half-sync-half-async-auction-context-scope": (
        "half-sync-half-async",
        "Bids are handled synchronously while settlement checks are queued to a worker tier, so the blocking and asynchronous layers meet at the queue.",
    ),
    "half-sync-half-async-diagnostics-vehicle-table": (
        "half-sync-half-async",
        "A synchronous acceptance tier is bridged to an asynchronous analysis worker by a queue, which is the two-layer structure the worker sits in.",
    ),
    "half-sync-half-async-dispatch-audit-chain": (
        "half-sync-half-async",
        "Synchronous command handling is bridged to a worker tier by a queue, which is the two-layer structure this system is built on.",
    ),
    "half-sync-half-async-grid-invalid-frame-store": (
        "half-sync-half-async",
        "The substation gateway answers synchronously while processing runs on a queue-fed worker tier, so the two layers are joined at a queue hand-off.",
    ),
    "half-sync-half-async-grid-measurement-units": (
        "half-sync-half-async",
        "Samples cross from a synchronous acceptance tier to a conversion worker through a queue, an explicit bridge between a blocking front and an asynchronous back.",
    ),
    "half-sync-half-async-hospital-record-cloning": (
        "half-sync-half-async",
        "Synchronous admission handling is bridged to a queue-fed preparation tier, and the cloning decision sits on the asynchronous side of that bridge.",
    ),
    "half-sync-half-async-hotel-folio-extras": (
        "half-sync-half-async",
        "Bookings are answered synchronously while postings are processed asynchronously by a worker tier reached through a queue, an explicit two-layer bridge.",
    ),
    "half-sync-half-async-hotel-session-identity": (
        "half-sync-half-async",
        "The blocking front-desk tier and the asynchronous folio tier are connected by a queue, so the two layers meet at an explicit hand-off rather than calling each other directly.",
    ),
    "half-sync-half-async-hvac-connection-acceptance": (
        "half-sync-half-async",
        "Connection handlers serve requests synchronously and hand longer work to a worker tier through a queue, so the synchronous and asynchronous layers are joined at one explicit hand-off.",
    ),
    "half-sync-half-async-maintenance-lease-grant": (
        "half-sync-half-async",
        "Submissions are accepted synchronously and planning runs on a worker tier reached through a queue, an explicit boundary between the blocking and asynchronous layers.",
    ),
    "half-sync-half-async-procurement-quote-bridge": (
        "half-sync-half-async",
        "The portal is request-response with thread-per-request handling while the processing tier is an event loop, and the two are joined by a queue whose job is to absorb bursts rather than let requests block.",
    ),
    "half-sync-half-async-provisioning-authorised-call": (
        "half-sync-half-async",
        "A synchronous acceptance tier hands work to a worker tier through a queue, and the restricted operation executes on the asynchronous side of that bridge.",
    ),
    "half-sync-half-async-replenishment-long-orders": (
        "half-sync-half-async",
        "Request threads never block on queue work: a synchronous tier hands work over a queue to a worker pool and collects outcomes later, which is the bridge the description turns on.",
    ),
    "half-sync-half-async-sensor-rule-objects": (
        "half-sync-half-async",
        "The ingest endpoint acknowledges synchronously and a worker tier consumes queued batches, an explicit synchronous-to-asynchronous bridge with the queue as its load-bearing element.",
    ),
    "half-sync-half-async-ticketing-seat-kinds": (
        "half-sync-half-async",
        "Five sales channels are answered synchronously while allocation runs on a worker tier fed by a queue, so synchronous acceptance and asynchronous work meet at a queue hand-off.",
    ),
    "half-sync-half-async-till-journal-walk": (
        "half-sync-half-async",
        "Till requests are served synchronously while a worker tier consumes queued journal entries, the explicit bridge this description is built around.",
    ),
    "half-sync-half-async-till-page-assembly": (
        "half-sync-half-async",
        "The render path answers inline while a worker tier prepares fragments that arrive through a hand-off queue, so a synchronous front and an asynchronous back are bridged by the queue rather than by blocking calls.",
    ),
    "half-sync-half-async-trial-visit-neighbourhood": (
        "half-sync-half-async",
        "The console serves requests synchronously while spatial analysis is queued to a worker tier, an explicit bridge between blocking and asynchronous processing.",
    ),
    "hexagonal-blazor-dispense-state-passing": (
        "hexagonal",
        "Style evidence: the dispensing core is 'a plain C# library with no reference to ASP.NET, Entity Framework or the printer SDK' and declares three boundaries with server-side adapters, one of which is a fake in tests; one process is deployed. Nearest alternative is layered-monolith, but there is no horizontal request-to-persistence chain - the core is framework-free and external concerns arrive through adapters it names.",
    ),
    "hexagonal-call-detail-rating-transform-stack": (
        "hexagonal",
        "Style evidence: the rating core has 'no Kafka client, no JDBC and no file-format library' and declares three boundaries with adapters for Kafka, a subscriber cache and a Parquet writer, plus a fixture replay path. Nearest alternative is pipe-and-filter, because transforms are applied in sequence, but the text shows no chain of independently running stages - it shows one core owning boundaries and a composed value.",
    ),
    "hexagonal-charging-session-channel-endpoint": (
        "hexagonal",
        "Style evidence: settlement rules 'import only the standard library; no MQTT client, no SQL driver and no HTTP package appears', with three declared interfaces and adapters for PostgreSQL, a tariff API and OCPP. Nearest alternative is event-driven, given published topics, but no independent consumers with their own offsets are described; the mechanism is one core behind owned boundaries plus a translating endpoint.",
    ),
    "hexagonal-counsel-review-operation-primitives": (
        "hexagonal",
        "Style evidence: the review core 'imports no Spring, no JDBC and no PDF library', declares four boundaries, and each adapter is 'replaceable with fixtures in the rule tests' inside one service process. Nearest alternative is layered-monolith, which would require horizontal stages calling each other; here the core reaches out only through boundaries it owns, and no layer ordering is described.",
    ),
    "hexagonal-dispensing-formulary-vendor-retry": (
        "hexagonal",
        "The style evidence is structural and named: rules classes with 'no package reference to ASP.NET, Dapper or any HTTP library', four interfaces the worker declares, adapters for REST, SOAP, a printer and the clock, and a stub adapter used by tests. One worker process, so not independently deployed services; the nearest alternative, layered-monolith, would show request-path layers calling each other, which this text never does.",
    ),
    "hexagonal-fare-search-expression-interpreter": (
        "hexagonal",
        "Style evidence: the search core 'imports no web framework, no ORM and no vendor SDK' and declares three boundaries whose adapters own their own schemas, with the core's vocabulary independent of each. Nearest alternative is layered-monolith, but no layer chain is described; the text separates a core from replaceable external interpretations, which is the boundary discipline being labelled.",
    ),
    "hexagonal-freight-document-operation-registry": (
        "hexagonal",
        "Style evidence: the core carries 'no Spring annotations, no JPA and no EDI library on the classpath' and declares four boundaries, with adapters for a rate database, a broker API, a carrier API and an EDI mailbox, plus separately compiled modules registered at start-up. Nearest alternative is microkernel - extensions do register at start-up - but the extensions here extend domain operations over a framework-free core rather than add capability to a minimal product core, and the core's port ownership is what the text establishes.",
    ),
    "hexagonal-legacy-report-dependency-registry": (
        "hexagonal",
        "Style evidence: the maintenance core has 'zero references to the web framework, the time-series client or the plant protocol stack' and declares the four boundaries, with adapters for PostgreSQL, SAP, OPC UA and InfluxDB that are substitutable. Nearest alternative is monolithic - one deployable is true - but the load-bearing structure is a framework-free core reached only through interfaces it owns, so the one-deployable label alone would miss the mechanism.",
    ),
    "hexagonal-mediation-provider-interaction-fixtures": (
        "hexagonal",
        "Style evidence: the mediation core 'imports no Spring, no JMS client and no vendor SDK', declares three boundaries with AMQP, subscriber and SOAP adapters faked in tests, and the system under decision is 'one Java 21 deployable, released as a unit'. Nearest alternative is microservices, because a second team's tool consumes it, but the rubric counts the deployables of the system being decided: one here, with the consumer's separate release being the reason the recorded interactions exist.",
    ),
    "hexagonal-procurement-tender-deadline-queries": (
        "hexagonal",
        "Style evidence: a framework-free evaluation core with three declared boundaries and adapters for SQL Server, a supplier API and an approval workflow, in one service process. Nearest alternative is monolithic, since one deployable is described, but the text makes the core's independence and its owned boundaries load-bearing, which is more than a plain one-unit build.",
    ),
    "hexagonal-purchase-order-single-procedure": (
        "hexagonal",
        "Style evidence: the business core 'references neither the web framework, nor Dapper, nor the ERP client' and reads and writes through four boundaries it declares, with adapters for the procurement database, the ERP API, the ledger and an audit store. Nearest alternative is layered-monolith, but the text shows no layer calling the next - it shows a core behind interfaces it owns, inside one service process.",
    ),
    "hexagonal-refill-authorisation-domain-naming": (
        "hexagonal",
        "Style evidence: the logic package 'imports no Spring, no JDBC and no printer library' and depends on three boundaries it declares, with adapters for PostgreSQL, a clearinghouse API and an SMS gateway. Nearest alternative is monolithic, since one process is described, but the text makes the framework-free package and its owned boundaries the structure, and the decision is about naming inside it.",
    ),
    "hexagonal-regional-broadband-plan-selection": (
        "hexagonal",
        "The style evidence is the framework-free business core: 'a package of plain Java 21 classes that imports nothing from Spring, Jakarta or the NETCONF client', three boundaries it declares, adapters that can be faked in the build, and one deployable. Nearest alternative is layered-monolith, but nothing here calls layer to layer - the core reaches outside only through interfaces it owns, which is what a reader must use to separate the two.",
    ),
    "hexagonal-streaming-catalogue-inspection-queue": (
        "hexagonal",
        "The description makes the style structural: 'The core is a plain Java module ... whose import list holds no Spring, no JDBC and no vendor SDK', with 'every outside concern ... behind an interface the core declares' and three named adapters that can be swapped for fixtures. One deployable, so not independently deployed services; the nearest alternative is layered-monolith, but calls do not flow presentation to domain to persistence - they enter a framework-free core through boundaries the core owns.",
    ),
    "hexagonal-substation-snapshot-unit-of-work": (
        "hexagonal",
        "Style evidence: the aggregate 'imports nothing beyond the standard library; no SQL driver, no MQTT client, no HTTP package appears in its imports', with three interfaces it declares and adapters for JSONB, MQTT and the asset API, in one binary per region. Nearest alternative is monolithic, which the one-binary fact supports, but the text's load-bearing structure is the framework-free core behind interfaces it owns.",
    ),
    "hexagonal-tender-pricing-specialisation": (
        "hexagonal",
        "Style evidence: the pricing core is 'a module of pure functions ... with no reference to the web host, the estimating database or the ERP connector', declaring three boundaries with adapters for the index warehouse, a tax service and a treasury feed. Nearest alternative is layered-monolith, but the text shows no horizontal call chain - it shows a core behind interfaces it owns plus a composition-time decision.",
    ),
    "hexagonal-ticketing-request-drain-loop": (
        "hexagonal",
        "Style evidence: the ledger 'imports only the standard library: no SQL driver, no gRPC package, no payment SDK', with four declared interfaces and adapters for PostgreSQL, a processor, turnstiles and an audit stream, in one binary per venue. Nearest alternative is monolithic, which the single binary satisfies, but the load-bearing structure is the framework-free ledger behind boundaries it owns, with a queued invocation decision inside it.",
    ),
    "hexagonal-trip-offer-self-describing-unit": (
        "hexagonal",
        "Style evidence: the matching core has 'no web framework, no database driver and no queue client on its import list' and declares four boundaries, with adapters for PostGIS, a presence store, a pricing service and a push gateway. Nearest alternative is event-driven, since offers do travel to a push gateway, but the text never shows several consumers reacting on their own offsets; it shows one core behind boundaries it owns deciding a payload's shape.",
    ),
    "hexagonal-utility-interval-store-family": (
        "hexagonal",
        "Style evidence: the ledger 'imports only the standard library, never a database driver', the three interfaces it defines, and adapters for PostgreSQL, TimescaleDB, CSV and the coming InfluxDB writer, all inside one binary. Nearest alternative is monolithic - one deployable is true - but the description shows an interior boundary discipline where the core owns the interfaces and every external concern is reached through an adapter, so the deciding style is not the plain one-deployable label.",
    ),
    "hybrid-cloud-adjudication-behaviour-test-shape": (
        "hybrid-cloud",
        "Style evidence: \"A motor insurer adjudicates claims on two estates: the claim store, document images and the policy master sit in the insurer's own data centre under a data-residency condition, while the adjudication engine, rules service and payment orchestration run in one public cloud region.\" Nearest alternative style is `multi-cloud`: one public provider appears, and the load-bearing boundary is premises versus it.",
    ),
    "hybrid-cloud-booking-window-refuse-on-state": (
        "hybrid-cloud",
        "Style evidence: \"A container terminal runs crane and straddle-carrier control on servers inside the terminal, where a command must reach a machine within 30 milliseconds and must keep working when the terminal's link to the internet is cut;\" Nearest alternative style is `edge-computing`: the terminal servers are the operator's own fixed estate, and the decision is where state lives and how it refuses, not proximity to users.",
    ),
    "hybrid-cloud-capacity-allocation-operation": (
        "hybrid-cloud",
        "Style evidence: \"A network service provider keeps its inventory of ducts, fibre and wavelength capacity in an on-premises OSS because the regulator requires the record of installed plant to remain in-country, and runs order orchestration and customer-facing APIs in one public cloud region.\" Nearest alternative style is `multi-cloud`: only one public provider is in play.",
    ),
    "hybrid-cloud-cloud-signer-before-first-use": (
        "hybrid-cloud",
        "Style evidence: \"A pharmaceutical cold-chain operator keeps sensor gatewaying in a validated on-premises data centre and runs analytics and alerting in one public cloud region.\" Nearest alternative style is `multi-cloud`: a single public provider appears, and the boundary that matters is premises versus that provider.",
    ),
    "hybrid-cloud-dispatch-fallible-step-values": (
        "hybrid-cloud",
        "Style evidence: \"the rider app, pricing experiments and cross-city analytics run in one public cloud region.\" Nearest alternative style is `multi-cloud`: one public provider appears, and the split that matters is the city data centres versus it.",
    ),
    "hybrid-cloud-downlink-concurrency-permits": (
        "hybrid-cloud",
        "Style evidence: \"A satellite ground-segment operator moves data on two estates: downlink ingest and frame decoding run at the ground station on the operator's own servers, because raw telemetry may not leave the site until it is calibrated, and calibration, archive and customer delivery run in one public cloud region.\" Nearest alternative style is `edge-computing`: the station server is the operator's fixed facility, and the question is the concurrency bound inside it.",
    ),
    "hybrid-cloud-drawing-payload-redeem-ticket": (
        "hybrid-cloud",
        "Style evidence: \"Drawings, model files and inspection evidence stay in the contractor's on-premises object store because the project agreements forbid the models leaving the site network;\" Nearest alternative style is `multi-cloud`: one public provider, and the split that matters is premises versus it.",
    ),
    "hybrid-cloud-feeder-ownership-mailboxes": (
        "hybrid-cloud",
        "Style evidence: \"A distribution network operator collects substation telemetry at four regional data centres on its own premises and runs forecasting and market-facing aggregation in one public cloud region.\" Nearest alternative style is `multi-cloud`: one public provider appears, and the load-bearing divide is premises versus that provider.",
    ),
    "hybrid-cloud-guest-page-fragment-regions": (
        "hybrid-cloud",
        "Style evidence: \"A hotel group runs its property management system in its own data centre and its booking site in one public cloud region.\" Nearest alternative style is `multi-cloud`: one public provider, and the load-bearing split is premises versus it.",
    ),
    "hybrid-cloud-lab-dependency-breaker-state": (
        "hybrid-cloud",
        "Style evidence: \"A clinical laboratory network runs instrument integration and result validation on servers inside each hospital, because unvalidated results may not leave the hospital;\" Nearest alternative style is `multi-cloud`: one public provider, and the split that matters is hospital premises versus it.",
    ),
    "hybrid-cloud-loyalty-issuer-override-test": (
        "hybrid-cloud",
        "Style evidence: \"A coalition loyalty programme runs its points ledger in its own data centre, where the balance of record must stay under national accounting rules, and runs campaign management and the member app in one public cloud region.\" Nearest alternative style is `multi-cloud`: one public provider, with the premises-versus-provider split carrying the constraint.",
    ),
    "hybrid-cloud-mediation-config-named-setters": (
        "hybrid-cloud",
        "Style evidence: \"A telecom operator runs call-detail-record mediation in its own data centre, because law requires records to remain in-country, and runs rating and the invoice service in one public cloud region.\" Nearest alternative style is `multi-cloud`: one public provider is present, and the boundary that matters is premises versus it.",
    ),
    "hybrid-cloud-provisioning-order-single-flush": (
        "hybrid-cloud",
        "Style evidence: \"A network service provider splits its estate: the inventory of installed plant stays in an on-premises Postgres cluster that the regulator inspects, while order management and the customer portal run in one public cloud region.\" Nearest alternative style is `multi-cloud`: a single public provider, with the boundary between premises and it carrying the decision.",
    ),
    "hybrid-cloud-review-console-two-role-ipc": (
        "hybrid-cloud",
        "Style evidence: \"The human review console, the audit archive and the appeals queue run in one public cloud region, reached over a dedicated interconnect on a private address range.\" Nearest alternative style is `multi-cloud`: only one public provider appears, and the load-bearing split is premises versus that provider.",
    ),
    "hybrid-cloud-rights-record-designated-fields": (
        "hybrid-cloud",
        "Style evidence: \"the catalogue API and the client-facing search index run in one public cloud region.\" Nearest alternative style is `multi-cloud`: a single public provider appears, and the load-bearing split is the broadcaster's own estate versus it.",
    ),
    "hybrid-cloud-store-bound-access-handles": (
        "hybrid-cloud",
        "Style evidence: \"Each store's till server keeps the local sale journal and can complete a sale when the wide-area link is down, because the chain's agreement with its card processor accepts a two-hour offline window;\" Nearest alternative style is `multi-cloud`: one public provider appears.",
    ),
    "hybrid-cloud-tariff-value-effective-dates": (
        "hybrid-cloud",
        "Style evidence: \"A charging-network operator runs charge-point management at substations in its own facilities and the driver app, billing and settlement in one public cloud region.\" Nearest alternative style is `edge-computing`: the on-premises charger is the operator's own facility equipment, and the decision is where billing state lives, not how near the user computation sits.",
    ),
    "hybrid-cloud-wholesale-amount-currency-pair": (
        "hybrid-cloud",
        "Style evidence: \"invoice finalisation and the statutory invoice archive run in the provider's own data centre, because tax authorities require the invoice of record to be produced in-country.\" Nearest alternative style is `multi-cloud`: a single public provider appears.",
    ),
    "kappa-architecture-billing-stream-outline": (
        "kappa-architecture",
        "Live and reprocessed views are produced by one streaming pipeline replaying the same call-record log.",
    ),
    "kappa-architecture-charging-external-stream-doc": (
        "kappa-architecture",
        "The live dashboard and the reprocessed settlement window come from one streaming pipeline replaying the same session log.",
    ),
    "kappa-architecture-claims-stream-view-formatting": (
        "kappa-architecture",
        "Live throughput and the reprocessed audit window are produced by one streaming pipeline over the same decision log.",
    ),
    "kappa-architecture-diagnostics-stream-stop-request": (
        "kappa-architecture",
        "Live and reprocessed diagnostic figures come from one streaming pipeline replaying the same session log.",
    ),
    "kappa-architecture-dispatch-stream-columns": (
        "kappa-architecture",
        "Live dashboards and the reprocessed promise-accuracy study both come from one streaming pipeline over the same log.",
    ),
    "kappa-architecture-forecast-stream-result-set": (
        "kappa-architecture",
        "Live map layers and the reprocessed verification study come from one streaming pipeline over the same log.",
    ),
    "kappa-architecture-freight-stream-intervals": (
        "kappa-architecture",
        "Live and reprocessed compliance figures come from one streaming pipeline over the same declaration log.",
    ),
    "kappa-architecture-grocery-stream-content-routing": (
        "kappa-architecture",
        "Live dashboards and the reprocessed promise-accuracy study come from one streaming pipeline over the same delivery log.",
    ),
    "kappa-architecture-hvac-stream-algorithm-skeleton": (
        "kappa-architecture",
        "Live and reprocessed energy figures come from one streaming pipeline replaying the same sensor log.",
    ),
    "kappa-architecture-identity-stream-explicit-test": (
        "kappa-architecture",
        "Live and reprocessed compliance figures come from one streaming pipeline over the same log.",
    ),
    "kappa-architecture-imagery-stream-deferred-values": (
        "kappa-architecture",
        "Live coverage and the reprocessed calibration study come from one streaming pipeline replaying the same scene log.",
    ),
    "kappa-architecture-matching-stream-least-rights": (
        "kappa-architecture",
        "Live and reprocessed fare figures come from one streaming pipeline replaying the same ride log.",
    ),
    "kappa-architecture-review-stream-class-tables": (
        "kappa-architecture",
        "Live and reprocessed matter figures come from one streaming pipeline replaying the same review log.",
    ),
    "kappa-architecture-telematics-stream-mapping": (
        "kappa-architecture",
        "Both the live map and the reprocessed insurance study come from one streaming pipeline over the same log.",
    ),
    "kappa-architecture-transit-stream-intermediates": (
        "kappa-architecture",
        "Live and reprocessed figures come from one streaming pipeline replaying the same tap log.",
    ),
    "kappa-architecture-waste-event-log-immutable": (
        "kappa-architecture",
        "A single streaming pipeline serves the live view and the reprocessed study alike, and reprocessing is a replay of the same operators rather than a second batch path.",
    ),
    "kappa-architecture-waste-stream-ingress-validation": (
        "kappa-architecture",
        "Live and reprocessed route figures come from one streaming pipeline replaying the same collection log.",
    ),
    "lambda-architecture-agronomist-console-binding-state": (
        "lambda-architecture",
        "Two paths, both computing: 'the overnight rebuild reprocesses the whole history ... into a complete immutable per-field and per-crop-cycle view', 'the live path keeps only present readings, the last hour of trend and the irrigation advisories currently queued', and the tool 'reads the merged view'. The nearest alternative is model-view-controller, since a desktop screen is described, but no controller-per-request-flow is present; the binding engine and the observable view-model are what the decision is about.",
    ),
    "lambda-architecture-analyser-result-store-isolation": (
        "lambda-architecture",
        "Two paths over one dataset: the completeness path 'rebuilds nightly from every result ever released, producing a complete immutable view ... and rewrites the partition', the freshness path 'keeps only results released in the last four hours and the instruments currently flagged', and 'the clinician query API merges them'. The nearest alternative is kappa-architecture, which would serve today's window and history from one streaming computation; here the full rebuild is separate and merged at read.",
    ),
    "lambda-architecture-applicant-set-criterion-handle": (
        "lambda-architecture",
        "Two computations over one dataset: 'the full-history job recomputes each night from every application, offer and enrolment event a complete immutable admissions view ... rewriting the partition', the projection 'holds only current seat counts and the live waitlist', and 'the enrolment API combines them'. The nearest alternative is kappa-architecture, which would rebuild settled intakes by replaying the stream; here a nightly full recompute is a separate path merged with the live projection.",
    ),
    "lambda-architecture-assurance-record-two-shapes": (
        "lambda-architecture",
        "Two code paths over one dataset are the load-bearing mechanism: 'the completeness path is a nightly Spark job over the whole retained history' and 'republishes an immutable partition for every circuit and month', while 'the freshness path ... keeps only the current incident state and the last hour of transitions', and 'the query API answers ... by taking the settled partition for closed months and overlaying live state'. The nearest alternative is kappa-architecture, which would keep one streaming computation for both live and reprocessed views; here two distinct computations exist and only their outputs are joined.",
    ),
    "lambda-architecture-bay-allocation-guard-lifetime": (
        "lambda-architecture",
        "Two computations over one dataset: 'the completeness path rebuilds monthly from every inspection and repair order ever recorded, producing a complete immutable view' and 'rewrites the partition', while 'the freshness path ... keeps only current bay occupancy' and 'the depot console merges the two'. The nearest alternative is kappa-architecture, which would serve both views from one streaming computation; here a monthly full rebuild and a live path coexist and are joined at read.",
    ),
    "lambda-architecture-bulletin-child-rows-through-parent": (
        "lambda-architecture",
        "Two paths over the same forecasts: 'a historical recompute rewrites, from every forecast run ever published, a complete immutable verification view' and the live path 'holds only today's bulletins and the pending issuance queue', with 'the dissemination service merges them when a subscriber asks'. The nearest alternative is data-mesh, since several analytical outputs exist, but no team-owned datasets with published contracts and a self-serve plane are described; the system is one dissemination service over a two-path computation.",
    ),
    "lambda-architecture-consumer-recorded-interaction-replay": (
        "lambda-architecture",
        "Two paths over one estate: 'the nightly recompute rebuilds, from every submission, referral and acceptance record, a complete immutable risk-appetite and pricing view' and 'rewrites the partition', while 'the live path ... holds only current referral queues and acceptance counts', and 'the underwriting API merges them'. The nearest alternative is kappa-architecture, which would serve history and today from one streaming computation; here the nightly rebuild is a separate path whose output is merged with a live one.",
    ),
    "lambda-architecture-encode-stage-illegal-moves": (
        "lambda-architecture",
        "Both paths are visible and distinct: the completeness path 'recomputes, from the full rights and encode history, a complete availability view' and 'rewrites the whole partition', while the freshness path 'consumes encoder and rights events from Kafka and holds only the current state of in-flight titles', and 'the catalogue API merges them'. The nearest alternative is command-query-responsibility-segregation, since a catalogue API reads what other jobs write, but no separate write model serving commands is described - the two sides are a recompute and a live path merged at read.",
    ),
    "lambda-architecture-fault-neighbourhood-position-lookup": (
        "lambda-architecture",
        "Both computations are named: 'a periodic full recompute over all retained measurements produces an immutable per-feeder load and interruption view ... rewritten from scratch each cycle', while 'the live feed path holds only present breaker states, protection flags and the last ten minutes of alarms', and the API 'answers a reported fault by merging the rebuilt feeder history with the live alarm window'. The nearest alternative is kappa-architecture, which would derive the filing view from one streaming computation; here a separate periodic recompute is merged with the live path.",
    ),
    "lambda-architecture-hazmat-approval-token-construction": (
        "lambda-architecture",
        "Two paths over one dataset: the completeness path 'recomputes each night from the full history ... producing a complete immutable view of berth utilisation and demurrage exposure', the freshness path 'keeps only current vessel positions, queue order and the next six hours of berth windows', and the dashboard and billing 'merge the two'. The nearest alternative is kappa-architecture, one streaming computation for live and reprocessed views; here a nightly full recompute is merged with a live path at read time.",
    ),
    "lambda-architecture-job-transaction-single-flush": (
        "lambda-architecture",
        "Two computations, one dataset: the completeness path 'recomputes the complete service-level and cost position per site and contract, and rewrites an immutable partition', the freshness path 'maintains only today's jobs, engineer positions and open escalations', and 'the operations API merges the two'. The nearest alternative is kappa-architecture, which serves live and reprocessed views from a single streaming computation; here the recompute is a separate weekly job whose output is merged at read.",
    ),
    "lambda-architecture-lot-sequence-single-instance": (
        "lambda-architecture",
        "Two computations over one dataset: the completeness path 'recomputes each night from the entire auction history a complete immutable settlement view ... and rewrites the partition', the freshness path 'keeps only the current high bid, bidder count and remaining seconds for lots closing today', and 'the bidding API merges them'. The nearest alternative is space-based, since live bidding state sits in memory, but there is no distributed in-memory tier or shared space; the live state is one service's own projection.",
    ),
    "lambda-architecture-merchandising-row-membership-rules": (
        "lambda-architecture",
        "Two computations over the same data: 'the nightly rebuild recomputes, from all play, rights and catalogue events, a complete immutable view' and 'rewrites the partition', while 'the delta path ... holds only the current top-performing titles and newly published ones', and 'the merchandising API combines them'. The nearest alternative is kappa-architecture, which would answer history from one streaming computation; here an immutable nightly rebuild is joined with a delta path at request time.",
    ),
    "lambda-architecture-merged-lot-sequential-walk": (
        "lambda-architecture",
        "Two computations over one dataset: the completeness path 're-runs monthly over the whole bid history in Spark' and 'rewrites the immutable settlement partition', the freshness path 'consumes bid and withdrawal events from Kafka and maintains only each lot's current high bid', and the service 'merges the two ... into one ordered result'. The nearest alternative is kappa-architecture, which would reprocess by replaying a single streaming computation; here a monthly full recompute and a live path both exist.",
    ),
    "lambda-architecture-model-host-capability-wrapping": (
        "lambda-architecture",
        "Two paths are explicit: 'the completeness path recomputes nightly, from the full booking and rate history, a complete occupancy and rate view' and 'rewrites the immutable partition', while 'the freshness path ... holds only current occupancy, house count and the next seven nights', and 'the dashboard merges them'. The nearest alternative is kappa-architecture, which would replay one streaming computation to serve history; here a nightly full recompute exists beside the live path.",
    ),
    "lambda-architecture-playback-capability-absence": (
        "lambda-architecture",
        "Two computations are named: the completeness path 'rebuilds, from all licence contracts and device-certification history, a complete immutable view ... and rewrites the partition', the freshness path 'keeps only the current state for titles whose rights changed in the last week', and 'the playback decision API merges them'. The nearest alternative is microkernel-plugin, since clients ask for capabilities, but nothing is installed or discovered at start-up in a core - the capabilities are answered from a merged view of two data paths.",
    ),
    "lambda-architecture-plugin-component-runtime-registry": (
        "lambda-architecture",
        "Two paths are visible and both compute: 'a full recompute job runs each night over the entire call and outage history and produces a complete, immutable handle-time and callback view', while 'an incremental path ... holds only current queue depth, agent state and open outages', and 'the console joins the two at request time'. The nearest alternative is kappa-architecture, which would answer the settled quarters by replaying one streaming computation; here the recompute is a distinct batch job and only its output is joined with the live path.",
    ),
    "lambda-architecture-routing-audit-seam-outcomes": (
        "lambda-architecture",
        "Two paths over one dataset: the completeness path 'recomputes ... the complete routing outcome ... from the beginning of the retention window, and rewrites an immutable partition', the freshness path 'keeps current queue depth, agent state and the last two hours of transfers', and 'the wallboard and the compliance query merge the two'. The nearest alternative is kappa-architecture, which would serve both live and reprocessed views from one streaming computation; here a separate full rebuild and a separate live path are merged.",
    ),
    "lambda-architecture-sensor-ingest-completion-dispatch": (
        "lambda-architecture",
        "Two paths over one dataset are explicit: the completeness path 'rebuilds nightly from every raw frame ever received ... producing a complete immutable season view', the freshness path 'holds only the current readings, the last hour of trend', and 'the advisory API merges them'. The nearest alternative is reactive-architecture, since a storm of reconnects and load shedding are mentioned, but no end-to-end backpressure contract or demand signalling is described; the mechanism asked about is how completions are dispatched.",
    ),
    "lambda-architecture-substation-capacity-partitioning": (
        "lambda-architecture",
        "Two paths over one dataset: the completeness path 'recomputes nightly from all retained telemetry, producing a complete immutable load-profile and constraint view ... and rewrites the partition', the freshness path 'keeps only current flows, breaker states and the last fifteen minutes of alarms', and 'the grid API merges them'. The nearest alternative is reactive-architecture, since bounded capacity and a saturation policy are mentioned, but the mechanism is per-dependency partitioning rather than an end-to-end demand-signalling design; and the two-path structure is what the estate is built around.",
    ),
    "layered-monolith-applicant-field-protection-carrier": (
        "layered-monolith",
        "Label layered-monolith: 'one FastAPI process in a single container' with the chain router -> service -> domain -> statement owner and SQL confined to the bottom layer. Nearest alternative monolithic; modular-monolith and hexagonal are excluded because the text shows horizontal tiers and no vertical modules or adapters.",
    ),
    "layered-monolith-booking-step-list-on-command": (
        "layered-monolith",
        "Label layered-monolith: one Ktor service on a single Deployment with the chain route -> service -> domain -> store, SQL confined to the store. Nearest alternative monolithic, excluded by the layer order; the steps run in-process, so pipe-and-filter and event-driven readings do not fit, and no vertical modules or ports are described.",
    ),
    "layered-monolith-delivery-stage-transition-encoding": (
        "layered-monolith",
        "Label layered-monolith: the platform is 'one Node 22 service ... in a single deployable', and `routes.ts` -> `AssignmentService` -> `Delivery` -> `pgAssignmentStore` is an ordered path where only the store holds SQL. Nearest alternative monolithic, which is what the same deployable would be with no interior order; modular-monolith would need vertical slices owning data and hexagonal a port boundary, and the text shows neither.",
    ),
    "layered-monolith-issuer-adapter-uniform-holding": (
        "layered-monolith",
        "Label layered-monolith: 'one Rust service ... deployed as a single container' with the chain handler -> service -> domain -> sqlx module and SQL confined to the module. Nearest alternative monolithic, excluded by the layer order; modular-monolith would need vertical modules and hexagonal would need an adapter ring around a core, which the text does not describe.",
    ),
    "layered-monolith-ledger-adjustment-caller-restriction": (
        "layered-monolith",
        "Label layered-monolith: one Spring Boot jar and the chain controller -> service -> aggregate -> JDBC module with SQL confined to the bottom. Nearest alternative monolithic, which would need no interior order; modular-monolith would need vertical modules owning their own data and hexagonal would need ports and adapters.",
    ),
    "layered-monolith-match-score-readonly-holder": (
        "layered-monolith",
        "Label layered-monolith: one Spring Boot deployable with the chain controller -> service -> domain -> repository and SQL confined to the repository. Nearest alternative monolithic, excluded by the stated layer order; modular-monolith needs vertical modules owning data, hexagonal needs an adapter boundary, neither shown.",
    ),
    "layered-monolith-on-sale-request-state-carrier": (
        "layered-monolith",
        "Label layered-monolith: 'one ASP.NET Core 9 process deployed as a single image' plus an explicit chain controller -> service -> aggregate -> store, with 'Nothing skips a layer'. Nearest alternative monolithic; modular-monolith would require vertical modules owning their own data, which the text never shows, and hexagonal would require a port boundary rather than the stated horizontal order.",
    ),
    "layered-monolith-overnight-transmitter-stop-protocol": (
        "layered-monolith",
        "Label layered-monolith: 'one Spring Boot process ... a single deployable, no cluster' and the chain controller -> service -> domain -> repository. Nearest alternative monolithic, excluded by the explicit layer order; modular-monolith and hexagonal would require vertical modules or ports, neither described.",
    ),
    "layered-monolith-prerequisite-graph-reference-discipline": (
        "layered-monolith",
        "Label layered-monolith: one deployable ('one Rust binary ... shipped as a single container image') with a stated chain handler -> application -> domain -> sqlx module, and the mapper owns all statements. Nearest alternative monolithic, which is what a single deployable would be without a layer order; modular-monolith would need vertical modules owning their own data, and hexagonal would need ports and adapters, neither of which the text describes.",
    ),
    "layered-monolith-replenishment-wire-dialect-mapping": (
        "layered-monolith",
        "Label layered-monolith: 'one ASP.NET Core process ... no second deployable' with the chain controller -> service -> domain -> Dapper persistence layer stated. Nearest alternative monolithic; modular-monolith would need vertical modules and hexagonal would need adapters around a core, neither of which is described.",
    ),
    "layered-monolith-robot-and-supervisor-boundary-check": (
        "layered-monolith",
        "Label layered-monolith: one NestJS application on a single Deployment with the chain controller -> service -> domain -> repository. Nearest alternative monolithic, excluded by the stated layer order; the two credential kinds are handled by one in-process guard, so modular-monolith and hexagonal are not shown either.",
    ),
    "layered-monolith-seat-allocation-sql-owner": (
        "layered-monolith",
        "Label layered-monolith: one Django deployment with the chain view -> service -> domain -> table module, and SQL confined to the bottom. Nearest alternative monolithic, which would apply if the layer order were unstated; model-view-controller tempts through Django, but the described mechanism is the ordered horizontal path, and no vertical modules or ports exist for the other candidates.",
    ),
    "layered-monolith-sensor-measurement-value-type": (
        "layered-monolith",
        "Label layered-monolith: one jar, one deployable, and the chain controller -> service -> domain -> mapper stated with SQL confined to the mapper. Nearest alternative monolithic, which would need no layer order; modular-monolith and hexagonal are excluded because no vertical modules own data and no port boundary is described.",
    ),
    "layered-monolith-settlement-signing-public-primitives": (
        "layered-monolith",
        "Label layered-monolith: one Go binary in a single container with the chain handler -> service -> domain -> store and SQL confined to the store. Nearest alternative monolithic, excluded by the explicit layer order; the shared infrastructure belongs to counterparties, not to this system, so multi-cloud, hybrid-cloud and modular readings do not apply.",
    ),
    "layered-monolith-shard-rebuild-phase-coordination": (
        "layered-monolith",
        "Label layered-monolith: one Python process in a single container with the chain router -> service -> domain -> index store. Nearest alternative monolithic, excluded because the horizontal order is stated; the eight threads are a mechanism inside the application layer, not separate deployables, so modular-monolith and hexagonal do not apply.",
    ),
    "layered-monolith-slot-override-stale-write-detection": (
        "layered-monolith",
        "Label layered-monolith: 'one ASP.NET Core process in a single container' with the chain controller -> service -> domain record -> DbContext and all queries confined to the bottom layer. Nearest alternative monolithic, which would apply without the layer order; modular-monolith and hexagonal are excluded by the absence of vertical modules or a port boundary.",
    ),
    "layered-monolith-tariff-plan-staged-construction": (
        "layered-monolith",
        "Label layered-monolith: a single Fastify process ('one container, one release') whose requests pass `CdrRouter` -> `RatingService` -> `TariffPlan` -> `cdrStore`, with SQL confined to the last hop. Nearest alternative monolithic, excluded because the layer sequence is explicit; modular-monolith and hexagonal would require vertical modules or adapters, which the mediation path never mentions.",
    ),
    "layered-monolith-trip-billing-row-set-rules": (
        "layered-monolith",
        "Label layered-monolith: one ASP.NET Core process with the chain controller -> service -> billing logic -> data-access module. Nearest alternative monolithic, excluded by the stated horizontal order; modular-monolith would need vertical modules owning their data and hexagonal an adapter boundary, neither described.",
    ),
    "microkernel-plugin-booking-revocable-grant": (
        "microkernel-plugin",
        "Pricing, availability and loyalty each arrive as a separately packaged capability the core discovers at start-up, so the core stays minimal.",
    ),
    "microkernel-plugin-dispatch-module-report": (
        "microkernel-plugin",
        "The core is minimal and each field report arrives as its own package discovered at start-up, with no core change to add one.",
    ),
    "microkernel-plugin-hvac-shared-protocol": (
        "microkernel-plugin",
        "Capabilities are separately packaged units the core discovers, so the platform's own surface is a minimal core plus those packages.",
    ),
    "microkernel-plugin-hvac-worker-fanout": (
        "microkernel-plugin",
        "The core holds building state and extension points; each control strategy is a separately packaged unit activated per building.",
    ),
    "microkernel-plugin-inventory-measured-values": (
        "microkernel-plugin",
        "Supplier and forecasting capabilities arrive as separate packages registered with a minimal core that holds inventory records.",
    ),
    "microkernel-plugin-inventory-warehouse-table": (
        "microkernel-plugin",
        "The core holds inventory records and extension points, while each forecasting, allocation and supplier capability is a separately packaged module registered with it.",
    ),
    "microkernel-plugin-loyalty-module-initialisation": (
        "microkernel-plugin",
        "Every earn rule is a separate package registered with the core at start-up, and the core itself only holds ledger storage and member access.",
    ),
    "microkernel-plugin-loyalty-rule-request": (
        "microkernel-plugin",
        "Each earn and burn rule is a separate package registered with the core, and the core's own surface is limited to rule registration and member state.",
    ),
    "microkernel-plugin-matching-test-collaborator": (
        "microkernel-plugin",
        "The core is minimal and each matching policy is a separate package, and the test replaces one collaborator of the core.",
    ),
    "microkernel-plugin-origination-flow-control": (
        "microkernel-plugin",
        "Pipeline stages are separately packaged units discovered by a minimal core, so the core itself holds little beyond stage registration.",
    ),
    "microkernel-plugin-review-narrow-index": (
        "microkernel-plugin",
        "The core is minimal and every analysis is a separately packaged unit discovered from a drop directory.",
    ),
    "microkernel-plugin-telematics-unknown-vehicle": (
        "microkernel-plugin",
        "Decoding and enrichment arrive as separate packages registered with a core that holds only frame storage and extension points.",
    ),
    "microkernel-plugin-ticketing-module-fares": (
        "microkernel-plugin",
        "Every fare capability is a separately packaged module the core discovers at start-up, with the core holding only station and product records.",
    ),
    "microkernel-plugin-ticketing-module-rights": (
        "microkernel-plugin",
        "Every capability - gate driver, fare product, top-up - is a separately packaged unit that a minimal core activates per station.",
    ),
    "microservices-condition-set-self-traversal": (
        "microservices",
        "Label microservices: four applications with their own stores, release pipelines and scaling metrics and no shared schema. Nearest alternative monolithic, ruled out because the applications are separately deployed and released; api-gateway and service-mesh are ruled out because no client entry point or proxy layer exists.",
    ),
    "microservices-duty-period-cohesive-unit": (
        "microservices",
        "Label microservices: the roster, rules, licence and pairing applications keep separate schemas and independent releases, each scaling on its own metric. Nearest alternative monolithic, which the four deployables exclude; modular-monolith would need vertical modules inside one deployable, and no client entry point or proxy layer suggests api-gateway or service-mesh.",
    ),
    "microservices-excursion-detector-orthogonal-units": (
        "microservices",
        "Label microservices: four applications, each with a separate store and release cadence and its own scaling metric, and no shared schema. Nearest alternative event-driven, because detection consumes a sensor stream, but that stream is one application's input rather than the carrier among all components; api-gateway and service-mesh are excluded as no entry point or proxy plane is described.",
    ),
    "microservices-interaction-validation-result-collection": (
        "microservices",
        "Label microservices: four applications with distinct stores, release cadences and scaling signals and no shared schema. Nearest alternative monolithic, excluded because each application deploys on its own; the collected result is returned synchronously to the caller, so observer and publisher-subscriber readings do not apply.",
    ),
    "microservices-journey-expand-collapse-recursion": (
        "microservices",
        "Label microservices: four applications with their own stores, pipelines, release cadences and scaling metrics, none sharing a schema. Nearest alternative monolithic, ruled out because the applications are separately deployed; api-gateway is ruled out because clients call the builder directly.",
    ),
    "microservices-journey-row-translation-layer": (
        "microservices",
        "Label microservices: four applications, each with its own database, deployment pipeline and release schedule. Nearest alternative monolithic, ruled out because the deployables are separate; api-gateway and service-mesh are ruled out because no client entry point or proxy layer exists.",
    ),
    "microservices-late-model-run-partial-answer": (
        "microservices",
        "Label microservices: four applications with their own data stores, pipelines, release cadences and scaling metrics, and no cross-reads of tables. Nearest alternative monolithic, ruled out because the applications are separately deployed and released; api-gateway is ruled out because clients call the forecast application directly.",
    ),
    "microservices-mediation-decision-outside-writers": (
        "microservices",
        "Label microservices: intake, rating, invoice assembly and dispute ledger each own a database and ship separately, and no application reads another's tables. Nearest alternative service-mesh, because a mediation boundary sits between writers and storage, but that boundary is an application-owned decision component rather than a proxy layer with a control plane.",
    ),
    "microservices-merchant-risk-rule-set-type": (
        "microservices",
        "Label microservices: the switch, registry, risk and settlement applications hold separate schemas and are released independently, each scaling on its own metric. Nearest alternative monolithic, which four separately released deployables exclude; api-gateway is also excluded since clients and peers reach the switch directly rather than through one entry point.",
    ),
    "microservices-pick-instruction-step-chain": (
        "microservices",
        "Label microservices: four applications with their own stores, release pipelines and scaling metrics, each owning its data. Nearest alternative monolithic, ruled out because the applications are separately deployed; pipe-and-filter tempts because four stages are described, but they are pure functions inside one application, not a data-stream topology between deployables.",
    ),
    "microservices-plot-rollup-statement-owner": (
        "microservices",
        "Label microservices: four applications with their own stores, pipelines and releases, none sharing a schema, each scaling on its own metric. Nearest alternative monolithic, ruled out because the deployables and releases are separate; api-gateway and service-mesh are ruled out because no single client entry point or proxy layer exists.",
    ),
    "microservices-quality-check-collaborator-registry": (
        "microservices",
        "Label microservices: four applications with separate stores, pipelines, release cadences and scaling metrics and no shared schema. Nearest alternative hexagonal, since the quality checks live behind a frozen plug-in interface, but that interface sits inside one application while the deployables are independent; api-gateway and service-mesh are ruled out because no client entry point or proxy layer exists.",
    ),
    "microservices-recommendation-consumer-committed-position": (
        "microservices",
        "Label microservices: catalogue, analytics, recommendations and entitlements each own a schema and deploy on their own schedule, scaling separately. Nearest alternative event-driven, since a retained topic feeds the recommender, but that topic is one application's inbound feed while the system shape is the independent deployables; broker is excluded as no single registration point exists.",
    ),
    "microservices-registration-commit-time-validation": (
        "microservices",
        "Label microservices: four applications with their own database, pipeline and release schedule, and no shared schema. Nearest alternative saga, because a registration spans several steps, but every step lands in the enrolment application's own store and the coordination is commit-time validation inside it, not compensation across parties.",
    ),
    "microservices-resident-session-single-authority": (
        "microservices",
        "Label microservices: four applications with their own schemas, pipelines and scaling metrics, and three of them accept cookies directly. Nearest alternative api-gateway, ruled out because no single entry point fronts the applications: each accepts and validates the cookie itself, calling the shared session owner rather than routing through a proxy.",
    ),
    "microservices-room-night-availability-guard": (
        "microservices",
        "Label microservices: four applications with separate databases, pipelines and release cadences, none sharing a schema. Nearest alternative monolithic, ruled out because the deployables are separate and independently released; service-mesh and api-gateway are ruled out because no traffic layer or single client entry is described.",
    ),
    "microservices-route-focus-immutable-day-edit": (
        "microservices",
        "Label microservices: `work-orders`, `technician-availability`, `route-planning` and `parts-logistics` each own a schema and deploy on their own cadence. Nearest alternative monolithic, ruled out by the four separate deployables; api-gateway and service-mesh are excluded because dispatch clients reach the planning application directly and no proxy layer is named.",
    ),
    "microservices-verification-flow-variant-cloning": (
        "microservices",
        "Label microservices: five applications, each independently deployed with its own store and its own pipeline. Nearest alternative monolithic, ruled out because the deployables are separate and released separately; api-gateway and service-mesh are ruled out because no single client entry and no proxy layer are described.",
    ),
    "model-view-controller-auction-past-tense": (
        "model-view-controller",
        "Bids are controller actions over an auction model and the lot page is a logic-free view, which is the split the console is organised on.",
    ),
    "model-view-controller-billing-fanout-totals": (
        "model-view-controller",
        "Invoice runs are controller actions over a billing model whose invoice page is a logic-free view, the three-part split of this console.",
    ),
    "model-view-controller-claims-format-canonical": (
        "model-view-controller",
        "Claim intake and adjudication are controller actions over a claim model whose sheet is a logic-free view, the split this console uses.",
    ),
    "model-view-controller-claims-sensitive-panel": (
        "model-view-controller",
        "Review sheets are controller actions over a claim model whose templates render them, the three-part split of this console.",
    ),
    "model-view-controller-coldchain-shipped-command": (
        "model-view-controller",
        "Excursion reports are controller actions over a shipment model that the views render without logic, the three-part split of this console.",
    ),
    "model-view-controller-grid-report-rules": (
        "model-view-controller",
        "Report handling is a controller over an outage model, and the rendered report carries no logic, which is the split this console is built on.",
    ),
    "model-view-controller-identity-validity-window": (
        "model-view-controller",
        "Verification outcomes are controller actions over an identity model and logic-free views, the split this console is built on.",
    ),
    "model-view-controller-ledger-instance-handles": (
        "model-view-controller",
        "Statement requests are controller actions over a member model, rendered by views that carry no logic, which is the split this console uses.",
    ),
    "model-view-controller-moderation-concrete-tables": (
        "model-view-controller",
        "Triage requests are handled by controllers over a content model whose views carry no logic, which is the split this console is built on.",
    ),
    "model-view-controller-moderation-queue-vocabulary": (
        "model-view-controller",
        "Requests are handled by controllers that drive a decision model and hand logic-free views to the renderer, which is the split this console is organised around.",
    ),
    "model-view-controller-mortgage-lender-channel": (
        "model-view-controller",
        "Request handling is a controller that drives a model and hands a logic-free view to the renderer, which is the model-view-controller split the description is built on.",
    ),
    "model-view-controller-onboarding-neighbourhood": (
        "model-view-controller",
        "Review sheets are controller actions over an applicant model with logic-free views, the split this console uses.",
    ),
    "model-view-controller-port-statement-formatting": (
        "model-view-controller",
        "The statement is a controller action over a movement model that a logic-bearing view currently renders, which is exactly the split this description is about.",
    ),
    "model-view-controller-schedule-state-replacement": (
        "model-view-controller",
        "Each reschedule is a controller action over an appointment model that the view renders, the three-part split the description is built on.",
    ),
    "model-view-controller-telematics-frame-immutable": (
        "model-view-controller",
        "Map and query handling is a controller over a vehicle model rendered by views, which is the UI split this console is organised around.",
    ),
    "model-view-controller-telemetry-consumer-pool": (
        "model-view-controller",
        "Console requests are controller actions over a vehicle model with logic-free views, and the consumer pool sits behind that UI split.",
    ),
    "model-view-controller-till-durable-outbox": (
        "model-view-controller",
        "The sale flow is a controller over a sale model with a logic-free receipt view, the split this console is built on.",
    ),
    "modular-monolith-auction-lot-creation": (
        "modular-monolith",
        "The application is one deployment whose modules are vertical and own their tables, never released separately.",
    ),
    "modular-monolith-billing-module-independence": (
        "modular-monolith",
        "Deployment is one unit and the interior is a set of vertical modules that own their tables and speak through interfaces.",
    ),
    "modular-monolith-booking-collaboration-record": (
        "modular-monolith",
        "The application is one deployment whose interior modules are vertical and own their tables, with interfaces between them.",
    ),
    "modular-monolith-booking-large-reads": (
        "modular-monolith",
        "The application is one deployment whose modules are vertical and own their tables, talking through interfaces rather than shared tables.",
    ),
    "modular-monolith-booking-query-objects": (
        "modular-monolith",
        "Deployment is one unit; interior modules are vertical and own their tables, with interfaces rather than shared schema.",
    ),
    "modular-monolith-claims-module-boundary-restriction": (
        "modular-monolith",
        "The platform is a single deployment whose interior modules are vertical, own their tables and communicate through narrow interfaces rather than shared schema.",
    ),
    "modular-monolith-claims-source-canonical": (
        "modular-monolith",
        "Deployment is a single unit whose interior modules are vertical and own their data, with interfaces rather than shared schema.",
    ),
    "modular-monolith-diagnostics-remote-operations": (
        "modular-monolith",
        "The system ships as a single deployment whose modules are vertical, each owning its tables, with interfaces rather than shared schema between them.",
    ),
    "modular-monolith-dispatch-valid-fixtures": (
        "modular-monolith",
        "The platform is a single deployment whose modules are vertical and own their tables, with interfaces rather than shared schema.",
    ),
    "modular-monolith-imagery-concurrent-edits": (
        "modular-monolith",
        "The platform is a single deployment whose modules are vertical and own their data, with interfaces between modules instead of shared tables.",
    ),
    "modular-monolith-imagery-duplicate-fetches": (
        "modular-monolith",
        "The platform is one deployment whose interior modules are vertical, own their tables and speak through interfaces.",
    ),
    "modular-monolith-port-container-effects": (
        "modular-monolith",
        "The system is a single deployable whose modules are vertical, own their own tables and communicate through narrow interfaces rather than shared schema access.",
    ),
    "modular-monolith-procurement-form-pages": (
        "modular-monolith",
        "Deployment is a single unit whose interior modules are vertical and own their tables, with interfaces instead of shared schema.",
    ),
    "modular-monolith-review-clause-combinators": (
        "modular-monolith",
        "The whole platform is one deployable, but its interior is a set of vertical modules that own their data and expose narrow interfaces to each other, with no module reaching into another's tables.",
    ),
    "modular-monolith-scheduling-code-shape": (
        "modular-monolith",
        "The codebase is a single deployment whose modules are vertical and data-owning, with interfaces instead of shared tables.",
    ),
    "modular-monolith-settlement-channel-nonclient": (
        "modular-monolith",
        "The platform is a single deployment whose interior modules are vertical, own their tables and communicate through interfaces rather than shared tables.",
    ),
    "modular-monolith-ticketing-identity-field": (
        "modular-monolith",
        "The platform ships as one deployment whose modules are vertical and own their data, with interfaces between them.",
    ),
    "modular-monolith-till-effective-prices": (
        "modular-monolith",
        "The application is a single deployment whose modules are vertical, own their tables and communicate through interfaces.",
    ),
    "modular-monolith-underwriting-agent-humble-panel": (
        "presentation-abstraction-control",
        "Both blind annotators labelled this presentation-abstraction-control where the authoring intent was modular-monolith",
    ),
    "monolithic-agronomy-advisory-naming": (
        "monolithic",
        "Style evidence: 'one Rails application: a single deployment' doing ingest, advice and portal against one database, 'released as a whole app'. Nearest alternative layered-monolith if ingest, advice and portal read as stages; no stage-only call chain or layer discipline is described, and the release unit decides.",
    ),
    "monolithic-archive-payload-type-discriminator": (
        "monolithic",
        "Style evidence: 'a single FastAPI deployment' ingesting, scheduling and serving, 'one image and one release' with the replay tool a subcommand of it. Nearest alternative event-driven if the archive and dead-letter files read as a record flow; they are a replay path inside one process, not a carrier between independent consumers.",
    ),
    "monolithic-bid-interaction-split": (
        "monolithic",
        "Style evidence: 'one Go binary: a single service' serving submission pages, award batch and notices, deployed 'as one binary ... and restarted whole' with the batch a subcommand of it. Nearest alternative microservices, since the three roles sound separable; they are objects inside one binary with one build.",
    ),
    "monolithic-capability-wrapping-for-partner-hooks": (
        "monolithic",
        "Style evidence: 'one Node deployment' with front end, worker and console starting 'from the same process entry point', and 'a rollout replaces the process'. Nearest alternative microkernel-plugin, since customer-supplied code executes inside the product; the supplied bodies are configuration rows evaluated by the product's own worker, with no core-plus-extension packaging or start-up discovery, so the single deployable carries the label.",
    ),
    "monolithic-chained-compliance-journal": (
        "monolithic",
        "Style evidence: 'a single statically linked binary' that 'serves the catalogue admin API, runs the nightly rights-window sweep and writes the compliance record', released by copying that binary and restarting. Nearest alternative event-sourcing: the chained record is an evidence journal beside the catalogue store, not the authoritative state, so state is never rebuilt from it.",
    ),
    "monolithic-charger-state-cells": (
        "monolithic",
        "Style evidence: 'one Kotlin deployment: a single Ktor application in one JVM' doing connections, metering and billing, 'one artifact and one restart'. Nearest alternative actor-based if the mailboxes are read as the architecture; the cells are an in-process concurrency structure inside one deployable, with no remoting, clustering or per-unit deployment.",
    ),
    "monolithic-charging-command-backlog": (
        "monolithic",
        "Style evidence: 'one Go service: a single binary' doing websockets, driver API and reconciliation, 'rolled out as a unit - one binary, one config, one restart'. Nearest alternative half-sync-half-async, since handlers enqueue to workers; the buffer and workers are inside that single process, and no separately deployed asynchronous tier exists.",
    ),
    "monolithic-crew-qualification-date-ranges": (
        "monolithic",
        "Style evidence: 'a single Rust binary' storing qualifications, building rosters and serving the portal, shipped 'as one artifact and restarts whole'. Nearest alternative event-sourcing if dated facts read as history; nothing is replayed, and current rows plus ranges are authoritative.",
    ),
    "monolithic-diagnostics-frame-dispatch": (
        "monolithic",
        "Style evidence: 'a single Rust binary: one executable' decoding frames and serving the tablet, shipped 'as one artifact and restarted during service'. Nearest alternative modular-monolith if decoding, logging and API read as modules; they share one binary and one release with nothing independently deployable.",
    ),
    "monolithic-fleet-link-domain-calls": (
        "monolithic",
        "Style evidence: 'a single Ktor application in one JVM' handling waves, allocation and console, 'deployed from one image and restarted whole', with the vendor link 'the only thing that leaves the process'. Nearest alternative broker, since one queue is named; it carries one outbound integration, not the wiring between the product's own components.",
    ),
    "monolithic-forecast-run-markup-delegate": (
        "monolithic",
        "Style evidence: 'one ASP.NET Core Blazor Server application: a single deployment' hosting all pages, with no separate front-end deployment. Nearest alternative modular-monolith if pages read as modules; they are components inside one app with one build and one database.",
    ),
    "monolithic-freight-validator-adaptation": (
        "monolithic",
        "Style evidence: 'one Spring Boot application: a single jar' with 'the whole product is one deployment' and capacity added by 'more copies of the same jar'. Nearest alternative modular-monolith if booking, customs and console read as modules; they share classes and one release, with no module-owned stores.",
    ),
    "monolithic-lab-rule-grammar-evaluation": (
        "monolithic",
        "Style evidence: 'one Go binary: a single service' receiving analyser output, evaluating rules and serving review, with one thin proxy in front of that single upstream and no routing table. Nearest alternative api-gateway, since a TLS-terminating proxy is named; it forwards all traffic to exactly one backend and carries no edge policy or backend catalogue, so the deployable is the shape.",
    ),
    "monolithic-layout-lookup-retention": (
        "monolithic",
        "Style evidence: 'one Python deployment: a single FastAPI process' computing routes, tracking robots and serving the dashboard, with capacity added 'by adding whole copies of that image'. Nearest alternative space-based, since state is retained in memory; the retained values are derived lookups, not the system's source of truth, and one database stays authoritative.",
    ),
    "monolithic-moderation-score-operator-family": (
        "monolithic",
        "Style evidence: 'one C++20 executable: a single statically linked binary' that scores posts and serves the console endpoint, 'restarted to change the rules bundle'. Nearest alternative none, since the decision looks like a local type technique; the description names a deployed binary with a runtime endpoint and a release, so topology is visible and the single deployable governs.",
    ),
    "monolithic-mortgage-case-version-tokens": (
        "monolithic",
        "Style evidence: 'a single Rust service: one axum binary' serving the portal and running the checklist job, shipping as 'one container image with one migration set'. Nearest alternative layered-monolith if the form, job and store are read as stages; no stage-by-stage call discipline is described, and the release unit is the decisive shape.",
    ),
    "monolithic-ocr-session-single-worker": (
        "monolithic",
        "Style evidence: 'one JVM serving the capture upload, the liveness challenge and the decision callback', with 'the whole application restarts to pick up a model change'. Nearest alternative layered-monolith if upload, orchestration and decision read as stages; the described shape is one deployable plus an internal task executor.",
    ),
    "monolithic-pick-face-traversal-handoff": (
        "monolithic",
        "Style evidence: 'a single executable jar, one JVM, one PostgreSQL database', rolled out 'by replacing that jar', and 'nothing inside it ... can be restarted or capacity-tuned on its own'. Nearest alternative layered-monolith, because handlers, aggregate and store appear in that order; the text shows no strict stage-only call chain, so the deployment unit carries the label.",
    ),
    "monolithic-planner-config-holder-base": (
        "monolithic",
        "Style evidence: 'one native binary' running the reorder calculation and a status page, 'installed on one on-premises server' and restarted whole. Nearest alternative none, since the decision looks like a local layout technique; a deployed binary with a runtime status endpoint is described, so a deployable exists and is the unit of build and restart.",
    ),
    "monolithic-product-pricing-rules": (
        "monolithic",
        "Style evidence: 'one Rust deployment: a single binary' serving quotes, batch and documents, with 'one artifact and one restart'. Nearest alternative modular-monolith if products read as modules; product variation lives in a trait selected per request inside one binary, not in separately deployable units.",
    ),
    "monolithic-provisioning-order-lock": (
        "monolithic",
        "Style evidence: 'one Spring Boot application: a single jar' driving workflow, carrier submissions and console, 'deployed as one unit'. Nearest alternative modular-monolith if the workflow and console read as modules; they share one release and one store, and no module owns private data.",
    ),
    "monolithic-purchase-order-subcontractor-reference": (
        "monolithic",
        "Style evidence: 'one TypeScript deployment on Node' handling onboarding, orders and receipts, 'one container image and one release'. Nearest alternative modular-monolith if onboarding, orders and receipts read as modules; they share one image and one database with no module-owned data.",
    ),
    "monolithic-replenishment-lookup-absence": (
        "monolithic",
        "Style evidence: 'one Java deployment: a single Spring Boot jar' running the calculation, serving approval and publishing suggestions, with 'one process and one release'. Nearest alternative modular-monolith if calculation, approval and publication read as modules; they share one build and one database.",
    ),
    "monolithic-robot-domain-model-split": (
        "monolithic",
        "Style evidence: 'one TypeScript deployment on Node' assigning tasks, tracking cycles and serving the console, with 'one image, one release, one restart'. Nearest alternative hexagonal, since a repository boundary is described; that boundary covers only the store, with no ports for every external concern and no framework-free core discipline stated.",
    ),
    "monolithic-routing-graph-rebuild-grant": (
        "monolithic",
        "Style evidence: 'a single Ktor application in one JVM' holding presence, matcher and statistics, 'deployed by replacing the jar and restarted as a unit'. Nearest alternative modular-monolith if presence, matching and statistics are read as modules; they share one release and there is no data-owning module boundary described.",
    ),
    "monolithic-sensor-domain-translators": (
        "monolithic",
        "Style evidence: 'a single TypeScript service on Node' with ingest, advice and API sharing 'one image and one release'. Nearest alternative layered-monolith if translators are read as a persistence stage below the domain; the text describes translator objects called from the service with no strict layer-by-layer call chain, and the deployable is one.",
    ),
    "monolithic-substation-adaptor-capability-lookup": (
        "monolithic",
        "Style evidence: 'a single worker host' with ingest loop, aggregation and dashboard endpoints in 'one process and one deployment slot', restarted to apply config. Nearest alternative modular-monolith if the ingest and dashboard parts read as modules; they share one configuration object and one lifetime, with no per-module release.",
    ),
    "monolithic-tax-filing-submission-workers": (
        "monolithic",
        "Style evidence: 'a single jar' doing accept, validate and submit, with 'a jar swap and a restart' as deployment and no separate submission worker. Nearest alternative half-sync-half-async, since HTTP hand-off to workers on a queue is described; the queue and workers are internal to the one application, and there is no separately deployed asynchronous service tier.",
    ),
    "monolithic-telemetry-query-staged-construction": (
        "monolithic",
        "Style evidence: 'a single TypeScript service on Node' holding ingest, query API and report generator, 'deployed from one container and restarted together'. Nearest alternative modular-monolith if the three parts read as modules; they share one image and one release, with no module-owned data.",
    ),
    "monolithic-tender-eligibility-combiners": (
        "monolithic",
        "Style evidence: 'one Spring Boot application: a single jar' serving portal, evaluation batch and registry, with the logic 'a plain call inside the same JVM'. Nearest alternative rule-based-system, since eligibility rules are described; the rules are compiled combinators, changed only by a release, and no rule engine evaluates declarative input.",
    ),
    "monolithic-theatre-schedule-tabular-result": (
        "monolithic",
        "Style evidence: 'one deployable: a single Java service running in one JVM' serving console, scheduler and report, replaced whole on release. Nearest alternative layered-monolith if console, scheduler and report read as stages; they are callers of one process and one scheduled job, with no strict stage-to-stage chain.",
    ),
    "monolithic-till-basket-step-sequence": (
        "monolithic",
        "Style evidence: 'one Rust binary: a single executable' holding catalogue, promotions, printer and till API, with 'one deployment, one rollback'. Nearest alternative modular-monolith if catalogue and promotions read as modules; they compile into one binary with a single release and no module-owned store.",
    ),
    "monolithic-trip-record-access-gate": (
        "monolithic",
        "Style evidence: 'one deployable: a single service' matching, streaming and closing fares, 'one artifact and one restart'. Nearest alternative hexagonal if the repository and check point read as ports; the check point is internal discipline inside one deployable, not a set of adapters for external concerns.",
    ),
    "multi-cloud-auction-two-provider-mediation": (
        "multi-cloud",
        "Auction state and settlement live on two different public providers, with the payment provider fixing the second.",
    ),
    "multi-cloud-contact-two-provider-events": (
        "multi-cloud",
        "State and the event service run on different public providers, with the managed service the reason for the second.",
    ),
    "multi-cloud-diagnostics-two-provider-result-set": (
        "multi-cloud",
        "The operational store and the analytics engine run on two different public providers, with the split part of the platform's design.",
    ),
    "multi-cloud-enrolment-two-provider-expiry": (
        "multi-cloud",
        "Records and the messaging service sit on different public providers, so cross-provider delivery is part of the platform's design.",
    ),
    "multi-cloud-forecast-two-provider-clients": (
        "multi-cloud",
        "The platform's compute runs in one public cloud and its storage and delivery in another, with per-region placement load-bearing.",
    ),
    "multi-cloud-grid-two-provider-derived": (
        "multi-cloud",
        "Ingest and the forecasting service sit on two different public providers, with that split load-bearing for the platform's design.",
    ),
    "multi-cloud-imagery-two-provider-render": (
        "multi-cloud",
        "Tiles and rendering run on different public providers, with the split load-bearing for the console.",
    ),
    "multi-cloud-imagery-two-provider-walk": (
        "multi-cloud",
        "Storage and the GPU pool sit on two different public providers, with the split determined by where accelerators are available.",
    ),
    "multi-cloud-lab-two-provider-cleanup": (
        "multi-cloud",
        "Lab records and the analysis service run on different public providers, so the lease spans a provider boundary.",
    ),
    "multi-cloud-loyalty-two-provider-filter-chain": (
        "multi-cloud",
        "Regulated member data and the promotions service sit on different public providers, with per-regulator placement load-bearing.",
    ),
    "multi-cloud-matching-two-provider-key": (
        "multi-cloud",
        "Trip data and the routing service run on different public providers, with the placement decided by where routing is offered.",
    ),
    "multi-cloud-review-two-provider-metadata": (
        "multi-cloud",
        "Storage and indexing are placed on different public providers, with per-matter residency deciding which region holds which.",
    ),
    "multi-cloud-roster-two-provider-concept": (
        "multi-cloud",
        "Regulated crew data and the optimisation jobs run on different public providers, with per-regulator placement load-bearing.",
    ),
    "multi-cloud-trial-two-provider-release": (
        "multi-cloud",
        "Regional participant data and analytics run on different public providers, with placement fixed by each regulator's requirement.",
    ),
    "multi-cloud-waste-two-provider-cluster": (
        "multi-cloud",
        "The dispatch core and the mapping service run on two different public providers, with per-depot placement load-bearing.",
    ),
    "multi-cloud-waste-two-provider-lookups": (
        "multi-cloud",
        "The dispatch core and the geocoding capability sit on two different public providers, with the split load-bearing for routing decisions.",
    ),
    "multi-cloud-waste-two-provider-metadata": (
        "multi-cloud",
        "Operational data and analytics sit on two different public providers, with the warehouse chosen for its managed service.",
    ),
    "none-adjudication-outcome-shapes": (
        "none",
        "The decision is a type shape \u2014 'one immutable record per outcome carries exactly that case's fields' and 'an outcome that is denied and also paid cannot be written down'. The nearest alternative style is monolithic, and it is wrong because no deployable appears: the table with a type token is a storage detail for the same objects, and the constraints are about nullable fields, map keys and compiler checks.",
    ),
    "none-attempt-schedule-helper": (
        "none",
        "The deliverable is an algorithm \u2014 'the helper computes the delay before attempt n as a base interval multiplied by 2^(n-1)' \u2014 parameterised by budgets and a random source. The nearest alternative style is monolithic, and it is wrong because the text never describes a deployable or a boundary: the three client objects are dependencies passed in by callers, and the row is about the schedule they share.",
    ),
    "none-circuit-graph-single-save": (
        "none",
        "The subject is an object graph and its mapping \u2014 'one object that owns its parts ... persisted and loaded as one unit'. The nearest alternative style is monolithic, and it is wrong because naming five tables does not describe a deployable: no build, serving or scaling unit is in question, and the constraints are about write ordering, versions and query counts.",
    ),
    "none-credential-form-normalisation": (
        "none",
        "The proposal is an interface and result shape \u2014 'one small contract: a verifier accepts a credential value and returns either a normalised principal ... or a failure value'. The nearest alternative style is monolithic, and it is wrong because the text names four callers of one library, not one deployable: no build, release or scaling unit is described, and the constraints are about substitution, provisioning and log hygiene.",
    ),
    "none-effective-dated-material-cost": (
        "none",
        "The request is a property-interface decision \u2014 'agreement.cost().on(LocalDate) returns the value in force at that instant' \u2014 not a statement about deployed shape. The nearest alternative style is monolithic, and it is wrong because no deployable is described at all: the text stays inside a domain class and its accessors, so a label about build and scale units would be inventing facts.",
    ),
    "none-fare-search-shard-merge": (
        "none",
        "The scenario is explicitly a decomposition inside one method \u2014 'partitions the candidate set into equal shards, hands each shard to one of a fixed pool of four executor threads'. The nearest alternative style is monolithic, and it is wrong because nothing here is shown as one deployable serving work; thread count is a parameter of a function, not a deployment shape, and the text never says how anything is built, shipped or scaled.",
    ),
    "none-fused-instalment-expansion": (
        "none",
        "The row is about allocation and recursion structure \u2014 'a single recursion carries a seed ... and an accumulator ... so the intermediate structure is never materialised'. The nearest alternative style is monolithic, and it is wrong because nothing is deployed or scaled: the batch figure is a loop count over 400,000 in-memory computations, and the transaction boundary is a persistence detail of the same codebase.",
    ),
    "none-loyalty-earning-rule-objects": (
        "none",
        "The text is a code-shape question inside one module \u2014 'each rule should become a small named object with one isSatisfiedBy(candidate) method' \u2014 with no deployable, no inter-component boundary and nothing observable about a runtime. The nearest alternative style is monolithic, and it is wrong because monolithic is a positive finding about one deployable that serves and scales as a unit; here there is no unit to point at, only the representation of a rule in an existing codebase.",
    ),
    "none-ordered-slot-booking-stages": (
        "none",
        "The proposal is compile-time API shape \u2014 'constrains construction with types instead of validation' and 'only the last stage declares build()'. The nearest alternative style is monolithic, and it is wrong because the text describes no deployable: it describes which expressions a caller can write inside one Kotlin module, and the shipping platform is mentioned only to rule out reflection.",
    ),
    "none-per-row-sql-ownership": (
        "none",
        "The text is about where statements live \u2014 'an InventorySlot value ... exposes Find, Insert, Update and Delete methods that own the SQL for that table'. The nearest alternative style is monolithic, and it is wrong because naming a database does not describe a deployable: the decision never leaves the codebase's own packages, and nothing about serving, building or scaling a unit is at stake.",
    ),
    "none-recipe-tree-node-children": (
        "none",
        "The choice is a representation \u2014 'node classes declaring children in typed fields with their own names' versus 'one base class with a uniform list of children'. The nearest alternative style is monolithic, and it is wrong because nothing is deployed: the memory ceiling and the pass count describe the parser's own data structure, not the shape of a built or scaled system.",
    ),
    "none-roster-pipeline-owned-sequence": (
        "none",
        "The question is who calls whom inside one codebase \u2014 'a RosterPipeline class owns the sequence once ... the three rules sets plug in by supplying only the steps that differ'. The nearest alternative style is monolithic, and it is wrong because no deployable is described: the pipeline and the rules sets are types in one codebase, and the constraints are about hooks, exceptions and substitution, not about build or scale units.",
    ),
    "none-segment-kind-construction-table": (
        "none",
        "The decision is about a data structure and its population \u2014 'the importer owns an empty table of named construction functions, and each segment kind contributes its own entry'. The nearest alternative style is monolithic, and it is wrong because nothing is deployed: the importer is a function inside one binary, and the constraints concern trim safety, duplicates and error reporting, not deployment units.",
    ),
    "none-self-protecting-rights-envelope": (
        "none",
        "The requirement is on the object itself \u2014 'the carrier to protect its own designated fields' and 'encrypted ... at construction' \u2014 so the decision is a class contract, not a deployment shape. The nearest alternative style is monolithic, and it is wrong because the text names collaborators as call sites inside one codebase, not as parts of a deployable whose unit of build or scale is being chosen.",
    ),
    "none-serialized-roster-board-calls": (
        "none",
        "The decision is about how calls reach one object \u2014 'each caller receives a small object implementing the board's interface ... enqueues that request ... and returns the future immediately'. The nearest alternative style is monolithic, and it is wrong because no deployable is named: the contention and the queue live inside one JVM and one codebase, so a statement about build or scaling units would be unsupported.",
    ),
    "none-shared-model-pending-changes": (
        "none",
        "The text decides how in-memory objects and pending changes are shared \u2014 'each edit is recorded in a pending-change list keyed by the model object rather than by the session'. The nearest alternative style is monolithic, and it is wrong because no deployable of the codebase is described: there is no unit that is built, released or scaled, only a shared model graph and a flush ordering inside the editor's extension host.",
    ),
    "none-shared-ticketing-behaviour-object": (
        "none",
        "The question is who owns a method \u2014 'define the behaviours once, outside all four types, in a stand-alone collaborator whose functions take the ticketed item as an explicit parameter'. The nearest alternative style is monolithic, and it is wrong because there is no deployable being described: no build, serving or scaling unit appears, only the placement of shared logic among classes.",
    ),
    "none-single-collaborator-private-view": (
        "none",
        "The whole request is about compile-time access \u2014 'a RouteRebalancer class becomes the sole friend of RoundPlan' and re-exported nested aliases. The nearest alternative style is monolithic, and it is wrong because nothing is deployed: the constraints are RTTI, allocation and derivation rules, which describe how source compiles, not how a system is built or scaled.",
    ),
    "none-slab-walk-position-object": (
        "none",
        "Everything in the text is a data-structure decision \u2014 'adds a separate traversal value ... keeps the current slot position' \u2014 governed by borrow rules and allocation budgets. The nearest alternative style is monolithic, and it is wrong because no unit is built or scaled: the only runtime facts are shared borrows and per-item allocations inside one crate.",
    ),
    "none-supplier-client-interception": (
        "none",
        "The row is about a wrapper class and its delegation contract \u2014 'a wrapper class implementing the same five-method interface ... returns exactly what the vendor returned or rethrows exactly what it threw'. The nearest alternative style is monolithic, and it is wrong because the vendor client is a library dependency, not a described topology, and no deployable of the codebase under decision is characterised at all.",
    ),
    "none-tick-coalesced-status-flush": (
        "none",
        "The text decides a buffer's shape \u2014 'a change list keyed by technician id keeps only the latest value per row' and 'a single ordered flush at the end of the tick'. The nearest alternative style is monolithic, and it is wrong because although a database is mentioned, no deployable is described: there is no serving or scaling unit in question, only how a codebase talks to rows it already owns.",
    ),
    "none-total-case-analysis-outcomes": (
        "none",
        "The proposal is a type-closure and dispatch decision \u2014 'rewrites each of the three functions as a case analysis over the whole enum with no default branch'. The nearest alternative style is monolithic, and it is wrong because nothing is deployed: the persisted status token is a storage detail of a table the module reads, and the decision is about what compiles, not about a system's build or scale unit.",
    ),
    "none-trade-stage-type-parameter": (
        "none",
        "The evidence is compile-time API structure \u2014 'only Trade<Cleared> has settle' and 'a zero-sized type parameter marks the stage' \u2014 with nothing about deployment. The nearest alternative style is monolithic, and it is wrong because there is no deployable to be a unit: the scenario is which calls compile inside one library, and the persistence mention is a table the library reads, not a system shape.",
    ),
    "none-transitive-capability-wrapping": (
        "none",
        "The decision is about runtime values at a boundary inside one codebase \u2014 'wraps every value that crosses the boundary ... memoised in a WeakMap'. The nearest alternative style is monolithic, and it is wrong because nothing is deployed or scaled: the boundary is a capability seam inside one process, and the interesting artefacts are proxies and closures, not deployment units.",
    ),
    "pipe-and-filter-analyser-capability-wrapping": (
        "pipe-and-filter",
        "Style evidence: four successive stages fed by bounded channels, each on its own thread with its own queue and no shared mutable state, one stage movable and new analysers installed at start-up without editing the chain. Nearest alternative is microkernel, since extensions are added as packages, but the load-bearing mechanism shown is the stream through independent stages, and the decision concerns what crosses a stage boundary.",
    ),
    "pipe-and-filter-authorisation-case-exhaustive-model": (
        "pipe-and-filter",
        "Style evidence: stages joined by bounded queues, each on its own thread with its own queue and no reference to another, with the risk stage replayable over stored requests. Nearest alternative is layered-monolith, but the text describes a stream through independent, independently replayable stages, not a request path of layers calling one another.",
    ),
    "pipe-and-filter-billing-internal-view-forwarder": (
        "pipe-and-filter",
        "Style evidence: four stages connected by bounded channels, each on its own thread, unaware of the others, consuming and producing values with no cross-stage references, and stages removable or rerunnable independently. Nearest alternative is layered-monolith: no layer calls the next and no request path is described; the stream through independent stages is the mechanism.",
    ),
    "pipe-and-filter-charging-row-self-persistence": (
        "pipe-and-filter",
        "Style evidence: stages joined by bounded queues, each on its own worker with its own queue and no shared state, with the settlement stage replayable over a stored session. Nearest alternative is layered-monolith, but the text describes a stream of stages over sessions, not a request path with persistence layers beneath a presentation layer.",
    ),
    "pipe-and-filter-claims-intake-sealed-payload": (
        "pipe-and-filter",
        "Style evidence: a chain of independent stages wired by bounded channels, each a task of its own, holding no handle on another, reading and writing only the value handed to it, with one stage swappable or movable in the order. Nearest alternative is layered-monolith: these are not request-path layers inside one call but stream stages that can be reordered, which is what the text demonstrates.",
    ),
    "pipe-and-filter-cold-chain-reply-matching": (
        "pipe-and-filter",
        "Style evidence: stages joined by bounded queues, each on its own worker with its own queue and no reference to another, with the excursion stage replayable over stored shipments. Nearest alternative is layered-monolith, but the text describes a stream through independent stages, and the matching decision belongs to one stage rather than to a request path.",
    ),
    "pipe-and-filter-contact-centre-assignment-procedure": (
        "pipe-and-filter",
        "Style evidence: stages joined by bounded queues, each on its own worker with its own queue and no reference to another, with the priority stage replaceable by a replay simulation. Nearest alternative is layered-monolith, and the single-procedure shape inside a stage may suggest it, but the system's load-bearing mechanism is the stream through independent stages, not layers of one call.",
    ),
    "pipe-and-filter-crane-call-abandonment": (
        "pipe-and-filter",
        "Style evidence: stages joined by bounded channels, each in its own coroutine on its own dispatcher, holding no reference to another, with the crane stage rerunnable alone over a revised value. Nearest alternative is layered-monolith, but stages here are independent stream processors, not layers calling down a request path.",
    ),
    "pipe-and-filter-diagnostic-signal-subscription-hub": (
        "pipe-and-filter",
        "Style evidence: stages joined by bounded channels, each in its own coroutine on a dedicated dispatcher with its own bounded queue, holding no reference to another, with a stage removable and decoders added independently. Nearest alternative is event-driven, because notifications fan out to subscribers, but the load-bearing structure is a stream through replaceable stages, and the hub is wiring inside it, not the system's shape.",
    ),
    "pipe-and-filter-diagnostics-work-item-registry": (
        "pipe-and-filter",
        "Style evidence: stages connected by bounded channels, each in its own coroutine on its own dispatcher, no stage holding a reference to another or to shared mutable state, with stages replaceable and rerunnable independently. Nearest alternative is layered-monolith, but these are stream stages over work items, not request-path layers calling one another inside a single call.",
    ),
    "pipe-and-filter-ecrf-optional-capability-handle": (
        "pipe-and-filter",
        "Style evidence: stages joined by bounded channels, each on its own thread with its own queue and no reference to another, with the edit-check stage replayable alone over stored input. Nearest alternative is layered-monolith, but these are independent stream stages rather than request-path layers, and each can be replayed or replaced on its own.",
    ),
    "pipe-and-filter-entitlement-claim-signing-chain": (
        "pipe-and-filter",
        "Style evidence: stages joined by bounded channels, each its own task on its own thread with its own bounded queue and no reference to another, with the claim stage replaceable and the stage list assembled from configuration so a stage can be inserted. Nearest alternative is layered-monolith, but these are independently running stream stages rather than request-path layers.",
    ),
    "pipe-and-filter-feeder-lock-version-token": (
        "pipe-and-filter",
        "Style evidence: stages joined by bounded queues, each on its own thread with its own queue and no in-memory reference to another, with the constraint stage independently replaceable. Nearest alternative is layered-monolith, but the mechanism shown is a stream through independent stages; the shared database row the text calls out is a persistence fact, not a layer structure.",
    ),
    "pipe-and-filter-identity-proofing-switch": (
        "pipe-and-filter",
        "Style evidence: stages joined by bounded queues, each on its own worker thread with its own queue, holding no reference to another, with the match stage replayable alone over stored captures. Nearest alternative is layered-monolith, but no request path of layers is described; the mechanism is a stream through independent stages, one of which switches implementations.",
    ),
    "pipe-and-filter-irrigation-access-family": (
        "pipe-and-filter",
        "Style evidence: a chain of stages joined by bounded queues, each in its own thread holding no reference to another, with the model stage swappable without touching the rest. Nearest alternative is layered-monolith, but these stages process a stream and are independently replaceable rather than layers of a request path calling one another.",
    ),
    "pipe-and-filter-label-spec-staged-assembly": (
        "pipe-and-filter",
        "Style evidence: stages joined by bounded channels, each in its own coroutine on its own dispatcher with its own queue, holding no reference to another stage, with the composition stage rerunnable over the same value. Nearest alternative is layered-monolith, but these are stream stages, not request-path layers calling down; the text makes that explicit by giving each stage its own thread and queue.",
    ),
    "pipe-and-filter-optical-port-permit-pool": (
        "pipe-and-filter",
        "Style evidence: a chain of stages over an order stream, each stage a worker on its own thread with its own bounded queue, holding no reference to the others, with the verification stage detachable and rerunnable against the same value. Nearest alternative is layered-monolith: the stages are stream stages rather than layers of one request, and the text says so by making them independently threaded and reorderable.",
    ),
    "pipe-and-filter-rating-rule-grammar-functions": (
        "pipe-and-filter",
        "Style evidence: stages joined by bounded channels, each its own task on its own thread with its own bounded queue, holding no reference to another, with the discount stage reorderable per plan. Nearest alternative is monolithic, since one service is described, but the load-bearing mechanism in view is the stream through independent stages that the compiled rules feed.",
    ),
    "pipe-and-filter-rating-step-driver-loop": (
        "pipe-and-filter",
        "Style evidence: stages joined by bounded channels, each in its own coroutine on its own dispatcher, holding no reference to another, with the tax stage swappable per country. Nearest alternative is layered-monolith, but the stages are independent stream processors with their own threads and queues, not layers calling down inside one request.",
    ),
    "pipe-and-filter-settlement-instrument-sharing": (
        "pipe-and-filter",
        "Style evidence: a chain of stages joined by bounded queues, each on its own thread, unable to see another's state, with stages skippable and reorderable per market. Nearest alternative is layered-monolith, since stages might read as layers, but they process a stream and can be reordered - unlike a request path whose layers call the next one down.",
    ),
    "pipe-and-filter-yard-stage-base-type": (
        "pipe-and-filter",
        "Style evidence: stages joined by bounded channels, each its own goroutine with its own queue, no shared mutable state, with the gate stage replayable over the same value. Nearest alternative is modular-monolith, since one deployable is described, but the load-bearing structure shown is a stream through independent, reorderable stages rather than vertical modules inside a process.",
    ),
    "presentation-abstraction-control-auction-agent-self-describing": (
        "presentation-abstraction-control",
        "Each screen is a presentation/abstraction/control triple, and only control agents send information across unit boundaries.",
    ),
    "presentation-abstraction-control-billing-agent-typed-groups": (
        "presentation-abstraction-control",
        "Console screens are triples of presentation, abstraction and control agents, with control agents owning inter-screen flow.",
    ),
    "presentation-abstraction-control-booking-agent-capability": (
        "presentation-abstraction-control",
        "Each unit is three cooperating agents and inter-unit flows pass only through control agents.",
    ),
    "presentation-abstraction-control-charging-agent-active-records": (
        "presentation-abstraction-control",
        "Every panel is a hierarchy of presentation, abstraction and control agents, and panels interact only through control agents.",
    ),
    "presentation-abstraction-control-claims-agent-restricted-creation": (
        "presentation-abstraction-control",
        "Screens are hierarchies of presentation, abstraction and control agents, and only control agents mediate between units.",
    ),
    "presentation-abstraction-control-contact-centre-agent-locking": (
        "presentation-abstraction-control",
        "Each panel pairs presentation, abstraction and control agents, and only control agents mediate between panels.",
    ),
    "presentation-abstraction-control-contact-centre-dialog-agents": (
        "presentation-abstraction-control",
        "Every screen is a triple of presentation, abstraction and control agents, and screens talk only through their control agents.",
    ),
    "presentation-abstraction-control-dispatch-agent-contracts": (
        "presentation-abstraction-control",
        "Each console screen is a presentation/abstraction/control triple, with control agents owning cross-screen flows.",
    ),
    "presentation-abstraction-control-enrolment-agent-operations": (
        "presentation-abstraction-control",
        "Every screen unit pairs presentation, abstraction and control agents, and flows between units pass only through control agents.",
    ),
    "presentation-abstraction-control-freight-agent-transform-chain": (
        "presentation-abstraction-control",
        "Every screen is a presentation/abstraction/control triple, with flows mediated by control agents.",
    ),
    "presentation-abstraction-control-inventory-agent-state-flow": (
        "presentation-abstraction-control",
        "Every screen pairs a presentation agent with an abstraction agent and a control agent, with control mediating between screens.",
    ),
    "presentation-abstraction-control-maintenance-agent-variant-elimination": (
        "presentation-abstraction-control",
        "Each panel is a hierarchy of presentation, abstraction and control agents, with control owning inter-panel flows.",
    ),
    "presentation-abstraction-control-procurement-agent-construction": (
        "presentation-abstraction-control",
        "Each screen in the console is a presentation/abstraction/control triple, with only control agents mediating between screens.",
    ),
    "presentation-abstraction-control-sensor-agent-layer-contract": (
        "presentation-abstraction-control",
        "Each unit of the console is a triple of cooperating agents, with no direct presentation-to-presentation coupling.",
    ),
    "presentation-abstraction-control-telematics-agent-cluster-lock": (
        "presentation-abstraction-control",
        "Each panel is a triple of cooperating agents, and panels interact only through their control agents.",
    ),
    "presentation-abstraction-control-telematics-agent-row-access": (
        "presentation-abstraction-control",
        "Each panel is a hierarchy of presentation, abstraction and control agents, with control mediating between panels.",
    ),
    "presentation-abstraction-control-waste-agent-permission": (
        "presentation-abstraction-control",
        "Each depot screen is a triple of cooperating agents, and every cross-screen request passes through a control agent.",
    ),
    "reactive-architecture-billing-reference-monitor": (
        "reactive-architecture",
        "Each consumer pulls rated events at its own rate, so the design is bounded by demand rather than buffering.",
    ),
    "reactive-architecture-catalogue-command-ship": (
        "reactive-architecture",
        "The service never pushes beyond what a client requests, so its flow is governed by explicit downstream demand.",
    ),
    "reactive-architecture-charging-connector-families": (
        "none",
        "Both blind annotators labelled this none where the authoring intent was reactive-architecture",
    ),
    "reactive-architecture-claims-stable-identity": (
        "reactive-architecture",
        "Work arrives only as fast as assessor benches request it, keeping the pipeline within a bounded window.",
    ),
    "reactive-architecture-claims-thread-scoped-context": (
        "reactive-architecture",
        "Results reach each workbench only as fast as it requests them, which bounds the pipeline.",
    ),
    "reactive-architecture-claims-transitive-restriction": (
        "reactive-architecture",
        "Emission is paced by each ledger consumer's request window, not by unbounded queueing.",
    ),
    "reactive-architecture-diagnostics-event-channel": (
        "reactive-architecture",
        "Subscribers' declared capacity governs emission, so the flow is bounded by demand instead of queue growth.",
    ),
    "reactive-architecture-enrolment-role-interface": (
        "reactive-architecture",
        "Event delivery follows each client's requested window rather than a push loop.",
    ),
    "reactive-architecture-forecast-window-values": (
        "reactive-architecture",
        "Emissions follow each subscriber's requested rate, so the pipeline is bounded by demand rather than a queue.",
    ),
    "reactive-architecture-imagery-band-associations": (
        "reactive-architecture",
        "The source publishes at the rate downstream consumers signal they can accept, so the design is organised around bounded demand rather than unbounded buffering.",
    ),
    "reactive-architecture-imagery-restricted-construction": (
        "reactive-architecture",
        "Consumers pull tiles at their own rate and the server emits within that window.",
    ),
    "reactive-architecture-maintenance-derived-cache": (
        "reactive-architecture",
        "Subscribers pull at the rate they can process and the service emits accordingly, so the flow is demand-driven.",
    ),
    "reactive-architecture-matching-currency-amounts": (
        "reactive-architecture",
        "Fares are emitted only as fast as the client pulls them, so the flow is bounded by demand.",
    ),
    "reactive-architecture-matching-foreign-model-boundary": (
        "reactive-architecture",
        "Apps request offers at their own rate and the service never exceeds that window, which is what shapes the design.",
    ),
    "reactive-architecture-pos-isolated-counters": (
        "reactive-architecture",
        "The store serves price deltas at the rate each till requests them, so the flow is governed by demand.",
    ),
    "reactive-architecture-procurement-batched-writes": (
        "reactive-architecture",
        "Update emission is bounded by what each bidder requests, so the service's flow is demand-shaped.",
    ),
    "reactive-architecture-roster-store-family": (
        "reactive-architecture",
        "Each crew app pulls legality results at its own rate, so the service is bounded by demand.",
    ),
    "reactive-architecture-telematics-sensitive-stream": (
        "reactive-architecture",
        "Emission follows each consumer's request window, so the pipeline is bounded by downstream demand.",
    ),
    "reactive-architecture-ticketing-now-or-never": (
        "reactive-architecture",
        "Each box office pulls availability at its own rate, so the flow is bounded by demand.",
    ),
    "reactive-architecture-trial-threshold-language": (
        "reactive-architecture",
        "Monitors pull readings at their own rate, which bounds the service's emission.",
    ),
    "reactive-architecture-waste-stream-enhancement": (
        "reactive-architecture",
        "Emission follows each dashboard's requested window rather than an unbounded push.",
    ),
    "reflection-architecture-auction-rule-specialisation": (
        "reflection-architecture",
        "Rule classes are inspected at start-up and their annotated parameters drive per-house adaptation.",
    ),
    "reflection-architecture-banking-eligibility-combinators": (
        "reflection-architecture",
        "Rule assemblies are discovered by inspecting annotated classes at start-up, so the pipeline's shape is decided at run time.",
    ),
    "reflection-architecture-billing-store-bound-family": (
        "reflection-architecture",
        "The mediation runtime inspects its own deployment descriptor and switches behaviour at start-up rather than at build time.",
    ),
    "reflection-architecture-catalogue-cluster-guard": (
        "reflection-architecture",
        "Entitlement rules are discovered by inspecting annotated classes at start-up and adapt behaviour per territory.",
    ),
    "reflection-architecture-contact-centre-behaviour-test": (
        "reflection-architecture",
        "Routing rules are discovered by inspecting annotated classes at start-up, so behaviour comes from what the runtime finds.",
    ),
    "reflection-architecture-dispatch-shared-schedule": (
        "reflection-architecture",
        "Routing policies are discovered by inspecting annotated classes at start-up and the runtime adapts per region.",
    ),
    "reflection-architecture-freight-runtime-registration": (
        "reflection-architecture",
        "The application scans its own assemblies at start-up and builds its processing chain from what it discovers.",
    ),
    "reflection-architecture-identity-collection-encapsulation": (
        "reflection-architecture",
        "Model collections are discovered by inspecting annotated fields at start-up and drive the runtime's validators.",
    ),
    "reflection-architecture-lab-scoped-access-point": (
        "reflection-architecture",
        "The application inspects its own assemblies at start-up, discovers annotated units and rewires its object graph without a compile-time reference to them.",
    ),
    "reflection-architecture-loyalty-runtime-templates": (
        "reflection-architecture",
        "The runtime inspects its own annotated model at start-up and generates behaviour from what it finds, which is self-adaptation at run time.",
    ),
    "reflection-architecture-matching-compile-time-tag": (
        "reflection-architecture",
        "Policies are discovered by inspecting annotated classes at start-up and mounted without a compile-time reference.",
    ),
    "reflection-architecture-origination-offline-consumer": (
        "reflection-architecture",
        "Consumers are discovered by inspecting annotated classes at start-up and wired without a compile-time reference.",
    ),
    "reflection-architecture-pos-partial-focus": (
        "reflection-architecture",
        "View descriptors are discovered by inspecting the runtime's own classes at start-up and bindings follow them.",
    ),
    "reflection-architecture-scheduling-value-operations": (
        "reflection-architecture",
        "Value classes and their operations are discovered by inspecting annotations at start-up and exposed automatically.",
    ),
    "reflection-architecture-telematics-table-runtime": (
        "reflection-architecture",
        "Handlers are discovered by inspecting annotated classes at start-up, so the runtime's structure follows what it finds.",
    ),
    "reflection-architecture-ticketing-optional-capabilities": (
        "reflection-architecture",
        "The runtime reads annotated capabilities out of its own driver set at start-up and enables features per terminal.",
    ),
    "reflection-architecture-ticketing-thin-framework-element": (
        "reflection-architecture",
        "Lifecycle handlers are discovered by inspecting annotated classes at start-up and invoked by the runtime.",
    ),
    "reflection-architecture-trial-fallible-verdict": (
        "reflection-architecture",
        "The validation chain is assembled from validators the runtime discovers in its own classes at start-up.",
    ),
    "reflection-architecture-trial-record-row-behaviour": (
        "reflection-architecture",
        "The visit runtime discovers annotated fields by inspecting its own classes at start-up and adapts its persistence behaviour.",
    ),
    "reflection-architecture-underwriting-burst-buffer": (
        "reflection-architecture",
        "Extensions are discovered by inspecting annotated classes at start-up and mounted into the pipeline, so the runtime assembles itself from what it finds.",
    ),
    "rule-based-system-authorization-unevaluable-input": (
        "rule-based-system",
        "Style evidence: 'maintained by risk analysts in a decision-table editor, published as versioned data, and evaluated by a TypeScript service built on json-rules-engine'. Nearest alternative style event-driven: messages are the carrier, but what the platform sells is the authored clause evaluation of each authorization; the queue's blocking behaviour is the symptom, not the style.",
    ),
    "rule-based-system-building-schedule-edit-lock": (
        "rule-based-system",
        "Style evidence: 'Facility engineers ... author occupancy and setpoint clauses in a web editor', published 'as a versioned package that a decision service evaluates every 15 minutes against zone sensor facts'; 'a threshold change is a data release'. Nearest alternative style aiml-centric: the service predicts nothing; every setpoint is derived from authored clauses and evaluated, so there is no training, registry or served model.",
    ),
    "rule-based-system-catalogue-rating-band-type": (
        "rule-based-system",
        "Style evidence: 'Editorial policy is authored as clauses ... published as versioned data to a Go decision service that evaluates them for each catalogue request'; 'the bundle is hot-swapped and every response records the bundle version'. Nearest alternative style modular-monolith: the interior is partitioned, but the system is one deployed evaluation service whose value is the authored clause base, and the decision is about one type's dependencies.",
    ),
    "rule-based-system-claim-action-implementation-choice": (
        "rule-based-system",
        "Style evidence: 'Claims specialists author the clause set in a decision-table workbook ... the build publishes it as a signed bundle that a decision service evaluates over each claim's fact set', and 'nothing in the adjudication service is rebuilt when clause wording changes'. Nearest alternative style microkernel-plugin: action families vary, but they are compiled-in specialisations, not extensions discovered and installed at start-up.",
    ),
    "rule-based-system-cold-chain-position-lookup": (
        "rule-based-system",
        "Style evidence: 'a Go decision service evaluates excursion clauses ... authored by quality assurance staff as versioned sets and published to the service as data, so a stability-profile change never ships a binary'. Nearest alternative style none: shipments, a deployed service, loggers and an archive are all topology, so the position-structure question sits inside a real system rather than replacing one.",
    ),
    "rule-based-system-deviation-outcome-variants": (
        "rule-based-system",
        "Style evidence: 'Site monitors maintain deviation clauses as DMN decision tables ... deployed as versioned packages to a decision service that evaluates each subject visit as it is entered', with a clause package retrievable for audit. Nearest alternative style monolithic: the surveillance service is one deployable, but the text's load-bearing mechanism is the authored decision-table base plus its evaluation cycle, which is what a reader must not lose.",
    ),
    "rule-based-system-dispatch-reservation-now-or-refuse": (
        "rule-based-system",
        "Style evidence: 'several hundred JSON rules authored by dispatch supervisors', 'loaded by a Python evaluator ... which re-reads the published rule set whenever the bundle is deployed; no application release is involved'. The rules are data owned by dispatchers, and the engine's evaluation supplies eligibility and priority. Nearest alternative aiml-centric: the text says 'scores', but no model is trained, registered or versioned \u2014 the ranking is a rule outcome.",
    ),
    "rule-based-system-fare-tree-editing-position": (
        "rule-based-system",
        "Style evidence: 'Tariff analysts maintain them in a hierarchical editor', and published clause versions 'are signed and loaded by the fare calculation service, which evaluates them per journey and returns a fare plus the clause chain that produced it'. Nearest alternative style none: a deployed service, a versioned data release and a browser app are all described, so the tree-editing decision is a code-shape decision inside a real topology.",
    ),
    "rule-based-system-form-evaluation-callback-delivery": (
        "rule-based-system",
        "Style evidence: 'The clause base is maintained by medical monitors in a decision-table console ... published as a signed package to a decision service that evaluates each submitted form version; the engine writes intermediate grades back into the evaluation context'. Nearest alternative style event-driven: forms do arrive as records, but the described mechanism and its cost are the authored clause evaluation, not the transport.",
    ),
    "rule-based-system-guidance-fetch-attempt-schedule": (
        "rule-based-system",
        "Style evidence: 'a Drools decision service that applies forecaster-authored corrections ... in a versioned container the pipeline picks up without a restart'. Behaviour on every forecast is authored clause data evaluated by the engine. Nearest alternative style aiml-centric: the phrase 'model guidance' names upstream numerical products, and no component here is trained, registered or served as a prediction.",
    ),
    "rule-based-system-maintenance-parts-lookup-table": (
        "rule-based-system",
        "Style evidence: 'reliability engineers author the trigger clauses ... in a small declarative syntax', 'clause bundles are published from a maintenance portal and reloaded in place, so a threshold change is a data release rather than a firmware update'. Nearest alternative style microkernel-plugin: adapters are swapped per site, but the load-bearing mechanism is the authored clause base and its evaluation, not a core extended by packaged extensions.",
    ),
    "rule-based-system-moderation-verdict-creation": (
        "rule-based-system",
        "Style evidence: 'about 900 JSON rule documents ... authored by policy analysts in an internal console, versioned, and evaluated by json-rules-engine inside a Node service that hot-swaps the rule set on a signed publish'. Policy change is a data publish. Nearest alternative style aiml-centric: the rules read model scores as inputs, but the value-bearing component is the authored policy, with no training or registry in the text.",
    ),
    "rule-based-system-pharmacy-rule-package-exchange": (
        "rule-based-system",
        "Style evidence: 'about 4,000 declarative entries ... authored by a clinical governance board in a rule editor and published as versioned packages to a Kotlin decision service that evaluates each prescription'; changes 'land several times a day' as data. Nearest alternative style reactive-architecture: the package swap sounds like a hot-swap concern, but the system's value is the authored clause evaluation and its atomic visibility, not backpressure.",
    ),
    "rule-based-system-reservation-evaluation-worker-handoff": (
        "rule-based-system",
        "Style evidence: 'a versioned set of declarative packages ... maintained by revenue managers in an editor and published to a decision service that reloads them from a signed bundle, so a rate change never waits for an application release', plus the evaluation step between intake and the inventory hold. Nearest alternative style reactive-architecture: bounded queues and backpressure appear, but the load-bearing mechanism is the authored rule base; responsiveness is a budget, not the design's purpose.",
    ),
    "rule-based-system-settlement-outcome-announcement": (
        "rule-based-system",
        "Style evidence: 'Operations staff author the clause set ... the compiled package is deployed to a settlement decision service that evaluates each open instruction', with clause changes as a reviewed data release. Nearest alternative style event-driven: the record is published to receivers, but transport is not the mechanism under decision \u2014 the settlement service's value is the authored clause evaluation, and the item's immutability is the question.",
    ),
    "rule-based-system-specimen-record-two-hierarchies": (
        "rule-based-system",
        "Style evidence: 'Laboratory medicine staff maintain about 1,900 DMN decision tables' and 'publish new table versions ... without any of the four consuming applications being rebuilt'; the service 'matches the tables in dependency order, writes intermediate outputs back into the evaluation context'. Behaviour is authored declarative data evaluated by an engine. Nearest alternative style aiml-centric: there is no fitted scorer, no training pipeline and no registry \u2014 every threshold is a table row a clinician owns.",
    ),
    "rule-based-system-telematics-row-gateway-boundary": (
        "rule-based-system",
        "Style evidence: 'clauses are written in GRL text by fleet compliance analysts, stored in a Git repository, and loaded into a grule instance at start-up and on a signed hot reload; a clause change is a pull request and a bundle publish, never a service rebuild'. Nearest alternative style layered-monolith: the persistence boundary looks like a layer, but the system is several deployables and what it would lose is the authored clause base, not layer ordering.",
    ),
    "rule-based-system-telematics-score-and-justification": (
        "rule-based-system",
        "Style evidence: 'safety clauses authored by fleet-safety consultants ... maintained in a rule editor and published as versioned data that the engine evaluates over each trip's event stream; a clause change never ships a new binary'. Nearest alternative style aiml-centric: there is a numeric score, but no fitted component, no training pipeline and no registry \u2014 the score is defined by authored clauses.",
    ),
    "rule-based-system-tender-stage-composition": (
        "rule-based-system",
        "Style evidence: 'procurement officers' author a 'clause base ... maintained in a clause register and published as versioned data to an evaluation service; a tender round can change the clause set with no deployment', and every award is reproducible against the clause version. Nearest alternative style monolithic: one service, but the load-bearing mechanism is the authored clause base plus its evaluation, and the text's decision is about composing that evaluation.",
    ),
    "rule-based-system-underwriting-referral-row-behaviour": (
        "rule-based-system",
        "Style evidence: 'Underwriters maintain the appetite clauses themselves in a guided editor ... published as a versioned package to a Drools service'. Clause changes are data; the engine's evaluation of them sets the band. Nearest alternative style layered-monolith: the workbench has a persistence edge, but what the system would lose is the authored clause base, not a layer stack, and the clause service is shared by three callers.",
    ),
    "rule-based-system-verification-batch-scoring-module": (
        "rule-based-system",
        "Style evidence: 'verification scientists maintain a clause table ... as a Git-tracked CSV, and a Rust service compiles it into a package that KusionStack's engine evaluates over each forecast-observation pair'. Nearest alternative style none: there is a deployed nightly service with a real inter-component boundary, so the table-shaped code decision does not remove the topology.",
    ),
    "rule-based-system-warehouse-dispatch-neighbour-calls": (
        "rule-based-system",
        "Style evidence: 'Dispatch policy is authored as clauses by operations engineers ... published as versioned data to the dispatch decision service, which evaluates a robot's situation and issues a goal', with 'no vehicle firmware or dispatch binary rebuilt'. Nearest alternative style actor-based: robots are addressed units, but the dispatch decision is produced by evaluating authored clauses, and the robots' own isolation is not what the text is deciding.",
    ),
    "saga-banking-crossparty-enhancer": (
        "saga",
        "Account opening runs local steps against three parties' own APIs, registering a compensation for each step.",
    ),
    "saga-billing-crossparty-entity": (
        "saga",
        "Number porting executes local steps against three parties' APIs, reversing earlier steps when the registry rejects.",
    ),
    "saga-charging-crossparty-ignorable": (
        "saga",
        "Roaming settlement runs local steps against three parties' APIs, reversing the session when a network rejects.",
    ),
    "saga-claims-crossparty-singleflight": (
        "saga",
        "Reimbursement executes local steps against three parties' own APIs, reversing reservations when payment fails.",
    ),
    "saga-claims-crossparty-strategy": (
        "saga",
        "Recovery runs local steps against three parties' APIs, reversing the claim when the reinsurer declines.",
    ),
    "saga-contact-crossparty-worker": (
        "saga",
        "Restoration runs as local steps against two parties' own APIs, reversing the parts order when supply fails.",
    ),
    "saga-diagnostics-crossparty-audit": (
        "saga",
        "The repair's steps execute against three parties' own APIs, with reversal when the manufacturer declines.",
    ),
    "saga-dispatch-crossparty-temporal": (
        "saga",
        "Warranty repair runs local steps against three parties' APIs, reversing the parts order when the manufacturer declines.",
    ),
    "saga-enrolment-crossparty-repository": (
        "saga",
        "Enrolment executes local steps against three parties' own APIs, reversing earlier steps when a later one fails.",
    ),
    "saga-identity-crossparty-grant": (
        "saga",
        "Verification executes local steps against three parties' own APIs, reversing the biometric request when consent is withdrawn.",
    ),
    "saga-loyalty-crossparty-converter": (
        "saga",
        "Redemption executes local steps against three parties' own APIs, releasing the reservation when fulfilment fails.",
    ),
    "saga-origination-crossparty-thin": (
        "saga",
        "The application's steps execute against three parties' own APIs, with reversal when the valuation is unacceptable.",
    ),
    "saga-pharmacy-crossparty-version": (
        "saga",
        "The authorisation runs local steps against three parties' APIs, reversing the reservation when the insurer refuses.",
    ),
    "saga-review-crossparty-command": (
        "saga",
        "Filing executes local steps against three parties' own APIs, compensating when the court rejects.",
    ),
    "saga-robot-crossparty-concept": (
        "saga",
        "Each handover runs as local steps against three parties' own APIs, with re-booking as the compensation when a later step fails.",
    ),
    "saga-settlement-crossparty-decomposition": (
        "saga",
        "The run's steps execute against four custodians' own APIs, with compensation when a submission is rejected.",
    ),
    "saga-settlement-money-crossparty": (
        "saga",
        "The margin call executes local steps against three parties' own APIs, reversing the reservation when the bank rejects.",
    ),
    "saga-settlement-swap-handles": (
        "saga",
        "The netting run coordinates steps across three custodians the platform does not control, with compensations when one rejects.",
    ),
    "serverless-banking-function-capability-mark": (
        "serverless",
        "Each document invokes a platform-managed function that scales to zero between events.",
    ),
    "serverless-card-function-concurrent-calls": (
        "serverless",
        "Each authorization invokes a function the platform may schedule concurrently in a reused container.",
    ),
    "serverless-claims-event-handler-shape": (
        "serverless",
        "Each submission event invokes a provider-managed function that scales to zero, and the platform owns the runtime and the scaling.",
    ),
    "serverless-claims-function-health": (
        "serverless",
        "Each document step invokes a platform-managed function, and routing decisions happen per invocation.",
    ),
    "serverless-construction-function-grant": (
        "serverless",
        "Each bid invokes a platform-managed function that scales to zero between events, so the grant must survive outside the process.",
    ),
    "serverless-contact-function-restricted-construction": (
        "serverless",
        "Each interaction invokes a function the platform runs and scales, with no long-lived process.",
    ),
    "serverless-grid-function-audit": (
        "serverless",
        "Each threshold crossing invokes a function the platform runs and scales, so the runtime and scaling are not the team's.",
    ),
    "serverless-hotel-function-tier-contract": (
        "serverless",
        "Edge, aggregation and adapter functions each run on a platform that scales them to zero independently.",
    ),
    "serverless-hvac-function-single-instance": (
        "serverless",
        "Each schedule tick invokes a platform-managed function whose container may be reused across ticks.",
    ),
    "serverless-identity-function-latch": (
        "serverless",
        "Each verification invokes a platform-managed function that scales to zero and must finish inside one invocation timeout.",
    ),
    "serverless-imagery-function-configured-double": (
        "serverless",
        "Handlers run as platform-managed functions invoked per tile event, so the collaborator boundary is the test seam.",
    ),
    "serverless-industrial-function-open-protocol": (
        "serverless",
        "Each machine event invokes a function the platform runs and scales to zero.",
    ),
    "serverless-lab-function-partial-response": (
        "serverless",
        "Each read request invokes a platform-managed function with a hard invocation timeout and no warm server.",
    ),
    "serverless-mortgage-function-quantities": (
        "serverless",
        "Each upload event invokes a function the platform runs and scales to zero between events.",
    ),
    "serverless-pharmacy-function-table-access": (
        "serverless",
        "Each dispense invokes a function the platform runs and scales to zero, with no database connection held between events.",
    ),
    "serverless-roster-function-typestate": (
        "serverless",
        "Every roster submission invokes a function the platform runs and scales to zero between events.",
    ),
    "serverless-streaming-job-creation": (
        "serverless",
        "Every upload invokes a function the platform runs and scales, with no server held warm between events.",
    ),
    "serverless-telematics-function-lazy": (
        "serverless",
        "Each trip event invokes a function the platform runs and scales to zero, so per-invocation cost dominates.",
    ),
    "serverless-travel-function-pricing": (
        "serverless",
        "Each supplier event invokes a platform-managed function that scales to zero, with no warm process held.",
    ),
    "serverless-trial-function-behaviour-test": (
        "serverless",
        "Each visit invokes a platform-managed function that scales to zero between events.",
    ),
    "service-mesh-booking-event-hub": (
        "service-mesh",
        "Peer authentication, retries and timeouts are applied by a proxy beside every instance rather than by the services.",
    ),
    "service-mesh-container-port-temporal": (
        "service-mesh",
        "Every instance's traffic is authenticated and retried by an infrastructure layer beside it, so resilience and identity live outside the service code.",
    ),
    "service-mesh-diagnostics-frame-sharing": (
        "service-mesh",
        "Traffic between instances is authenticated, timed and version-split by a proxy layer, so no service carries networking or resilience code.",
    ),
    "service-mesh-dispatch-traffic-infrastructure": (
        "service-mesh",
        "Every pod carries an injected proxy that terminates mutual TLS, applies per-route timeouts and retries and reports telemetry, all governed by a control plane rather than by the services themselves.",
    ),
    "service-mesh-enrolment-transitive-restriction": (
        "service-mesh",
        "All inter-service traffic is authenticated and timed by an infrastructure layer beside each instance rather than by the services.",
    ),
    "service-mesh-forecast-traffic-vocabulary": (
        "service-mesh",
        "Instance-to-instance traffic is authenticated, retried and split by an infrastructure layer the services do not contain, with weights changed by the operator rather than by a deployment.",
    ),
    "service-mesh-forecast-usecase-boundary": (
        "service-mesh",
        "Instance traffic is authenticated and retried by a proxy layer rather than by the services, so the estate's traffic discipline is infrastructure.",
    ),
    "service-mesh-grid-producer-evolution": (
        "service-mesh",
        "Peer authentication, retry budgets and version splits are applied by a proxy beside every instance rather than inside the services.",
    ),
    "service-mesh-hvac-tamper-evident-log": (
        "service-mesh",
        "Instance-to-instance calls are authenticated and retried by a proxy layer, so the services carry no transport security code of their own.",
    ),
    "service-mesh-imagery-narrow-grant": (
        "service-mesh",
        "Peers authenticate through a proxy beside every instance and retries and timeouts are applied there, so the services carry no traffic-management code.",
    ),
    "service-mesh-imagery-scope-symbols": (
        "service-mesh",
        "Traffic between services is authenticated and retried by a proxy layer beside each instance, so the service carries no networking code.",
    ),
    "service-mesh-lab-analyzer-preopen": (
        "service-mesh",
        "Peer authentication, timeouts and retries are applied by a proxy beside each instance, so the service itself carries no traffic-management code.",
    ),
    "service-mesh-review-memory-footprint": (
        "service-mesh",
        "Peers authenticate through a proxy beside each instance and the service holds no networking code, so traffic management lives entirely in that infrastructure layer.",
    ),
    "service-mesh-sensor-coarse-facade": (
        "service-mesh",
        "Inter-service calls are authenticated and retried by an infrastructure layer beside each instance, so the estate's traffic discipline is not written in the services.",
    ),
    "service-mesh-sensor-read-cancellation": (
        "service-mesh",
        "The readers' traffic is authenticated and timed by an infrastructure layer beside each instance, with the service holding no such code.",
    ),
    "service-mesh-settlement-batched-flush": (
        "service-mesh",
        "Peer authentication, retry budgets and traffic splits are applied by a proxy beside each instance, not inside the services.",
    ),
    "service-mesh-till-payload-column": (
        "service-mesh",
        "Peer authentication, retry budgets and traffic splitting are applied by a proxy beside each instance, so no service contains traffic-management code.",
    ),
    "service-mesh-till-session-authority": (
        "service-mesh",
        "Instance-to-instance authentication, timeouts and retries are infrastructure beside each instance, and the session authority sits above that layer.",
    ),
    "service-oriented-architecture-banking-capability-readers": (
        "service-oriented-architecture",
        "Nine independently owned capability contracts are versioned in a registry and composed through it.",
    ),
    "service-oriented-architecture-billing-plan-association": (
        "service-oriented-architecture",
        "Eight capabilities owned by different teams are consumed through governed, versioned contracts held in a registry.",
    ),
    "service-oriented-architecture-charging-capability-window": (
        "service-oriented-architecture",
        "Eight independently owned capability contracts are composed through a registry, so the spanning session crosses published contracts only.",
    ),
    "service-oriented-architecture-claims-capability-distinct-ids": (
        "service-oriented-architecture",
        "Ten independently owned capability contracts are composed through a registry, and every interaction crosses a published contract.",
    ),
    "service-oriented-architecture-contact-capability-narrow-view": (
        "service-oriented-architecture",
        "Eight independently owned capability contracts are versioned in a registry and composed only through those contracts.",
    ),
    "service-oriented-architecture-inventory-capability-identity": (
        "service-oriented-architecture",
        "Ten independently owned capability contracts are composed through a registry, so every interaction crosses a published contract.",
    ),
    "service-oriented-architecture-inventory-capability-parser": (
        "service-oriented-architecture",
        "Contract documents owned by ten different partners are discovered and versioned through one registry.",
    ),
    "service-oriented-architecture-loyalty-capability-deferred": (
        "service-oriented-architecture",
        "The rewards estate composes nine independently owned capability contracts held in a registry, and every call crosses a published contract.",
    ),
    "service-oriented-architecture-loyalty-capability-restriction": (
        "service-oriented-architecture",
        "Nine independently owned capability contracts are composed through a registry, so every call crosses a published contract boundary.",
    ),
    "service-oriented-architecture-port-capability-columns": (
        "service-oriented-architecture",
        "The customs estate composes nine capability contracts published by six organisations and versioned in a registry, so the contracts are the integration surface.",
    ),
    "service-oriented-architecture-port-capability-deferred-read": (
        "service-oriented-architecture",
        "Nine independently owned capability contracts are versioned in a registry and composed only through those contracts.",
    ),
    "service-oriented-architecture-pos-capability-invalid-input": (
        "service-oriented-architecture",
        "Nine independently owned capability contracts are versioned in a registry and composed only through it.",
    ),
    "service-oriented-architecture-pos-capability-payload": (
        "service-oriented-architecture",
        "Nine independently owned capability contracts are composed through a registry, so every consumer interaction crosses a published contract.",
    ),
    "service-oriented-architecture-roster-capability-skeleton": (
        "service-oriented-architecture",
        "Nine independently owned capability contracts are composed through a registry, so every invocation crosses a published contract.",
    ),
    "service-oriented-architecture-scheduling-adr-contract": (
        "service-oriented-architecture",
        "Eleven capabilities owned by different health authorities are composed through governed contracts held in a registry.",
    ),
    "service-oriented-architecture-scheduling-capability-audit": (
        "service-oriented-architecture",
        "Eleven independently owned capability contracts are published and versioned in a registry, and every invocation crosses one.",
    ),
    "service-oriented-architecture-telecom-cdr-contract-generation": (
        "service-oriented-architecture",
        "Twelve capabilities owned by six organisations are exposed over governed, versioned contracts registered in a catalogue, and consumption happens only through those contracts.",
    ),
    "service-oriented-architecture-transit-capability-facade": (
        "service-oriented-architecture",
        "Ten independently owned capabilities are composed through governed contracts in a registry, so contract composition is the integration mechanism.",
    ),
    "service-oriented-architecture-waste-capability-history": (
        "service-oriented-architecture",
        "Nine independently owned capability contracts are versioned in a registry and composed through it.",
    ),
    "space-based-billing-grid-audit": (
        "space-based",
        "Rating state is held in a replicated in-memory fabric across seven nodes, with the database written behind the fabric.",
    ),
    "space-based-billing-grid-registry": (
        "space-based",
        "Rating state is held in a replicated in-memory fabric across seven nodes, with the billing store written behind it.",
    ),
    "space-based-billing-grid-window": (
        "space-based",
        "Rating state is kept in a replicated in-memory fabric across six nodes, and the database is written behind it rather than on the request path.",
    ),
    "space-based-card-grid-completion-token": (
        "space-based",
        "Inflight authorization state is held in a replicated in-memory fabric across twelve nodes, with the ledger written behind the fabric.",
    ),
    "space-based-construction-grid-readers": (
        "space-based",
        "Bid and supplier state lives in a replicated in-memory fabric across seven nodes, with the tender store written behind it.",
    ),
    "space-based-diagnostics-grid-identity": (
        "space-based",
        "Diagnostic state is held in a replicated in-memory fabric across ten nodes, with the workshop store written behind it.",
    ),
    "space-based-forecast-grid-named-construction": (
        "space-based",
        "Model-run state is held in a replicated in-memory fabric across ten nodes, with the run store written behind the fabric.",
    ),
    "space-based-freight-grid-policy-guard": (
        "space-based",
        "Shipment state lives in a replicated in-memory fabric across eight nodes, with the shipment store written behind it.",
    ),
    "space-based-legal-grid-active-object": (
        "space-based",
        "Matter state is held in a replicated in-memory fabric across six nodes, with the matter store written behind it.",
    ),
    "space-based-loyalty-grid-authenticator": (
        "space-based",
        "Member state is held in a replicated in-memory fabric across six nodes, with the ledger store written behind the fabric.",
    ),
    "space-based-matching-grid-absent-value": (
        "space-based",
        "Driver state lives in a replicated in-memory fabric across twelve nodes, with the trip store written behind it.",
    ),
    "space-based-matching-grid-monostate": (
        "space-based",
        "Surge pricing state is held in a replicated in-memory fabric across twelve nodes, and the pricing store is written behind it.",
    ),
    "space-based-robot-grid-translation": (
        "space-based",
        "Robot task state is held in a replicated in-memory fabric and reaches the database asynchronously behind the fabric.",
    ),
    "space-based-sports-grid-degradation": (
        "space-based",
        "Seat and hold state is held in a replicated in-memory fabric across twelve nodes, with the ticketing store written behind the fabric.",
    ),
    "space-based-tax-grid-master-worker": (
        "space-based",
        "The working set lives in a replicated in-memory fabric across eight nodes, with writes reaching the database asynchronously rather than on the request path.",
    ),
    "space-based-transit-grid-restricted-creation": (
        "space-based",
        "Gate state is held in a replicated in-memory fabric across nine nodes, with the transit store written behind it instead of on the request path.",
    ),
    "space-based-transit-grid-settlement-rule": (
        "space-based",
        "Tap and fare state is held in a replicated in-memory fabric and reaches the settlement database behind it.",
    ),
    "space-based-travel-grid-operations-open": (
        "space-based",
        "Itinerary state is held in a replicated in-memory fabric across nine nodes, with the booking store written behind the fabric.",
    ),
    "space-based-travel-grid-typed-groups": (
        "space-based",
        "Booking state is held in a replicated in-memory fabric across nine nodes, with the booking store written behind it.",
    ),
    "task-control-architecture-agriculture-controller-decomposition": (
        "task-control-architecture",
        "Weeding decomposes through a field controller whose segment controllers arbitrate competing blade behaviours.",
    ),
    "task-control-architecture-auction-robot-contract-test": (
        "task-control-architecture",
        "Item handling decomposes through a task controller whose arm controllers set points for gripper behaviours.",
    ),
    "task-control-architecture-card-atm-controller-wiring": (
        "task-control-architecture",
        "Commands decompose through a ticking controller hierarchy whose levels publish setpoints to behaviours and arbitrate preemption.",
    ),
    "task-control-architecture-hotel-loop-stages": (
        "task-control-architecture",
        "A room-service task controller decomposes work for joint controllers and gripper behaviours at increasing tick rates.",
    ),
    "task-control-architecture-imagery-behaviour-heterogeneous": (
        "task-control-architecture",
        "Acquisition decomposes through a hierarchy of controllers ticking at fixed rates, with safety arbitration preempting lower behaviours.",
    ),
    "task-control-architecture-lab-controller-long-step": (
        "task-control-architecture",
        "Sample processing decomposes through a task controller whose pipette controllers drive pump behaviours at a higher rate.",
    ),
    "task-control-architecture-lab-robot-large-payload": (
        "task-control-architecture",
        "Retrieval decomposes through a task controller whose gantry controllers set points for gripper behaviours.",
    ),
    "task-control-architecture-maintenance-controller-pure-step": (
        "task-control-architecture",
        "Inspection decomposes through a task controller whose pan-tilt controllers drive camera behaviours at fixed rates.",
    ),
    "task-control-architecture-port-crane-activation": (
        "task-control-architecture",
        "Container work decomposes through a task controller whose trolley controllers set points for spreader behaviours.",
    ),
    "task-control-architecture-provisioning-controller-command": (
        "task-control-architecture",
        "A tick hierarchy decomposes splicing into motion setpoints and clamp behaviour commands.",
    ),
    "task-control-architecture-scheduling-hierarchy-filters": (
        "task-control-architecture",
        "The dispensing task controller decomposes work consumed by motion and gripper behaviours in a fixed-rate hierarchy.",
    ),
    "task-control-architecture-settlement-controller-boundary": (
        "task-control-architecture",
        "A cell task controller decomposes cycles for joint controllers whose setpoints drive gripper behaviours at a higher tick rate.",
    ),
    "task-control-architecture-settlement-robot-partial-acquire": (
        "task-control-architecture",
        "Counting decomposes through a task controller whose transport controllers drive shutter behaviours.",
    ),
    "task-control-architecture-settlement-robot-result-set": (
        "task-control-architecture",
        "Vault cycles decompose through a task controller whose drawer controllers set points for latch behaviours.",
    ),
    "task-control-architecture-tax-robot-process-manager": (
        "task-control-architecture",
        "Batch scanning decomposes through a task controller whose scanner controllers set points for feed behaviours.",
    ),
    "task-control-architecture-transit-robot-authorisation": (
        "task-control-architecture",
        "Gate actions decompose through a task controller whose barrier controllers set points for motor behaviours.",
    ),
}


def _scenario_ids(corpus: list[dict[str, Any]]) -> list[str]:
    return [item["scenario_id"] for item in corpus]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line: corpus in, typed style rows out (default: the repo paths)."""
    parser = argparse.ArgumentParser(description="Derive qa/typed/style.jsonl from the benchmark scenarios.")
    parser.add_argument("--corpus", default="tests/benchmark/scenarios/full.json")
    parser.add_argument("--out", default="qa/typed/style.jsonl")
    return parser.parse_args(argv)


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Read the scenario corpus (list or {"scenarios": [...]} shape)."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("scenarios", payload)
    if not isinstance(items, list) or not items:
        raise SystemExit(f"{path} carries no scenarios")
    return items


def build_criteria() -> dict[str, str]:
    """The served style criteria: shared glosses in affinity order + ``none`` last."""
    from src.style_affinity import STYLE_AFFINITY

    criteria = {key: seam_prompts.STYLE_GLOSSES[key] for key in STYLE_AFFINITY}
    criteria[seam_prompts.STYLE_NONE_KEY] = seam_prompts.STYLE_NONE_GLOSS
    return criteria


def to_style_row(scenario: dict[str, Any], criteria: dict[str, str], corpus: str, corpus_sha12: str) -> dict[str, Any]:
    """One scenario → one typed style row (raw description state, reviewed label)."""
    scenario_id = scenario["scenario_id"]
    if scenario_id not in STYLE_LABELS:
        raise SystemExit(f"scenario {scenario_id!r} has no reviewed style label (add it to STYLE_LABELS)")
    label, rationale = STYLE_LABELS[scenario_id]
    if label not in criteria:
        raise SystemExit(f"scenario {scenario_id!r}: label {label!r} is not among the served criteria keys")
    # The served answer space is the style vocabulary, and serving asks it in shards that
    # together cover it exactly — so a label outside it could never be produced. This is the
    # invariant whose absence let 16 of 22 holdout labels sit outside the shortlist the seam
    # actually offered (guide §10.1: affinity is a prior, never a filter).
    if label != seam_prompts.STYLE_NONE_KEY and label not in seam_prompts.STYLE_VOCABULARY:
        raise SystemExit(f"scenario {scenario_id!r}: label {label!r} is outside the served style vocabulary")
    if set(criteria) - {seam_prompts.STYLE_NONE_KEY} != set(seam_prompts.STYLE_VOCABULARY):
        raise SystemExit(
            f"scenario {scenario_id!r}: criteria keys drifted from the served style vocabulary (re-derive the rows)"
        )
    return {
        "id": f"{scenario_id}:style",
        "state": scenario["description"],
        "type": "choice",
        "instructions": seam_prompts.STYLE_INSTRUCTIONS,
        "criteria": criteria,
        "label": label,
        "rationale": rationale,
        "source": {
            "corpus": corpus,
            "corpus_sha12": corpus_sha12,
            "scenario_id": scenario_id,
            "split": scenario["split"],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Corpus → style rows; refuses to silently skip a corpus scenario."""
    args = parse_args(argv)
    corpus_path = Path(args.corpus)
    items = load_corpus(corpus_path)
    corpus_sha12 = hashlib.sha256(corpus_path.read_bytes()).hexdigest()[:12]
    criteria = build_criteria()
    rows = [to_style_row(item, criteria, args.corpus, corpus_sha12) for item in items]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} style rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
