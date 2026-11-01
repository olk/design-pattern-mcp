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
#: rationale_ok) and ``clm_calibrate.py`` refuses to recommend a confidence floor
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
        "layered-monolith",
        "One web endpoint with controller steps over a shared store; the team explicitly rejects extra business objects",
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
        "layered-monolith",
        "One tenant portal whose login controller delegates to deployment-selected verifiers behind a boundary",
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
        "model-view-controller",
        "Admin console screens behind per-screen test wrappers — UI screen/view separation is the shape",
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
        "model-view-controller",
        "A search page whose query state moves from page level to the owning field component",
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
        "api-gateway",
        "Compliance clients reaching a growing event table through one bounded paged export endpoint",
    ),
    "remote-contracts-order-submission": (
        "api-gateway",
        "A desktop client completing checkout through one coarse SubmitOrder operation instead of dozens of calls",
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
        "event-driven",
        "Checkout/Ledger/Gateway payment attempts captured as uniform security events at one invocation boundary",
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
