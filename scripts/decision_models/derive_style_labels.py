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
        "saga",
        "A wire disbursement coordinated as create -> approve -> release steps across console, workflow, and adapter",
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
