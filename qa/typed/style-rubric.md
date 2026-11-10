# Ch2 style label rubric (frozen)

The annotation sheet for `qa/typed/style.jsonl`. Two annotators label every row from this
sheet alone; a third adjudicates every disagreement. `qa/typed/review.json` records this
file's sha256 in `rubric_sha256`; editing this file voids that record and the batch
calibration below must be re-run.

The label vocabulary and the glosses are **verbatim** from `src/decisions/seam_prompts.py`
(`STYLE_GLOSSES`, `STYLE_NONE_GLOSS`, `STYLE_INSTRUCTIONS`) — the served prompt, not a
paraphrase of it. `tests/unit/test_style_label_gap_scripts.py` fails if they drift.

## The judgment

Label the **dominant architectural style of the system under decision** — the one the
brief's `architecture_style` should resolve to. Not the style the requester might like, not
the pattern the decision statement asks about: the shape of the system the description
shows. Exactly one label per row.

## Decision procedure

Apply in order, stop at the first that settles it:

1. **Is a system topology described at all?** A bare library, algorithm, data-structure or
   code-shape decision with no deployable, no runtime and no inter-component boundary is
   `none`. A single named service that receives requests and returns answers counts as
   topology — it is not `none`.
2. **Is it one deployable?** Then it is `monolithic` unless the description shows the
   organization inside it: strictly horizontal request→domain→persistence stages with each
   layer calling only the next is `layered-monolith`; well-separated vertical modules that
   own their own data and talk through narrow interfaces inside the same process is
   `modular-monolith`.
3. **Otherwise name the structural evidence that dominates.** The decisive question is what
   the description's *load-bearing* mechanism is — what the system would lose if you removed
   it. Use the near-miss pairs below; when two remain plausible, choose the one the
   `architecture_style` decision should resolve to and record the other as `alternative`.

`none` is never a fallback for "the style is hard to name", and `monolithic` is never a
fallback for "one deployable plus a detail I am unsure about": both are positive findings
about the described shape.

## Near-miss pairs

| pair | what separates them |
|---|---|
| `none` / `monolithic` | `none` has no topology visible; `monolithic` has one deployable that we can see serve, deploy and scale as a unit |
| `monolithic` / `layered-monolith` / `modular-monolith` | layers (horizontal, strict) vs vertical modules (own data, narrow interfaces) vs neither |
| `microservices` / `service-oriented-architecture` | many independently deployable services of one product vs business capabilities exposed over standard protocols to other systems, with the integration contract (WSDL/SOAP/registry) load-bearing |
| `microservices` / `service-mesh` | the services themselves vs a dedicated infrastructure layer (proxies, control plane) through which service-to-service traffic is managed |
| `microservices` / `api-gateway` | the decomposition vs the single client-facing entry point that fronts several backend services and carries cross-cutting edge policy |
| `api-gateway` / `backend-for-frontend` | one entry point for all clients vs one dedicated backend per client type, each shaped to that client |
| `api-gateway` / `monolithic` | a proxy the deployable happens to have vs the deployable being the gateway; a thin TLS-terminating proxy fronting one service is not a gateway |
| `api-gateway` / `enterprise-service-bus` | request routing and edge policy for clients vs a central integration backbone mediating protocol/format translation between heterogeneous systems |
| `event-driven` / `broker` / `enterprise-service-bus` | events as the general carrier vs one central broker all components must register with vs protocol-translating integration backbone |
| `event-driven` / `reactive-architecture` | asynchronous notification flows vs responsiveness under backpressure as the design's purpose (bounded queues, demand signalling, non-blocking throughout) |
| `event-driven` / `half-sync-half-async` | events end-to-end vs an explicit bridge where a synchronous front layer hands work to asynchronous workers through queues |
| `event-sourcing` / `command-query-responsibility-segregation` | state *is* the replayed event log vs separate write and read models; often stated together — pick the one the description makes load-bearing |
| `saga` / `pipe-and-filter` | a long-running cross-party transaction with compensations vs a staged data pipeline with no transaction semantics |
| `pipe-and-filter` / `layered-monolith` | independent stages over a data stream vs request-path layers inside one deployable |
| `hexagonal` / `layered-monolith` | domain core behind ports with adapters for every external concern vs horizontal layers calling each other |
| `hexagonal` / `microservices` | in-process boundary discipline vs independently deployable processes |
| `model-view-controller` / `presentation-abstraction-control` | controller-per-request-flow with a shared model vs a hierarchy of cooperating presentation/abstraction/control agents per UI unit |
| `microkernel-plugin` / `reflection-architecture` | a minimal core plus separately packaged extensions vs runtime self-inspection and adaptation of the system's own structure |
| `microkernel-plugin` / `modular-monolith` | discoverable plug-ins extending a small core vs fixed vertical modules of one product |
| `rule-based-system` / `aiml-centric` | declarative if-then rules evaluated by an engine vs learned models as the value-bearing components |
| `aiml-centric` / `event-driven` | model inference is what the system sells vs the transport that carries work between components |
| `serverless` / `edge-computing` | provider-managed functions that scale to zero vs computation placed near the data source or user on nodes the provider does not schedule for you |
| `serverless` / `microservices` | functions the platform runs and scales vs long-lived services you deploy and operate |
| `edge-computing` / `hybrid-cloud` | where the computation sits (near the source) vs which estates the workload spans (on-prem plus public cloud) |
| `hybrid-cloud` / `multi-cloud` | on-premises plus one public cloud vs two or more public clouds, with the split load-bearing |
| `space-based` / `lambda-architecture` | distributed in-memory units and spaces for scalability vs batch and speed layers serving complete and fresh views |
| `lambda-architecture` / `kappa-architecture` | two code paths (batch + stream) vs one streaming path for live and reprocessed views |
| `data-mesh` / `lambda-architecture` | analytical data owned as products by domain teams vs a batch/speed topology for the analytics platform |
| `actor-based` / `event-driven` | isolated stateful units exchanging messages by address, one message at a time vs a general event flow between components |
| `actor-based` / `task-control-architecture` | general concurrent computation vs hierarchical task controllers driving robotic or real-time behaviours |
| `blockchain-based` / `event-sourcing` | consensus-backed shared ledger vs local state rebuilt from an event log |
| `service-oriented-architecture` / `enterprise-service-bus` | the services and their standard contracts vs the bus that mediates between them |

## Label vocabulary (served glosses, verbatim)

- `event-driven` — Components communicate primarily through asynchronous events and message flows
- `broker` — Independent components interact only through a central message broker
- `enterprise-service-bus` — Heterogeneous services integrate through a central enterprise service bus
- `event-sourcing` — State is derived by replaying an append-only log of domain events
- `command-query-responsibility-segregation` — Command and query paths use separate models optimized for writes and reads
- `saga` — A long-running transaction is coordinated as a sequence of local steps with compensations
- `microservices` — The system is composed of small independently deployable services
- `service-oriented-architecture` — Business capabilities are exposed as interoperable services over standard protocols
- `hexagonal` — Domain logic sits at the core behind ports, with adapters for all external concerns
- `layered-monolith` — A single deployable is organized in strict horizontal layers
- `modular-monolith` — A single deployable is organized in well-separated vertical modules
- `monolithic` — The whole system is built, deployed, and scaled as one unit; also the answer for one deployable with no other style signal
- `pipe-and-filter` — Data flows through a chain of independent processing stages
- `model-view-controller` — UI concerns split into model, view, and controller responsibilities
- `presentation-abstraction-control` — Each UI unit pairs presentation, abstraction, and control cooperating agents
- `microkernel-plugin` — A minimal core is extended by plug-ins for most capabilities
- `reflection-architecture` — The system inspects and adapts its own structure and behavior at runtime
- `reactive-architecture` — The design is built around asynchronous message-driven responsiveness and backpressure
- `actor-based` — Concurrent computation is carried by isolated actors exchanging messages
- `half-sync-half-async` — Synchronous and asynchronous processing layers are bridged by queues
- `task-control-architecture` — Hierarchical task controllers coordinate concurrent robotic or real-time behaviors
- `serverless` — Code runs as provider-managed functions scaled to zero when idle
- `edge-computing` — Computation is pushed to nodes near the data source or user
- `space-based` — Scalability comes from distributed in-memory processing units and spaces
- `data-mesh` — Analytical data is owned as products by decentralized domain teams
- `lambda-architecture` — Batch and speed layers are combined to serve complete and fresh views
- `kappa-architecture` — A single streaming pipeline serves both real-time and reprocessed views
- `api-gateway` — Clients reach backend services through a single gateway entry point
- `backend-for-frontend` — Each frontend gets a dedicated backend tailored to its needs
- `service-mesh` — Service-to-service traffic is managed by a dedicated infrastructure layer
- `rule-based-system` — Behavior is driven by a set of declarative if-then rules
- `aiml-centric` — AI/ML models are the primary value-bearing components of the system
- `blockchain-based` — Trust and state rely on a distributed ledger with consensus
- `hybrid-cloud` — Workloads span on-premises infrastructure and public cloud
- `multi-cloud` — Workloads span more than one public cloud provider
- `none` — No listed architecture style applies, or the state describes only a library, algorithm, or code-shape decision with no system topology

## Review protocol

1. **Two blind annotators per row.** Annotator A and Annotator B label the same rows
   independently, from this sheet only. Neither sees the other's labels, the label source
   (`scripts/decision_models/derive_style_labels.py`), `qa/typed/style.jsonl`,
   `qa/typed/review.json`, or git history for those rows while labeling. Relabelling an old
   row blind is the audit; copying the current label is not.
2. **Batch acceptance.** A batch (~75 rows) is accepted when raw agreement between A and B is
   `>= 0.80`. Cohen's kappa is recorded per batch regardless; `kappa < 0.67` triggers a
   rubric review even when raw agreement passes, because raw agreement hides class imbalance.
3. **Adjudication, never discard.** A third party adjudicates every disagreement and writes
   the rationale, naming the evidence that decides it. A rejected batch is re-annotated in
   full after a written clarification note — rows are never salvaged one at a time and never
   dropped without a recorded reason.
4. **Recorded provenance.** Every accepted row carries its batch id, the two annotator
   labels, the adjudicator's rationale when they differ, and the reviewer identity in
   `qa/typed/review.json`. Labels are model review passes unless a human expert signs; the
   ledger states which, and never upgrades one to the other.

## Clarifications after the 2026-11 calibration batch

The 30-row audited holdout was relabelled blind by two annotators under this rubric
(sha256 `66dd2e055c6b9da9744b6f47dbae66a410032dd975261283d4bf0defdc78c970`, i.e. this file
without this section): raw agreement 0.8333, Cohen's kappa 0.7774, accepted. Five rows were
disputed and adjudicated, and they name the two places the procedure was being read two ways.
Both are now explicit:

1. **A component called a "service" is not deployment evidence.** `Policy Service`,
   `Ledger Reserve service`, `Refund service` read as domain-level units inside one request path
   unless the text says the component is deployed, released or scaled independently. Two of the
   five disputes turned on this; "microservices" was chosen on the strength of a noun alone.
2. **"The system under decision" is the system whose decision the text asks about.** A provider
   service whose published contract a separately released client consumes is one system with one
   deployable, not a service landscape: the client's independent release is why the contract is
   verified in the provider's build, not a second capability. Count the deployables of the
   system being decided, not every artifact the text mentions.
3. **A name does not make a structure.** A narrow published interface over hidden internals is
   not `hexagonal` unless the text describes a domain core plus adapters for external concerns,
   and a central registration point every producer and consumer wires into is `broker` even when
   the surrounding application is one deployable — interior partitioning alone is
   `modular-monolith`, and the two are not interchangeable.

4. **Label the system described, not the refactor it asks for.** Most rows pair a described
   estate with a requested change; when the request would introduce a structure the text does not
   yet show, the label is the described structure. A row that says "one deployable, no per-charger
   process exists" and then asks how per-charger state should be isolated is `monolithic`, not
   `actor-based`: the second reading labels the intent, and intent does not deploy. This came out of
   the first reviewed train batch, where it was the single dispute.

Two of the 30 disputed rows are recorded as *rubric tensions* rather than resolved by these
rules: `enforcement-owned-secret-vault` (reviewed as `layered-monolith`, whose layer ordering the
text does not state) and `regression-boundary-independent-mobile-api` (reviewed as
`microservices` for a client/provider pair). They stay in the holdout with their reviewed labels
and are reported, not silently relabelled: the holdout is the instrument, and editing it to fit
the rubric would destroy the measurement it exists for. Annotators must still label them under
the rubric; a disagreement on a tension row is adjudicated, not scored as annotator error.

## Authoring rules for labeled scenarios (not part of the judgment)

These govern how a *scenario* may be written, so a label cannot be read off a string:

- The description never names the style, its served gloss, or its everyday name
  (`tests/benchmark/harness/draft.py::_STYLE_LEAK_TOKENS`), and never names the catalog
  pattern it is about.
- The style evidence is structural: deployment shape, ownership, protocols, where policy
  lives, what carries work between components.
- Organic scenarios describe a plausible real system (grounded in published documentation
  for the stack they name); synthetic ones are drafted from the teacher pipeline and are
  capped so organic stays `>= 70 %` of new train rows and `100 %` of holdout rows.
