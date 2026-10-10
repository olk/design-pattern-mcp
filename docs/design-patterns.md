# Software Design Patterns — Decision-First Reference

A **design pattern** is a named, reusable arrangement of collaborating code that solves a recurring implementation problem under stated trade-offs. It belongs here only when it structures a part of a program: an object, module, component, message flow, persistence boundary, concurrent task, or specialised runtime.

Architectural styles, deployment topology, organisation-level design, and data-platform choices are deliberately excluded. They decide how a whole system is cut rather than how one implementable part is arranged.

## Quick routing

| If the decision is about… | Start in |
|---|---|
| domain terms, invariants, identity, or time | [1. Model domain and values](#1-model-domain-and-values) |
| object construction, dependency wiring, or code composition | [2. Construct and compose components](#2-construct-and-compose-components) |
| selecting behaviour, a variant, or an extension point | [3. Select behaviour and extend variants](#3-select-behaviour-and-extend-variants) |
| UI, API, remote, serialization, or message endpoint boundaries | [4. Define interaction boundaries](#4-define-interaction-boundaries) |
| commands, events, channels, routing, or workflow coordination | [5. Coordinate workflows and messages](#5-coordinate-workflows-and-messages) |
| persistence, transactions, cache ownership, or offline reconciliation | [6. Persist-own-and-reconcile-state](#6-persist-own-and-reconcile-state) |
| threads, async tasks, cancellation, deadlines, or overload | [7. Execute-under-concurrency-time-and-failure](#7-execute-under-concurrency-time-and-failure) |
| access enforcement, validation, test doubles, or contracts | [8. Secure-validate-and-verify](#8-secure-validate-and-verify) |
| a client UI, simulation loop, parser, or language implementation | [9. Implement-specialised-component-types](#9-implement-specialised-component-types) |

## How to use this catalog

1. Start from the smallest named problem surface: a concrete component, boundary, workflow, resource, or runtime object.
2. Select one **primary pattern** that owns that decision. Add supporting patterns only where they solve distinct constraints.
3. Use the facet views to find provenance, style, scope, and language affinity after identifying the primary home; they are not competing taxonomies.

### Primary homes and facets

Every candidate pattern appears in one canonical table row. A pattern may have roots in GoF, EAA, EIP, DDD, POSA, a language idiom, or a functional catalog; provenance is retained through the description and source notes, but does not determine where the pattern is selected. This avoids duplicate recommendations such as listing the same pattern once by source and again by paradigm.
## 1. Model domain and values

*Choose this group when the decision changes domain facts, rules, identities, time, or which states are valid.*
### Domain logic and application services

| Pattern | Use when |
|---|---|
| **Transaction Script** | the logic for one request is a straight line of steps; a procedure is honest |
| **Domain Model** | behaviour and data belong together in objects named after domain concepts |
| **Table Module** | the logic is one set-oriented pass over a whole table or record set |
| **Service Layer** | several callers need one coherent boundary of domain operations and transactions |
| **Typed Collection** | a collection should own its elements' invariants, not leak a bare list |
| **Domain Factory** | rebuilding domain objects from raw data needs one dedicated owner |
| **Property** | an object needs dynamic, discoverable property names instead of fixed fields |

### Tactical domain modelling

| Pattern | Use when |
|---|---|
| **Entity** | something is identified by continuity, not by its attributes |
| **Aggregate / Aggregate Root** | invariants must hold across a cluster, and one owner must enforce them |
| **Domain Service** | an operation is a domain concept but belongs to no entity or value |
| **Application Service** | a use case must orchestrate the model and own the transaction |
| **Domain Event** | something happened in the domain that others must react to |
| **Factory** | building a consistent aggregate is complex enough to deserve an owner |
| **Specification** | a business rule must be reusable, combinable and testable as an object |
| **Module** | related concepts should name one coherent unit of the model |
| **Separated Domain Model** | the domain model stays explicit but is deliberately not the data model |

### Values, types and correctness by construction

| Pattern | Use when |
|---|---|
| **Special Case** | callers shouldn't test for nil/zero/"unconfirmed"; give that case behaviour |
| **Null Object** | callers keep doing the same thing when the collaborator is absent |
| **Type Object** | the differences between types are data you'd rather edit than recompile |
| **Value Object** | a cluster of attributes that always changes as a whole |
| **Money** | arithmetic on currency needs units and rounding discipline |
| **Immutable Object** | share and cache freely; no visible mutation, no defensive copies |
| **Type-Safe Enum** | a fixed set of constants that must not mix with ints or strings |
| **Newtype** | a primitive needs its own identity so two can't be swapped by accident |
| **Typestate** | encode state in type parameters so an illegal transition is a compile error |
| **Safe Bool** | a boolean conversion must not accept arithmetic or comparisons |

### Time and temporal modelling

| Pattern | Use when |
|---|---|
| **Quantity**[^quantity] | a measurement must carry its unit with its amount; put both in a value object so conversions and unit-compatible arithmetic have one owner — Money (§12) is the currency-specific case |
| **Range**[^range] | bounds repeatedly travel and are compared together; use an interval value with containment/overlap operations and explicit inclusive, exclusive and open-ended semantics |
| **Effectivity**[^effectivity] | a fact or relationship is valid only during a period; attach a validity Range and test dates against it, with explicit rules for overlapping or missing periods |
| **Temporal Property**[^temporal-property] | only a few attributes need historical reads; hide history behind date-indexed accessors rather than making every caller search effective-dated records |
| **Temporal Object**[^temporal-object] | most attributes change over time, or the domain explicitly refers to versions such as contract amendments; separate stable identity from dated version objects rather than versioning each property independently |

### Model clarity and rules

| Pattern | Use when |
|---|---|
| **Intention-Revealing Interface** | a caller shouldn't have to read the implementation to use the object |
| **Side-Effect-Free Function** | reasoning and testing need results that depend only on inputs |
| **Assertion** | invariants should be declared where they're checked, not in a comment |
| **Standalone Class** | a class should be understandable without the rest of the model |
| **Closure of Operations** | combining two model objects should yield another model object |
| **Conceptual Contour** | conceptually related behaviour should be grouped, not scattered |

### Functional data modelling

| Pattern | Use when |
|---|---|
| **Algebraic Data Type (sum + product)** | a value is one of a fixed set of variants, each with known fields |
| **Pattern Matching** | every variant must be handled, and the compiler should check it |
| **Sealed / case class** | the variant set is closed and the data immutable |
| **Make Illegal States Unrepresentable** | combinations that can't happen shouldn't be expressible |
| **Lens** | update one nested field of immutable data without hand-writing copies |
| **Zipper**[^zipper] | repeatedly navigate and edit nearby nodes in an immutable tree; keep a focused subtree plus its surrounding context, rather than restarting each edit from the root as with a fixed-path update |
| **Optional / Traversal** | the nested field may be absent, or there may be several |

## 2. Construct and compose components

*Choose this group when the decision is how to create, wire, encapsulate, or locally compose program parts.*
### Creation and configuration

| Pattern | Use when |
|---|---|
| **Factory Method** | the exact type is only known at runtime; let a subclass decide |
| **Abstract Factory** | you must create *families* of related objects that stay consistent |
| **Builder** | construction has many steps or optional parameters |
| **Prototype** | cloning a configured instance beats constructing from scratch |
| **Singleton** | exactly one shared instance is genuinely required (rare — prefer DI, §12) |
| **Functional Options**[^functional-options] | a constructor has growing optional settings and useful defaults; accept named configuration functions without expanding its required parameter list, and define option ordering and conflicts |
| **Named Constructor / Static Factory Method** | construction needs a name, a cache, or a subtype choice |
| **Type-Safe Builder / Step Builder** | required construction steps must be ordered and unskippable |
| **Virtual Constructor (Clone)** | you need to copy an object through its base interface |

### Composition and encapsulation

| Pattern | Use when |
|---|---|
| **Adapter** | an existing class's interface doesn't match what callers expect |
| **Bridge** | an abstraction and its implementation must vary independently |
| **Composite** | part–whole trees that clients should treat uniformly |
| **Decorator** | add responsibilities to an object without subclassing |
| **Facade** | a complex subsystem needs one simple entry interface |
| **Flyweight** | huge numbers of similar objects must share state |
| **Proxy** | control, defer, or record access to another object |
| **Virtual Proxy** | the object being stood in for is expensive to build; construction is what's deferred |
| **Extension Interface** | grow a component's interface via lookup, without bloating the base interface or breaking clients |
| **Wrapper Facade** | a non-OO API (C, syscalls, wire protocol) needs one cohesive, safe object interface |
| **Monostate** | shared state, but normal value-semantics objects — Singleton without the static accessor |
| **Multiton** | one instance *per key*, not one per process |
| **Servant** | several unrelated classes need the same behaviour and none of them should own it |
| **Role Object** | one entity legitimately presents different interfaces per context, and roles come and go at runtime |
| **Twin** | two parallel hierarchies (abstraction vs. platform) must evolve independently yet stay in step |
| **Abstract Document** | type-safe views over loosely typed, dynamically composed property maps |
| **Marker Interface** | a type must be recognisable with no methods to implement |
| **Private Class Data** | fields must be readable but not writable, even by subclasses |
| **Interface Class** | a class exists to declare a contract and hold no state |
| **Component** | an object gets behaviour from pluggable parts, not from a deep hierarchy |

### Dependency wiring and discovery

| Pattern | Use when |
|---|---|
| **Dependency Injection (constructor / setter / interface)** | an object shouldn't create what it depends on; hand it in |
| **Inversion of Control** | a framework owns the call flow and calls back into your code |
| **Service Locator** | no injection point exists; look the dependency up in one registry |
| **Separated Interface** | the interface must be deployable where the implementation can't go |
| **Registry** | many objects need one well-known place to find one another |
| **Layer Supertype** | every object in a layer should share one declared type |
| **Delegation** | reuse behaviour by forwarding to a contained object, not by inheriting |

### Method and local implementation structure

| Pattern | Use when |
|---|---|
| **Fluent Interface** | a small DSL reads better as a chain than as nested calls |
| **Composed Method** | a method should stay at one level of abstraction |
| **Method Object** | a method is too big, but its locals belong together — make the invocation an object |
| **Macro Method** | a short method composed entirely of calls to other methods |
| **Encapsulate Collection** | expose behaviour over a collection, never the collection itself |
| **Inline Message** | a one-line delegating method that makes the caller read better |
| **Collecting Parameter** | several callees must append to one accumulator you pass in |
| **Parameter Object** | the same parameters keep travelling together |
| **Interesting Return Value** | return something the caller can act on, not nothing |
| **Simple Enumeration** | walk a collection with one message instead of an index loop |
| **Rectangular Block** | keep blocks visually balanced so nesting is obvious |
| **Copy-and-Swap** | you need the strong exception guarantee from assignment |
| **Non-Throwing Swap** | swap must never throw, so rollback paths stay safe |
| **NVI / Non-Virtual Interface** | pre/post-conditions on a virtual must not be bypassable by subclasses |
| **Execute Around** | a method should own setup and teardown and take the body as an argument |

### Functional composition

| Pattern | Use when |
|---|---|
| **Function Composition** | a transformation is a chain of small pure steps |
| **Pipeline** | stages should be named and ordered, read top to bottom |
| **Functional Core, Imperative Shell**[^functional-core] | a module mixes decision logic and I/O; keep decisions pure over values and let a thin shell interpret them — avoid forcing the split around I/O-dominated code |
| **Collection Pipeline** | map/filter/reduce over a collection, declaratively |
| **Combinator library** | embed a DSL of `andThen` / `orElse` / `many` instead of a class hierarchy |

## 3. Select behaviour and extend variants

*Choose this group when the decision is which algorithm, implementation variant, dispatch mechanism, or extension point executes.*
### Algorithms, traversal and variation

| Pattern | Use when |
|---|---|
| **Interpreter** | you need a grammar and evaluator for a small expression language |
| **Iterator** | traverse a collection without exposing its representation |
| **Strategy** | interchangeable algorithms chosen at runtime |
| **Template Method** | fixed algorithm skeleton with steps filled in by subclasses |

### Dispatch and extensibility

| Pattern | Use when |
|---|---|
| **Visitor** | new operations over a stable object structure without touching its classes |
| **Double Dispatch** | behaviour depends on the runtime type of *two* objects |
| **Acyclic Visitor** | visitors must not depend on concrete elements, so a new element doesn't touch existing visitors |
| **Extensible Visitor** | the opposite trade: add elements without recompiling visitors, paying with implementation inheritance |
| **Sealed hierarchy + exhaustive match** | variants closed, operations open — let the compiler prove coverage (frequently retires Visitor) |
| **Dispatch table / function table** | behaviour is data; key a table by opcode or type instead of branching |
| **Type Erasure** | one interface must hold unrelated types with shared behaviour, with no common base |
| **Capability Query** | ask an object whether it supports an optional interface, rather than testing its type |
| **Dynamic Proxy** | the proxy is produced at runtime by reflection, not written per class |

### Extension, policy and compile-time variation

| Pattern | Use when |
|---|---|
| **Plugin** | behaviour must be addable without recompiling the host |
| **Extension Object** | attach attributes to a class without touching it or its subclasses |
| **Pimpl / Handle-Body / Compilation Firewall** | implementation details must not leak into the header (Cheshire Cat; Fast Pimpl) |
| **Envelope-Letter** | one interface must dispatch to interchangeable implementations |
| **CRTP** | static polymorphism without virtual dispatch |
| **Barton–Nackman** | mixin operators for a family of related types via CRTP |
| **Non-copyable Mixin** | copying this type is meaningless and must be a compile error |
| **Empty Base Optimization** | a stateless base class must not consume object size |
| **Tag Dispatching** | pick the implementation at compile time from a type tag, with no virtuals |
| **Traits** | select behaviour by type properties rather than by inheritance |
| **Policy-based design** | compose behaviour from interchangeable template policies |
| **Feature Toggle** | behaviour must switch by configuration, not by deploy |
| **Filterer** | filtering logic deserves to be a first-class object |
| **Factory Kit** | one factory dispatches over many registered types |
| **Rule Engine** | the rules change more often than the code |
| **Subclass Sandbox** | subclasses should compose behaviour only from base primitives |

### Functional composition and interpretation

| Pattern | Use when |
|---|---|
| **Immutability** | shared data must not change under a reader |
| **Functor** | you only need to map the inside of a context |
| **Applicative** | several independent effects are combined |
| **Monad (incl. IO Monad)** | each step's effect depends on the previous value |
| **Contravariant / `Decidable` validators** | reusable checks that narrow on a condition |
| **Tagless Final** | abstract program logic over the interpretation of its effects |
| **Interpreter-as-data** | describe effects as data and run them differently in test vs. production |
| **Fold / Catamorphism** | consume a recursive structure by combining its cases |
| **Unfold / Anamorphism** | generate a structure by expanding a seed |
| **Hylomorphism** | unfold then fold — one recursion, no intermediate structure |
| **Tail Recursion** | the recursive call is last, so it can run as a loop |
| **Trampoline** | mutual recursion must not grow the stack |
| **Currying / Partial Application** | specialise a function now and call it later |

## 4. Define interaction boundaries

*Choose this group when the decision defines a presentation, remote, message, translation, or endpoint boundary.*
### Server-side presentation

| Pattern | Use when |
|---|---|
| **Page Controller** | one page or form owns its request handling |
| **Front Controller** | every request needs shared dispatch, auth and common behaviour |
| **Intercepting Filter** | cross-cutting pre/post processing must wrap the core handler |
| **Template View** | the markup is static except for holes the model fills |
| **Transform View** | presentation is a programmatic translation of domain data into markup |
| **Two-Step View** | build a logical presentation first, then render it per channel |
| **Template Script** | a script embedded in the view pulls model data and emits markup |
| **View Helper** | the template needs logic, but not inline scripting |
| **Composite View** | assemble a page from reusable view fragments |
| **Page Cache** | output is identical across requests; serve it pre-generated |
| **Application Controller** | workflow decisions must be centralised, out of both view and page controller |
| **Action Domain Responder** | one request, one handler, one response — the minimal alternative to Page Controller |
| **Multiple Views** | one model must serve several view technologies side by side |
| **Service to Worker** | the front controller should dispatch to command-like workers |
| **Session Facade** | one conversation needs a coarse facade over fine-grained services |
| **Business Delegate** | hide remote lookup and transport behind a local-looking delegate |
| **Context Object** | carry per-request state through a flow without threading parameters |

### Remote contracts and data shipping

| Pattern | Use when |
|---|---|
| **Remote Facade** | many chatty local calls must become one network round trip |
| **Data Transfer Object** | carry a whole payload over the boundary in one serialisable object |
| **Gateway** | access to another system should look like a plain method call |
| **Mapper** | representations differ on either side and must be translated, in one place |
| **Service Stub** | the client needs a stand-in for a remote service (also for tests) |
| **Command Base** | ship a serialised command to the far side to be executed there |
| **Cursor Pagination**[^cursor-pagination] | a boundary returns a large or changing ordered collection; send bounded pages and an opaque continuation token, pinning the filter/order and defining token expiry |
| **Record Set** | carry a set of rows as a first-class value between layers |
| **Forwarder-Receiver** | transparent point-to-point messaging between two parts; boundary-adjacent |

### Message contracts

| Pattern | Use when |
|---|---|
| **Message** | you need the unit of communication: headers plus body, sent as a package |
| **Command Message** | tell the receiver to do something |
| **Event Message** | tell any receiver that something happened |
| **Document Message** | transfer data and let recipients decide what to do with it |
| **Request-Reply**[^eip-request-reply] | the sender needs a result or domain error; send a request and correlated reply, with a timeout and cancellation rule for a missing reply |
| **Return Address** | the reply must go somewhere other than the sender's default |
| **Correlation Identifier** | match replies to requests without shared state |
| **Message Sequence** | a batch of messages must be handled as one ordered group |
| **Message Expiration** | a message that arrived too late is worse than no message |
| **Content-Based Serializer** *(incl. format-introduced)* | the payload must carry enough to pick the right deserialiser |
| **Property-Based Serializer** | put the type hints in properties/headers so routing stays cheap |

### Translation and compatibility

| Pattern | Use when |
|---|---|
| **Converter** | two representations need bidirectional mapping without either knowing the other |
| **Message Translator** | two systems disagree on format; translate at the seam |
| **Envelope Wrapper**[^eip-envelope-wrapper] | the payload lacks required infrastructure metadata or a compatible wire shape; wrap it at the seam and unwrap it before application code sees it |
| **Content Enricher** | the message needs data the sender didn't have |
| **Content Filter** | send only the fields this receiver needs |
| **Claim Check** | pass a reference to the payload instead of the payload |
| **Normalizer**[^eip-normalizer] | several input formats express the same semantic message; map each to one normal form before shared downstream logic runs |
| **Tolerant Reader** | the producer adds fields you don't know; ignore what you don't need |
| **Partial Response** | the caller names the fields it actually wants |
| **Client Session** | conversation state belongs on the client |
| **Server Session** | conversation state belongs on the server |
| **Anti-Corruption Layer** | translate an alien model at the seam so it never leaks inward |
| **Open Host Service** | many different callers need one published protocol from your subsystem |
| **Published Language** | the exchange format itself must be standard and documented |

### Message endpoints

| Pattern | Use when |
|---|---|
| **Message Endpoint** | you need the code that sends and receives — named and isolated |
| **Channel Adapter**[^eip-channel-adapter] | an application must send or receive messages without coupling to the messaging API; translate between its interface/data and the channel |
| **Messaging Gateway** | callers should make a method call, not write messaging code |
| **Messaging Mapper** | convert domain objects ⇄ messages without polluting either side |
| **Service Activator**[^eip-service-activator] | an incoming message must invoke ordinary service code; map the message to a service method and map its result or error back into the flow |
| **Event-Driven Consumer** | let incoming messages invoke your code instead of polling |
| **Polling Consumer**[^eip-polling-consumer] | the transport cannot push, or the consumer must control its pull rate; poll bounded batches with explicit idle/backoff and acknowledgement behaviour |
| **Durable Subscriber**[^eip-durable-subscriber] | a publish-subscribe consumer may be offline; retain its subscription backlog so it can resume, with explicit retention and catch-up limits |

## 5. Coordinate workflows and messages

*Choose this group when the decision coordinates commands, events, messages, routing, or an end-to-end workflow.*
### Handlers, state and local events

| Pattern | Use when |
|---|---|
| **Chain of Responsibility** | several handlers could serve a request; pass it along until one does |
| **Command** | requests need to be queued, logged, undone, or replayed |
| **Mediator** | many-to-many object chatter should go through one hub |
| **Memento** | capture and restore state (undo, snapshots) |
| **Observer** | one state change must notify many dependents |
| **State** | behaviour changes with an object's internal state |
| **Command Processor** | Command, plus a queue and an executor — invocation separated from execution, with logging, undo and macros |
| **Event Aggregator** | many components raise events and many subscribe; one hub owns the wiring |
| **Notification** | return the errors you collected instead of throwing on the first one |
| **Interceptor** | a service must attach transparently to a framework and fire on named events |
| **Asynchronous Completion Token** | responses to async calls you issued to many different services need demultiplexing |
| **Callback** | the callee must return before the work it started is finished |
| **Publisher-Subscriber** | decoupled multi-consumer notification — at part level a Mediator/Observer usually suffices (§9) |

### Messaging channels and delivery

| Pattern | Use when |
|---|---|
| **Point-to-Point Channel** | exactly one consumer may handle each message |
| **Publish-Subscribe Channel** | every interested consumer must receive it |
| **Datatype Channel** | each message type deserves its own channel so the contract is explicit |
| **Invalid Message Channel** | malformed messages need somewhere to go, not a black hole |
| **Dead Letter Channel** | expired or unroutable messages should be redirected for diagnosis |
| **Guaranteed Delivery** | the channel must survive failure and still deliver exactly once |
| **Competing Consumers** | several consumers on one channel buy throughput and availability |
| **Message Dispatcher** | something must choose which consumer serves which message |

### Routing and process control

| Pattern | Use when |
|---|---|
| **Message Router** | where a message goes depends on the message |
| **Content-Based Router** | route on what's inside the payload |
| **Message Filter** | the consumer should see only a slice of the original message |
| **Dynamic Router** | the routing rules change at runtime and shouldn't be hard-coded |
| **Recipient List** | the set of recipients changes and must be data |
| **Splitter** | a large message must be broken into processable pieces |
| **Aggregator** | pieces must be merged back once all — or enough — have arrived |
| **Resequencer** | parallel processing scrambled the order; restore it |
| **Composed Message Processor** | split a message, route the pieces independently, then aggregate |
| **Scatter-Gather** | ask many endpoints and merge the answers that arrive in time |
| **Routing Slip** | let the message carry its own route |
| **Process Manager** | a multi-step flow needs coordination, timeouts and compensation |
| **Control Bus** | manage and monitor integration components out of band |
| **Detour** | a component or environment must be bypassable by toggling a route |
| **Legacy Window** | a legacy interface or format must keep seeing the old shape |
| **Pipe and Filter** | a document transformation is a chain of independent stages |

### Message reliability, observability and management

| Pattern | Use when |
|---|---|
| **Idempotent Receiver** | duplicate delivery must be harmless by design |
| **Wire Tap** | you must observe messages without changing the flow |
| **Service Proxy** | front a messaging service with an equivalent, controllable interface |
| **Message History** | debugging needs to know where a message has already been |
| **Message Store**[^eip-message-store] | audit or operations need an independent durable record of selected message data; copy it outside the main flow, distinguish it from delivery durability, and redact/expire it deliberately |
| **Watchman** | someone must be told when messages stop flowing |
| **Store-and-Forward** | sender and receiver can't both be up; persist locally, then send |
| **Transactional Message Queue** | the send must commit or roll back with the business work |
| **Careful Buffering** | the buffer must be bounded, with a decided outcome when it fills |
| **Retry** | the failure is known to be transient |

## 6. Persist, own and reconcile state

*Choose this group when the decision governs persisted state, resource ownership, caching, or reconciliation over time.*
### Data access gateways

| Pattern | Use when |
|---|---|
| **Table Data Gateway** | one object should mediate all access to an entire table; rows stay records |
| **Row Data Gateway** | one object per row, holding that row's data *and* its find/insert/update |
| **Data Access Object** | isolate the quirks of one specific data store behind its own interface |
| **DAO Factory** | the store choice varies; hand out the DAO that matches |
| **DAO / DAO Factory** | the access path to a specific store needs isolating, and choosing |

### Domain-to-store mapping

| Pattern | Use when |
|---|---|
| **Composite Entity** | a cluster of related objects must persist and synchronise as one unit |
| **Active Record** | the domain is simple enough that an object can carry its own SQL |
| **Data Mapper** | domain objects must stay persistence-ignorant; a separate mapper does the moving |
| **Dependent Mapping**[^dependent-mapping] | a child has no independent persistence lifecycle; the parent's mapper loads and saves it with the parent |

### Change tracking and identity

| Pattern | Use when |
|---|---|
| **Unit of Work** | everything changed in one transaction must be flushed once, consistently |
| **Identity Map** | the same row must never be two different in-memory objects in one session |
| **Lazy Load** | the value is expensive to fetch and often never touched |
| **Write Behind** | defer and coalesce writes until commit, so intermediate states never hit the DB |
| **Version Number** | a per-row token detects a write based on stale data (see §8) |

### Relational structure mapping

| Pattern | Use when |
|---|---|
| **Identity Field** | mappers and identity maps need a key inside the object to recognise it by |
| **Foreign Key Mapping** | one object refers to another; store the other's key |
| **Association Table Mapping** | a many-to-many needs a join table to exist relationally |
| **Embedded Value** | a value object's fields belong in the owning row's columns |
| **Serialized LOB** | a whole object graph is fine to store opaque, in one column |
| **Single Table Inheritance** | a small hierarchy fits one table plus a type code |
| **Class Table Inheritance** | one table per class, joined on the key, to keep columns honest |
| **Concrete Table Inheritance** | one table per concrete class, accepting duplicated base columns |
| **Inheritance Mappers** | the table strategy you chose dictates which mapper shape you write |

### Queries and mapping metadata

| Pattern | Use when |
|---|---|
| **Repository** | callers should think in collections of domain objects, not in queries |
| **Query Object** | a query must be an object: composable, storable, passable |
| **Metadata Mapping** | mapping knowledge belongs in metadata/config, duplicated in no class |
| **Lightweight Entity** | keep entities dumb so metadata can drive the mapping |
| **Serialized Entity** | store the whole entity serialised rather than column-mapped |

### Long-running change and consistency

| Pattern | Use when |
|---|---|
| **Optimistic Offline Lock** | conflicts are rare; detect them at commit instead of holding a DB lock |
| **Pessimistic Offline Lock** | conflicts are expensive; take an explicit lock for the whole edit session |
| **Coarse-Grained Lock** | changing one object really means changing a cluster; lock the cluster |
| **Implicit Lock** | locks should fall out of structure and containment, not from explicit calls |
| **Idempotent Client** | a retry may arrive after the first attempt already succeeded |

### Ownership and whole–part lifecycle

| Pattern | Use when |
|---|---|
| **Whole-Part** | a composite's *Whole* must own its *Parts'* lifecycle and present one interface |

### Resource lifecycle and caching

| Pattern | Use when |
|---|---|
| **Lazy Initialization** | a field is expensive to build and often unused |
| **Object Pool** | construction is heavy and instances are short-lived and interchangeable |
| **RAII / Execute-Around Object** | release must happen on *every* exit path, including exceptions |
| **Scope Guard** | run cleanup (or any action) at scope exit without inventing an owner object |
| **Construct-On-First-Use / Nifty (Schwarz) Counter** | static initialisation order across translation units is undefined |
| **Smart Pointer** | ownership of a heap object must be expressible in the type |
| **Lookup** | you know the resource's name but not its location; resolve it, and re-resolve when it moves |
| **Lazy Acquisition** | acquisition is costly and may never be needed — defer to first use |
| **Eager Acquisition** | use-time cost must be predictable and small — acquire and initialise before the deadline |
| **Partial Acquisition** | split acquisition into stages so the expensive part is deferred or shared |
| **Caching** | re-acquisition is expensive and reuse is unchanged — keep and invalidate |
| **Cache-Aside**[^cache-aside] | a source of truth is outside a hot read cache; read the cache first, populate on miss, then update the source before invalidating or refreshing the entry — define TTL and stale-read behaviour |
| **Resource Return** | a borrowed resource must be handed back rather than closed by the borrower |
| **Dispose / Deterministic Release**[^dispose] | a garbage-collected object owns resources whose lifetime is not one lexical scope; expose idempotent close/dispose and cascade cleanup to owned resources, not borrowed ones, instead of waiting for collection |
| **Weak Reference**[^weak-reference] | a cache, listener list or back-reference must not keep its target alive; obtain a strong reference before use and tolerate absence — not a substitute for deterministic resource cleanup |

## 7. Execute under concurrency, time and failure

*Choose this group when the decision governs parallel work, cancellation, deadlines, overload, or resource-cost trade-offs.*
### Dependency failure and overload

| Pattern | Use when |
|---|---|
| **Circuit Breaker** | a dependency keeps failing; stop calling it and let it recover |
| **Bulkhead** | one slow dependency must not drain the shared capacity of all the others |
| **Retry with Backoff** | the failure is transient; retry with growing, jittered delay |
| **Timeout** | a call that can hang must stop hanging |
| **Deadline Propagation**[^deadline-propagation] | an inbound request makes downstream calls under one latency budget; pass the remaining deadline and cancellation onward, never a fresh full timeout per hop |
| **Rate Limiting / Throttling** | volume must stay inside what you or the dependency can serve |
| **Load Leveller** | absorb bursts with a queue instead of shedding work or scaling out |
| **Fail Fast** | the request can't be served; say so immediately and free the resource |
| **Graceful Degradation** | keep serving the parts that still work and drop the rest |
| **Health-Check Endpoint** | an orchestrator or proxy needs to decide whether you're usable |

### Coordination primitives

| Pattern | Use when |
|---|---|
| **Producer–Consumer** | production and consumption proceed at different rates |
| **Thread Pool** | thread-per-task is too expensive; reuse a bounded set of workers |
| **Thread-Local Storage** | a "logically global" access point that is really per-thread, lock-free |
| **Guardian Object** | one dedicated object serialises access to shared state |
| **Double-Checked Locking** | the expensive initialisation happens once; check cheaply first *(needs correct publication to be sound)* |
| **Reactor** | demultiplex readiness events from many handles on one thread |
| **Proactor** | dispatch on the *completion* of asynchronous operations |
| **Half-Sync/Half-Async** | keep async I/O in one layer and straightforward sync logic in another |
| **Active Object** | an object with its own thread, scheduling its method requests from a queue |
| **Future / Promise / async-await** | the result doesn't exist yet, but it should be a value |
| **Lock-Free Queue** | pass work between threads without locks, where compare-and-swap suffices |
| **Counting Semaphore**[^semaphore] | at most N operations may use a finite resource concurrently; release each acquired permit on every exit path — this bounds concurrency, not requests per second |
| **Barrier / Countdown Latch**[^barrier-latch] | workers must rendezvous at each phase (reusable barrier), or a caller must await N completions (one-shot latch); define failure/cancellation handling so a missing participant cannot hang the group |
| **Lock Striping**[^lock-striping] | one lock over keyed state causes contention but one lock per key costs too much memory; hash keys onto a bounded set of locks, accepting collisions and acquiring distinct stripes in a consistent order |

### Concurrent objects and I/O

| Pattern | Use when |
|---|---|
| **Monitor Object** | an object must serialise its own method calls and schedule them cooperatively |
| **Balking** | act only if state permits *right now*; otherwise return immediately |
| **Guarded Suspension** | same check, but wait until the condition holds |
| **Read-Write Lock** | many readers, rare writers; readers must not exclude each other |
| **Thread Confinement** | the object is touched by one thread only, so it needs no locking |
| **Two-Phase Termination** | a worker must stop cleanly: release locks, flush state, notify |
| **Thread-Safe Interface** | intra-component calls must not re-acquire the component's own lock |
| **Strategized Locking** | the synchronisation mechanism should be a pluggable policy |
| **Scoped Locking** | acquire on scope entry, release on every exit path |
| **Leader/Followers** | threads take turns over one shared set of event sources |
| **Acceptor-Connector** | connection setup separated from the processing after connect |
| **Component Configurator** | link and unlink component implementations at runtime |
| **Lockable Object** | the object publishes its own lock so collaborators serialise against it |

### Task lifetimes and asynchronous work

| Pattern | Use when |
|---|---|
| **Poison Pill** | a sentinel message drains and closes a consumer |
| **Master-Worker** | decompose a task, dispatch it, aggregate the parts |
| **Fork/Join** | recursive subtasks, with idle threads stealing the remainder |
| **Async Method Invocation** | invoke now, collect the result later, without blocking the caller |
| **Event-Based Asynchronous** | model a non-blocking call as start-now, result-later |
| **Backpressure** | a slow downstream stage must tell upstream to slow down, not overflow |
| **Actor** | state plus a mailbox; no shared memory, only messages |
| **Structured Concurrency**[^structured-concurrency] | related subtasks must not outlive their parent operation; own them in a scope that joins them and coordinates failure and cancellation — children must still cooperate with cancellation |
| **Cancellation Token**[^cancellation-token] | a caller must stop work throughout an existing call tree; pass a cancellation signal to each operation and check it at safe points — a request to cancel is not forced termination |
| **Singleflight / In-Flight Request Coalescing**[^singleflight] | concurrent equivalent requests should share one computation and its result or error; key by all result-affecting context, including tenant/authorisation scope — this does not cache results for later callers |
| **Master-Slave** | decomposition and redundant execution coordinated by one master (= Master-Worker, §11); crosses the part boundary, so listed for reference |

### Performance and data locality

| Pattern | Use when |
|---|---|
| **Copy-on-Write** | copies are frequent and mutation rare; share until someone writes |
| **Thread-safe Copy-on-Write** | the same, with the reference count protected |
| **Intrusive Reference Counting / Counted Body** | the object owns its count, so handles stay cheap |
| **Small Object Optimization** | small instances must avoid a per-object allocation |
| **Base-from-Member** | a base must be initialised from a derived object's own data |
| **Erase-Remove** | remove from a sequence without shifting per element |
| **Shrink-to-fit** | capacity, not size, is holding the memory |
| **Generator** | produce a sequence on demand, one item at a time |
| **Lazy Collection** | the sequence is infinite or larger than memory |
| **Memoization** | a pure function is expensive and repeats its inputs |
| **Dirty Flag** | recompute lazily, only when someone asks (see §20) |
| **Transducer** | compose transformations independently of the source or destination |
| **Deforestation** | chained transformations shouldn't materialise intermediates |
| **Divide and Conquer** | independent subproblems plus a combine step |

## 8. Secure, validate and verify

*Choose this group when the decision enforces security, validates input or failures, or proves a consumer-visible contract.*
### Enforcement points

| Pattern | Use when |
|---|---|
| **Protected System / Protection Proxy** | one object owns the resource and every access must pass through it |
| **Reference Monitor**[^reference-monitor] | a protected operation must be unbypassable; every access goes through a small, tamperproof, independently testable decision point, not merely an optional guard |
| **Policy + Guard** | the authorisation rule should live apart from the resource code |
| **Secure Proxy** | gates must nest, each checked independently (defence in depth locally) |
| **Complete Mediation** | every access must be checked, not just the first one |

### Identity and session

| Pattern | Use when |
|---|---|
| **Authenticator** | the authentication mechanism must be swappable behind one interface |
| **Subject Descriptor** | code must ask who is acting without knowing the credential format |
| **Security Context** | the authenticated principal has to travel to the guard |
| **Secure Session Manager** | session state must be centralised, validated and invalidated |
| **Assertion Builder** | you must produce a signed authentication/authorisation assertion |

### Request security pipeline

| Pattern | Use when |
|---|---|
| **Authentication Enforcer** | *whether* to authenticate is central, *how* is delegated |
| **Authorization Enforcer** | centralise the decision, delegate the enforcement |
| **Intercepting Validator** | all inbound data must be validated in one intercepting layer |
| **Secure Base Action** | shared security handling belongs in one base handler |

### Audit and sensitive data

| Pattern | Use when |
|---|---|
| **Audit Interceptor** | record security-relevant events as a decorator, not inline |
| **Secure Logger** | the log must survive tampering by whoever it records |
| **Obfuscated Transfer Object** | sensitive fields travel between layers and shouldn't be readable in transit |

### Security design rules

| Pattern | Use when |
|---|---|
| **Fail-Safe Defaults** | access is denied unless explicitly granted |
| **Least Privilege (per component)** | each part gets only the rights its one job needs |
| **Separation of Duties** | no single object or role completes a sensitive workflow alone |
| **Open Design** | protection must survive the design being public |

### Capabilities and isolation

| Pattern | Use when |
|---|---|
| **Revocable Forwarder / Caretaker**[^capability-wrappers] | a collaborator needs temporary access to an object; supply a forwarding wrapper with separately controlled revocation, never the raw target, and define what happens to calls already in flight |
| **Membrane**[^capability-wrappers] | access restrictions must also cover objects exchanged through an interface; wrap references in both directions, including returned objects and callbacks, rather than guarding only the initial object |

### Compile-time access control

| Pattern | Use when |
|---|---|
| **Attorney-Client** | grant a helper the ability to call a private implementation |
| **Passkey**[^passkey] | selected C++ collaborators may call particular constructors or methods without blanket friendship; require a token only those collaborators can construct — compile-time access control, not a runtime security boundary |

### Validation and error contracts

| Pattern | Use when |
|---|---|
| **Result / Railway-Oriented Result** | failure is expected and must travel with the value, not around it |
| **Option / Maybe** | absence is a legitimate answer, so force it into the signature |
| **Either** | two outcomes, with the error branch carrying a reason |
| **Guard Clause** | reject bad input up front and keep the happy path flat |
| **Mute Idiom** | one specific call's failure is provably ignorable |
| **Deferred Validation** | validate only when the object is actually used |

### Test setup and interaction

| Pattern | Use when |
|---|---|
| **Object Mother** | many tests need valid, ready-made domain objects from one place |
| **Test Data Builder** | a test should state only what it asserts, with valid defaults for the rest |
| **Arrange-Act-Assert / Given-When-Then** | a test method should have three legible beats |
| **Page Object** | a UI change should touch one class, not fifty assertions |

### Test doubles

| Pattern | Use when |
|---|---|
| **Dummy** | a parameter is required but never used by this path |
| **Stub** | the test needs canned answers to calls it makes |
| **Spy** | you'll assert on what was called, after running real code |
| **Mock** | the expected interaction sequence *is* the behaviour under test |
| **Fake** | a working but cheap implementation (in-memory store) is enough |
| **Recording fake** | assert on interactions recorded, rather than on return values |
| **Configurable Test Double**[^test-double-structures] | several tests need the same collaborator role with different responses or failures; configure a reusable double through a test-facing interface separate from the interface production code calls |
| **Self Shunt**[^test-double-structures] | a test can itself implement a small collaborator interface; pass the test object as that collaborator and observe callbacks there — extract a separate double when its responsibilities grow |

### Regression and boundary contracts

| Pattern | Use when |
|---|---|
| **Characterization Test / Golden Master** | pin current behaviour before changing legacy code |
| **Consumer-Driven Contract Test**[^consumer-driven-contract] | independently released consumer and provider must keep working; capture only the consumer's required request/reply/error interactions and verify the provider, alongside—not instead of—end-to-end tests |

### Testability seams

| Pattern | Use when |
|---|---|
| **Humble Object**[^humble-object] | framework callbacks, UI controls or entry points are hard to exercise; leave minimal glue there and move decisions into a plain testable collaborator, while still checking the remaining integration |
| **Test-Specific Subclass**[^test-double-structures] | legacy code has an overridable dependency call but no injection point; substitute that boundary in a test-only subclass — a constrained fallback, since tests now depend on inheritance details |

## 9. Implement specialised component types

*Choose this group when the decision implements a specialised UI, runtime loop, parser, or language-processing component.*
### Presentation state and coordination

| Pattern | Use when |
|---|---|
| **Presentation Model** | you want a screen's state and commands testable without a UI |
| **Model-View-Presenter** | the view should be a passive interface the presenter drives |
| **Model-View-ViewModel** | the view binds declaratively to a view-shaped observable model |
| **Model-View-Intent** | all user intent flows as actions into one reducer of state |
| **View Handler** | one object must own creating, switching and syncing a model's several views (§21) |
| **Flux / Redux** | many components mutate shared state; route it through one store and one-way updates |
| **Bloc** | state changes should be explicit events and states, streamed to the UI |
| **Container component** | state lives in a wrapper while children stay stateless |
| **Context / Provider** | cross-cutting props shouldn't be threaded down every level |
| **State Colocation** | keep state at the smallest scope that actually needs it |
| **Props Drilling** | *the smell*: state is being relayed through components that ignore it |

### Client-side composition and reuse

| Pattern | Use when |
|---|---|
| **Compound Components** | parent and children must share implicit state, like `select` and `option` |
| **Higher-order Component** | add behaviour to many components without editing each one |
| **Render Props** | hand a function the state, and let the caller decide the markup |
| **Custom Hook / Composable Logic** | reuse stateful behaviour across unrelated components |
| **Presentational vs Container** | separate how it looks from where the data comes from |
| **Headless component** | export the behaviour and let the consumer supply the markup |

### Per-object runtime and virtual machines

| Pattern | Use when |
|---|---|
| **Update Method** | every entity needs a per-frame step; one driver calls them all |
| **Game Loop / fixed-step driver** | simulation rate must be decoupled from I/O or render rate |
| **Double Buffer** | two buffers swapped atomically, so no reader sees a half-written state |
| **Event Queue** | buffer events and dispatch later, decoupling sender from receiver timing |
| **Spatial Partition** | queries should inspect neighbours by position, not scan everything |
| **Data Locality** | objects should be laid out for the cache, not for the class hierarchy |

### Language implementation

| Pattern | Use when |
|---|---|
| **Recursive-Descent Parser**[^recursive-descent] | a small grammar should map directly to readable functions, one per rule; use lookahead to choose alternatives and eliminate unsupported left recursion |
| **Pratt Parser / Top-Down Operator Precedence**[^pratt-parser] | an expression grammar has many prefix, infix or postfix operators; associate parsing behaviour and binding power with tokens instead of writing a function for every precedence level |
| **Memoizing Parser / Packrat Parser**[^packrat-parser] | backtracking repeatedly parses the same rule at the same input position; memoize successes and failures, trading input-sized storage for predictable work — stateful rules need more than a position-only cache key |
| **Normalized Heterogeneous AST**[^ast-shapes] | syntax nodes need different types and node-specific data, but generic walkers need a uniform children collection; preserve that common traversal shape |
| **Irregular Heterogeneous AST**[^ast-shapes] | named, typed child fields such as `condition`, `left` and `right` make node invariants clearer; accept node-specific traversal code instead of a uniform children collection |
| **Symbol Table for Nested Scopes**[^scope-symbols] | names may shadow outer declarations; represent enclosing scopes explicitly and resolve from the innermost outward, keeping name lookup separate from syntax recognition |
| **Tree Pattern Matcher**[^tree-matching] | analysis or rewriting targets particular subtree shapes; separate pattern/action rules from traversal so unrelated syntax can be skipped — unlike a plain Visitor, dispatch depends on structure, not just node type |

## Facet views

Use these as filters over the canonical homes above, never as a second recommendation path.

| Facet | Use it to answer | Primary-home reading |
|---|---|---|
| **Provenance** | Which source language and established catalog explains the pattern? | GoF is concentrated in §§2–5; EAA in §§1, 4 and 6; EIP in §§4–5; DDD in §§1, 4 and 6; POSA and concurrency catalogs in §§6–7. Inline citations and the source notes remain the authority. |
| **Style / paradigm** | Does the local code favour OO collaboration, functional composition, or data-oriented processing? | OO-heavy choices appear throughout. Functional construction is in §2, interpretation in §3, immutable data in §1, and lazy/dataflow execution in §7. |
| **Implementation scope** | Is the decision local to a value, object, module, boundary, message flow, persistence unit, or task lifetime? | Sections 1–3 are primarily local; §4 is boundary-facing; §5 is message/workflow-facing; §6 owns durable state; §7 owns execution lifetimes. |
| **Language affinity** | Is this an idiom whose payoff depends on C++, Rust, Java, or a UI/runtime ecosystem? | Language-sensitive mechanics live beside their implementation decision: C++ and Rust idioms mainly §§1–3, resource rules in §6, JVM/.NET concurrency in §7, and UI/parsing techniques in §9. |

## Background, principles and excluded material

These names supply context or describe systems at a larger scale. They are intentionally not primary MCP implementation recommendations.

### Strategic DDD context relationships

| Pattern | Use when |
|---|---|
| **Bounded Context** | one model, one language, one explicit boundary of validity |
| **Ubiquitous Language** | the team and the code must use the same terms inside a context |
| **Context Map** | you need to document how the contexts relate and where the seams are |
| **Upstream/Downstream** | one context's model shapes the other's whether you like it or not |
| **Partnership** | two contexts succeed or fail together and commit to coordinated change |
| **Shared Kernel** | a small model is deliberately shared, and changed only jointly |
| **Customer/Supplier** | the downstream team plans its needs into the upstream team's cycle |
| **Conformist** | you can't influence the upstream model, so you adopt it wholesale |

The part-level seam patterns **Anti-Corruption Layer**, **Open Host Service**, **Published Language**, and **Separated Domain Model** remain candidates in their canonical homes because each can be implemented by an owned boundary or model component.

### System-scale architecture patterns

| Pattern | Use when |
|---|---|
| **Client-Dispatcher-Server** | the client shouldn't know where the service lives; boundary-adjacent |
| **Broker** | location and discovery are resolved by an intermediary; boundary-adjacent |

These two POSA names are retained as background references because their normal use is whole-system architecture, not one primary implementation component.

### Principles, anti-patterns and documentation techniques

- **Principles:** SOLID · DRY · KISS · Separation of Concerns · Least Knowledge · Composition over Inheritance · high cohesion / low coupling. They guide when to apply a pattern; they are not reusable solutions themselves.
- **Anti-patterns:** God Object · Anemic Domain Model · Golden Hammer · Spaghetti Code · Boilerplate · Dead Code · Shoot-the-Footgun.
- **Documentation techniques:** diagrams and decision records describe a design; they are not the design.
- **Out of scope by design:** layered, hexagonal, clean, onion, microkernel, microservices versus monolith, API Gateway, service discovery, CQRS, Event Sourcing, Saga, Transactional Outbox, Strangler Fig, sharding, replication, change-data-capture, data engineering, autoscaling, and fleet-level resiliency. Where a name has a locally implementable part, that part is catalogued here instead—for example ACL, Presentation Model, Write-Ahead Log, Poison Pill, Backpressure, and Version Number.

## Sources

Gamma, Helm, Johnson & Vlissides, *Design Patterns: Elements of Reusable Object-Oriented Software* (1994) — classic object patterns distributed through §§2–5.  
Fowler et al., enterprise application pattern catalog (2002), martinfowler.com/eaaCatalog — primary source for domain, boundary, persistence and offline patterns in §§1, 4 and 6.  
Hohpe & Woolf, *Enterprise Integration Patterns* (2003), enterpriseintegrationpatterns.com — messaging contracts, routing and delivery in §§4–5.  
Evans, *Domain-Driven Design* (2003); Vernon, *Implementing Domain-Driven Design* (2013) — tactical modelling and context-mapping background.  
Schmidt, Stal, Rohnert & Buschmann (2000), concurrent & networked object pattern catalog — execution and coordination patterns in §7.  
Aziz Atiyeh and others, cloud operations pattern catalogs — dependency-failure and overload patterns in §7.

**Expanded inventory and provenance notes.**  
Buschmann, Meunier, Rohnert, Sommerlad & Stal (1996), *POSA Vol. 1: A System of Patterns* — background system-scale entries and design-level patterns.  
Frank, Schmidt, Rohnert & Stal (2009), *POSA Vol. 3: Patterns for Resource Management* — resource lifecycle patterns in §6.  
Schumacher, Fernandez-Buglioni, Hybertson, Buschmann & Sommerlad (2005), *Security Patterns*; Nagappan & Steel (2005), *Core Security Patterns* — security patterns in §8.  
Nystrom (2014), *Game Programming Patterns*, gameprogrammingpatterns.com — real-time runtime patterns in §9.  
Beck (1997), *Smalltalk Best Practice Patterns* — local code-structure patterns in §2.  
Lea (1999), *Concurrent Programming in Java* — Balking, Guarded Suspension, Read-Write Lock, Two-Phase Termination, Thread Confinement in §7.  
Tyagi & Volchko, *More C++ Idioms*, en.wikibooks.org — C++ idioms in §§1–3 and §6.  
Java Design Patterns catalog, java-design-patterns.com — named entries distributed across the catalog.  
ddd-crew/context-mapping and Evans/Vernon strategic material — background and boundary seams.  
Functional pattern catalogs (softwarepatternslexicon.com/functional; patterns.dev) — functional modelling, composition and execution across §§1–3 and §7.  
Kent C. Dodds on compound components, kentcdodds.com/blog/compound-components-with-react-hooks; Microsoft Rust patterns book (newtype, typestate) — §9 UI composition and §§1–3 language-sensitive value patterns.  
Security pattern inventory, en.wikipedia.org/wiki/Security_pattern.  
POSA series identification incl. Vol. 4, dre.vanderbilt.edu/~schmidt/POSA/.  
POSA3 resource-acquisition names (Lookup, Lazy/Eager/Partial Acquisition, Caching), oreilly.com/library/view/pattern-oriented-software-architecture/9780470845257.

> **Unverified attributions.** Items originating in the PLoP (*Pattern Languages of Program Design*) volumes are included, but the specific volume/author attributions were not confirmed in this pass and should be checked before being quoted. POSA Vol. 4 (*A Pattern Language for Distributed Computing*, 2007) was not surveyed because it is largely system-level.

**Web-verified additions (2026-09-16).**
The footnotes on new rows link to the sources consulted for this expansion; they do not retroactively verify the older inventory. Selection guidance is scoped to component implementation, even where a source also discusses wider applications.

[^semaphore]: Oracle, [Java SE `Semaphore`](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/concurrent/Semaphore.html) — permit-based bounds on simultaneous resource users.
[^barrier-latch]: Oracle, [Java SE `CyclicBarrier`](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/concurrent/CyclicBarrier.html) and [`CountDownLatch`](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/concurrent/CountDownLatch.html) — reusable rendezvous versus one-shot completion gate.
[^lock-striping]: Guava, [`Striped` design and contract](https://javadoc.io/static/com.google.guava/guava/33.4.0-jre/com/google/common/util/concurrent/Striped.html) — concurrency/memory trade-off, key collisions and ordered multi-stripe acquisition.
[^structured-concurrency]: OpenJDK, [JEP 499: Structured Concurrency](https://openjdk.org/jeps/499), motivation and description; Nathaniel J. Smith, [Notes on structured concurrency](https://vorpus.org/blog/notes-on-structured-concurrency-or-go-statement-considered-harmful/) — task lifetime nesting, joining, failure and cancellation policies.
[^cancellation-token]: Microsoft, [Cancellation in Managed Threads](https://learn.microsoft.com/en-us/dotnet/standard/threading/cancellation-in-managed-threads); Sameer Ajmani, [Go Concurrency Patterns: Context](https://go.dev/blog/context) — cooperative cancellation propagated through related operations.
[^singleflight]: Go authors, [`singleflight` documentation and implementation](https://raw.githubusercontent.com/golang/sync/master/singleflight/singleflight.go) — one in-flight execution per key with shared results and errors.
[^functional-options]: Dave Cheney, [Functional options for friendly APIs](https://dave.cheney.net/2014/10/17/functional-options-for-friendly-apis) — named option functions, defaults and extensible construction.
[^passkey]: Chromium, [The Passkey Pattern](https://chromium.googlesource.com/chromium/src/+/HEAD/docs/patterns/passkey.md) — restricting selected entry points through privately constructible tokens.
[^dispose]: Microsoft, [Dispose Pattern](https://learn.microsoft.com/en-us/dotnet/standard/design-guidelines/dispose-pattern) and [Implement a Dispose method](https://learn.microsoft.com/en-us/dotnet/standard/garbage-collection/implementing-dispose) — deterministic, idempotent cleanup and ownership-aware cascading.
[^weak-reference]: Oracle, [Java SE `WeakReference`](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/lang/ref/WeakReference.html); Python documentation, [`weakref`](https://docs.python.org/3/library/weakref.html) — non-owning references and safe retrieval of possibly collected targets.
[^zipper]: Gérard Huet, [The Zipper](https://www.st.cs.uni-saarland.de/edu/seminare/2005/advanced-fp/docs/huet-zipper.pdf), *Journal of Functional Programming* 7(5), 1997 — a focused subtree paired with its context for local functional editing.
[^capability-wrappers]: Deian Stefan and John C. Mitchell, [Analysing Object-Capability Patterns With Murϕ](https://cseweb.ucsd.edu/~dstefan/pubs/stefan%3A2011%3Aocap.pdf), §§2.2 and 4.1–4.2 — revocable forwarders, membranes, model assumptions and revocation races.
[^recursive-descent]: Robert Nystrom, [Crafting Interpreters, §6.2: Recursive Descent Parsing](https://craftinginterpreters.com/parsing-expressions.html#recursive-descent-parsing); Terence Parr, [Language Implementation Patterns, contents](https://media.pragprog.com/titles/tpdsl/toc.pdf), patterns 1–4.
[^pratt-parser]: Robert Nystrom, [Pratt Parsers: Expression Parsing Made Easy](https://journal.stuffwithstuff.com/2011/03/19/pratt-parsers-expression-parsing-made-easy/) — token-associated parselets, precedence and associativity.
[^packrat-parser]: Bryan Ford, [Packrat Parsing: Simple, Powerful, Lazy, Linear Time](https://bford.info/pub/lang/packrat-icfp02.pdf), 2002, especially §§2 and 5 — memoization, storage cost and statelessness assumptions; Parr's [contents](https://media.pragprog.com/titles/tpdsl/toc.pdf) names Memoizing Parser as pattern 6.
[^ast-shapes]: Terence Parr, [Language Implementation Patterns, publisher excerpt](https://media.pragprog.com/titles/tpdsl/patterns.pdf), patterns 10–11 — uniform child lists versus node-specific named fields.
[^scope-symbols]: Terence Parr, [Language Implementation Patterns, contents](https://media.pragprog.com/titles/tpdsl/toc.pdf), pattern 17; Robert Nystrom, [Crafting Interpreters: Resolving and Binding](https://craftinginterpreters.com/resolving-and-binding.html) — explicit scope stacks, shadowing and lexical resolution.
[^tree-matching]: Terence Parr, [Language Implementation Patterns, static-typing excerpt](https://media.pragprog.com/titles/tpdsl/static.pdf), pp. 203–204 — applies Tree Pattern Matcher (pattern 15) to selected expression subtrees independently of surrounding syntax.
[^quantity]: Martin Fowler, [Application Facades](https://martinfowler.com/apsupp/appfacades.pdf), “Implementing Quantity”, p. 4 — the local amount-plus-unit pattern from *Analysis Patterns*, not the article's broader presentation structure.
[^range]: Martin Fowler, [Range](https://martinfowler.com/eaaDev/Range.html), eaaDev draft — interval objects, bound inclusivity, open ends and range operations.
[^effectivity]: Martin Fowler, [Effectivity](https://martinfowler.com/eaaDev/Effectivity.html), eaaDev draft — validity periods attached to model objects.
[^temporal-property]: Martin Fowler, [Temporal Property](https://martinfowler.com/eaaDev/TemporalProperty.html), eaaDev draft — time-parameterised properties that encapsulate history.
[^temporal-object]: Martin Fowler, [Temporal Object](https://martinfowler.com/eaaDev/TemporalObject.html), eaaDev draft — continuity/version roles and the explicit distinction from Temporal Property.
[^humble-object]: Martin Fowler, [Humble Object](https://martinfowler.com/bliki/HumbleObject.html), summarising Michael Feathers and Gerard Meszaros — extracting decisions from hard-to-test elements into testable collaborators.
[^test-double-structures]: Gerard Meszaros, [xUnit Test Patterns, Chapter 23 (publisher sample)](https://ptgmedia.pearsoncmg.com/images/9780131495050/samplechapter/0131495054_CH23.pdf) — Configurable Test Double (p. 558), Self Shunt (p. 570), and Test-Specific Subclass (p. 579, including its fragility trade-off and behaviour-modifying variant).
[^dependent-mapping]: Fowler, [Dependent Mapping](https://martinfowler.com/eaaCatalog/dependentMapping.html) — a parent mapper handles mappings for dependent child objects.
[^cursor-pagination]: Google, [AIP-158: Pagination](https://google.aip.dev/158) — bounded pages, opaque page tokens and `next_page_token` progression.
[^eip-request-reply]: Hohpe & Woolf, [Request-Reply](https://www.enterpriseintegrationpatterns.com/patterns/messaging/RequestReply.html) — paired request and reply messages, with synchronous blocking or asynchronous callback delivery.
[^eip-envelope-wrapper]: Hohpe & Woolf, [Envelope Wrapper](https://www.enterpriseintegrationpatterns.com/patterns/messaging/EnvelopeWrapper.html) — wrap application data in a messaging-infrastructure-compliant envelope, then unwrap it at the destination.
[^eip-normalizer]: Hohpe & Woolf, [Normalizer](https://www.enterpriseintegrationpatterns.com/patterns/messaging/Normalizer.html) — convert semantically equivalent messages from different formats into one canonical form.
[^eip-channel-adapter]: Hohpe & Woolf, [Channel Adapter](https://www.enterpriseintegrationpatterns.com/patterns/messaging/ChannelAdapter.html) — publish application data to a channel and invoke application functions from incoming messages.
[^eip-service-activator]: Hohpe & Woolf, [Service Activator](https://www.enterpriseintegrationpatterns.com/patterns/messaging/MessagingAdapter.html) — invoke a service in response to a message.
[^eip-polling-consumer]: Hohpe & Woolf, [Polling Consumer](https://www.enterpriseintegrationpatterns.com/patterns/messaging/PollingConsumer.html) — an application-controlled loop actively receives messages.
[^eip-durable-subscriber]: Hohpe & Woolf, [Durable Subscriber](https://www.enterpriseintegrationpatterns.com/patterns/messaging/DurableSubscription.html) — preserve messages for a subscriber while it is unavailable.
[^eip-message-store]: Hohpe & Woolf, [Message Store](https://www.enterpriseintegrationpatterns.com/patterns/messaging/MessageStore.html) — retain message information for reporting and analysis without coupling the main flow.
[^deadline-propagation]: gRPC, [Deadlines](https://grpc.io/docs/guides/deadlines/) — downstream calls honor the original deadline, propagated as a remaining timeout to tolerate clock skew.
[^cache-aside]: Microsoft, [Cache-Aside Pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/cache-aside) — on-demand fill on a miss; update the source before cache invalidation to control staleness.
[^functional-core]: Gary Bernhardt, [Functional Core, Imperative Shell](https://www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell/) — pure core surrounded by I/O code, improving isolated testability and local reasoning.
[^reference-monitor]: NIST CSRC, [reference monitor](https://csrc.nist.gov/glossary/term/reference_monitor) — complete mediation, tamper resistance and verifiability requirements.
[^consumer-driven-contract]: Pact, [Introduction to contract testing](https://docs.pact.io/) — consumers capture exercised interactions; providers verify those contracts without replacing end-to-end tests.
