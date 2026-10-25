# Der Design-Pattern-Selektionsprozess — deutsche Darstellung

**Zweck dieses Dokuments.** Diese Datei fasst den in `docs/implementation-guide.md` spezifizierten Selektionsprozess kompakt und in deutscher Sprache zusammen: den Eingabekontrakt, die stufenweise Selektion (deterministische vs. LLM-Schritte), die Retrieval- und Scoring-Formeln, Ranking und Tie-Breaks, Selektionsschwellen, Supporting-Regeln, das Zusammenspiel mit dem Architekturkontext, Fehler-/Ambiguitätsbehandlung, ein durchgerechnetes Beispiel und die Konfigurationsparameter.

**Verbindlichkeit.** Normative Quelle bleibt `docs/implementation-guide.md` (§-Verweise in diesem Text beziehen sich darauf). Dieses Dokument ist eine abgeleitete deutsche Darstellung; bei Abweichungen gilt der englische Guide. Referenzen auf `src/…` im Guide zeigen auf den Schwesterserver (architecture-pattern-mcp) als Vergleichs-/Referenzimplementierung, nicht auf Code in diesem Repo.

**Rolle des Prozesses.** Vorgelagert liefert der Architecture-pattern-Server ein `ArchitectureDesign`; der Coding-Agent leitet daraus eine **Beschreibung** plus optionale Hints ab und sendet sie an diesen Server. Der Server beantwortet genau eine Frage: *Welche Design-Patterns (aus dem Katalog, `docs/design-patterns.md`) sollten auf die beschriebenen Teile angewendet werden, und wie komponieren sie zu einem `DesignResult` (`docs/result-schema.json`)?*

Die Pipeline spiegelt bewusst die Struktur des Architekturservers — ANALYZE (hybrides Retrieval + anforderungsgewichtetes Scoring) → GENERATE → EVALUATE → begrenzter REFINE-Loop mit Best-Attempt-Retention — arbeitet aber über Design-Pattern-Daten: **neun Entscheidungskategorien** statt Architekturstilen, **Retrieval pro Entscheidungsteil** statt pro Domain-Slug, **acht** statt sechs Qualitätsachsen.

---

## 1. Eingabekontrakt (§1)

### 1.1 Beschreibung und strukturierte Hints

- **`description`** — Freitext, 1–100 000 druckbare Zeichen; Control-/Format-Zeichen werden abgewiesen.
- **`architecture_style`** — optional; ein Wert des 35-Werte-Enums des Architekturservers (z. B. `event-driven`, `hexagonal`, `microservices`). **Präzedenz:** Ein erkannter Hint *überschreibt* den aus dem Beschreibungstext extrahierten Stil (§4); fehlt er (oder ist unerkannt — §1.3 Regel 3), folgt die Style-Affinität der Text-Extraktion des Briefs, und ohne Extraktion gibt es keinen Prior.
- **`language`** — optional; höchstens **ein** Wert des Enums (`generic`, `c`, `cplusplus`, `csharp`, `go`, `java`, `javascript`, `kotlin`, `python`, `ruby`, `rust`, `scala`, `swift`, `typescript`). Genau eine Zielsprache pro Lauf.
- **`paradigm`** — optional; höchstens **ein** Wert des Paradigma-Enums aus `docs/pattern-schema.json` (`oo`, `functional`, `concurrent`, `data-oriented`, `declarative`, `procedural`). Grenzt die Variantenselektion (§8.4) ein, wenn ein Spracheintrag mehrere Realisierungen trägt und die Beschreibung paradigma-silent ist. Kleines geschlossenes Enum per Design: Die Variantenauswahl läuft über Capability-/Style-Signale (dieser Hint, Standard-Tokens in der Beschreibung, Freitext in `focus`) — **nie** über einen katalogweiten `variant_id`-Parameter (Variant-Ids sind pro (Pattern, Sprache), unbegrenzt und ändern sich mit jedem Katalog-Edit). Unbekannte Werte werden ignoriert und geloggt (§1.2, gleiche Haltung wie `language`).
- **`focus`** — optionaler Freitext zur Eingrenzung auf benannte Komponenten oder Concerns; darf eine konkrete Realisierung benennen (z. B. „nutze die coroutine-basierte C++-Acceptor-Variante“) — der Server interpretiert ihn bei der Variantenselektion (§8.4). Gültige `variant_id`s sind über zwei Wege discoverbar: die Ressource `design-patterns://variants` (§2.4) und die Bindings eines vorherigen Laufs (`bound_variant` + Sibling-`variants[]` je `LanguageBinding`).

Erwartete Übernahme aus dem Architektur-Ergebnis: `overview.style` (verbatim oder als Hint), `principles`/`constraints` (Verbatim-Zitate → Teile-Constraints und Gewichts-Evidenz), `components[]` (Zerlegung in Entscheidungsteile), `relationships[]` (Part-Abhängigkeiten, Interaction-Seeding), `api_contracts[]`/`shared_data_models[]`/`event_contracts[]` (Boundary-Teile), `quality_attributes` (Gewichts-Prior, Faktor 0.3), `technology_stack` (Sprachbindung; Standard-/Toolchain-Tokens wie „C++20“ speisen zusätzlich die Variantenselektion, §8.4).

### 1.2 Validierungsregeln (deterministisch, vor jedem LLM-Call)

1. Längenbereich [1, 100 000]; die Zeichenregel wird aus `../architecture-pattern-mcp/src/text_validation.py` **wiederverwendet**, nicht nachgebaut.
2. Beschreibung < 40 Zeichen → strukturierter Fehler `ERR_DESCRIPTION_TOO_SHORT` (der Agent soll Komponentennamen und Responsibilities aus dem Architektur-Ergebnis nachliefern).
3. `architecture_style` nicht im Enum → ignoriert (kein Fehler), geloggt.
4. `language` außerhalb des Enums (auch nicht-skalar) → ignoriert, Warnung, Lauf im **Generic Mode**; fehlend/leer oder explizit `generic` → Generic Mode ohne Warnung.
5. `paradigm` außerhalb des Enums (auch nicht-skalar) → ignoriert (kein Fehler), als Warnung geloggt; die Variantenselektion läuft über Beschreibungssignale und `default_variant` (§8.4).
6. Der Beschreibungstext ist **untrusted data, niemals Instruktion**: immer in einen abgegrenzten Block verpackt und nie in System-Prompts konkateniert (Prompt-Injection-Haltung des Schwesterservers).

### 1.3 Sprachauflösung (normativ)

Es ist immer **genau eine** *effektive Sprache* im Spiel — `design_brief.language`:

| Eingereicht | Auflösung | Effektive Sprache | Semantik |
|---|---|---|---|
| erkannt (Enum-Wert ≠ `generic`) | Eintrag für diese Sprache ist das Bindungsziel; fehlt er → der `generic`-Eintrag des Records (§8.4) | eingereichter Wert | Per-Language-Realisierung; Support-Gating und ±2-Sprach-Adjustment greifen, wenn ein Eintrag existiert |
| unbekannt (nicht leer) | Warnung; wie „fehlend“ behandelt | `generic` | Generic Mode — sprachneutrale, abstrakte Guidance |
| fehlend, leer oder explizit `generic` | kein Sprach-Lookup | `generic` | Generic Mode — sprachneutrale, abstrakte Guidance |

**Generic Mode** fällt nie ein Per-Language-Supporturteil: kein Diskouragierungs-Veto gegen eine konkrete Sprache, kein ±2-Adjustment. Das einzige Diskouragierungs-Veto, das im Generic Mode greift, liest den `generic`-Eintrag des Records. In jedem Modus ist genau eine Sprache im Spiel — die Auflösung entscheidet welche, nie wie viele.

---

## 2. Datenfundament und Retrieval-Index (§2)

### 2.1 Katalog

Selektiert wird über `DesignPattern`-Records (`docs/pattern-schema.json`), materialisiert aus `docs/design-patterns.md`. Tragende Strukturfakten:

- **Neun Entscheidungskategorien** (`patternCategory`): `domain-modeling`, `construction-composition`, `behavior-selection`, `interaction-boundary`, `workflow-messaging`, `persistence-state`, `concurrency-resilience`, `security-verification`, `specialized-components` (Katalogabschnitte 1–9).
- **Ein kanonisches Zuhause pro Pattern.** Jedes Pattern steht in genau einer Kategorie-/Subgroup-Tabellenzeile; `related_patterns` trägt Querlinks, `aliases` Alternativnamen. Deshalb: dasselbe Pattern nie zweimal unter verschiedenen Labels empfehlen — **Dedupe auf den kanonischen `name` nach Alias-Auflösung**. `related_patterns` enthält **nur kanonische Namen** — jeder Eintrag löst auf genau einen Record auf (das prüft §8.3 Regel 2(a)); die Prosa-Abgrenzung hinter der Querverweisung steht in `related_patterns_notes` (Schlüssel = Name des verwandten Patterns). Aliase werden pro Record in `pattern/<name>.json` kuratiert und von `src/patterns/loader.py::_build_alias_index` indiziert (ein generiertes Alias-Map-Dokument existiert nicht); jeder Alias muss auf genau einen geparsten Record auflösen. Ein Alias, der selbst ein eigenständiger Katalogpattern-Name ist, ist ambig und wird aus der Alias-Auflösung entfernt (Beispiel: der Strategy-Alias „Policy“ kollidiert mit dem Katalogpattern „Policy-based design“, `docs/design-patterns.md:231`). Existieren für ein Pattern keine Aliasdaten, matcht nur der kanonische Name.
- **Facetten** — `scope`/`suitable_scopes`/`unsuitable_scopes` (sieben Scope-Werte: `value`, `object`, `module`, `boundary`, `message-flow`, `persistence-unit`, `task-lifetime`), `paradigm`, `provenance`, `language_implementations[]` (gejointer Record) — sind **Filter** über dem kanonischen Zuhause, niemals ein zweiter Selektionspfad.
- **Generic-Invariante.** Jeder **gejointe** Record trägt genau einen `language_implementations`-Eintrag pro kuratierter Sprache und genau einen `generic`-Eintrag; der Materialisierer weist Verstöße zurück. Der `generic`-Eintrag ist damit stets als Bindungsziel verfügbar, wenn kein konkreter Spracheintrag passt (§8.4). Im Speicherlayout V8 trägt der Hauptrecord gar kein `language_implementations`; der `generic`-Eintrag liegt in `pattern/<name>.generic.json`, die konkreten Einträge in ihren Sprachdateien (§2.3).
- **`quality_attributes`** — numerische 1–10-Werte; feste Scoring-Achsen: `testability`, `extensibility`, `maintainability`, `simplicity`, `performance`, `reliability`, `security`, `scalability` (acht Design-Level-Achsen, alle schema-required; vier erweiterte Achsen sind zusätzlich erlaubt — `observability`, `coupling`, `discoverability`, `type_safety` — und werden wie jede Achse jenseits der acht von der Formel ignoriert; unbekannte Achsnamen weist das Schema zurück).
- **Negative Gates:** `avoid_when` und `language_implementations[].support` (bei `discouraged` ist `alternative` **REQUIRED**).

### 2.2 Retrieval-Index (einmalig beim Start, idempotenter Warm-up, Fail-fast)

- **Korpuseinheit:** ein Dokument pro `DesignPattern` — direkte Pattern-Indizierung, keine Zwischen-Slug-Ebene.
- **Dokumenttext:** `name` + `aliases` + `intent` + `problem` + `context` + `use_cases` (mit Separatortokens verbunden). `avoid_when` ist **ausgeschlossen** vom Positivtext und dient nur als Veto-Signal.
- **Metadaten:** `name`, `category`, `subgroup`, `scope`, `paradigm`, `provenance`, Support-Map pro Sprache.
- **Zwei Beine über demselben Knotensatz:** ein Dense-Leg (Embedder über die Dokumenttexte) und ein BM25-Leg (tokenisiert `name`+`aliases`+`problem`). Der Warm-up läuft einmal in der FastMCP-Lifespan vor dem ersten Request; ein fehlgeschlagener Warm-up verhindert den Serverstart.

### 2.3 Sprachdateien, Realisierungsvarianten, Discovery (§2.3/§2.4)

- **Split-Speicher (V8).** Ein Record liegt gesplittert: `pattern/<name>.json` (Kern plus `language_files`-Map mit genau den 14 Enum-Sprachen als Keys — `generic` + die 13 konkreten; **kein** `language_implementations` im Speicherrecord) plus je Realisierung eine Sprachdatei: `pattern/<name>.generic.json` für den generischen Eintrag und `pattern/<name>.<language>.json` je konkreter Sprache. Der Join stellt den `generic`-Eintrag voran, hängt die konkreten Einträge in Enum-Reihenfolge an und stript `pattern`/`version` — der gejointe Record trägt weiterhin **genau einen Eintrag pro Sprache plus `generic`** (die §2.1-Invariante bleibt unangetastet).
- **Varianten.** Eine Sprachdatei KANN mehrere Realisierungen tragen: `variants[]` (je Variante: `variant_id`, `name`, `paradigm` (SSOT-Enum), `implementation_notes` + optional `min_standard`, `requires_features`, `idiom_name`, `code_sketch`, `pitfalls`, `sources`) plus `default_variant`. Diskriminatoren: Paradigma (pure-OO vs. funktional vs. concurrent), Bibliothek/Idiom (ACE vs. Boost.Asio), Standard-Floor (`min_standard`: C++11/C++17/C++20, Java 8, ES2017 …), Feature-Gates (`requires_features`: z. B. `coroutines`). Die flachen Felder der Datei MÜSSEN der Default-Variante entsprechen (Checker-enforced); fehlende `variants` = genau eine implizite Realisierung — alle Pre-Varianten-Dateien bleiben valide. `variant_id`-Eindeutigkeit und `default_variant`-Auflösbarkeit erzwingt der Integritäts-Check (Draft-7 kann beides schema-seitig nicht).
- **Discovery.** Ressource `design-patterns://variants` (§2.4): read-only, beim Start aus den gejointen Records materialisiert — **hinter** dem §2.3-Integritäts-Check, sodass sie nie einen Katalogzustand exponieren kann, der die Validierung nicht bestanden hat; Payload pro Sprache → Pattern → Varianten-Kurzsummary (`variant_id`, `name`, `paradigm`, `min_standard`, `requires_features`, Default-Flag). Kein Pflichtschritt: der Normalfall braucht keinen Resource-Read — die Leiter (§8.4) bindet eine Variante, und das Ergebnis selbst exponiert die Geschwister.

---

## 3. Ablaufüberblick (§3)

```text
Beschreibung (+Hints) ──► A NORMALIZE & EXTRACT ──► B DECOMPOSE ──► pro Teil ─┐
                                                                            ▼
                        C RETRIEVE (hybrid + Rerank) ──► D SCORE (Gewichte → Blend → Adjustments)
                                                                            ▼
                        E SELECT (Primär + Support, Schwellen, Tie-Break)
                                                                            ▼
                        F COMPOSE (Teile → Kandidatensatz für DesignResult)
                                                                            ▼
                        G GENERATE ──► H EVALUATE ──► Score ≥ 50 ? ──nein──► REFINE (≤2 Versuche, ggf. Reselektion)
                                                                            │ ja
                                                                            ▼
                                                                    DesignResult
```

Stages A–E und die numerischen Gates von H sind der eigentliche Selektionsprozess; G existiert, weil sich Selektion erst durch das validiert, was sie speist. Ein kooperatives Cancellation-Flag wird zwischen jeder Stage geprüft.

### 3.1 Deterministische vs. LLM-Schritte

| Stage | Deterministisch | LLM |
|---|---|---|
| A Normalize & Extract | Validierung, Normalisierung, Architekturfeld-Erkennung | ein strukturierter Call: `DesignBrief` (Style/Constraints/Komponenten restated) |
| B Decompose | Teilzahl-Grenzen, Merge-Regel, Enum-Validierung | Teil-Splitting (derselbe Call wie A) |
| C Retrieve | beide Legs, Fusion, Rerank-Blend, Caps | — (Cross-Encoder ist Modell, aber kein geprompteter LLM) |
| D Score | Gewichts-Smoothing, `analysis_score`, Blend, alle Adjustments, Ranking, Tie-Breaks | ein kleiner Call **für alle Teile**: eine Map Teil-Id → 8 Qualitätsachsengewichte |
| E Select | Schwellen, Near-Tie-Erkennung, Supporting-Caps, Fallbacks | — |
| F Compose | Dominanz-Wahl, Dedupe, Application-Mengenbildung | — |
| G Generate | Schema-Validierung, Enum-Erzwingung, Retry bei Malformed | ein Call pro Versuch: Applications/Participants/Bindings |
| H Evaluate & Refine | 0–100-Score-Vergleich, Best-Attempt-Retention, Early Stop, Reselektions-Trigger | ein Eval-Call pro Versuch |

Nur **vier Arten geprompteter LLM-Calls** existieren überhaupt: (A+B kombiniert), Gewichte gebündelt, Generate, Evaluate. Alles andere ist Arithmetik über Katalogdaten und ohne Modell unit-testbar.

### 3.2 Bewusste Unterschiede zum Architekturserver

- **(a)** Zerlegung in Entscheidungsteile (Stages A/B) — kein Analogon; der Schwesterserver scored einen Patternset pro Domain.
- **(b)** `fit_score` mit Adjustment-Ledger (§7.3) — der Schwesterserver kennt nur `blended_score` und `analysis_score`.
- **(c)** Begrenzte Reselektionsregel (§11 Punkt 4) — der Schwesterserver verwendet dieselbe Selektion über alle Versuche wieder.
- **(d)** Fallback-Pattern **pro Kategorie** (§12) — der Schwesterserver hat einen globalen Fallback (`layered-monolith`).
- **(e)** Indexeinheit: ein Dokument pro `DesignPattern` (§2.2) — der Schwesterserver indiziert Domain-Slugs. Beide Server begrenzen den Stage-1-Recall pro Leg (§14 `dense_top_k`/`bm25_top_k`, gleiche Defaults 20/20), sodass der fusionierte Pool begrenzt bleibt, obwohl die indizierten Dokumente hier der volle Pattern-Text und nicht Slugs sind.
- **(f)** Acht Qualitätsachsen (§2.1) — der Schwesterserver nutzt sechs.
- **(g)** Parametername `top_k_candidates` (§14) — beim Schwesterserver `top_k_patterns` (gleicher Default 5).
- **(h)** Default 5.0 für fehlende Achse (§7.2) — der Schwesterserver zählt 0.0; hier ist eine fehlende Achse *ungemessen*, kein Worst Case.
- **(i)** `max_tries` 2 (§14) — konsistent mit dem Config-Default des Schwesterservers.

---

## 4. Stage A — Normalize & Extract (`DesignBrief`) (§4)

**Eingabe:** validierte Beschreibung + Hints. **Ausgabe:** `DesignBrief` mit `restated_goal` (≤ 2 Sätze), `architecture_style` (Enum oder null), `constraints[]` (Verbatim-Zitate), `components[]` (Name, Typ, Responsibilities, Technologien), `flows[]` (source→target, kind), `negative_signals[]` (explizite „must not / no X“-Aussagen), `language` (die effektive Sprache — immer genau ein Wert), `paradigm` (der validierte Paradigma-Hint oder null — speist die Variantenselektion, §8.4).

**Entscheidungsregeln:**

1. Ein einziger strukturierter LLM-Call erzeugt Brief **und** Teile (Stage B). Validiert die Antwort nicht gegen das Brief+Parts-Schema, wird einmal wiederholt; danach fällt der Server auf einen **Minimal-Brief mit einem einzigen Teil** über der ganzen Beschreibung zurück (Scope `module`, Kategorie-Hypothese null). Dieser Fallback-Einzelteil umgeht Stage B und den Anti-Scope-Guard vollständig und speist Stage C direkt.
2. Der System-Prompt enthält die **aus `docs/pattern-schema.json` abgeleiteten** Kategorie- und Scope-Enums (nie handgepflegte Listen — genau daran hatte der Schwesterserver Drift-Bugs) und instruiert: *nur Design-Level-Entscheidungen; Architekturstile, Deployment-Topologie und Datenplattform-Wahl sind out of scope und dürfen keine Teile werden*.
3. `negative_signals` werden verbatim extrahiert (z. B. „no reflection“, „must stay single-threaded“, „avoid singletons“) und dem Stage-C-Veto zugeführt.
4. Benennt der Brief nur Architektur-Konstrukte (z. B. nur „use microservices“), garantiert die Pipeline trotzdem ≥ 1 implementierbaren Teil: die Konstruktion der beschriebenen Komponente selbst wird ein Teil in `construction-composition` (siehe E9, §12).
5. **Stil-Präzedenz (deterministisch).** Lieferte der Caller einen erkannten `architecture_style`-Hint (§1.2), ist `DesignBrief.architecture_style` der Hint — die eigene Extraktion des Briefs wird verworfen. Andernfalls gilt der aus dem Text extrahierte Stil oder null. Die Auflösung ist reine Fallback-Arithmetik (`hint or extracted`), nie ein Merge.

---

## 5. Stage B — Zerlegung in Entscheidungsteile (§5)

**Eingabe:** `DesignBrief`. **Ausgabe:** 1–`max_parts` (Default 8) `DecisionPart`s, jeweils: `id` (kebab-case), `statement` (selbstständiges Ein-Satz-Entscheidungsproblem), `component_refs[]` (Brief-Komponentennamen), `scope` (einer der sieben Enum-Werte), `category_hypothesis` (eine der neun Kategorien oder null), `constraints[]`, `priority` (0–1; genau ein Teil darf 1.0 sein), `negative_signals[]` (komponentenscoped vom Brief geerbt).

**Entscheidungsregeln:**

1. **Ein Teil = eine Primärpattern-Entscheidung.** Ein Teil entsteht pro distinkter Entscheidungsfläche aus dem Brief (Responsibility-Cluster, Boundary-Obligation, Workflow-Obligation, Runtime-Constraint) — nicht pro Komponente, nicht pro Satz.
2. **Grenzen.** < 1 oder > `max_parts` Teile → deterministische Reparatur: die niedrigstpriorisierten Teile, die sich `category_hypothesis` **und** `scope` teilen, werden zusammengeführt (Statements mit `"; "` konkateniert), bis die Grenze eingehalten ist. **Vertragsgetriebene Teile (§10.2) sind von der Merge-Reparatur ausgenommen**; überschreitet die Zahl danach weiterhin `max_parts`, wird der strukturierte Fehler `ERR_TOO_MANY_BOUNDARY_PARTS` mit Auflistung der Obligations emittiert — niemals stilles Kollabieren.
3. **Scope-Validierung.** `scope` nicht im Enum → fällt auf `module` zurück. `category_hypothesis` nicht im Enum → null.
4. **Priorität.** Genau ein Teil wird auf Priorität 1.0 normalisiert (die Entscheidung, die der Agent am dringendsten beantwortet braucht) — das LLM schlägt vor, die deterministische Prüfung normalisiert: Ist kein Teil auf 1.0, wird die höchste Rohpriorität 1.0; stehen mehrere auf 1.0, behält der erste in Pipeline-Reihenfolge die 1.0, die übrigen werden auf den nächsthöchsten distinkten Rohwert herabgestuft.
5. **Anti-Scope-Guard.** Teile, deren Statement nur eine Architekturwahl wiederholt (etwas von der Out-of-Scope-Liste des Katalogs: layered, hexagonal, clean, onion, microkernel, Microservices vs. Monolith, API Gateway, Service Discovery, CQRS, Event Sourcing, Saga, Transactional Outbox, Strangler Fig, Sharding, Replikation, Change-Data-Capture, Data Engineering, Autoscaling, Fleet-Resilienz), werden **verworfen** — es sei denn, der Katalog hostet explizit eine lokal implementierbare Teilmenge davon (Anti-Corruption Layer, Presentation Model, Poison Pill, Backpressure, Version Number); dann bleibt der Teil mit dieser Kategorie-Hypothese erhalten. Die Out-of-Scope-Menge wird aus dem geparsten Background-Abschnitt von `docs/design-patterns.md` („Out of scope by design“) abgeleitet, nicht als Literal gepflegt; „lokal katalogisiert“ entscheidet sich daran, ob ein geparster `DesignPattern`-Record existiert, nie an der Prosa-Beispielliste.

---

## 6. Stage C — Hybrides Retrieval pro Teil (§6)

Der Guide spiegelt hier die Retrieval-Mechanik des Architekturservers exakt, tauscht aber Korpus und Queries gegen Design-Pattern-Daten.

**Eingabe:** ein `DecisionPart` + der vorbereitete Index. **Ausgabe:** rangierte `(DesignPattern, retrieval_score)`-Paare, nach Scoring (Stage D) auf `top_k_candidates` (Default 5) begrenzt, plus der Rohpool für die Normalisierung.

**Algorithmus:**

1. **Query-Bau.** Dense-Leg-Query: `statement` + `constraints` + Scope-Phrase (z. B. „task-lifetime“). BM25-Leg-Query: derselbe Text normalisiert (lowercase, Bindestriche → Leerzeichen). Unterschiedliche Repräsentation pro Leg ist Absicht.
2. **Zwei Beine.** Dense-Leg (Embedder) und BM25-Leg über demselben Knotensatz; per-Leg-Top-k = `dense_top_k` 20 / `bm25_top_k` 20 (§14), auf die Korpusgröße geklemmt, `0` = voller Korpus (verlustfrei). Die fusionierte Union umfasst damit höchstens `dense_top_k + bm25_top_k` Namen und begrenzt so den Cross-Encoder-Pool aus Schritt 4; die Caps müssen `dense_top_k + bm25_top_k` ≤ Reranker-`max_batch_size` (48) erfüllen, damit ein Teil nie mehr als eine Cross-Encoder-Anfrage umfasst.
3. **Fusion.** Relative-Score-Fusion mit `dense_weight` 0.7 / `bm25_weight` 0.3 (müssen zu 1.0 ± 1e-3 summieren; validiert).
4. **Cross-Encoder-Rerank.** Pflicht, sobald der fusionierte Pool > 1 Eintrag hat: Score `(Query = Teil-Statement, Dokument = Pattern-Dokumenttext)`; Überlebende auf `rerank_top_n` 10 begrenzt über den Vespa-artigen Reciprocal-Rank-Blend. `rerank_text` (Default `full`) wählt den Dokumenttext: `capsule` = Name + Intent + Problem + die ersten zwei Use Cases — rund ein Drittel der Zeichen und gemessen 3,6–4,3x schneller auf dem CPU-Sidecar (2026-09-25), aber die Rankings sind NICHT äquivalent (Top-1 weicht ab; Top-5-Schnitt 2/5, Top-10 7/10, Top-20 18/20 bei einer 40-Dokument-Liveprobe), daher ist `capsule` Opt-in und erfordert eine erneute Validierung der Auswahlqualität, bevor man sich darauf verlässt

   $$\text{blend}(r_{\text{fused}}, r_{\text{ce}}) = \frac{1}{60 + r_{\text{fused}} - 1} + \frac{1}{60 + r_{\text{ce}} - 1}, \qquad r \ge 1, \qquad \max = \tfrac{2}{60} \approx 0.0333$$

   (numerisch identisch zum `reciprocal_rank_score` des Schwesterservers). Der Pool, den er scored, ist die fusionierte Union aus Schritt 2 (≤ `dense_top_k + bm25_top_k`), nie der ganze Katalog. Der Blend läuft **immer** (er schützt konsensgestützte Patterns vor Cross-Encoder-Ausreißern); er ist nur dann ordnungsäquivalent zu CE-only, wenn fused- und Cross-Encoder-Ranking in der relativen Ordnung übereinstimmen — bei Abweichung gewinnt der Fusionsrang.

5. **Deterministische Vorfilter (auf dem Überlebens-Pool, vor dem Scoring):**
   - **Scope-Veto:** Patterns, deren `unsuitable_scopes` den `part.scope` enthält, werden verworfen.
   - **Negativsignal-Veto:** Patterns, deren kanonischer `name` oder ein aufgelöster `alias` token-mäßig mit einem `negative_signal` überlappt (Signal „avoid singletons“ vetoed `singleton` und den Alias „Borg“). Es zählen nur kanonischer Name plus aufgelöste Aliase (Alias-Kontrakt §2.1).
   - **Sprach-Diskouragierungs-Veto:** Patterns, deren `language_implementations`-Eintrag für die **effektive Sprache** (`design_brief.language`) `support = "discouraged"` trägt, werden verworfen — im Recognized Mode der konkrete Spracheintrag, im Generic Mode der `generic`-Eintrag. Ein Pattern **ohne** Eintrag für die effektive Sprache wird *nicht* vetoed (Abwesenheit ist keine Diskouragierung). Die REQUIRED `alternative` des verworfenen Patterns wird im Veto-Ledger des Teils festgehalten (für `alternatives_considered` in Stage G).
6. **Relevanz-Floor.** Ist der Überlebens-Pool leer **oder** der beste Blend < `min_fusion_score` (Default 0.0, Validator deckelt auf 2/60) → Kategorie-Fallback (§12) und der Teil wird als `is_fallback` markiert.
7. **Alias-Dedupe.** Aliase auf den kanonischen `name` auflösen; ein Eintrag pro Pattern. Ohne Aliasdaten matcht nur der kanonische Name.

---

## 7. Stage D — Anforderungsbewusstes Scoring pro Teil (§7)

**Eingabe:** Kandidatenpool mit `retrieval_score` (dem RR-Blend), der Teil, die effektive Sprache, der Architekturstil. **Ausgabe:** jeder Kandidat annotiert mit `analysis_score`, `fusion_score_normalized`, `fit_score` und einem Adjustment-Ledger.

### 7.1 Gewichtsextraktion (ein kleiner LLM-Call für alle Teile)

Der Call erhält **nur** die Id, das Statement und die Constraints jedes Teils plus die Brief-Constraints (keine Pattern-Daten — billig und unvoreingenommen) und liefert eine Map von Entscheidungsteil-Id auf acht Floats in [0, 1], einen pro Design-Achse: `testability, extensibility, maintainability, simplicity, performance, reliability, security, scalability`. Jede gelieferte Teil-Id muss genau einmal vorkommen und keine andere; eine fehlende oder zusätzliche Id ist ein Validierungsfehler, den der Repair-Retry sieht. Das Batching koppelt die Bewertungen nicht — jeder Teil wird aus seinem eigenen Statement und seinen eigenen Constraints bewertet — und ersetzt N Provider-Roundtrips durch einen.

**Kalibrierungsanker (im Prompt, analog zum Schwesterserver):**

| Anker | Bedeutung | Beispiele für Design-Level-Evidenz |
|---|---|---|
| 0.0 | explizit ausgeschlossen / anti-required | „no test harness“, „must not add layers“ |
| 0.1–0.2 | nicht erwähnt (nie alles null) | — |
| 0.3–0.5 | beiläufige/generische Erwähnung | „should be clean“, „some tests“ |
| 0.6–0.8 | klare Priorität mit konkretem Ziel | „p99 < 50 ms on the enrich step“, „every handler unit-testable“ |
| 0.9–1.0 | nicht verhandelbar | „zero duplicate writes“, „PCI-DSS validation on every field“ |

**Harte Regeln:** Werte in [0, 1]; Operationsreihenfolge propose → cap → max-normalize → smooth → score; 1.0 für höchstens ein bis zwei Achsen pro Teil reservieren, notfalls Peak bei 0.85–0.9 — danach so normalisieren, dass das Maximum 1.0 ist (Gleichstände dürfen alle 1.0 erreichen); jede Gewichtung muss auf einer tatsächlich vorhandenen Phrase beruhen; numerische SLOs schlagen vage Adjektive; sparse/ambige Teile bekommen 0.1–0.2 auf unerwähnten Achsen.

**Deterministische Nachverarbeitung (pro Teil, nach dem Batch):** Teile, deren Vektor all-zero zurückkam, werden in genau einem zweiten Batch erneut angefragt, der nur diese Teile enthält; bleibt ein Vektor all-zero, greift das ungewichtete Mittel (0.125 je Achse, E4). Konvexe Glättung: $w'_k = \alpha\,w_k + (1-\alpha)/8$ mit `weight_smoothing_alpha` $\alpha = 0.7$. Trägt der Teil keine Prioritäts-Evidenz **und** hat der Agent Architektur-`quality_attributes` (numerische Map) weitergegeben, werden diese (max-normalisiert, also ohne Bereichsannahme) vor dem Smoothing mit **0.7 extrahiert / 0.3 architektonischer Prior** gemischt.

### 7.2 Deterministische Scores

1. **`analysis_score`** (0–100): $10 \cdot \dfrac{\sum_k w'_k \cdot qa_k(\text{pattern})}{\sum_k w'_k}$ über die acht festen Achsen, mit den `quality_attributes` des Patterns; eine fehlende Achse nimmt den neutralen Mid-Scale-Default **5.0**. Das weicht bewusst vom Schwesterserver ab (dort 0.0): Eine fehlende Achse ist *ungemessen*, und 0.0 würde Record-Unvollständigkeit als Qualitätsurteil tarnen.
2. **`fusion_score_normalized`** (0–100): Min–Max-Normalisierung des `retrieval_score` über den Kandidatenpool des Teils.
3. **`blended`**: `analysis_blend_weight` · `analysis_score` + `fusion_blend_weight` · `fusion_score_normalized` mit Defaults 0.7 / 0.3 (Summe 1.0 ± 1e-3; validiert).

### 7.3 Alignment-Adjustments (deterministisch auf `blended`, Ergebnis auf [0, 100] geklemmt)

| Adjustment | Bedingung | Delta |
|---|---|---|
| Scope-Bonus | `part.scope ∈ pattern.suitable_scopes` | **+3** |
| Kategorie-Hypothese-Treffer | `pattern.category == part.category_hypothesis` (nicht null) | **+5** |
| Kategorie-Hypothese-Mismatch | `pattern.category ≠ part.category_hypothesis` (nicht null) | **−10** |
| Style-Affinität | `pattern.category` in der Affinitätsmenge des Architekturstils (§10.1) | **+5** |
| Sprachsupport (Recognized Mode, nur wenn der Record einen Eintrag für die angefragte Sprache hat) | `native` / `idiomatic` / `library` | **+2** |
| | `manual` | **−2** |
| | `discouraged` | bereits in Stage C vetoed — erreicht das Scoring nie |
| Sprachsupport (Generic Mode oder kein Eintrag für die angefragte Sprache) | — | kein Adjustment |

Stacking ist erlaubt (z. B. Kategorie-Treffer + Style-Affinität + Scope = +13); die Klemmung begrenzt den Effekt. Jedes angewendete Delta wird im **Adjustment-Ledger** festgehalten (für den `overview.reasoning`-Trace und für Tests).

---

## 8. Stage E — Ranking, Schwellen, Selektion (§8)

**Eingabe:** annotierte Kandidaten pro Teil. **Ausgabe pro Teil:** ein Primärpattern (oder Fallback-/Ambiguitätsausgang), 0–`max_supporting` Supporting-Patterns, Saatgut für `alternatives_considered`.

### 8.1 Ranking und Tie-Breaking (deterministische Totalordnung)

Sortierung: `fit_score` ↓ → `analysis_score` ↓ → `fusion_score_normalized` ↓ → **Katalogordnung** (Kategorierang 1–9, dann Subgroup-Reihenfolge, dann Zeilenreihenfolge — die Katalogdatei definiert eine stabile kanonische Ordnung) → `name` alphabetisch. Die Kette garantiert eine eindeutige, reproduzierbare Rangfolge selbst bei exakten Score-Gleichständen.

### 8.2 Selektionsschwellen

| Schwelle | Default | Regel |
|---|---|---|
| `primary_score_threshold` | 50 | Der Top-Kandidat wird nur Primär, wenn `fit_score` ≥ 50 **und** `analysis_score` ≥ 50 (Dual Gate); sonst Kategorie-Fallback mit `is_fallback = true`. Das Dual Gate **erweitert** den Schwesterserver (der nur über `analysis_score` gated); das `fit`-Floor hat dort kein Analogon. |
| `supporting_score_threshold` | 40 | Ein Nicht-Primär-Kandidat ist erst ab `fit_score` ≥ 40 als Supporting geeignet. |
| `near_tie_margin` | 5 | Jeder Kandidat innerhalb der Marge des Leaders wird geflaggt: Der Teil gilt als `ambiguous`, und jeder solche Kandidat **MUSS** mit Delta in `alternatives_considered` der gewinnenden Application erscheinen; nur die Top 2 konkurrieren um Primär. |
| `max_supporting` | 3 | harte Obergrenze pro Teil |
| `top_k_candidates` | 5 | pro Teil in Stage F getragen (Selektion passiert **nach** dem Scoring, nie davor — gleiche Reihenfolge wie beim Schwesterserver) |

### 8.3 Supporting-Pattern-Regeln

Ein Kandidat tritt dem Supporting-Set des Primärs bei, **nur wenn alle** Bedingungen gelten:

1. `fit_score ≥ supporting_score_threshold` und er ist nicht der Primär (in diesem Teil oder, nach Stage-F-Dedupe, in irgendeinem Teil).
2. **Distinct-Constraint-Regel:** Er löst eine Constraint, die der Primär nachweislich nicht löst — belegt entweder durch (a) eine Kante aus `related_patterns` des Primärs **plus** eine nicht-überlappende Katalog-Subgroup, oder (b) ein explizites Statement unter den `constraints` des Teils. Ein Kandidat, dessen einzige Begründung das Problem des Primärs wiederholt, wird abgelehnt (Katalog-Nutzungsregel 2: „supporting patterns only where they solve distinct constraints“).
3. **Kein bekannter Konflikt** mit dem Primär (Konflikttabelle neben dem Katalog; z. B. `singleton` vs. `dependency-injection`, `service-locator` vs. `dependency-injection`, `template-method` vs. `strategy` am selben Variationspunkt, `mock` vs. `stub` am selben Test-Seam). Konfliktpaare dürfen nie beide für denselben Teil selektiert werden.
4. **Same-Subgroup-Limit:** Zwei Patterns aus derselben Katalog-Subgroup dürfen nur unter Regel 2(b) beide Supporting sein — je eine explizite, distinkte, genannte Constraint.
5. Nicht discouraged in der effektiven Sprache (bereits upstream vetoed, §6 Schritt 5); ein Pattern ohne Eintrag für diese Sprache besteht die Prüfung.

Jeder `role`-String eines Supporting-Patterns (vom `SupportingPattern`-Schema im Result verlangt) benennt die distinkte Constraint und den Teil, z. B. „bound hung geo-lookups (enrichment)“.

### 8.4 Sprachbehandlung

Eine Sprache pro Lauf: die effektive Sprache (`design_brief.language`, §1.3/§4) — die eingereichte Sprache im Recognized Mode, `generic` im Generic Mode.

**Konsequenzen zur Selektionszeit:**

- **Diskouragierungs-Veto:** angewendet in Stage C gegen den Eintrag der effektiven Sprache (konkreter Spracheintrag im Recognized Mode, `generic`-Eintrag im Generic Mode). Vetoed Patterns erreichen das Scoring nie; ihre REQUIRED `alternative` gelangt über das Veto-Ledger in `alternatives_considered` (§6 Schritt 5, §12).
- **Sprach-Adjustment:** das +2/−2-Support-Adjustment (§7.3) gilt nur im Recognized Mode und nur, wenn der Record einen Eintrag für die angefragte Sprache trägt; ein Generic-Mode-Lookup fällt kein Supporturteil.
- **Supporting-Eligibility:** §8.3 Regel 5 liest denselben Eintrag.

**GENERATE-Zeit-Bindung (pro Application).** Der gebundene Eintrag ist der `language_implementations`-Eintrag für die effektive Sprache; ist die effektive Sprache `generic` oder hat eine erkannte Sprache keinen Record-Eintrag, ist der gebundene Eintrag der `generic`-Eintrag (immer vorhanden — §2.1-Invariante). `LanguageBinding.language` nennt den Sprachwert des gebundenen Eintrags (eine Fallback-Bindung meldet also `generic`), und `support`, `idiom_name`, `implementation_notes`, `code_sketch` und `pitfalls` kommen verbatim von dort (`code_sketch` darf null sein; `generic`-Einträge tragen sprachneutrale Guidance).

**Variantenauswahl (§8.4, GENERATE-Zeit).** Trägt der gebundene Eintrag `variants`, bindet GENERATE **genau eine** Variante nach dieser Präzedenz (erster Treffer gewinnt):

1. **Explizite Nennung** — `focus` zitiert eine `variant_id` wörtlich (z. B. `ace-reactor`, „nutze die `ace-reactor`-Variante“). Nur exakte Treffer; eine Paraphrase wie „nutze die ACE-Variante“ matcht keine Id und fällt auf die nächste Ebene durch.
2. **Paradigma-Hint** — der `paradigm`-Hint (§1.1) matcht eine Variante; teilen mehrere das Paradigma, entscheidet Ebene 3.
3. **Capability-Floor** — Standard-/Toolchain-Tokens der Beschreibung (`components[].technology_stack`: „C++20“, „Java 8“, „ES2017“ …) werden gegen `min_standard`/`requires_features` jeder Variante gematcht; unter den erfüllbaren Varianten gewinnt die höchste Floor (die Toolchain läuft sie tatsächlich).
4. **Default** — `default_variant`.

Die Felder der gewählten Variante ersetzen die flachen Felder in der `LanguageBinding` (der §2.3-Checker garantiert, dass die Default-Variante ihnen ohnehin gleicht); die Bindung setzt `bound_variant` auf die gewählte `variant_id` (maschinenlesbar, in einem Folge-`focus` zitierbar) und trägt `variants[]` — kompakte Summaries aller **nicht** gebundenen Geschwister (`variant_id`, `name`, `paradigm`, `min_standard`, `requires_features`) —, sodass der Agent Floor-/Paradigma-Fit gegen den echten Codebase beurteilen und bei falscher Realisierung mit `focus` auf eine Geschwister-`variant_id` erneut aufrufen kann. Bewusst gibt es **keinen `variant_id`-Tool-Parameter**: Variant-Ids sind pro (Pattern, Sprache) unbegrenzt und churning — Capability-/Style-Signale plus diese Präzedenz halten das Input-Schema klein und stabil (analog CMake `CXX_STANDARD`/`cxx_std_17`, Cargo `edition`, tsconfig `target`: Nutzer nennt die Floor, das Tool resolved die Implementierung). Support, Veto und Adjustment (§7.3) bleiben per-Sprache — Support ist eine Eigenschaft der Sprache, nicht einer Variante.

`LanguageBinding.alternative` wird nie befüllt: Das Stage-C-Veto garantiert, dass keine Bindung `support = "discouraged"` trägt.

**Request-Echo.** `DesignResult.overview.languages` (Array im Result-Schema) enthält genau ein Element: die effektive Sprache.

**Gelöschte Multi-Language-Semantik.** Weil genau eine Sprache angefragt wird, haben die frühere partielle Diskouragierungsregel, das Worst-Value-Support-Adjustment und die Best-Support-Bindung keine Bedeutung mehr; jede Frage, die sie beantworteten, beantwortet jetzt der einzelne Effective-Language-Lookup.

---

## 9. Stage F — Komposition zum Selektionsergebnis (§9)

**Eingabe:** Selektionen pro Teil mit Scores und Ledgern. **Ausgabe:** der von GENERATE konsumierte Kandidatensatz, organisiert für `DesignResult` (`docs/result-schema.json`).

**Entscheidungsregeln:**

1. **Dominant-Part-Wahl (deterministisch).** $\text{dominance} = \text{part.priority} \times \bigl(\text{fit\_score}(\text{primary})/100\bigr)$. Der Teil mit der höchsten Dominanz definiert `DesignOverview`: sein Primär wird `overview.primary_pattern`, seine Kategorie `overview.category`. Tie-Breaks: `fit_score`, dann `analysis_score`, dann Teilreihenfolge.
2. **`overview.supporting_patterns`** = Vereinigung aller Supporting-Sets der Teile (dedupliziert auf den kanonischen Namen, insgesamt auf `max_supporting` begrenzt nach `fit_score` ↓), jeweils mit `role`-String.
3. **Applications.** Jedes selektierte (Teil, Pattern)-Paar — Primärs sekundärer Teile eingeschlossen — wird ein `PatternApplication`-Seed: `id` = `{part-id}-{pattern-name}` (kebab-case, `^[a-z][a-z0-9_-]*$`), `pattern`, `category`, `name` (menschenlesbar, aus den Component-Refs des Teils), `problem` (aus Teil-Statement + Constraints).
4. **Dedupe.** Dasselbe Pattern in zwei Teilen ergibt zwei Applications (verschiedene Probleme), erscheint aber nur einmal in `overview.supporting_patterns`. Ein Pattern, das in einem Teil Primär und in einem anderen Supporting ist, behält seine Primärapplikation; die Supporting-Referenz löst auf deren Application-Id auf.
5. **Interactions-Seed.** Brief-`flows[]` zwischen Komponenten, die von verschiedenen Applications berührt werden, säen `DesignResult.interactions[]` (`source`/`target` = Application-Ids) zur Verfeinerung durch Stage G.
6. **Traceability.** `overview.reasoning` wird deterministisch aus den Ledgern zusammengesetzt: Anzahl der Teile, das `statement` jedes Teils (die gesampfte Zerlegungsfrage, wortgetreu), Top-3-Retrieval-Blends pro Teil, Gewichts-Zusammenfassung plus des vollständigen acht-Achsen-Gewichtsvektors (§7.2-Achsenreihenfolge), angewendete Adjustment-Deltas, Schwellenentscheidungen, Fallback-/Ambiguitätsflags — sodass ein Aufrufer Lauf-zu-Lauf-Variation ohne Server-Log zuordnen kann. **Keine** LLM-Narration nötig.

---

## 10. Zusammenspiel mit dem Architekturkontext (§10)

### 10.1 Style→Kategorie-Affinität (Soft Prior, +5 über §7.3)

| Architekturstil (`architecture_style`-Hint / Brief) | Affinitätskategorien |
|---|---|
| `event-driven`, `broker`, `enterprise-service-bus` | workflow-messaging, interaction-boundary |
| `event-sourcing` | persistence-state |
| `command-query-responsibility-segregation` | persistence-state, domain-modeling |
| `saga` | workflow-messaging |
| `microservices`, `service-oriented-architecture` | interaction-boundary, workflow-messaging, concurrency-resilience |
| `hexagonal` | domain-modeling, behavior-selection, interaction-boundary |
| `layered-monolith`, `modular-monolith`, `monolithic` | domain-modeling, construction-composition |
| `pipe-and-filter` | workflow-messaging, behavior-selection |
| `model-view-controller`, `presentation-abstraction-control` | specialized-components |
| `microkernel-plugin`, `reflection-architecture` | behavior-selection, construction-composition |
| `reactive-architecture`, `actor-based`, `half-sync-half-async`, `task-control-architecture` | concurrency-resilience, workflow-messaging |
| `serverless`, `edge-computing` | concurrency-resilience, interaction-boundary |
| `space-based` | concurrency-resilience, persistence-state |
| `data-mesh`, `lambda-architecture`, `kappa-architecture` | interaction-boundary, persistence-state |
| `api-gateway`, `backend-for-frontend`, `service-mesh` | interaction-boundary |
| `rule-based-system`, `aiml-centric` | behavior-selection, domain-modeling |
| `blockchain-based` | persistence-state, security-verification |
| `hybrid-cloud`, `multi-cloud` | interaction-boundary, security-verification |
| (jeder nicht gelistete Stil) | kein Bonus |

Die Tabelle ist ein **Prior, nie ein Filter**: Eine `event-driven`-Beschreibung, deren Teil klar Test-Isolation betrifft, selektiert trotzdem nach Merit aus `security-verification`/`construction-composition`.

### 10.2 Weitere Kontextinteraktionen

- **Components → Teile:** jedes Responsibility-Cluster, in das GENERATE Code platzieren muss, wird ein Teil; `maps_to`-Felder der Participants müssen Entitäten benennen, die aus den Komponentennamen des Briefs ableitbar sind, damit der Agent sie lokalisieren kann.
- **Relationships → Interactions:** Sync-/Async-/Event-Stream-Kanten werden `DesignResult.interactions[]` zwischen den Applications, die ihre Endpunkte implementieren; die Selektion seedet sie nur, GENERATE verfeinert die Typen (`notifies`, `delegates-to`, `invalidates`, …).
- **Contracts → Boundary-Teile:** Jede `api_contracts[]`/`event_contracts[]`-Obligation auf einer benannten Komponente erzwingt mindestens einen Teil mit `category_hypothesis` ∈ {`interaction-boundary`, `workflow-messaging`} und `scope` `boundary` oder `message-flow`. Vertragsgetriebene Teile sind von der Merge-Reparatur (Stage B Regel 2) ausgenommen; überschreitet die resultierende Teilzahl `max_parts`, wird `ERR_TOO_MANY_BOUNDARY_PARTS` mit den Obligations emittiert — niemals stilles Kollabieren.
- **Quality Attributes → Gewichte:** nur als 0.3-Prior aus §7.1 — Architektur-Level-Scores dürfen Design-Level-Evidenz aus der Beschreibung nicht überstimmen.
- **Exclusion-Disziplin:** Architektur-Level-Namen tauchen nie als selektierte Patterns auf (Stage B Regel 5); die Background-Sektion des Katalogs (strategisches DDD, Client-Dispatcher-Server, Broker) ist nicht selektierbar — diese Records tragen keine Kategorie und sind vom Index ausgeschlossen.

---

## 11. Stage G/H — Validierung der Selektion durch Generator und Evaluation (§11)

Selektion ist erst vollständig, wenn das gewählte Set Generation und Evaluation übersteht; der Loop spiegelt den `design_loop` des Schwesterservers.

1. **GENERATE** erzeugt das konkrete `DesignResult`: `solution`, `participants` (Rolle → konkrete Entität via `maps_to`), `interactions`, `implementation_steps`, pro Application eine `LanguageBinding` (gebundener Eintrag = `language_implementations`-Eintrag für die effektive Sprache, sonst der `generic`-Eintrag des Records; `language` = Sprache des gebundenen Eintrags; `support`, `idiom_name`, `implementation_notes`, `code_sketch` von dort; `bound_variant` + Sibling-`variants[]` gemäß §8.4, wenn der gebundene Eintrag Varianten trägt; kein `alternative` — das Stage-C-Veto garantiert, dass keine Bindung `discouraged` ist), `test_notes` (aus test-relevanten Benefits des Patterns und dem `testability`-Gewicht des Teils) und `alternatives_considered` (Runner-up je Near-Tie plus jeder Veto-Ledger-Eintrag im Format „considered but discouraged in {entry language}; use {alternative}“).
2. **EVALUATE** bewertet das Ergebnis 0–100 gegen die acht Design-Achsen, **gewichtet mit den Teilgewichten** (ein auf Einfachheit gewichteter Teil darf nicht für fehlende Extensibilität abgewertet werden), plus Cross-Application-Konsistenz: keine duplizierte Entscheidungsownership, kein Konfliktpaar aus §8.3 Regel 3, jeder Supporting-`role` auf eine genannte Constraint zurückführbar.
3. **REFINE-Loop:** bis `max_tries` 2 Versuche; jeder Retry bettet die Findings der vorherigen Evaluation ein; **Best-Attempt-Retention** behält das höchstbewertete valide Ergebnis (nie einfach das letzte); Early Stop, sobald `overall ≥ min_quality_score` 50. Cancellation wird zwischen Versuchen geprüft.
4. **Reselektionsregel:** Zeigen die Evaluations-Findings einen *Selektionsdefekt* (Primär besitzt die Entscheidung nicht; eine Supporting-Constraint bleibt ungelöst; ein Konfliktpaar ist durchgerutscht), betritt der Retry **Stage E** erneut mit dem betroffenen Kandidaten exkludiert und dem nächstplatzierten promoviert — eine begrenzte, geledgerte Reselektion, kein vollständiger Pipeline-Rerun. Maximal **eine** Reselektion pro Loop (nicht pro Versuch); Exklusionen sind **sticky** — ein per Reselektion ausgeschlossener Kandidat darf im selben Loop nicht wieder promoviert werden. `attempts > 1` und das Reselektions-Ledger erscheinen in den Logs für den Agenten.
5. **Präzedenz:** Best-Attempt-Retention regiert, sobald irgendein Versuch validiert; der Terminalfehler greift nur, wenn **alle** Versuche die Validierung verfehlen — dann propagiert der letzte Validierungsfehler als strukturierter Tool-Fehler (nie ein still degradiertes Ergebnis).

---

## 12. Fallbacks, Fehler und Ambiguitätsbehandlung (§12)

| # | Bedingung | Verhalten |
|---|---|---|
| E1 | Beschreibung < 40 Zeichen | `ERR_DESCRIPTION_TOO_SHORT` — kein Pipeline-Lauf |
| E2 | Control-/Format-Zeichen | Validierungsfehler beim Intake |
| E3 | Brief/Teile-LLM-Call scheitert zweimal | Minimaler Ein-Teil-Brief (ganze Beschreibung, Scope `module`, Hypothese null) — degradiert, aber funktional; umgeht Stage B und den Anti-Scope-Guard vollständig und speist Stage C direkt |
| E4 | Gewichtsextraktion in beiden Batches all-zero | ungewichtetes Mittel (0.125 je Achse) für genau die Teile, die all-zero geblieben sind |
| E5 | Retrieval-Pool leer oder unter `min_fusion_score` | Kategorie-Fallback-Pattern, `is_fallback = true`, Fallback im `overview.reasoning` vermerkt |
| E6 | Top-Kandidat scheitert am Dual-Gate 50/50 | Kategorie-Fallback-Pattern für den Teil (Semantik wie E5, pro Teil) |
| E7 | Teil in `specialized-components` erreicht E5/E6 | **kein universeller Fallback für diese Kategorie** → strukturierter Fehler `ERR_NO_PRIMARY_PATTERN` mit den Top-3-Kandidaten samt Scores und der Bitte an den Agenten, den konkreten Komponententyp zu benennen (UI, Loop, Parser) — nie eine erfundene Empfehlung |
| E8 | Near-Tie (≤ `near_tie_margin`) | Selektion läuft mit dem deterministischen Tie-Break weiter, `ambiguous`-Flag gesetzt, Runner-up forciert in `alternatives_considered` |
| E9 | Beschreibung nennt nur Architektur-Level-Entscheidungen | ein synthetischer `construction-composition`-Teil für die benannte Komponente (Stage A Regel 4) |
| E10 | unbekannter `architecture_style`/`language`/`paradigm` | ignoriert/verworfen, geloggt (Warnung bei unbekannter Sprache oder Paradigma); Style-Affinität entfällt einfach; Sprache → **Generic Mode**: das Ergebnis formuliert die Patterns sprachneutral (§8.4); `paradigm` → Variantenselektion läuft über Beschreibungssignale und `default_variant` (§8.4) |
| E11 | Cancellation angefordert | Pipeline stoppt an der nächsten Stage-Grenze; partieller Zustand wird verworfen |
| E12 | GENERATE-Output nach allen Retries invalide | letzter Validierungsfehler propagiert; kein Fallback-Ergebnis |

**Kategorie-Fallback-Patterns** (das Pattern mit der geringsten Zeremonie und universeller Anwendbarkeit je Kategorie, gewählt aus den „Use when“-Zeilen):

| Kategorie | Fallback-Primär |
|---|---|
| domain-modeling | `transaction-script` |
| construction-composition | `dependency-injection` |
| behavior-selection | `strategy` |
| interaction-boundary | `data-transfer-object` |
| workflow-messaging | `observer` |
| persistence-state | `repository` |
| concurrency-resilience | `timeout` |
| security-verification | `guard-clause` |
| specialized-components | — (stattdessen E7) |

Ein Fallback-Primär erhält **nie** Supporting-Patterns (keine Evidenzbasis, die sie rechtfertigen könnte) und trägt `is_fallback` in den Reasoning-Trace.

**Ambiguität wird sichtbar gemacht, nie versteckt:** Near-Ties (E8), Fallbacks (E5/E6/E7), Veto-Ledger-Einträge und Reselektionen sind alle in `overview.reasoning`, `alternatives_considered` und den Logs sichtbar — der Agent kann immer sehen, warum ein Pattern gewonnen hat und was fast gewonnen hätte.

---

## 13. Durchgerechnetes Beispiel (§13)

**Eingabe an diesen Server** (Beschreibung, abgeleitet aus einem Architecture-pattern-mcp-Ergebnis: Style `event-driven`, Komponenten Kafka-ingest / JSON-parser / geo-enricher / InfluxDB-sink / S3-sink, Quality Attributes scalability 9.1, maintainability 8.2, Relationship „geo-enricher → Redis (sync)“, Technologien TypeScript/Node):

> „Event-driven ETL (architecture style: event-driven). Component `geo-enricher` (TypeScript/Node worker) calls a Redis geo-lookup synchronously while enriching parsed sensor events; enrichment must stay within the stage's p99 latency budget, Redis transient failures and outages must not stall the pipeline, and duplicate deliveries from upstream retries must not produce duplicate sink writes. Also: the JSON parser must tolerate unknown fields from heterogeneous device firmware. Constraints: at-least-once delivery, p99 < 50 ms per event.“

**Stage A/B** erzeugen drei Teile:

| Teil | Statement (gekürzt) | Scope | Kategorie-Hypothese | Priorität |
|---|---|---|---|---|
| `enrichment-resilience` | den synchronen Redis-Lookup unter p99 begrenzen und überstehen | task-lifetime | concurrency-resilience | 1.0 |
| `sink-idempotency` | Duplikat-Lieferungen dürfen keine Doppel-Writes in Sinks erzeugen | message-flow | workflow-messaging | 0.8 |
| `parser-tolerance` | unbekannte Felder dürfen das Parsing nicht brechen | object | interaction-boundary | 0.7 |

**Stage C/D für `enrichment-resilience`.** Extrahierte (rohe) Gewichte: reliability 1.0, performance 0.8, simplicity 0.6, testability 0.5, maintainability 0.4, scalability 0.3, extensibility 0.2, security 0.2 → gesmootht (α = 0.7, +0.0375 je Achse): 0.7375, 0.5975, 0.4575, 0.3875, 0.3175, 0.2475, 0.1775, 0.1775 (Σ = 3.1). Kandidaten nach Rerank und Vetos (Scope `task-lifetime` vetoed keinen dieser Kandidaten; keine Negativsignale):

| Pattern | analysis_score | Retrieval-Blend | fusion_norm | blended | Adjustments | fit |
|---|---|---|---|---|---|---|
| `timeout` | 76.4 | 0.0317 | 100.0 | 0.7·76.4 + 0.3·100 = 83.5 | Scope +3, Kategorie +5, Style (`event-driven` → concurrency-resilience nicht in der Affinitätsmenge) +0, Sprache `native` +2 | **93.5** |
| `retry-with-backoff` | 72.0 | 0.0300 | 89.8 | 77.3 | +3 +5 +2 | 87.3 |
| `circuit-breaker` | 72.0 | 0.0283 | 79.6 | 74.3 | +3 +5 +2 | 84.3 |
| `bulkhead` | 68.0 | 0.0260 | 65.9 | 67.4 | +3 +5 +2 | 77.4 |
| `guarded-suspension` | 61.0 | 0.0200 | 29.9 | 51.7 | +3 +5 +0 (Style) −2 (`manual`) | 57.7 |

*(analysis_score-Beispiel `timeout`: Σ w′·qa = 23.70 über seine acht Achsenwerte; 10 × 23.70 / 3.1 = 76.4.)*

**Stage E.** Primär: `timeout` (fit 93.5 ≥ 50, analysis 76.4 ≥ 50 — Dual Gate bestanden). Near-Tie-Check: 93.5 − 87.3 = 6.2 > 5 → nicht ambig. Supporting (je ≥ 40, je mit distinkter genannter Constraint, keine Konflikte, ≤ 3): `retry-with-backoff` — Rolle „absorb transient Redis failures without caller awareness (enrichment)“; `circuit-breaker` — Rolle „stop calling Redis during an outage (enrichment)“. `bulkhead` abgelehnt: keine genannte Kapazitätsisolations-Constraint für diesen Teil (Distinct-Constraint-Regel scheitert); `cache-aside` (fit nach Adjustments unter der Schwelle) wird ein `alternatives_considered`-Seed. Dominanz 1.0 × 0.935 wählt diesen Teil als dominant → `overview.primary_pattern = timeout`, `overview.category = concurrency-resilience`.

Die anderen beiden Teile wählen `idempotent-receiver` (Primär, workflow-messaging) und `tolerant-reader` (Primär, interaction-boundary) mit derselben Arithmetik; sie gehen als `PatternApplication`s ein (`sink-idempotency-idempotent-receiver`, `parser-tolerance-tolerant-reader`) und — wo sie die Constraints der dominanten Entscheidung stützen — als `overview.supporting_patterns`-Einträge. GENERATE bindet dann jede Application an die TypeScript-Einträge der `language_implementations` der gewählten Patterns (z. B. `timeout` → Idiom „`AbortSignal`/`Promise.race` timeout wrapper“, Support `native`), mappt Participants auf konkrete Entitäten (`GeoEnrichmentClient`, `RedisGeoGateway`), seedet Interactions aus den Architektur-Relationships, und EVALUATE schließt den Loop; ein hypothetisches Evaluations-Finding „duplicate-write-Constraint durch das Supporting-Set von `timeout` ungelöst“ würde im nächsten Versuch die Stage-E-Reselektionsregel auslösen und dort nur für diese Constraint nächstplatzierte Kandidaten promovieren.

---

## 14. Konfigurationsparameter (Defaults) (§14)

| Parameter | Default | Zwang / Hinweis |
|---|---|---|
| `dense_weight` / `bm25_weight` | 0.7 / 0.3 | Summe 1.0 ± 1e-3 (validiert); < 0.05 auf einem Leg → Warnung |
| `rerank_top_n` | 10 | Überlebende nach dem Cross-Encoder-Schnitt |
| `reranker.rerank_text` | `full` | Dokumenttext für den Cross-Encoder (§6 Schritt 4): `full` = §2.2-Dokumenttext; `capsule` = Name + Intent + Problem + die ersten zwei Use Cases (~3x weniger Zeichen, gemessen 3,6–4,3x schneller auf dem CPU-Sidecar). Rankings sind NICHT äquivalent (gemessene Divergenz, §6 Schritt 4) — nur Opt-in |
| `rrf_k` | 60 | RR-Blend-Konstante; Blend-Maximum 2/60 ≈ 0.0333 |
| `min_fusion_score` | 0.0 | ≤ 2/60 (Validator- und Startup-Check) |
| `top_k_candidates` | 5 | pro Teil, angewendet nach dem Scoring |
| `dense_top_k` / `bm25_top_k` | 20 / 20 | Stage-1-Recall-Cap pro Leg (§6 Schritt 2); `0` = voller Korpus. Gemessen 2026-09-24 auf dem 438-Dokumente-Katalog: verlustfreier Recall speiste 438 Dokumente (10 Cross-Encoder-Chunks à ≤48, je ~58 s) in jeden Entscheidungsteil ≈ 9,8 min/Teil; 20/20 hält den fusionierten Pool ≤ 40 = ein Chunk |
| `retrieval.max_concurrent_parts` | 1 | Obergrenze für Teile, deren Fuse→Rerank→Blend-Schwanz in `retrieve_many` concurrently läuft (die gemeinsame Embed+FAISS-Stufe ist ein einziger Batch-Aufruf und nicht durch diesen Wert begrenzt). Semantikneutral: Die Gather-Argumentreihenfolge erhält die Ergebnisreihenfolge je Teil. A/B-gemessen 2026-09-25 auf dem CPU-Sidecar mit 8 Volltext-Teile-Queries (§14-Latenznotizen): Ein voller Kappe-1-`retrieve_many`-Durchlauf dauert 242,4 s (ungetimter Warm-up-Durchlauf 250,4 s), und der Kappe-2-Lauf schlägt komplett fehl — bereits eine Volltext-≤40-Dokumente-Anfrage sättigt den 16.384-Token-Inferenzpuffer des Sidecars, eine zweite gleichzeitige Anfrage erhält kein Inferenz-Permit, und TEI antwortet mit HTTP 429 "Model is overloaded" (TEI erzwingt `max_batch_requests=8`; der Client hat bewusst keine Retries), was die gesamte Anfrage scheitern lässt. Default daher 1; Kappe 2 ist Opt-in und nur sicher mit `capsule`-Rerank-Text oder schnellerem Sidecar |
| `weight_smoothing_alpha` | 0.7 | konvexe Glättung der extrahierten Gewichte |
| `analysis_blend_weight` / `fusion_blend_weight` | 0.7 / 0.3 | Summe 1.0 ± 1e-3 (validiert) |
| `scope_bonus` / `category_match` / `category_mismatch` / `style_affinity` | +3 / +5 / −10 / +5 | fit nach Adjustments auf [0, 100] geklemmt |
| `language_support_bonus` / `language_manual_penalty` | +2 / −2 | nur Recognized Mode und nur mit Eintrag für die angefragte Sprache; Generic Mode → kein Sprach-Adjustment; `discouraged` → Stage-C-Veto |
| `primary_score_threshold` | 50 | Dual Gate: fit ≥ 50 **und** analysis ≥ 50 |
| `supporting_score_threshold` | 40 | pro Supporting-Kandidat |
| `near_tie_margin` | 5 | Ambiguitätsflag + erzwungener `alternatives_considered`-Eintrag |
| `max_supporting` | 3 | pro Teil |
| `max_parts` | 8 | Zerlegungsgrenze mit Merge-Reparatur |
| `max_tries` | 2 | Generate–Evaluate-Versuche (Best-Attempt wird behalten) |
| `min_quality_score` | 50 | Early-Stop-Schwelle, 0–100; 100 deaktiviert |
| `generator.config.timeout_seconds` | 120.0 | Provider-Grenze für jeden Generator-Call **und** die Intent-Level-Deadline darum herum: ein Call, der sie überschreitet, endet mit `ERR_009` und dem dokumentierten E3/E4/E12-Fallback. Transport-Retries sind fest auf eine initiale Anfrage gesetzt (`max_retries=1`); `validation.max_retries` steuert den getrennten Structured-Output-Repair-Pfad |
| `generator.config.generation_batch_size` | 5 | Anwendungen pro GENERATE-Provider-Call (§11 Punkt 1). `seed.applications` werden in Batches dieser Größe geteilt, jeder mit genau den Katalog-Slices, die seine eigenen Anwendungen referenzieren; die konkatenierte `RawGeneration` wird gegen den vollständigen Seed assembliert (Batch-Reihenfolge erhalten). Ein Batch-Fehler propagiert unverändert in den Design-Loop/E12-Pfad (geschwisterliche laufende Batches werden abgebrochen) |
| `generator.config.generation_max_concurrency` | 2 | Obergrenze gleichzeitig laufender GENERATE-Batches; Ergebnisse werden in Batch-Reihenfolge konkateniert, die assemblierte `RawGeneration` ist identisch zur sequentiellen Schleife. `1` = sequentielles Verhalten vor dem 2026-09-25 |
| `generator.config.circuit_breaker_failure_threshold` | 3 | aufeinanderfolgende qualifizierende Transportfehler (Timeout, Verbindungsfehler, HTTP 408/429/5xx, leere Antwort), die den prozesslokalen Circuit dieses Agents öffnen. Authentifizierungs-/Konfigurations-4xx zählen nie |
| `generator.config.circuit_breaker_cooldown_seconds` | 60.0 | Ablehnungsfenster im offenen Zustand, bevor genau eine Half-Open-Probe zugelassen wird; Erfolg schließt und nullt den Zähler, ein Fehlschlag der Probe öffnet erneut |
| `tasks.deadline_seconds` | 1800.0 | vollständiges Ausführungsbudget eines Tool-Calls bzw. Job-Laufs, unabhängig von `generator.config.timeout_seconds`. Beim Ablauf wird die laufende Provider-Koroutine abgebrochen und der Lauf endet mit `ERR_014` (`select_design_patterns`, `analyze`, `generate`, `evaluate`, `submit_design_job`). Umsetzung: die Pipeline ist ein llama-index-`Workflow` mit internem Timeout `deadline_seconds * 0.9`; die Wrapper übersetzen das resultierende `WorkflowTimeoutError` in `TimeoutError`, das `run_with_deadline` als `ERR_014` markiert. Für legitim längere Läufe explizit erhöhen; nie entfernen |
| `reasoning.enabled` | `true` | Hauptschalter der Pre-LLM-Reasoning-Schicht; `false` degradiert jede Phase auf das In-Prompt-Thinking-Scaffold |
| `reasoning.per_phase.<phase>.enabled` | `true` (analyze, weights, generate, evaluate) | Schalter je Phase; eine deaktivierte Phase liefert einen leeren Reasoning-Kontext — kein Generator-Call, kein Reasoning-Subprozess, kein Cache-Eintrag. Overrides werden auf die ausgelieferte Agenda/Directive gemergt |
| `reasoning.per_phase.<phase>.pre_llm_thoughts` | 2 / 2 / 2 / 2 (analyze, weights, generate, evaluate) | Reasoning-Schritte vor dem Haupt-LLM-Call der Phase (2026-09-25 von 4 / 3 / 3 / 3 reduziert — siehe §14-Latenznotizen) |
| `reasoning.per_phase.<phase>.tools` | `["shannon", "code"]` in jeder ausgelieferten Phase | beide Scratchpad-Server sind per Default aktiv; die Liste wechselt pro Schritt, ruft also beide Server (ab zwei Gedanken). Der Startup-Health-Check prüft genau die Tools der aktivierten Phasen |
| `reasoning.max_total_steps` | 8 | Obergrenze der Reasoning-Schritte pro Phasenlauf |
| `reasoning.max_injected_chars` | `null` (unkapp; `REASONING_MAX_INJECTED_CHARS`) | Obergrenze für den gerenderten Reasoning-Trace, der in Downstream-Prompts injiziert wird: überschreitet der Render die Grenze, wird er gekappt und `(reasoning trace truncated at N chars)` angehängt; die Thoughts je Schritt bleiben ganz, nur die Downstream-Injektion wird gekürzt. Senkt Input-Tokens bei jedem Downstream-Call (Latenz-Hebel L6). Der Wire-Config-Notiz: eine nicht gesetzte Env (`:-`) mapt auf `null` = unkapp |
| `validation.truncation_rerun` | `repair` (`VALIDATION_TRUNCATION_RERUN`) | Was geschieht, wenn eine Antwort am Provider-Token-Limit gekappt (`finish_reason=length`) und nicht parsebar ist (§4 Regel 1a): `repair` (ausgelieferter Default, Verhalten unverändert) = der gekappte Payload durchläuft den normalen Repair-Retry; `boost` = Truncation-first — KEIN Repair-Recall in dasselbe Limit, genau EIN Rerun mit `truncation_boost_max_completion_tokens`, eine zweite Kappung wirft `LLMError(failure_kind="truncated")` (Transport-Klasse, niemals repair-gepromptet) |
| `validation.truncation_boost_max_completion_tokens` | `65536` (`VALIDATION_TRUNCATION_BOOST_TOKENS`) | Completion-Budget des Boost-Reruns (empfohlenes MiniMax-M2.x-Limit); erreicht den Provider als litellms `max_completion_tokens` |
| `prompts.stable_prefix_first` | `false` (`PROMPT_STABLE_PREFIX`) | L7-Reihenfolge: verlagert in jedem User-Prompt den ≥512-Zeichen statischen Response-Contract-Block (identisch über alle Calls der Operation) nach vorn und jeden Lauf-Payload (Beschreibung, Parts, Result-JSON, Reasoning-Kontext) nach hinten — maximiert MiniMax passive Prompt-Cache-Treffer (≥512-Token stabiler Präfix). Inhalt in beiden Modi identisch, nur die Reihenfolge ändert sich; Prompt-Cache-Verwertung wird über `usage.prompt_tokens_details.cached_tokens` beurteilt, wenn der Provider sie ausliefert |

Alle Parameter werden beim Start gelesen und mit denselben Summen- und Range-Checks validiert wie beim Schwesterserver; jede Entscheidung, die sie gaten, wird mit ihren Inputs geloggt (Teil-Id, Kandidat, Rohscores, angewendete Deltas), sodass jede Selektion aus den Logs reproduzierbar ist.

Vier Bounds bestimmen einen Lauf und dürfen nicht verwechselt werden: der **Provider-Call-Timeout** (`generator.config.timeout_seconds`, 120 s) begrenzt eine HTTP-Anfrage an das Modell; das **Structured-Output-Repair-Budget** (`validation.max_retries`, 1) stellt genau einen weiteren Call bereit und ausschließlich bei Modell-Output-Fehlern — ein Provider-Timeout, eine Provider-Exception, eine leere Antwort oder ein offener Circuit werden sofort weitergereicht und niemals repair-gepromptet (und mit `validation.truncation_rerun="boost"` wird eine `finish_reason=length`-Kappung vom Truncation-first-Rerun statt dem Repair-Loop behandelt); der **Design-Loop-Retry** (`max_tries`, 2) ist die bestehende GENERATE→EVALUATE-Versuchsgrenze, die E12 besitzt; und die **Deadline des gesamten Laufs** (`tasks.deadline_seconds`, 1800 s) bricht ab, was noch läuft — einschließlich der Provider-Koroutine — und beendet den Lauf mit `ERR_014`. Mechanisch läuft die Lauf-Deadline über den Workflow-Timeout der Pipeline (`deadline_seconds * 0.9`): Ablauf löst `WorkflowTimeoutError` im Graphen aus, jeder öffentliche Wrapper reicht ihn als `TimeoutError` weiter, und die Tool-Naht (`run_with_deadline`) mappt ihn auf `ERR_014`.

**Latenznotizen (2026-09-25).** Live-A/B-Lauf auf dem CPU-Dev-Stack (8-Teile-IoT/Kafka-Anfrage, 777 Zeichen): Before (sequenzielle Teil-Schwänze, sequenzielle GENERATE-Batches, 4/3/3/3 Reasoning-Schritte) 1169 s; die Teil-Schwänze allein machen 315 s aus (8 × ~40 s), die sechs sequenziellen GENERATE-Batches ~495 s. Der Kappe-2-Retrieval-Schwanz-Lauf scheitert an diesem Sidecar komplett (HTTP 429, eine Volltext-Anfrage sättigt den 16.384-Token-Inferenzpuffer — siehe `retrieval.max_concurrent_parts`-Zeile), daher ist der ausgelieferte Default 1; ein voller Kappe-1-`retrieve_many`-Durchlauf misst 242,4 s (Warm-up 250,4 s). `retrieval.max_concurrent_parts` 2 und `reranker.rerank_text: capsule` bleiben Opt-in; `capsule` war auf dem Sidecar 3,6–4,3x schneller gemessen, aber die Rankings weichen ab (siehe §6 Schritt 4).

---

## 15. Determinismus, Nebenläufigkeit und Testbarkeit (§15)

- **Deterministischer Kern.** Alles außer den vier LLM-Call-Arten (§3.1) ist reine Arithmetik über Katalogdaten: identische Inputs, Parameter und Index → identisches Ranking, identische Tie-Breaks, Schwellen und Fallbacks. Retrieval ist CPU + Embedder-HTTP und läuft deshalb abseits des Event-Loops in einem Worker-Thread (analog zur `asyncio.to_thread`-Nutzung des Schwesterservers), damit parallele MCP-Requests nicht blockieren.
- **Cancellation.** Ein kooperatives Token wird zwischen Stages und zwischen Versuchen geprüft; Wirkungen stoppen an der nächsten Grenze.
- **Workflow-Realisierung (2026-10-01).** `DesignPipeline` ist eine llama-index-`Workflow`-Unterklasse: Die fünf Einstiegspunkte starten einen Ereignisgraphen, dessen Schritte (`route`, `analyze_step`, `generate_step`, `evaluate_step`, `refine_step`) die §3-Stufenreihenfolge realisieren, und die E12-Loop-Rückkante ist das `SelectionReady`-Ereignis `refine_step` → `generate_step` statt einer Python-Schleife. Beobachtbares unverändert: Signaturen, Wire-Contract, Fehlercodes, E12-Semantik, Cancellation und der deterministische Kern (oben).
- **Teststrategie (vom Design impliziert).** Der deterministische Kern ist ohne jedes Modell testbar: Gewichts-Smoothing, `analysis_score`, Blend, Adjustments, Ranking-Kette, Dual Gate, Near-Tie-Marge, Supporting-Regeln (Distinct-Constraint, Konflikttabelle, Caps), Dominanz-Wahl, Dedupe, Fallbacks E5–E7 und die E7-Nicht-Fallback-Regel für `specialized-components` — dazu die kleinen puren Funktionen um die LLM-Calls: Part-Repair/Merge, Alias-Dedupe, Prioritätsnormalisierung, Reselektions-Ledger und Stickiness, Dominanz-Tie-Break, Contract-Part-Exemption, plus die Ein-Sprach-Auflösung: recognized / unbekannt→generic / fehlend→generic, die Effective-Language-Normalisierung in `design_brief.language`, das einheitliche Diskouragierungs-Veto (konkreter Eintrag und `generic`-Eintrag), die Generic-Mode-Nicht-Adjustment-Regel, die Bindungsreihenfolge (angefragter Spracheintrag → `generic`-Eintrag) und das Ein-Element-`overview.languages`-Echo. LLM-abhängige Stages werden mit aufgezeichneten Fixture-Antworten getestet, die Schema-Validität und die deterministische Nachverarbeitung dahinter prüfen.
- **Katalog-Synchronität.** Die Kategorie-/Scope-/Sprach-Enums in Prompts und Validatoren werden aus `docs/pattern-schema.json` abgeleitet (Single Source of Truth), und die Style-Affinitätstabelle hängt am publizierten Style-Enum des Architekturservers — Drift zwischen Katalog und Code ist ein Validierungsfehler, keine stille Verhaltensänderung. Die Ressource `design-patterns://variants` (§2.4) wird beim Start aus den gejointen Records materialisiert, hinter demselben §2.3-Integritäts-Check — sie kann nie einen Katalogzustand exponieren, der die Validierung nicht bestanden hat.

---

## 16. Kurzfazit

Der Selektionsprozess ist eine Pipeline aus zwei LLM-Stufen (Brief+Teile, gebündelte Gewichte), hybridem Retrieval (Dense 0.7 + BM25 0.3 mit Pflicht-Cross-Encoder-Blend), anforderungsgewichtetem Scoring mit deterministischem Adjustment-Ledger, streng sortierter Selektion mit Dual-Gate (50/50), Near-Tie-Markierung und fünf Supporting-Regeln, Komposition mit Dominanz-Wahl sowie einer begrenzten Generate–Evaluate–Refine-Schleife (max. 2 Versuche, Best-Retention, max. 1 Reselektion). Jede Ambiguität, jeder Fallback und jedes Veto bleibt im Reasoning-Trace sichtbar — der Agent sieht stets, warum ein Pattern gewonnen hat und was fast gewonnen hätte.
