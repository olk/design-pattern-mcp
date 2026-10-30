# Interpretation: `consistency-base-20-spark` → `consistency-jev-20`

Stand: 2026-10-30. Quelle: `comparison.md` / `comparison.json` plus eigene
Auszählungen aus den Szenario-Records beider Arme und den `manifest.json`-Dateien.

**Kurzfazit:** Sauberer Jev-A/B ohne Provider-Confound, Urteil
konstruktionsbedingt `inconclusive` (0 Holdout-Samples). Szenario-Bilanz
5 besser / 2 schlechter / 13 unverändert, alle Deltas insignifikant. Der
Seam feuerte in 5/20 Szenarien (15/60 Records) — alle 5 waren in beiden
Armen hit-einstimmig — und in keinem Flip- oder diskordanten Szenario.
**Kein messbarer Jev-Effekt; alle Deltas sind LLM-Varianz bzw.
Workload-Differenz.**

## 1. Versuchsaufbau

| | Baseline | Kandidat |
|---|---|---|
| Run | `consistency-base-20-spark` | `consistency-jev-20` |
| Modus | `analyze`, 60 Records (20 Szenarien × 3 Repeats, alle train) | identisch |
| Korpus | `full.json` sha `92a8558a…` | identisch (match: true) |
| Generator | `opencode-go / openai/muse-spark-1.3-contributor` | identisch |
| Reasoning | shannon + code ok | identisch |
| Warmup | aus (kein Cache-Artefakt) | aus |
| Unterschied | `DECISIONS_ENABLED=false` | `DECISIONS_ENABLED=true`, `jev-1.13-free` |

Integrität (`comparison.json`): einziger Config-Diff ist
`decisions.enabled`, einziger Env-Diff `DECISIONS_ENABLED`. Kein
Generator-, Reasoning- oder Korpus-Confound — ein echter Jev-A/B. Zwei
Einschränkungen: `git.head` ist in beiden Manifesten `null`
(Container-Artefakt, Code-Identität nicht belegbar), und die Läufe sind
sequenziell am selben Tag (Kandidat 11:35–14:54, Baseline 15:15–18:45 UTC).
`llm.weights`-Totale (−2 %) sprechen für einen stabilen Provider.

Nur der Ch2-Style-Seam war aktiv: `weights_source` ist in allen 149+182
Parts `llm`, kein Decision-Model-Weights-Eingriff.

## 2. Warum `inconclusive`

Die vorregistrierte Regel verlangt **Holdout-p95 −15 % UND
Kandidat-Hitrate ≥ Baseline-Wilson-Untergrenze**. Beide Arme haben
0 Holdout-Samples (Split war `train`), p95 ist undefiniert. Kein
Fehlschlag, sondern Design: der Run wurde für Konsistenz (Repeats)
gebaut, nicht für die Regel. Die Qualitäts-Hälfte wäre erfüllt
(66,7 % ≥ ~47,4 % Baseline-Wilson-Untergrenze), die Latenz-Hälfte ist
untestbar.

## 3. Trefferquote: Szenario-Bilanz 5/2/13

Record-Ebene (n=60, Repeats als unabhängig gezählt):

| Split | Baseline | Kandidat | Δ [95 % CI] |
|---|---|---|---|
| overall | 60,0 % | 66,7 % | +0,067 [−0,104, +0,232] |
| holdout | — | — | — |

Das Newcombe-Intervall schließt 0 ein. Zusätzlich übertreibt n=60 die
Präzision: 3 Repeats teilen sich u. a. einen LRU-gecachten Jev-Style
(`src/decisions/client.py:124`), effektives N ≈ 20 — das echte Intervall
ist breiter. Ehrliche Headline ist die Szenario-Ebene (eigene Auszählung):

**5 Szenarien besser / 2 schlechter / 13 unverändert** (netto +4 Records):

| Szenario | Baseline (Hits/3) | Jev (Hits/3) | Δ |
|---|---|---|---|
| `tactical-customer-identity` | 1 | 3 | +2 |
| `domain-logic-shipment-import` | 2 | 3 | +1 |
| `method-structure-buffer-assignment` | 2 | 3 | +1 |
| `queue-drain-shutdown` | 1 | 2 | +1 |
| `composition-encapsulation-checkout-entry` | 0 | 1 | +1 |
| `functional-composition-request-validation` | 1 | 0 | −1 |
| `model-clarity-ledger-contract-checks` | 1 | 0 | −1 |

Hit-Flips auf Record-Ebene: 7 Gewinne, 3 Verluste. Auffällig:
`extension-compile-time-matrix` flippt in #2 zum Guten und #3 zum
Schlechten — Signatur von LLM-Varianz, kein gerichteter Eingriff.

Selektionsmetriken (Mittel): `recall@3` +0,003, `recall@5` +0,004,
`acceptable_primary_f1` +0,047, `score` +0,005 (32 verbessert /
28 verschlechtert — Münzwurf), `veto_violations` 0/0. `supporting_f1`
0,003→0,038 bleibt ~0: leere Supporting-Sets im Analyze-Modus, kein Signal.

## 4. Konsistenz: Gewinn nicht Jev-attribuierbar

Pro-Szenario-Übereinstimmung über die 3 Repeats:

| | Baseline | Kandidat |
|---|---|---|
| Pattern-einstimmig (3/3 gleiches Final-Pattern) | 9/20 | 13/20 |
| Hit-einstimmig (3/3 gleiches Hit/Miss) | 13/20 | 17/20 |

McNemar auf Szenario-Ebene: 5 nur-Baseline-gemischt vs. 1 nur-Jev-gemischt
(p≈0,22, N=20) — insignifikant. Aber entscheidend: Der Test vergleicht
zwei Rauschprozesse, keinen unterpowerten Jev-Effekt, denn der Seam war
in jedem diskordanten Szenario inert (siehe §7). „Richtung stimmt / Power
reicht nicht" ist zurückgezogen.

Baseline-gemischt (7): `domain-logic-shipment-import`,
`extension-compile-time-matrix`, `functional-composition-request-validation`,
`method-structure-buffer-assignment`, `model-clarity-ledger-contract-checks`,
`queue-drain-shutdown`, `tactical-customer-identity`.
Jev-gemischt (3): `composition-encapsulation-checkout-entry`,
`extension-compile-time-matrix`, `queue-drain-shutdown`.

## 5. Latenz: kein belastbarer Gewinn

| | Baseline | Kandidat | Δ |
|---|---|---|---|
| e2e p50 | 209,7 s | 198,9 s | −5,2 % |
| e2e p95 | 264,7 s | 262,2 s | −0,9 % |
| gepaart Median / Mittel | — | — | −12,8 s / −10,4 s, 36 schneller / 24 langsamer, Sign-Test p=0,155 |

p=0,155 bei n=60 (effektiv ~20): Rauschen. Beachte: e2e ist hier eine
Phasen-Teilmenge (nur analyze), keine Full-Run-Latenz.

## 6. Stage-Deltas: Rerank-Effekt ist ein Parts-Artefakt

| Stage | Calls Δ | Total Δ | Erklärung |
|---|---|---|---|
| `rerank` | −33 (182→149) | −894 s | exakt = −33 Parts; pro Call 22,9 s → 22,0 s (−4 %, Rauschen) |
| `llm.analyze` | 0 | −107 s | −1,8 s/Record, Varianz |
| `llm.weights` | 0 | −50 s | −0,8 s/Record, Provider stabil |
| `reasoning.analyze` | 0 | −57 s | Varianz |
| `reasoning.weights` | 0 | +251 s | +4,2 s/Record; Jev berührt Reasoning nicht → Last-Varianz |
| `embed.query` | 0 | −15 s | vernachlässigbar |

Rechnung: gepaartes e2e-Δ −10,4 s/Record × 60 = **−624 s**; Rerank allein
**−894 s**, teilweise kompensiert durch `reasoning.weights` **+251 s**.
Der gesamte e2e-Unterschied steckt in der Parts-Zahl plus
Reasoning-Lastdrift — kein Latenz-Claim für den Decision-Seam.

Kausal unmöglich anders: `rerank`-Calls = Parts exakt (60/60 Records in
beiden Armen), und die Dekomposition läuft in `src/pipeline.py:1610-1611`
**vor** `_apply_decision_style` — Jev kann die Parts-Zahl gar nicht
verändern. Die −33 Calls sind Ch1-Dekompositionsdrift des LLM
(z. B. `composition-encapsulation-checkout-entry` 4→1 Parts in allen
3 Repeats, total Parts 182→149).

## 7. Attribution: Seam feuerte, entschied aber nichts

Fingerprint (`style_affinity+5`-Adjustments auf Kandidaten):

- **15/60 Kandidat-Records = 5 Szenarien × 3 Repeats** tragen Affinität;
  0/60 in der Baseline. Der Seam ist verdrahtet und aktiv.
- Die 5 Szenarien: `algorithms-traversal-variation-pricing-rules`,
  `concurrency-resilience-fanout-deadline`,
  `coordination-primitives-bounded-payment-handoff`,
  `dependency-wiring-api-handler-collaborators`,
  `functional-composition-inbound-normalization`.
- **Alle 5 sind hit-einstimmig in beiden Armen** (4× TTT, 1× FFF bei
  `coordination-handoff`). Auch dort, wo Jev griff, ist der Effekt auf
  Hits exakt null.
- **Überlappung Affinität × 8 Flip-Szenarien: leer. × Baseline-mixed (7):
  leer. × Jev-mixed (3): leer.**
- **0/10 Hit-Flips** haben `style_affinity` auf dem gewählten
  Final-Pattern (Projekt-Fingerprint-Regel: keine Affinität auf dem
  entscheidenden Part = LLM-Varianz).
- Die +5-Affinität ist klein gegen Fit-Scores ~70–90 — sie kippt nur
  knappe Rennen; hier kippte sie keines.

Caveat: Die Records loggen den aufgelösten Brief-Style nicht
(`pipeline: null` im Analyze-Modus); ein rein prompt-vermittelter Effekt
wäre unbeobachtbar. Nach der Fingerprint-Regel gilt er als nicht belegt.

## 8. Einordnung

Valider A/B-Aufbau, aber: (1) kein Holdout → keine Regel-Entscheidung;
(2) alle Deltas insignifikant und durch Repeats geschönt (Szenario-Bilanz
5/2/13 statt +6,7 pp zitieren); (3) beide auffälligen Zahlen (Hits,
Rerank) sind per Fingerprint- bzw. Pipeline-Analyse **nicht** auf Jev
attribuierbar — der Seam war in jedem diskordanten Szenario inert.

**Fazit des Arms: Seam verdrahtet und aktiv, kein messbarer Effekt.**
Für einen belastbaren Konsistenz-Claim: pro-Szenario-Agreement als
Primärmetrik mit exaktem Paartest bei größerem N; für das vorregistrierte
Urteil: Holdout-Arm mit ≥4 Samples pro Arm.
