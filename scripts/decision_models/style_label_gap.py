#!/usr/bin/env python
"""The §4.7.1 style-label gap: authoring briefs, pool validation, the leakage gate, application.

Why this exists
---------------
``docs/tacit-fine-tune.md`` §4.7.1 blocks style-seam enablement on data volume, not modelling:
139 reviewed train style labels and a 30-row holdout cannot certify a confidence floor. The
recorded requirement is ``>= 500`` reviewed train style labels and ``>= 100`` holdout rows that
are family-disjoint, with organic scenarios ``>= 70 %`` of new train rows and ``100 %`` of new
holdout rows, every row labelled by two blind annotators under ``qa/typed/style-rubric.md`` and
third-party adjudication. This module is the corpus machinery for that work; the review side
(packs, agreement, certification) is ``style_label_review.py``.

Subcommands
-----------
``briefs``
    Deterministic authoring slots: each slot fixes the split, the label the row must exemplify,
    the catalog family/subgroup, the expected primary ids, the category and the language. An
    author writes a description and a decision statement and never chooses a label, so the pool
    cannot drift toward convenient classes. Targets overshoot the 500 floor — 540 organic + 180
    synthetic train candidates (75 % organic) and 108 organic holdout candidates — so the gate
    survives a batch-rejection rate up to 50 %.
``validate``
    Schema + catalog + leak + duplicate + coverage checks over the authored pool. A description
    that *names* its style (or its pattern) teaches string matching instead of topology, so it
    is refused; so is a duplicate or near-duplicate of an existing row.
``dedup``
    The holdout leakage gate: word 5-gram Jaccard ``>= 0.8`` and any shared 50-token substring
    between a holdout row and ANY train row (existing corpus + new pool), plus family-set
    intersection and id collisions. Non-zero exit on any violation.
``apply``
    Append accepted rows to the corpus (``tests/benchmark/scenarios/full.json``), to
    ``STYLE_LABELS`` in ``derive_style_labels.py``, and regenerate ``qa/typed/style.jsonl`` plus
    the mirrored ``qa/typed/choice.jsonl``. Existing corpus rows keep their order and bytes, so
    a benchmark arm that cuts the first N scenarios still sees the same N.

Every artifact this writes is derived from the corpus and the label source; nothing here is
hand-copied, so ``dedup``/``validate`` are the only ways to pass.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.decisions import seam_prompts  # noqa: E402
from src.patterns import loader as pattern_loader  # noqa: E402
from src.style_affinity import STYLE_AFFINITY  # noqa: E402

DEFAULT_CORPUS = "tests/benchmark/scenarios/full.json"
DEFAULT_LABELS = "scripts/decision_models/derive_style_labels.py"
DEFAULT_STYLE_OUT = "qa/typed/style.jsonl"
DEFAULT_CHOICE_OUT = "qa/typed/choice.jsonl"

#: Candidate-pool targets. Sized for a ~610-label reviewed corpus at the plan's 65 % yield
#: assumption; the 500-label gate survives a yield as low as 50 % (361 / 720).
TRAIN_POOL_TOTAL = 720
TRAIN_ORGANIC_TOTAL = 540
HOLDOUT_POOL_TOTAL = 108
#: The audit found the only two api-gateway train labels were mislabelled, so the seam has no
#: train positive at all: the pool carries a deliberate over-sample instead of the class prior.
API_GATEWAY_TRAIN_TARGET = 28
#: Every class gets a floor so a rare style cannot be starved by a proportional draw.
TRAIN_STYLE_FLOOR = 12
HOLDOUT_STYLE_FLOOR = 3
#: §4.7.1: organic scenarios are at least 70 % of new train rows and all of the holdout.
ORGANIC_SHARE_FLOOR = 0.70
#: Languages a brief may request (schema enum members verified against the catalog at run time).
LANGUAGES: tuple[str, ...] = ("java", "python", "typescript", "go", "rust", "csharp", "kotlin")
#: Style labels the pool may exemplify, in served order (``none`` last, as served).
POOL_LABELS: tuple[str, ...] = (*STYLE_AFFINITY, seam_prompts.STYLE_NONE_KEY)
#: Leakage thresholds: the split must contain no near-duplicate pair at or above this Jaccard.
JACCARD_LIMIT = 0.8
SHARED_SUBSTRING_TOKENS = 50
NGRAM = 5
#: Near-duplicate reporting band inside the pool (a warning, never a pass/fail).
NEAR_DUPLICATE_BAND = 0.5
#: Application domains a brief may point an author at. Two rows for the same style drawn from
#: two different domains are two systems; two rows in one domain drift into the same system with
#: the nouns changed, which is what the leakage gate exists to catch.
DOMAIN_HINTS: tuple[str, ...] = (
    "freight forwarding",
    "hospital scheduling",
    "retail point of sale",
    "energy grid telemetry",
    "insurance claims intake",
    "airline crew rostering",
    "card payment authorization",
    "telecom billing mediation",
    "clinical laboratory results",
    "construction procurement",
    "fleet telematics",
    "video streaming catalogue",
    "grocery delivery dispatch",
    "legal document review",
    "consumer tax filing",
    "pharmaceutical cold chain",
    "securities trade settlement",
    "agricultural sensor network",
    "hotel reservations",
    "university enrolment",
    "public transit ticketing",
    "vehicle diagnostics",
    "smart building climate control",
    "online auctions",
    "loyalty programme ledger",
    "contact centre routing",
    "network service provisioning",
    "content moderation queue",
    "digital identity verification",
    "mortgage origination",
    "claims adjudication",
    "inventory replenishment",
    "warehouse robotics",
    "ride hailing matching",
    "travel itinerary booking",
    "subscription billing",
    "field service dispatch",
    "clinical trial monitoring",
    "weather forecasting pipeline",
    "satellite imagery ingestion",
    "electric vehicle charging",
    "port container logistics",
    "municipal waste collection",
    "insurance underwriting",
    "retail banking onboarding",
    "industrial machine maintenance",
    "sports event ticketing",
    "pharmacy dispensing",
)


# ---------------------------------------------------------------------------------------
# Deterministic planning
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """One authoring slot: the fixed facts an author may not change."""

    slot_id: str
    split: str
    kind: str
    style: str
    family: str
    primary: str
    acceptable_primary: tuple[str, ...]
    acceptable_supporting: tuple[str, ...]
    category: str
    language: str | None
    intent_hint: str
    domain_hint: str

    def brief(self) -> dict[str, Any]:
        """The JSON brief handed to an author (everything but the prose)."""
        return {
            "slot_id": self.slot_id,
            "split": self.split,
            "kind": self.kind,
            "style": self.style,
            "family": self.family,
            "primary": self.primary,
            "acceptable_primary": list(self.acceptable_primary),
            "acceptable_supporting": list(self.acceptable_supporting),
            "category": self.category,
            "language": self.language,
            "intent_hint": self.intent_hint,
            "domain_hint": self.domain_hint,
        }


def _seed(key: str) -> int:
    """Stable integer seed from a slot id (no global RNG state, reproducible across runs)."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _largest_remainder(weights: Mapping[str, float], total: int) -> dict[str, int]:
    """Distribute ``total`` integer slots over ``weights`` proportionally, largest remainder."""
    if total < 0:
        raise ValueError("total must be non-negative")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive number")
    quotas = {key: total * weight / weight_sum for key, weight in weights.items()}
    counts = {key: int(quota) for key, quota in quotas.items()}
    leftover = total - sum(counts.values())
    order = sorted(quotas, key=lambda key: (-(quotas[key] - counts[key]), key))
    for key in order[:leftover]:
        counts[key] += 1
    return counts


def train_targets(
    prior: Mapping[str, int],
    *,
    total: int = TRAIN_POOL_TOTAL,
    organic_total: int = TRAIN_ORGANIC_TOTAL,
) -> dict[str, tuple[int, int]]:
    """Per-label ``(organic, synthetic)`` train counts: floor-per-class, prior-weighted rest.

    The floor guarantees every served style has candidates to review, which is what closes the
    ``api-gateway`` gap; the prior-weighted remainder keeps the reviewed corpus roughly
    proportional to the traffic the seam actually sees, and ``none`` is capped at its prior
    share so abstention cannot dominate the label budget.
    """
    if total < organic_total:
        raise ValueError("train pool total must be at least the organic target")
    counts = dict.fromkeys(POOL_LABELS, TRAIN_STYLE_FLOOR)
    remaining = organic_total - sum(counts.values())
    counts["api-gateway"] = API_GATEWAY_TRAIN_TARGET
    remaining -= API_GATEWAY_TRAIN_TARGET - TRAIN_STYLE_FLOOR
    weights = {label: float(prior.get(label, 0)) for label in POOL_LABELS if label not in {"api-gateway"}}
    if remaining > 0:
        if sum(weights.values()) <= 0:
            # No usable prior: spread evenly instead of failing (first build on an empty corpus).
            weights = {label: 1.0 for label in POOL_LABELS if label not in {"api-gateway"}}
        spread = _largest_remainder(weights, remaining)
        for label, extra in spread.items():
            counts[label] += extra
    none_cap = max(TRAIN_STYLE_FLOOR, round(priorshare(prior, "none") * organic_total))
    counts[seam_prompts.STYLE_NONE_KEY] = min(counts[seam_prompts.STYLE_NONE_KEY], none_cap)
    synthetic = _largest_remainder({label: float(count) for label, count in counts.items()}, total - organic_total)
    return {label: (counts[label], synthetic[label]) for label in POOL_LABELS}


def priorshare(prior: Mapping[str, int], label: str) -> float:
    """The label's share of the existing reviewed train corpus (0.0 when the corpus is empty)."""
    total = sum(prior.values())
    if total <= 0:
        return 0.0
    return prior.get(label, 0) / total


def holdout_targets(total: int = HOLDOUT_POOL_TOTAL) -> dict[str, int]:
    """Per-label holdout counts: an even floor over every served style, then prior-free extras.

    The holdout is an instrument: a class with no holdout row contributes nothing to the
    measured accuracy, so the floor is per class rather than proportional. ``108 = 36 x 3``
    covers the whole served vocabulary, which is stricter than the 11-class 30-row holdout it
    replaces (the accuracy bar stays 0.80 over more options, not fewer).
    """
    counts = dict.fromkeys(POOL_LABELS, HOLDOUT_STYLE_FLOOR)
    remaining = total - sum(counts.values())
    if remaining < 0:
        raise ValueError(f"holdout total {total} is below the per-class floor budget")
    for label, extra in _largest_remainder(dict.fromkeys(POOL_LABELS, 1.0), remaining).items():
        counts[label] += extra
    return counts


def family_index(catalog: pattern_loader.PatternLoader) -> dict[str, tuple[str, ...]]:
    """``subgroup -> pattern ids`` in catalog order."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in catalog.order:
        grouped[catalog.records[name].subgroup].append(name)
    return {subgroup: tuple(names) for subgroup, names in grouped.items()}


def corpus_families(corpus: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    """``split -> family`` set of a corpus."""
    families: dict[str, set[str]] = {"train": set(), "holdout": set()}
    for scenario in corpus:
        families.setdefault(str(scenario["split"]), set()).add(str(scenario["family"]))
    return families


def _sibling_pair(names: Sequence[str], slot_id: str) -> tuple[str, tuple[str, ...]]:
    """Pick the expected primary and two sibling acceptable ids, deterministically per slot."""
    ordered = sorted(names)
    start = _seed(slot_id) % len(ordered)
    rotated = ordered[start:] + ordered[:start]
    primary = rotated[0]
    siblings = tuple(rotated[1:3])
    return primary, siblings


def _language_for(catalog: pattern_loader.PatternLoader, primary: str, slot_id: str) -> str | None:
    """A language the primary is not discouraged in, or ``None`` (generic mode) for a third."""
    seed = _seed(slot_id)
    if seed % 3 == 0:
        return None
    candidates = [name for name in LANGUAGES if name in catalog.enums.languages]
    record = catalog.records[primary]
    viable = [
        language
        for language in candidates
        if (entry := record.entry(language)) is None or entry.support != "discouraged"
    ]
    return viable[seed % len(viable)] if viable else None


def _intent_hint(record: pattern_loader.JoinedPattern, limit: int = 260) -> str:
    """A bounded prompt hint: the pattern's intent, never its name (authors never see the id)."""
    intent = " ".join(record.intent.split())
    return intent[:limit]


def build_slots(
    catalog: pattern_loader.PatternLoader,
    corpus: Sequence[Mapping[str, Any]],
    *,
    train_total: int = TRAIN_POOL_TOTAL,
    organic_total: int = TRAIN_ORGANIC_TOTAL,
    holdout_total: int = HOLDOUT_POOL_TOTAL,
) -> list[Slot]:
    """The whole authoring plan: deterministic slots for the train and holdout pools."""
    prior = Counter(
        str(row["label"]) for row in _existing_style_rows() if row.get("source", {}).get("split") == "train"
    )
    families = corpus_families(corpus)
    train_families = sorted(families.get("train", set()))
    holdout_families = sorted(families.get("holdout", set()))
    if not train_families or not holdout_families:
        raise SystemExit("corpus carries no train/holdout families to draw slots from")
    index = family_index(catalog)
    degenerate = [family for family in (*train_families, *holdout_families) if len(index.get(family, ())) < 2]
    if degenerate:
        raise SystemExit(f"families without a sibling pair cannot carry a scenario: {sorted(set(degenerate))}")
    targets = train_targets(prior, total=train_total, organic_total=organic_total)
    slots: list[Slot] = []
    counters: Counter[str] = Counter()
    for style in POOL_LABELS:
        organic_count, synthetic_count = targets[style]
        for kind, count in (("organic", organic_count), ("synthetic", synthetic_count)):
            for _ in range(count):
                counters["train"] += 1
                slots.append(
                    _make_slot(
                        catalog,
                        index,
                        kind=kind if kind == "synthetic" else _organic_kind(style, kind),
                        split="train",
                        style=style,
                        families=train_families,
                        ordinal=counters["train"],
                    )
                )
    for style, count in holdout_targets(holdout_total).items():
        for _ in range(count):
            counters["holdout"] += 1
            slots.append(
                _make_slot(
                    catalog,
                    index,
                    kind="organic",
                    split="holdout",
                    style=style,
                    families=holdout_families,
                    ordinal=counters["holdout"],
                )
            )
    return slots


def _organic_kind(style: str, kind: str) -> str:
    """Organic rows keep their provenance kind (``apigw`` documents get their own tag)."""
    del kind
    return "organic-apigw" if style == "api-gateway" else "organic-doc"


def _make_slot(
    catalog: pattern_loader.PatternLoader,
    index: Mapping[str, tuple[str, ...]],
    *,
    kind: str,
    split: str,
    style: str,
    families: Sequence[str],
    ordinal: int,
) -> Slot:
    """One slot: family by round-robin position, primary/siblings/language from the catalog."""
    slot_id = f"{split[:2]}-{kind.rsplit('-', maxsplit=1)[-1]}-{ordinal:04d}"
    family = families[(ordinal - 1) % len(families)]
    primary, siblings = _sibling_pair(index[family], slot_id)
    record = catalog.records[primary]
    return Slot(
        slot_id=slot_id,
        split=split,
        kind=kind,
        style=style,
        family=family,
        primary=primary,
        acceptable_primary=(primary, *siblings),
        acceptable_supporting=(),
        category=record.category,
        language=_language_for(catalog, primary, slot_id),
        intent_hint=_intent_hint(record),
        domain_hint=DOMAIN_HINTS[_seed(f"domain-{slot_id}") % len(DOMAIN_HINTS)],
    )


def _existing_style_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """The current ``qa/typed/style.jsonl`` rows (the class prior and dedup reference)."""
    source = path or (REPO_ROOT / DEFAULT_STYLE_OUT)
    if not source.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------------------
# Text similarity (the leakage and duplicate machinery)
# ---------------------------------------------------------------------------------------


def tokens(text: str) -> list[str]:
    """Lowercased word tokens (punctuation dropped) — the unit every similarity check uses."""
    cleaned = "".join(character.lower() if character.isalnum() else " " for character in text)
    return cleaned.split()


def shingles(text: str, size: int) -> set[tuple[str, ...]]:
    """The set of ``size``-word shingles of ``text``."""
    words = tokens(text)
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[start : start + size]) for start in range(len(words) - size + 1)}


def jaccard(left: Iterable[object], right: Iterable[object]) -> float:
    """Jaccard similarity of two token-shingle sets."""
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


@dataclass(frozen=True)
class DuplicatePair:
    """One holdout/train pair that is too similar to keep in separate splits."""

    holdout_id: str
    train_id: str
    similarity: float
    shared_run: int


def near_duplicates(
    train_rows: Sequence[tuple[str, str]],
    holdout_rows: Sequence[tuple[str, str]],
    *,
    limit: float = JACCARD_LIMIT,
    run: int = SHARED_SUBSTRING_TOKENS,
) -> list[DuplicatePair]:
    """Every holdout/train pair at or above the Jaccard limit or sharing a long token run.

    ``train_rows``/``holdout_rows`` are ``(id, text)`` pairs. The 50-token run check is the
    exact shared-substring rule from Lee et al. §4.1: a run that long is copied text, not
    coincidence, and no MinHash approximation is needed at these corpus sizes — the exact
    comparison over every pair is affordable and strictly stronger than LSH at the same bar.
    """
    train_shingled = [(name, shingles(text, NGRAM), shingles(text, run)) for name, text in train_rows]
    found: list[DuplicatePair] = []
    for holdout_id, text in holdout_rows:
        holdout_shingles = shingles(text, NGRAM)
        holdout_runs = shingles(text, run)
        for train_id, candidate_shingles, candidate_runs in train_shingled:
            similarity = jaccard(holdout_shingles, candidate_shingles)
            shared = len(holdout_runs & candidate_runs) if holdout_runs and candidate_runs else 0
            if similarity >= limit or shared:
                found.append(DuplicatePair(holdout_id, train_id, similarity, shared))
    return found


# ---------------------------------------------------------------------------------------
# Pool records: authored rows → validated scenarios
# ---------------------------------------------------------------------------------------


#: The keys an author must supply per slot (prose only — every label fact comes from the brief).
AUTHOR_FIELDS: tuple[str, ...] = ("slot_id", "topic", "description", "decision_statement", "notes", "rationale")
#: Provenance kinds; ``synthetic`` is capped so organic stays >= 70 % of new train rows.
PROVENANCE_KINDS: tuple[str, ...] = ("organic-doc", "organic-apigw", "synthetic")


def _kebab(text: str) -> str:
    """Lowercase kebab slug (schema-compatible scenario ids)."""
    cleaned = "".join(character if character.isalnum() else "-" for character in text.strip().lower())
    return "-".join(part for part in cleaned.split("-") if part)


def compose_scenario_id(style: str, topic: str, *, max_length: int = 80) -> str:
    """``<style>-<topic>``, truncated to the schema cap while keeping the style prefix."""
    slug = _kebab(topic) or "scenario"
    candidate = f"{style}-{slug}" if not slug.startswith(f"{style}-") else slug
    return candidate[:max_length].rstrip("-")


#: Multi-word aliases that are ordinary engineering vocabulary, not a way to name a pattern. A
#: description that says "use case" is describing a system, not string-matching a label, so the
#: pattern-phrase gate must not refuse it.
GENERIC_ALIASES: frozenset[str] = frozenset({"use case", "data source", "business rule", "value type"})


def _banned_phrases(record: pattern_loader.JoinedPattern) -> tuple[str, ...]:
    """Pattern spellings a description may not use: the kebab id and multi-word names/aliases.

    Single words are deliberately excluded — "adapter", "strategy" and "observer" are ordinary
    engineering prose, and banning them would reject accurate descriptions of real systems.
    """
    phrases: list[str] = [record.name.replace("-", " ")]
    phrases.extend(alias.replace("-", " ") for alias in record.aliases)
    return tuple(
        phrase
        for phrase in dict.fromkeys(phrases)
        if len(phrase.split()) >= 2 and phrase.lower() not in GENERIC_ALIASES
    )


def leak_reasons(
    text: str,
    *,
    style: str,
    record: pattern_loader.JoinedPattern,
    style_leaks: Any,
    style_target: Any,
) -> list[str]:
    """Reasons a description/decision statement is refused as a string-matchable row.

    ``none`` is the abstention criterion: it has no served gloss and no everyday name to leak
    (the seam lists it as an option, not as a style), so only the pattern-phrase check applies.
    """
    reasons: list[str] = []
    if style != seam_prompts.STYLE_NONE_KEY:
        reasons.extend(style_leaks(text, style_target(style)))
    lowered = " ".join(text.lower().split())
    for phrase in _banned_phrases(record):
        if phrase.lower() in lowered:
            reasons.append(f"names the pattern phrase {phrase!r}")
    return reasons


@dataclass(frozen=True)
class PoolIssue:
    """One refused candidate, with every reason it failed."""

    slot_id: str
    scenario_id: str
    reasons: tuple[str, ...]


def build_scenario(
    brief: Mapping[str, Any],
    authored: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the corpus scenario for one authored row (all label facts from the brief)."""
    scenario_id = compose_scenario_id(str(brief["style"]), str(authored["topic"]))
    return {
        "scenario_id": scenario_id,
        "family": brief["family"],
        "category": brief["category"],
        "language": brief["language"],
        "description": " ".join(str(authored["description"]).split()),
        "decision_statement": " ".join(str(authored["decision_statement"]).split())[:400],
        "expected": {
            "primary": brief["primary"],
            "acceptable_primary": list(brief["acceptable_primary"]),
            "acceptable_supporting": list(brief["acceptable_supporting"]),
        },
        "split": brief["split"],
        "notes": " ".join(str(authored["notes"]).split())[:1000],
    }


def validate_pool(
    authored_rows: Sequence[Mapping[str, Any]],
    briefs: Mapping[str, Mapping[str, Any]],
    catalog: pattern_loader.PatternLoader,
    corpus: Sequence[Mapping[str, Any]],
    *,
    style_leaks: Any,
    style_target: Any,
) -> tuple[list[dict[str, Any]], list[PoolIssue]]:
    """Check every authored row; return the accepted scenarios and every refusal with reasons.

    Checks: the row answers a real slot exactly once, every field is present, the description is
    long enough to carry evidence, nothing names the style or the pattern, the scenario id is
    unique across the corpus and the pool, and the assembled record satisfies the corpus schema
    and catalog invariants (which ``tests.benchmark.harness.config.load_scenarios`` re-checks
    independently at apply time).
    """
    schema_validator = _scenario_validator()
    known_ids = {str(scenario["scenario_id"]) for scenario in corpus}
    seen: Counter[str] = Counter()
    scenarios: list[dict[str, Any]] = []
    issues: list[PoolIssue] = []
    for index, authored in enumerate(authored_rows):
        slot_id = str(authored.get("slot_id", f"#{index}"))
        brief = briefs.get(slot_id)
        reasons: list[str] = []
        if brief is None:
            issues.append(PoolIssue(slot_id, "", ("slot_id is not in the plan",)))
            continue
        missing = [field for field in AUTHOR_FIELDS if not str(authored.get(field, "")).strip()]
        if missing:
            issues.append(PoolIssue(slot_id, "", (f"missing fields: {missing}",)))
            continue
        scenario = build_scenario(brief, authored)
        scenario_id = str(scenario["scenario_id"])
        seen[scenario_id] += 1
        if scenario_id in known_ids:
            reasons.append("scenario_id collides with an existing corpus scenario")
        if seen[scenario_id] > 1:
            reasons.append("scenario_id repeats inside the pool (topic slug is not unique)")
        if len(str(scenario["description"])) < 240:
            reasons.append("description too short to carry style evidence (min 240 chars)")
        provenance = authored.get("provenance")
        if provenance not in PROVENANCE_KINDS:
            reasons.append(f"provenance must be one of {PROVENANCE_KINDS}")
        if brief["kind"].startswith("organic") and provenance == "synthetic":
            reasons.append("a slot planned as organic cannot be filled with a synthetic row")
        if authored.get("source_url") is not None and not str(authored["source_url"]).startswith("http"):
            reasons.append("source_url must be an absolute http(s) URL when present")
        # An organic row is a documentation-grounded reading; without the page it is not checkable,
        # and the >= 70 % organic share (§4.7.1) would rest on an unverifiable self-declaration.
        if provenance != "synthetic" and not str(authored.get("source_url", "")).startswith("http"):
            reasons.append("organic provenance requires the source_url that grounded the row")
        record = catalog.records[str(brief["primary"])]
        for field in ("description", "decision_statement"):
            reasons.extend(
                f"{field} {reason}"
                for reason in leak_reasons(
                    str(scenario[field]),
                    style=str(brief["style"]),
                    record=record,
                    style_leaks=style_leaks,
                    style_target=style_target,
                )
            )
        errors = sorted(schema_validator.iter_errors(scenario), key=lambda error: list(error.absolute_path))
        reasons.extend(
            f"schema {'/'.join(str(part) for part in error.absolute_path)}: {error.message}" for error in errors
        )
        if reasons:
            issues.append(PoolIssue(slot_id, scenario_id, tuple(reasons)))
            continue
        scenarios.append(
            {
                **scenario,
                "_pool": {
                    "slot_id": slot_id,
                    "kind": authored.get("kind", brief["kind"]),
                    "style": str(brief["style"]),
                    "provenance": str(authored["provenance"]),
                    "source_url": authored.get("source_url"),
                    "rationale": " ".join(str(authored["rationale"]).split()),
                },
            }
        )
    return scenarios, issues


def _scenario_validator() -> Any:
    """The corpus scenario schema validator (the harness's own copy of the contract)."""
    from tests.benchmark.harness import config as harness_config

    return harness_config._scenario_schema_validator()


def coverage_report(scenarios: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-split label counts, organic share and family counts of a validated pool."""
    train = [row for row in scenarios if row["split"] == "train"]
    holdout = [row for row in scenarios if row["split"] == "holdout"]
    train_labels = Counter(str(row["_pool"]["style"]) for row in train)
    holdout_labels = Counter(str(row["_pool"]["style"]) for row in holdout)
    organic_train = sum(1 for row in train if row["_pool"]["provenance"] != "synthetic")
    return {
        "train": len(train),
        "holdout": len(holdout),
        "train_organic": organic_train,
        "train_synthetic": len(train) - organic_train,
        "train_organic_share": round(organic_train / len(train), 4) if train else 0.0,
        "train_labels": dict(sorted(train_labels.items())),
        "holdout_labels": dict(sorted(holdout_labels.items())),
        "train_missing_labels": sorted(set(POOL_LABELS) - set(train_labels)),
        "holdout_missing_labels": sorted(set(POOL_LABELS) - set(holdout_labels)),
        "train_families": sorted({str(row["family"]) for row in train}),
        "holdout_families": sorted({str(row["family"]) for row in holdout}),
        "selection": {
            # Selection is stratified by served class with a per-class floor, so class coverage is
            # guaranteed by construction rather than by an uncertainty draw. Scoring every
            # candidate with the base checkpoint (uncertainty sampling) needs a served sidecar;
            # when one is reachable the scores are recorded here under ``uncertainty``.
            "method": "stratified-by-class-with-floor",
            "uncertainty_scored": False,
            "uncertainty_source": None,
        },
        "duplicate_band": [],
    }


def coverage_failures(report: Mapping[str, Any], *, expected: Mapping[str, int]) -> list[str]:
    """Reasons the pool fails its build contract (counts, coverage, organic share)."""
    failures: list[str] = []
    if report["train"] < expected["train"]:
        failures.append(f"train pool {report['train']} < target {expected['train']}")
    if report["holdout"] < expected["holdout"]:
        failures.append(f"holdout pool {report['holdout']} < target {expected['holdout']}")
    if report["train_organic_share"] < ORGANIC_SHARE_FLOOR:
        failures.append(f"organic share {report['train_organic_share']} < {ORGANIC_SHARE_FLOOR}")
    if report["train_missing_labels"]:
        failures.append(f"train labels with no candidate: {report['train_missing_labels']}")
    if report["holdout_missing_labels"]:
        failures.append(f"holdout labels with no candidate: {report['holdout_missing_labels']}")
    return failures


# ---------------------------------------------------------------------------------------
# Application: corpus, label source, derived artifacts
# ---------------------------------------------------------------------------------------


def load_corpus(path: Path) -> dict[str, Any]:
    """The corpus document (``{"_meta": ..., "scenarios": [...]}``)."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), list):
        raise SystemExit(f"{path} is not a corpus document")
    return payload


def read_style_labels(path: Path) -> dict[str, tuple[str, str]]:
    """``STYLE_LABELS`` as written in the label source (AST-read, never executed)."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "STYLE_LABELS":
            if not isinstance(node.value, ast.Dict):
                raise SystemExit("STYLE_LABELS is not a dict literal")
            labels: dict[str, tuple[str, str]] = {}
            for key, value in zip(node.value.keys, node.value.values, strict=True):
                if not isinstance(key, ast.Constant) or not isinstance(value, ast.Tuple) or len(value.elts) != 2:
                    raise SystemExit("STYLE_LABELS carries a non-literal entry")
                label, rationale = value.elts
                labels[str(key.value)] = (str(getattr(label, "value", "")), str(getattr(rationale, "value", "")))
            return labels
    raise SystemExit("no STYLE_LABELS dict found")


def insert_style_labels(source: str, entries: Mapping[str, tuple[str, str]], header: str) -> str:
    """Insert ``entries`` before the dict's closing brace, preserving every existing byte."""
    module = ast.parse(source)
    target: ast.Dict | None = None
    end_line = 0
    for node in ast.walk(module):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "STYLE_LABELS":
            if not isinstance(node.value, ast.Dict):
                raise SystemExit("STYLE_LABELS is not a dict literal")
            target = node.value
            end_line = node.value.end_lineno or 0
    if target is None or not end_line:
        raise SystemExit("no STYLE_LABELS dict found")
    lines = source.splitlines(keepends=True)
    close_index = end_line - 1
    if "}" not in lines[close_index]:
        raise SystemExit(f"unexpected STYLE_LABELS layout: line {end_line} holds no closing brace")
    block = [f"    # -- {header} (reviewed, two blind annotators + adjudication) --\n"]
    for scenario_id in sorted(entries):
        label, rationale = entries[scenario_id]
        block.append(f'    "{scenario_id}": (\n')
        block.append(f'        "{label}",\n')
        block.append(f"        {json.dumps(rationale)},\n")
        block.append("    ),\n")
    return "".join([*lines[:close_index], *block, *lines[close_index:]])


def choice_rows(
    corpus: Mapping[str, Any], criteria: Mapping[str, str], instructions: str, digest: str
) -> list[dict[str, Any]]:
    """The mirrored J2 category rows: one per corpus scenario, in corpus order."""
    rows: list[dict[str, Any]] = []
    for scenario in corpus["scenarios"]:
        rows.append(
            {
                "type": "choice",
                "id": f"{scenario['scenario_id']}:choice",
                "state": scenario["description"],
                "instructions": instructions,
                "criteria": dict(criteria),
                "label": scenario["category"],
                "source": {
                    "corpus": DEFAULT_CORPUS,
                    "corpus_sha12": digest,
                    "scenario_id": scenario["scenario_id"],
                    "split": scenario["split"],
                },
            }
        )
    return rows


def corpus_text(corpus: Mapping[str, Any]) -> str:
    """Serialize a corpus document exactly as the repository stores it (verified round-trip)."""
    return json.dumps(dict(corpus), indent=2, ensure_ascii=False) + "\n"


def apply_pool(
    accepted: Sequence[Mapping[str, Any]],
    *,
    corpus_path: Path,
    labels_path: Path,
    style_out: Path,
    choice_out: Path,
    header: str,
) -> dict[str, Any]:
    """Append accepted rows to the corpus and label source, then regenerate both artifacts."""
    corpus = load_corpus(corpus_path)
    existing = {str(scenario["scenario_id"]) for scenario in corpus["scenarios"]}
    labels = read_style_labels(labels_path)
    new_scenarios: list[dict[str, Any]] = []
    new_labels: dict[str, tuple[str, str]] = {}
    for row in accepted:
        scenario_id = str(row["scenario_id"])
        if scenario_id in existing:
            raise SystemExit(f"{scenario_id!r} is already in the corpus")
        if scenario_id in labels:
            raise SystemExit(f"{scenario_id!r} already carries a reviewed label")
        new_scenarios.append({key: value for key, value in row.items() if not key.startswith("_")})
        new_labels[scenario_id] = (str(row["_pool"]["style"]), str(row["_pool"]["rationale"]))
    corpus["scenarios"] = [*corpus["scenarios"], *new_scenarios]
    meta = corpus.setdefault("_meta", {})
    meta["authoring"] = (
        f"{meta.get('authoring', '')} ; 2026-11 style-label gap: {len(new_scenarios)} appended "
        "scenarios (style-label-gap pool, two blind annotators + adjudication, rubric "
        "qa/typed/style-rubric.md)"
    ).strip(" ;")
    corpus_path.write_text(corpus_text(corpus), encoding="utf-8")
    labels_path.write_text(insert_style_labels(labels_path.read_text(encoding="utf-8"), new_labels, header), "utf-8")
    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()[:12]
    style_count = _regenerate_style_rows(labels_path, corpus_path, style_out)
    criteria, instructions = _choice_preamble(choice_out)
    rows = choice_rows(corpus, criteria, instructions, digest)
    choice_out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return {
        "appended_scenarios": len(new_scenarios),
        "appended_labels": len(new_labels),
        "style_rows": style_count,
        "choice_rows": len(rows),
        "corpus_sha12": digest,
    }


def _choice_preamble(choice_out: Path) -> tuple[Mapping[str, str], str]:
    """The criteria glosses and instructions already used by ``choice.jsonl`` (row 1)."""
    for line in choice_out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            return row["criteria"], row["instructions"]
    raise SystemExit(f"{choice_out} carries no rows to copy the criteria preamble from")


def _regenerate_style_rows(labels_path: Path, corpus_path: Path, style_out: Path) -> int:
    """Run ``derive_style_labels.py`` against the updated corpus (one source of truth)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_derive_style_labels", labels_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {labels_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    exit_code = int(module.main(["--corpus", str(corpus_path), "--out", str(style_out)]) or 0)
    if exit_code != 0:
        raise SystemExit(f"derive_style_labels failed with exit {exit_code}")
    return sum(1 for line in style_out.read_text(encoding="utf-8").splitlines() if line.strip())


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def _draft_helpers() -> tuple[Any, Any]:
    """The harness leak machinery, reused so authoring and drafting share one rule set."""
    from tests.benchmark.harness import draft

    return draft.style_leaks, draft.style_target


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dicts (blank lines ignored)."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows as JSONL with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def cmd_briefs(args: argparse.Namespace) -> int:
    """Write the authoring plan: one brief per line, plus a per-batch index."""
    catalog = pattern_loader.load_catalog(Path(args.pattern_dir), Path(args.docs_dir))
    corpus = load_corpus(Path(args.corpus))["scenarios"]
    slots = build_slots(
        catalog,
        corpus,
        train_total=args.train_total,
        organic_total=args.organic_total,
        holdout_total=args.holdout_total,
    )
    out = Path(args.out)
    _write_jsonl(out, [slot.brief() for slot in slots])
    counts = Counter((slot.split, slot.kind) for slot in slots)
    print(json.dumps({"briefs": len(slots), "by_kind": {f"{k[0]}/{k[1]}": v for k, v in counts.items()}}, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate authored rows against the plan; write the accepted pool and a report."""
    catalog = pattern_loader.load_catalog(Path(args.pattern_dir), Path(args.docs_dir))
    corpus = load_corpus(Path(args.corpus))["scenarios"]
    briefs = {str(brief["slot_id"]): brief for brief in _load_jsonl(Path(args.briefs))}
    authored: list[dict[str, Any]] = []
    for path in args.pool:
        authored.extend(_load_jsonl(Path(path)))
    style_leaks, style_target = _draft_helpers()
    scenarios, issues = validate_pool(
        authored, briefs, catalog, corpus, style_leaks=style_leaks, style_target=style_target
    )
    report = coverage_report(scenarios)
    report["duplicate_band"] = _near_duplicate_band(scenarios)
    report["issues"] = [issue.__dict__ for issue in issues]
    report["authored"] = len(authored)
    if args.out:
        _write_jsonl(Path(args.out), scenarios)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "authored": len(authored),
                "accepted": len(scenarios),
                "refused": len(issues),
                "train": report["train"],
                "holdout": report["holdout"],
                "train_organic_share": report["train_organic_share"],
                "train_missing_labels": report["train_missing_labels"],
                "holdout_missing_labels": report["holdout_missing_labels"],
            },
            indent=2,
        )
    )
    return 0 if not args.strict or not issues else 1


def _near_duplicate_band(scenarios: Sequence[Mapping[str, Any]], limit: float = NEAR_DUPLICATE_BAND) -> list[Any]:
    """Pool pairs inside the near-duplicate band (reported so a human can look before applying)."""
    shingled = [(str(row["scenario_id"]), shingles(str(row["description"]), NGRAM)) for row in scenarios]
    found: list[dict[str, Any]] = []
    for index, (left_id, left) in enumerate(shingled):
        for right_id, right in shingled[index + 1 :]:
            similarity = jaccard(left, right)
            if similarity >= limit:
                found.append({"left": left_id, "right": right_id, "jaccard": round(similarity, 4)})
    return found[:200]


def cmd_dedup(args: argparse.Namespace) -> int:
    """The holdout leakage gate: near-duplicates vs every train row, and family overlap."""
    catalog = pattern_loader.load_catalog(Path(args.pattern_dir), Path(args.docs_dir))
    del catalog
    corpus = load_corpus(Path(args.corpus))["scenarios"]
    # Train side only: the gate asks whether a holdout row repeats a train row. Existing holdout
    # rows are the same side as the candidates, so comparing against them would report a holdout
    # family as an overlap with itself.
    corpus_train = [row for row in corpus if str(row["split"]) == "train"]
    train: list[tuple[str, str]] = [(str(row["scenario_id"]), str(row["description"])) for row in corpus_train]
    train_families = {str(row["family"]) for row in corpus_train}
    holdout_rows: list[tuple[str, str]] = []
    holdout_families: set[str] = set()
    holdout_ids: set[str] = set()
    for path in args.holdout:
        for row in _load_jsonl(Path(path)):
            if str(row.get("split")) != "holdout":
                continue  # a pool carries both splits; the holdout side is what is being carved
            holdout_rows.append((str(row["scenario_id"]), str(row["description"])))
            holdout_families.add(str(row["family"]))
            holdout_ids.add(str(row["scenario_id"]))
    for path in args.train_pool:
        for row in _load_jsonl(Path(path)):
            if str(row.get("split")) != "train":
                continue  # a pool carries both splits; only its train rows are the reference side
            train.append((str(row["scenario_id"]), str(row["description"])))
            train_families.add(str(row["family"]))
    pairs = near_duplicates(train, holdout_rows)
    collisions = sorted({name for name, _ in holdout_rows} & {name for name, _ in train if name in holdout_ids})
    overlap = sorted(holdout_families & train_families)
    result = {
        "holdout_rows": len(holdout_rows),
        "train_rows": len(train),
        "near_duplicate_pairs": [pair.__dict__ for pair in pairs],
        "id_collisions": collisions,
        "family_overlap": overlap,
        "jaccard_limit": JACCARD_LIMIT,
        "shared_substring_tokens": SHARED_SUBSTRING_TOKENS,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    failed = bool(pairs or collisions or overlap)
    print(
        json.dumps(
            {
                "holdout_rows": len(holdout_rows),
                "near_duplicate_pairs": len(pairs),
                "id_collisions": len(collisions),
                "family_overlap": overlap,
                "verdict": "FAIL" if failed else "PASS",
            },
            indent=2,
        )
    )
    return 1 if failed else 0


def cmd_apply(args: argparse.Namespace) -> int:
    """Append accepted rows to the corpus and label source, then regenerate derived artifacts."""
    accepted = _load_jsonl(Path(args.accepted))
    if args.split:
        accepted = [row for row in accepted if str(row.get("split")) == args.split]
    result = apply_pool(
        accepted,
        corpus_path=Path(args.corpus),
        labels_path=Path(args.labels),
        style_out=Path(args.style_out),
        choice_out=Path(args.choice_out),
        header=args.header,
    )
    print(json.dumps(result, indent=2))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line: subcommands share the catalog/corpus paths."""
    parser = argparse.ArgumentParser(description="Style-label gap: briefs, validation, leakage gate, apply.")
    parser.add_argument("--pattern-dir", default=str(REPO_ROOT / "pattern"))
    parser.add_argument("--docs-dir", default=str(REPO_ROOT / "docs"))
    sub = parser.add_subparsers(dest="command", required=True)

    briefs = sub.add_parser("briefs", help="write the deterministic authoring plan")
    briefs.add_argument("--corpus", default=DEFAULT_CORPUS)
    briefs.add_argument("--out", default="data/style-gap/briefs.jsonl")
    briefs.add_argument("--train-total", type=int, default=TRAIN_POOL_TOTAL)
    briefs.add_argument("--organic-total", type=int, default=TRAIN_ORGANIC_TOTAL)
    briefs.add_argument("--holdout-total", type=int, default=HOLDOUT_POOL_TOTAL)
    briefs.set_defaults(func=cmd_briefs)

    validate = sub.add_parser("validate", help="validate authored rows against the plan")
    validate.add_argument("--corpus", default=DEFAULT_CORPUS)
    validate.add_argument("--briefs", default="data/style-gap/briefs.jsonl")
    validate.add_argument("--pool", nargs="+", required=True)
    validate.add_argument("--out", default="data/style-gap/pool.jsonl")
    validate.add_argument("--report", default="data/style-gap/pool-report.json")
    validate.add_argument("--strict", action="store_true", help="exit non-zero when any row is refused")
    validate.set_defaults(func=cmd_validate)

    dedup = sub.add_parser("dedup", help="holdout leakage gate")
    dedup.add_argument("--corpus", default=DEFAULT_CORPUS)
    dedup.add_argument("--holdout", nargs="+", required=True)
    dedup.add_argument("--train-pool", nargs="*", default=[])
    dedup.add_argument("--report", default="data/style-gap/dedup.json")
    dedup.set_defaults(func=cmd_dedup)

    apply_cmd = sub.add_parser("apply", help="append accepted rows and regenerate artifacts")
    apply_cmd.add_argument("--accepted", required=True)
    apply_cmd.add_argument("--split", choices=("train", "holdout"), default=None)
    apply_cmd.add_argument("--corpus", default=DEFAULT_CORPUS)
    apply_cmd.add_argument("--labels", default=DEFAULT_LABELS)
    apply_cmd.add_argument("--style-out", default=DEFAULT_STYLE_OUT)
    apply_cmd.add_argument("--choice-out", default=DEFAULT_CHOICE_OUT)
    apply_cmd.add_argument("--header", default="style-label gap 2026-11")
    apply_cmd.set_defaults(func=cmd_apply)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the subcommand."""
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
