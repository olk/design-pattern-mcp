"""LLM-assisted corpus drafting from the catalog (plan §11: tooling ships, expansion is user-gated).

Produces **candidate** scenarios for the ``full.json`` corpus by asking the configured generator
for one scenario per (family, sibling group), grounded in the real catalog records of that
family. The output is a draft under ``data/benchmark-runs/drafts/`` — it is *never* committed:
the plan's workflow requires the user to review the labels before they join
``tests/benchmark/scenarios/full.json`` (expert-reviewed labels are the whole point of the
corpus).

Every candidate is repaired and re-validated deterministically before it is written:

- pattern ids must resolve to ``pattern/<id>.json`` records (unresolvable ids are dropped);
- ``primary`` is forced into ``acceptable_primary``; at least two distinct acceptable primaries
  with at least one sibling (same ``###`` subgroup) are required, else the candidate is dropped;
- the split is assigned by family (families sorted, last third → holdout), never by the model.

Style-coverage mode (``--style-criterion <key>``) drafts *style exemplars* instead of plain
family members: the described system is dominated by one Ch2 architecture style (the criterion
gloss from ``src/decisions/seam_prompts.py``), while the decision the scenario asks about stays
an ordinary catalog decision grounded in a rotating family. Those scenarios are always
``split == "train"`` and their family is ``<criterion>-style-systems``, so a family-disjoint
holdout is untouched by construction. The never-name rule extends to the style: a candidate
whose description or decision statement contains the criterion key, quotes the gloss, or uses
the style's everyday name is dropped by :func:`repair_scenario` exactly like an unresolvable id.

``--dry-run`` prints the prompt for the first family and writes nothing, so the tooling is
verifiable without a key or a provider.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field

from src.agent import build_llm_client
from src.decisions import seam_prompts
from src.patterns.loader import PatternLoader

from . import config as cfg

DEFAULT_DRAFT_OUT = cfg.DEFAULT_OUT_DIR / "drafts" / "full.draft.json"

#: Characters of catalog text per grounding record (keeps the prompt bounded).
_GROUNDING_CHARS = 420

#: Schema cap on ``scenario_id`` (``tests/benchmark/scenarios/scenario.schema.json``).
_SCENARIO_ID_MAX = 80


@dataclass(frozen=True)
class StyleTarget:
    """One Ch2 architecture style to draft exemplar *systems* for (style-coverage mode)."""

    criterion: str
    gloss: str
    leak_tokens: tuple[str, ...]
    clarifier: str = ""

    @property
    def style_family(self) -> str:
        """Reviewer-facing family label recorded in ``notes`` (never the corpus ``family``)."""
        return f"{self.criterion}-style-systems"


#: Disambiguation against the nearest sibling style in the served vocabulary. Two styles that
#: describe the same components must not produce interchangeable descriptions, or the head can
#: only learn arbitrary tie-breaks; an exemplar has to be *about* the criterion and not about its
#: neighbour. Only the pairs where the drafts actually collided are listed.
_STYLE_CLARIFIERS: dict[str, str] = {
    "saga": (
        "The long-running transaction crosses parties the system does not own (suppliers, carriers,"
        " payment or inventory providers reached over their APIs), and coordinating that transaction"
        " is what the system is for. Do not make the system itself a set of your own independently"
        " deployed services; keep the topology of the system under decision as the coordinating"
        " service plus its external participants, so the description is not a service-decomposition"
        " story."
    ),
    "lambda-architecture": (
        "Both a batch path that recomputes from the full history and a real-time path must be visible;"
        " if only one path computes, the system is the kappa sibling, not this style."
    ),
    "kappa-architecture": (
        "Exactly one ordered log and exactly one streaming computation path must exist; a nightly"
        " full recompute as a second path makes it the lambda sibling instead."
    ),
    "service-oriented-architecture": (
        "Expose capabilities as interoperable services through published contracts (WSDL/XSD-style"
        " interface descriptions) and a directory of endpoints; do not route the traffic through a"
        " central mediation tier, which is the enterprise-service-bus sibling."
    ),
    "enterprise-service-bus": (
        "The parties speak incompatible protocols and payload shapes and a central mediation tier"
        " translates and routes between them; a uniform contract for every participant is the"
        " service-oriented sibling instead."
    ),
    "modular-monolith": (
        "One deployable and one release, but with module boundaries that are real (own tables, own"
        " public interface, no internal imports across modules); a plain layered deployment without"
        " vertical modules is the monolithic sibling."
    ),
    "monolithic": (
        "One deployment unit organized as a single whole, with no vertical module boundaries and no"
        " separately releasable parts; that is what separates it from the modular-monolith sibling."
    ),
    "backend-for-frontend": (
        "Two or more distinct client kinds, each with its own dedicated backend service and no shared"
        " aggregation code; one single entry point for all clients is the api-gateway sibling."
    ),
    "api-gateway": (
        "One entry point that all clients share, with routing and cross-cutting concerns in it; a"
        " per-client backend is the backend-for-frontend sibling."
    ),
}


#: Everyday names of each style: what a candidate description may not use to *name* the style.
#: Structural evidence ("a central routing tier every unit publishes into") is the point of an
#: exemplar and stays allowed; the criterion key, its gloss verbatim, and these names are what a
#: reader could string-match on without reasoning. Bare words that appear in ordinary engineering
#: prose ("edge", "model", "lambda") are deliberately absent.
_STYLE_LEAK_TOKENS: dict[str, tuple[str, ...]] = {
    "event-driven": ("event-driven", "event driven", "event-oriented"),
    "broker": ("message broker", "broker-based", "broker"),
    "enterprise-service-bus": ("enterprise service bus", "service bus", "esb", "message bus"),
    "event-sourcing": ("event sourcing", "event-sourced", "append-only log of events"),
    "command-query-responsibility-segregation": ("cqrs", "command-query", "command query", "command/query"),
    "saga": ("saga",),
    "microservices": ("microservices", "microservice", "micro-services"),
    "service-oriented-architecture": ("service-oriented", "service oriented", "soa"),
    "hexagonal": ("hexagonal", "ports and adapters", "port-and-adapter"),
    "layered-monolith": ("layered monolith", "layered architecture", "n-tier", "n-tiered"),
    "modular-monolith": ("modular monolith", "modular-monolithic"),
    "monolithic": ("monolith", "monolithic"),
    "pipe-and-filter": ("pipe and filter", "pipe-and-filter", "pipes and filters"),
    "model-view-controller": ("model-view-controller", "model view controller", "mvc"),
    "presentation-abstraction-control": (
        "presentation-abstraction-control",
        "presentation abstraction control",
    ),
    "microkernel-plugin": ("microkernel", "micro-kernel", "plugin architecture", "plug-in architecture"),
    "reflection-architecture": ("reflection", "reflective", "metaprogramming"),
    "reactive-architecture": ("reactive",),
    "actor-based": ("actor model", "actor-based", "actor", "actors"),
    "half-sync-half-async": ("half-sync", "half sync", "half-async"),
    "task-control-architecture": ("task control", "task-control", "task control architecture"),
    "serverless": ("serverless", "server-less", "faas", "function-as-a-service"),
    "edge-computing": ("edge computing", "edge-computed", "edge nodes", "edge devices", "edge site"),
    "space-based": ("space-based", "space based", "tuple space", "in-memory data grid"),
    "data-mesh": ("data mesh", "data products", "data-as-a-product"),
    "lambda-architecture": ("lambda architecture", "lambda-arch", "speed layer", "batch layer"),
    "kappa-architecture": ("kappa architecture", "kappa-arch", "kappa"),
    "api-gateway": ("api gateway", "api-gateway", "gateway aggregation"),
    "backend-for-frontend": ("backend for frontend", "backend-for-frontend", "bff"),
    "service-mesh": ("service mesh", "service-mesh", "sidecar proxy", "sidecar"),
    "rule-based-system": ("rule-based", "rule based", "rules engine", "rule engine"),
    "aiml-centric": ("ai/ml", "aiml", "machine learning", "artificial intelligence", "ml model", "ai model"),
    "blockchain-based": ("blockchain", "block-chain", "distributed ledger"),
    "hybrid-cloud": ("hybrid cloud", "hybrid-cloud"),
    "multi-cloud": ("multi-cloud", "multicloud", "multi cloud"),
}


def style_target(criterion: str) -> StyleTarget:
    """Resolve a served style key to its gloss, leak tokens and sibling clarifier."""
    if criterion not in seam_prompts.STYLE_GLOSSES:
        raise KeyError(f"{criterion!r} is not a served style criterion")
    tokens = _STYLE_LEAK_TOKENS.get(criterion)
    if tokens is None:
        raise KeyError(f"no leak tokens declared for style criterion {criterion!r}")
    return StyleTarget(
        criterion,
        seam_prompts.STYLE_GLOSSES[criterion],
        tokens,
        _STYLE_CLARIFIERS.get(criterion, ""),
    )


def style_leaks(text: str, target: StyleTarget) -> list[str]:
    """Reasons ``text`` *names* the style instead of embodying it (empty list = clean)."""
    lowered = " ".join(text.lower().split())
    leaks: list[str] = []
    if target.criterion in lowered:
        leaks.append(f"names the criterion key {target.criterion!r}")
    if " ".join(target.gloss.lower().split()) in lowered:
        leaks.append("quotes the criterion gloss verbatim")
    for token in target.leak_tokens:
        if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", lowered):
            leaks.append(f"uses the style's everyday name {token!r}")
    return leaks


class DraftExpected(BaseModel):
    """The label block the model must produce."""

    primary: str = Field(default="", description="Expected primary pattern id (kebab-case).")
    acceptable_primary: list[str] = Field(default_factory=list, description="All acceptable primary ids.")
    acceptable_supporting: list[str] = Field(default_factory=list, description="Acceptable supporting ids.")


class DraftScenario(BaseModel):
    """One drafted scenario (before deterministic repair)."""

    scenario_id: str = Field(default="", description="Stable kebab-case id.")
    language: str | None = Field(default=None, description="Requested language (catalog enum) or null.")
    description: str = Field(default="", description="The request text; never names the pattern.")
    decision_statement: str = Field(default="", description="The single decision, <= 400 chars.")
    expected: DraftExpected = Field(default_factory=DraftExpected)
    notes: str | None = Field(default=None, description="Why these labels (for the reviewer).")


class DraftScenarioList(BaseModel):
    """The structured response of one drafting call."""

    scenarios: list[DraftScenario] = Field(default_factory=list)


@dataclass
class Family:
    """One catalog family (``###`` subgroup) with its grounding records."""

    subgroup: str
    category: str
    names: list[str] = field(default_factory=list)


def family_index(catalog: PatternLoader) -> list[Family]:
    """Catalog records grouped by ``###`` subgroup, in canonical catalog order."""
    families: dict[str, Family] = {}
    for name in catalog.order:
        record = catalog.records[name]
        family = families.setdefault(
            record.subgroup or "(unsectioned)",
            Family(subgroup=record.subgroup or "(unsectioned)", category=record.category),
        )
        family.names.append(name)
    return sorted(families.values(), key=lambda family: (family.category, family.subgroup))


SYSTEM_PROMPT = (
    "You author evaluation scenarios for a design-pattern selection benchmark. "
    "You write realistic engineering requests: named components, responsibilities, cross-component "
    "flows and concrete constraints. You never name, hint at or paraphrase the catalog pattern you "
    "label as the expected answer — the description must be answerable only by reasoning about the "
    "decision, not by string matching. You answer with the requested JSON object only."
)

#: The style-mode system prompt: the *system shape* is specified, so the never-name rule also
#: covers the architecture style, not just the catalog pattern.
STYLE_SYSTEM_PROMPT = (
    SYSTEM_PROMPT + " The system's architecture style is given to you as a definition: make the system embody that"
    " definition structurally, and never name the style itself — not its key, not the definition's"
    " wording, and not its everyday name. A reviewer must be able to infer the style from the"
    " components, communication paths and deployment alone."
)


def build_system_prompt(style: StyleTarget | None = None) -> str:
    """The system prompt for one drafting call (style mode adds the style never-name clause)."""
    return STYLE_SYSTEM_PROMPT if style is not None else SYSTEM_PROMPT


def build_user_prompt(
    family: Family,
    grounding: list[dict[str, str]],
    per_family: int,
    language_hint: str | None,
    style: StyleTarget | None = None,
) -> str:
    """The user prompt for one family (also printed by ``--dry-run``)."""
    lines: list[str] = []
    if style is not None:
        lines += [
            f"Architecture-style exemplar batch: {style.criterion}",
            f"Definition of that style: {style.gloss}",
            "",
            "The system you describe IS an instance of that style: its components, their communication",
            "and its deployment must embody the definition, and nothing in the text may name it — not",
            f"the key {style.criterion!r}, not the definition's wording, and not its everyday names",
            f"({', '.join(style.leak_tokens)}). The decision the scenario asks about is an ordinary",
            "design-pattern decision inside that system and may concern any catalog pattern; it is not",
            "a question about the style.",
        ]
        if style.clarifier:
            lines += ["", f"Keep it distinct from the neighbouring styles: {style.clarifier}"]
        lines.append("")
    lines += [
        f"Family (catalog subsection): {family.subgroup}",
        f"Catalog category of this family: {family.category}",
        f"Write {per_family} distinct scenarios about decisions inside this family.",
        "",
        "Catalog records in this family (ground the request in the *problem*, never in the name):",
    ]
    for record in grounding:
        lines += [f"- id: {record['name']}", f"  intent: {record['intent']}", f"  problem: {record['problem']}"]
        if record.get("use_cases"):
            lines.append(f"  use cases: {record['use_cases']}")
        if record.get("avoid_when"):
            lines.append(f"  avoid when: {record['avoid_when']}")
    lines += [
        "",
        "Rules for every scenario:",
        "- description: 600-1500 characters, plain ASCII, a concrete system with named components and",
        "  constraints; mention no pattern name, no pattern alias, and no book author.",
        "- decision_statement: at most 400 characters, naming the ONE decision the description is about,",
        "  again without naming the pattern.",
        f"- language: one of {language_hint or 'generic'} or null for language-neutral.",
        "- expected.primary: the single best catalog id from the grounding list above.",
        "- expected.acceptable_primary: primary plus at least one sibling id from the SAME family",
        "  (near misses a reviewer could also accept); at least 2 ids.",
        "- expected.acceptable_supporting: 1-2 ids from the grounding list that plausibly support the",
        "  solution (may be empty).",
        "- scenario_id: a short kebab-case id prefixed with the family topic.",
        "- notes: one sentence explaining to a reviewer why these labels fit.",
        "",
        'Answer as {"scenarios": [ {scenario_id, language, description, decision_statement, expected:',
        "  {primary, acceptable_primary, acceptable_supporting}, notes}, ... ]}.",
    ]
    return "\n".join(lines)


def _grounding_records(family: Family, catalog: PatternLoader) -> list[dict[str, str]]:
    """Compact catalog text for the family's records (bounded per record)."""
    records: list[dict[str, str]] = []
    for name in family.names:
        record = catalog.records[name]
        records.append(
            {
                "name": name,
                "intent": record.intent[:_GROUNDING_CHARS],
                "problem": record.problem[:_GROUNDING_CHARS],
                "use_cases": "; ".join(record.use_cases[:3])[:_GROUNDING_CHARS],
                "avoid_when": "; ".join(record.avoid_when[:2])[:200],
            }
        )
    return records


def assign_split(family_position: int, family_count: int) -> str:
    """Family-disjoint split: the last third of the (sorted) families becomes holdout."""
    if family_count >= 3:
        holdout_from = family_count - max(1, family_count // 3)
        return "holdout" if family_position >= holdout_from else "train"
    return "train"


def repair_scenario(
    draft: DraftScenario,
    *,
    family: Family,
    catalog: PatternLoader,
    split: str,
    language_hint: str | None,
    style: StyleTarget | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Deterministically repair one drafted scenario; return (record, rejection reasons)."""
    reasons: list[str] = []
    known = catalog.records
    primary = draft.expected.primary.strip()
    if primary not in known:
        return None, [f"primary {primary!r} is not a catalog record"]
    acceptable: list[str] = []
    for raw_name in [primary, *draft.expected.acceptable_primary]:
        candidate_name = raw_name.strip()
        if candidate_name and candidate_name in known and candidate_name not in acceptable:
            acceptable.append(candidate_name)
    primary_subgroup = known[primary].subgroup
    siblings = [name for name in acceptable if name != primary and known[name].subgroup == primary_subgroup]
    if len(acceptable) < 2 or not siblings:
        return None, ["fewer than two distinct acceptable primaries with a same-family sibling"]
    description = " ".join(draft.description.split())
    if len(description) < 40:
        return None, [f"description too short ({len(description)} chars)"]
    description = description[:10_000]
    decision = " ".join(draft.decision_statement.split())[:400] or description[:200]
    if style is not None:
        leaks = [
            *(f"description {leak}" for leak in style_leaks(description, style)),
            *(f"decision_statement {leak}" for leak in style_leaks(decision, style)),
        ]
        if leaks:
            return None, leaks
    scenario_id = _compose_scenario_id(
        draft.scenario_id or family.subgroup, style.criterion if style is not None else None
    )
    language = draft.language or language_hint or None
    if language == "generic":
        language = None
    if language is not None and language not in catalog.enums.languages:
        reasons.append(f"language {language!r} unknown; recorded as null (generic mode)")
        language = None
    supporting = [
        name.strip()
        for name in draft.expected.acceptable_supporting
        if name.strip() in known and name.strip() not in acceptable
    ]
    hints = {"architecture_style": None, "paradigm": None, "focus": None}
    notes = (draft.notes or "").strip()[:1000]
    if style is not None:
        # The corpus ``family`` must stay the primary's catalog subgroup (harness invariant), so
        # the style a scenario exemplifies is recorded here instead.
        notes = (notes + " " if notes else "") + f"[style exemplar: {style.style_family}]"
        notes = notes[:1000]
    record = {
        "scenario_id": scenario_id,
        "family": family.subgroup,
        "category": known[primary].category,
        "language": language,
        "description": description,
        "decision_statement": decision,
        "hints": hints,
        "expected": {
            "primary": primary,
            "acceptable_primary": acceptable,
            "acceptable_supporting": supporting[:3],
        },
        "split": split,
        "notes": notes or "LLM draft — reviewer must confirm the labels",
    }
    if len(supporting) > 3:
        reasons.append(f"acceptable_supporting truncated from {len(supporting)} to 3 ids")
    return record, reasons


def _kebab(text: str) -> str:
    """Lowercase kebab-case token (schema-compatible scenario/family ids)."""
    cleaned = "".join(character if character.isalnum() else "-" for character in text.strip().lower())
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts)


def _compose_scenario_id(topic: str, prefix: str | None = None) -> str:
    """Kebab scenario id, optionally prefixed, always inside the schema's 80-char cap."""
    slug = _kebab(topic) or "draft"
    if prefix and not (slug == prefix or slug.startswith(f"{prefix}-")):
        slug = f"{prefix}-{slug}"
    if len(slug) <= _SCENARIO_ID_MAX:
        return slug
    head = slug[:_SCENARIO_ID_MAX].rsplit("-", 1)[0]
    return head or slug[:_SCENARIO_ID_MAX]


def _unique_scenario_id(candidate: str, seen: set[str]) -> str:
    """``candidate``, or a ``-<n>`` variant no other record in this draft file uses."""
    if candidate not in seen:
        seen.add(candidate)
        return candidate
    for attempt in range(2, 100):
        suffix = f"-{attempt}"
        trimmed = candidate[: _SCENARIO_ID_MAX - len(suffix)].rstrip("-")
        unique = f"{trimmed}{suffix}"
        if unique not in seen:
            seen.add(unique)
            return unique
    raise ValueError(f"cannot make scenario id unique: {candidate!r}")


def _style_key(value: str) -> str:
    """argparse type: a served Ch2 style criterion key."""
    if value not in seam_prompts.STYLE_GLOSSES:
        raise argparse.ArgumentTypeError(f"{value!r} is not a served style criterion")
    return value


def build_parser() -> argparse.ArgumentParser:
    """CLI of the drafting tool."""
    parser = argparse.ArgumentParser(
        prog="tests.benchmark.harness.draft",
        description="Draft full.json scenario candidates from catalog records (output needs user review).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_DRAFT_OUT, help="draft corpus path")
    parser.add_argument("--per-family", type=int, default=2, help="scenarios requested per family (2-3)")
    parser.add_argument("--families", default=None, help="only families whose name contains this substring")
    parser.add_argument("--limit-families", type=int, default=None, help="stop after N families")
    parser.add_argument("--language", default="python", help="default language hint for drafted scenarios")
    parser.add_argument(
        "--style-criterion",
        action="append",
        default=[],
        type=_style_key,
        metavar="KEY",
        help="draft style exemplars for this Ch2 architecture style (repeatable; forces train-only)",
    )
    parser.add_argument("--style-count", type=int, default=3, help="exemplars per style criterion (2-4)")
    parser.add_argument("--train-only", action="store_true", help="assign every drafted scenario to the train split")
    parser.add_argument("--dry-run", action="store_true", help="print the first prompt and exit (no LLM call)")
    return parser


def _style_families(args: argparse.Namespace, families: list[Family]) -> list[Family]:
    """Sibling-capable train-split families only: a style exemplar may never join a holdout family.

    ``family`` must stay the ``###`` subgroup of the primary (``harness/config.py``
    ``_assert_labels_resolvable``) and the split must stay family-disjoint, so the grounding
    family — not a synthetic ``<criterion>-style-systems`` name — carries both invariants. The
    split is computed over ``families`` *before* any ``--families``/``--limit-families`` filter,
    and a single-record family is dropped because ``repair_scenario`` needs a same-subgroup
    sibling that such a family cannot supply.
    """
    train = [family for position, family in enumerate(families) if assign_split(position, len(families)) == "train"]
    selected = [
        family
        for family in train
        if len(family.names) >= 2 and (args.families is None or args.families.lower() in family.subgroup.lower())
    ]
    return selected[: args.limit_families] if args.limit_families is not None else selected


def _draft_style_exemplars(
    args: argparse.Namespace,
    targets: Sequence[StyleTarget],
    catalog: PatternLoader,
    families: list[Family],
    language_hint: str | None,
    agent: Any,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """One drafting call per exemplar, rotating the grounding family across style criteria."""
    ground = _style_families(args, families)
    if not ground:
        raise SystemExit("no train-split families selected (style exemplars must stay train-only)")
    order = list(seam_prompts.STYLE_GLOSSES)
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    rejected: list[str] = []
    per_style: dict[str, int] = {}
    for target in targets:
        offset = order.index(target.criterion) * args.style_count
        for attempt in range(args.style_count):
            family = ground[(offset + attempt) % len(ground)]
            label = f"{target.criterion}/{family.subgroup}"
            prompt = build_user_prompt(family, _grounding_records(family, catalog), 1, language_hint, style=target)
            try:
                answer: DraftScenarioList = _run_structured(agent, prompt, style=target)
            except Exception as exc:  # noqa: BLE001 - one bad call must not abort the batch
                rejected.append(f"{label}: drafting call failed ({type(exc).__name__}: {exc})")
                continue
            if not answer.scenarios:
                rejected.append(f"{label}: empty response")
                continue
            draft = answer.scenarios[0]
            record, reasons = repair_scenario(
                draft, family=family, catalog=catalog, split="train", language_hint=language_hint, style=target
            )
            if record is None:
                rejected.extend(f"{label}: {reason}" for reason in reasons)
                continue
            record["scenario_id"] = _unique_scenario_id(record["scenario_id"], seen)
            records.append(record)
            per_style[target.criterion] = per_style.get(target.criterion, 0) + 1
            rejected.extend(f"{label}: {reason}" for reason in reasons)
    summary: dict[str, Any] = {
        "style_criteria": [target.criterion for target in targets],
        "style_count": args.style_count,
        "grounding_families": len(ground),
        "per_style": per_style,
    }
    return records, rejected, summary


def _draft_family_scenarios(
    args: argparse.Namespace,
    catalog: PatternLoader,
    families: list[Family],
    language_hint: str | None,
    agent: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """One drafting call per catalog family, split assigned by family (or forced train)."""
    records: list[dict[str, Any]] = []
    rejected: list[str] = []
    for position, family in enumerate(families):
        split = "train" if args.train_only else assign_split(position, len(families))
        prompt = build_user_prompt(
            family, _grounding_records(family, catalog), max(2, min(3, args.per_family)), language_hint
        )
        try:
            answer: DraftScenarioList = _run_structured(agent, prompt)
        except Exception as exc:  # noqa: BLE001 - a failed family must not abort the whole draft
            rejected.append(f"{family.subgroup}: drafting call failed ({type(exc).__name__}: {exc})")
            continue
        for draft in answer.scenarios:
            record, reasons = repair_scenario(
                draft, family=family, catalog=catalog, split=split, language_hint=language_hint
            )
            if record is None:
                rejected.append(f"{family.subgroup}: {reasons[0]}")
                continue
            records.append(record)
            for reason in reasons:
                rejected.append(f"{family.subgroup}: {reason}")
    return records, rejected


def _print_dry_run(
    args: argparse.Namespace,
    targets: Sequence[StyleTarget],
    catalog: PatternLoader,
    families: list[Family],
    language_hint: str | None,
) -> None:
    """Print the first prompt for the selected mode and write nothing."""
    if targets:
        target = targets[0]
        ground = _style_families(args, families)
        family = ground[list(seam_prompts.STYLE_GLOSSES).index(target.criterion) % len(ground)]
        print(
            f"# style criteria: {len(targets)}; exemplars per criterion: {args.style_count}; "
            f"grounding families: {len(ground)} (first: {family.subgroup})",
            file=sys.stderr,
        )
        print(build_system_prompt(target))
        print()
        print(build_user_prompt(family, _grounding_records(family, catalog), 1, language_hint, style=target))
        return
    family = families[0]
    print(f"# families: {len(families)} (first: {family.subgroup})", file=sys.stderr)
    print(build_system_prompt())
    print()
    print(
        build_user_prompt(family, _grounding_records(family, catalog), max(2, min(3, args.per_family)), language_hint)
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: draft candidate scenarios and write the review file."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 2 <= args.style_count <= 4:
        parser.error("--style-count must be between 2 and 4")
    targets = [style_target(key) for key in args.style_criterion]
    config = cfg.load_server_config("live")
    catalog = cfg.load_repo_catalog(config)
    families = family_index(catalog)
    if not targets:
        # Style mode keeps the full order: _style_families derives the train split before filtering.
        if args.families:
            families = [family for family in families if args.families.lower() in family.subgroup.lower()]
        if args.limit_families is not None:
            families = families[: args.limit_families]
    if not families:
        print("no families selected", file=sys.stderr)
        return 2
    language_hint = args.language if args.language in catalog.enums.languages else None
    if args.dry_run:
        _print_dry_run(args, targets, catalog, families, language_hint)
        return 0

    agent: Any = build_llm_client(config)  # runtime DesignAgent; the protocol has no generic call
    records: list[dict[str, Any]] = []
    rejected: list[str] = []
    style_summary: dict[str, Any] = {}
    if targets:
        records, rejected, style_summary = _draft_style_exemplars(
            args, targets, catalog, families, language_hint, agent
        )
    else:
        records, rejected = _draft_family_scenarios(args, catalog, families, language_hint, agent)

    payload = {
        "_meta": {
            "corpus": "full.draft",
            "harness_version": cfg.HARNESS_VERSION,
            "generator": config.generator.config.model,
            "families": len(families),
            "per_family": args.per_family,
            "generated_at": cfg.utc_now_iso(),
            "workflow": (
                "review the labels, then copy the reviewed scenarios into "
                "tests/benchmark/scenarios/full.json (the draft file lives under data/ and is never committed)"
            ),
            "rejections": rejected,
            **style_summary,
        },
        "scenarios": records,
    }
    cfg.write_json(args.out, payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "drafted": len(records),
                "rejected": len(rejected),
                "families": len(families),
                "splits": {
                    split: sum(1 for record in records if record["split"] == split) for split in ("train", "holdout")
                },
            },
            indent=2,
        )
    )
    return 0


def _run_structured(agent: Any, user_prompt: str, style: StyleTarget | None = None) -> DraftScenarioList:
    """One structured drafting call against the production agent."""

    async def _call() -> DraftScenarioList:
        answer = await agent.generate_structured(
            build_system_prompt(style),
            user_prompt,
            DraftScenarioList,
            operation="benchmark-draft",
        )
        return cast("DraftScenarioList", answer)

    return asyncio.run(_call())


if __name__ == "__main__":
    sys.exit(main())
