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

``--dry-run`` prints the prompt for the first family and writes nothing, so the tooling is
verifiable without a key or a provider.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field

from src.agent import build_llm_client
from src.patterns.loader import PatternLoader

from . import config as cfg

DEFAULT_DRAFT_OUT = cfg.DEFAULT_OUT_DIR / "drafts" / "full.draft.json"

#: Characters of catalog text per grounding record (keeps the prompt bounded).
_GROUNDING_CHARS = 420


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
            record.subgroup or "(unsectioned)", Family(subgroup=record.subgroup or "(unsectioned)", category=record.category)
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


def build_user_prompt(family: Family, grounding: list[dict[str, str]], per_family: int, language_hint: str | None) -> str:
    """The user prompt for one family (also printed by ``--dry-run``)."""
    lines = [
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
    draft: DraftScenario, *, family: Family, catalog: PatternLoader, split: str, language_hint: str | None
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
    scenario_id = _kebab(draft.scenario_id) or f"{_kebab(family.subgroup)}-draft"
    language = (draft.language or language_hint or None)
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
        "notes": (draft.notes or "").strip()[:1000] or "LLM draft — reviewer must confirm the labels",
    }
    if len(supporting) > 3:
        reasons.append(f"acceptable_supporting truncated from {len(supporting)} to 3 ids")
    return record, reasons


def _kebab(text: str) -> str:
    """Lowercase kebab-case token (schema-compatible scenario/family ids)."""
    cleaned = "".join(character if character.isalnum() else "-" for character in text.strip().lower())
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts)[:80]


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
    parser.add_argument("--dry-run", action="store_true", help="print the first prompt and exit (no LLM call)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: draft candidate scenarios and write the review file."""
    args = build_parser().parse_args(argv)
    config = cfg.load_server_config("live")
    catalog = cfg.load_repo_catalog(config)
    families = family_index(catalog)
    if args.families:
        families = [family for family in families if args.families.lower() in family.subgroup.lower()]
    if args.limit_families is not None:
        families = families[: args.limit_families]
    if not families:
        print("no families selected", file=sys.stderr)
        return 2
    language_hint = args.language if args.language in catalog.enums.languages else None
    if args.dry_run:
        family = families[0]
        print(f"# families: {len(families)} (first: {family.subgroup})", file=sys.stderr)
        print(SYSTEM_PROMPT)
        print()
        print(build_user_prompt(family, _grounding_records(family, catalog), max(2, min(3, args.per_family)), language_hint))
        return 0

    agent: Any = build_llm_client(config)  # runtime DesignAgent; the protocol has no generic call
    records: list[dict[str, Any]] = []
    rejected: list[str] = []
    for position, family in enumerate(families):
        split = assign_split(position, len(families))
        prompt = build_user_prompt(family, _grounding_records(family, catalog), max(2, min(3, args.per_family)), language_hint)
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
                    split: sum(1 for record in records if record["split"] == split)
                    for split in ("train", "holdout")
                },
            },
            indent=2,
        )
    )
    return 0


def _run_structured(agent: Any, user_prompt: str) -> DraftScenarioList:
    """One structured drafting call against the production agent."""

    async def _call() -> DraftScenarioList:
        answer = await agent.generate_structured(
            SYSTEM_PROMPT,
            user_prompt,
            DraftScenarioList,
            operation="benchmark-draft",
        )
        return cast("DraftScenarioList", answer)

    return asyncio.run(_call())


if __name__ == "__main__":
    sys.exit(main())
