"""Style-exemplar drafting (``draft.py --style-criterion``): the admission rules that keep a
style exemplar from becoming a string-matching answer.

The corpus rows this mode drafts are the train positives for Ch2 style criteria; a description
that *names* the style teaches the head to read a label instead of a topology, and a scenario id
without the criterion prefix makes the exemplar findable by id rather than by reasoning. Both
rules are enforced in :func:`tests.benchmark.harness.draft.repair_scenario`, so they are
asserted here against the real catalog (a namespace check, no LLM call, no side effects).
"""

from __future__ import annotations

from collections import Counter

from typing import Any

import pytest

from src.decisions import seam_prompts
from tests.benchmark.harness import config as cfg
from tests.benchmark.harness import draft as draft_mod
from tests.unit.conftest import REPO_ROOT


@pytest.fixture(scope="module")
def catalog() -> Any:
    """The repo catalog (needed by ``repair_scenario`` to resolve label ids)."""
    return cfg.load_repo_catalog(cfg.load_server_config("live"))


@pytest.fixture(scope="module")
def family(catalog: Any) -> Any:
    """One train-split grounding family (the decision home of a drafted exemplar)."""
    return draft_mod.family_index(catalog)[0]


def _draft(description: str) -> draft_mod.DraftScenario:
    """A drafted scenario shaped like the model's answer, with a resolvable sibling pair."""
    return draft_mod.DraftScenario(
        scenario_id="trip-holds",
        language="python",
        description=description,
        decision_statement="Which structure keeps the coordinator independent of provider call details?",
        expected=draft_mod.DraftExpected(
            primary="strategy",
            acceptable_primary=["strategy", "template-method"],
            acceptable_supporting=[],
        ),
        notes="draft for the test",
    )


def test_every_served_style_has_leak_tokens_and_a_target() -> None:
    """A criterion added to ``STYLE_GLOSSES`` without leak tokens would draft unchecked rows."""
    missing = [key for key in seam_prompts.STYLE_GLOSSES if key not in draft_mod._STYLE_LEAK_TOKENS]
    assert missing == []
    for key in seam_prompts.STYLE_GLOSSES:
        target = draft_mod.style_target(key)
        assert target.criterion == key
        assert target.gloss == seam_prompts.STYLE_GLOSSES[key]
        assert target.leak_tokens
    with pytest.raises(KeyError):
        draft_mod.style_target("no-such-style")


def test_style_leaks_flags_the_key_the_gloss_and_the_everyday_name() -> None:
    """The three ways a draft can name the style instead of embodying it."""
    target = draft_mod.style_target("saga")
    assert draft_mod.style_leaks("A trip saga coordinates providers", target)
    assert draft_mod.style_leaks(f"Note: {target.gloss.lower()}", target)
    assert draft_mod.style_leaks("The Saga follows each hold with a cancel", target)
    # Structural evidence of the same style stays admissible: it names no criterion.
    structural = (
        "One planning service drives holds against four external providers and reverses earlier "
        "holds through their cancel endpoints when a later step is refused."
    )
    assert draft_mod.style_leaks(structural, target) == []


def test_repair_drops_a_draft_that_names_its_style(family: Any, catalog: Any) -> None:
    """A leak is a rejection, exactly like an unresolvable label id."""
    target = draft_mod.style_target("saga")
    record, reasons = draft_mod.repair_scenario(
        _draft("Wanderlust runs a saga across four providers and reverses holds on refusal."),
        family=family,
        catalog=catalog,
        split="train",
        language_hint=None,
        style=target,
    )
    assert record is None
    assert any("saga" in reason for reason in reasons)


def test_repair_prefixes_the_id_and_keeps_the_catalog_family(family: Any, catalog: Any) -> None:
    """``harness/config.py`` requires family == primary subgroup; the criterion rides the id."""
    target = draft_mod.style_target("saga")
    record, reasons = draft_mod.repair_scenario(
        _draft(
            "Wanderlust runs one planning service that holds seats, rooms and a card charge "
            "against four external providers and reverses earlier holds when a later step fails."
        ),
        family=family,
        catalog=catalog,
        split="train",
        language_hint=None,
        style=target,
    )
    assert reasons == []
    assert record is not None
    assert record["scenario_id"] == "saga-trip-holds"
    assert record["family"] == catalog.records["strategy"].subgroup
    assert record["split"] == "train"
    assert "saga-style-systems" in record["notes"]


def test_compose_scenario_id_respects_the_schema_cap() -> None:
    """The schema caps ``scenario_id`` at 80 characters, prefix included."""
    long_id = draft_mod._compose_scenario_id(
        "a-very-long-topic-name-that-keeps-going-with-many-words-to-overflow-the-limit",
        "command-query-responsibility-segregation",
    )
    assert len(long_id) <= 80
    assert long_id.startswith("command-query-responsibility-segregation-")
    assert draft_mod._compose_scenario_id("trip-holds", "saga") == "saga-trip-holds"
    assert draft_mod._compose_scenario_id("saga-trip-holds", "saga") == "saga-trip-holds"


#: Served styles whose only train labels the 2026-11 audit corrected away, and the review that
#: closed the gap. The dict must stay empty now: the style-gap corpus work (docs/tacit-fine-tune.md
#: §4.7.1) added 25 reviewed api-gateway train exemplars, so any remaining gap is a regression.
_UNCOVERED_TRAIN_STYLES: dict[str, str] = {}

#: §4.7.1 data precondition on the holdout: `>= 100` family-disjoint rows, so a lowered holdout
#: fails here rather than silently weakening every calibration against the seam.
_HOLDOUT_FLOOR = 100
#: §4.7.1 precondition on train volume, and the floor every single class must clear.
_TRAIN_FLOOR = 500
_TRAIN_PER_CLASS_FLOOR = 10


def test_corpus_style_exemplars_carry_one_train_positive_per_criterion() -> None:
    """Every served style has a train positive, and the corpus meets the §4.7.1 volume floors."""
    styles = [
        line for line in (REPO_ROOT / "qa" / "typed" / "style.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    import json

    rows = [json.loads(line) for line in styles]
    train = [row for row in rows if row["source"]["split"] == "train"]
    holdout = [row for row in rows if row["source"]["split"] == "holdout"]
    uncovered = set(seam_prompts.STYLE_GLOSSES) - {row["label"] for row in train}
    assert uncovered <= set(_UNCOVERED_TRAIN_STYLES), f"new styles without a train positive: {uncovered}"
    assert len(train) >= _TRAIN_FLOOR, f"train style labels {len(train)} < {_TRAIN_FLOOR}"
    assert len(holdout) >= _HOLDOUT_FLOOR, f"holdout rows {len(holdout)} < {_HOLDOUT_FLOOR}"
    per_class = Counter(row["label"] for row in train)
    thin = {label: count for label, count in per_class.items() if count < _TRAIN_PER_CLASS_FLOOR}
    assert thin == {}, f"train styles below {_TRAIN_PER_CLASS_FLOOR} rows: {thin}"
