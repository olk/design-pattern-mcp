"""Style-label gap tooling: brief planning, pool validation, the leakage gate and certification.

The gap tooling decides what the reviewed style corpus becomes, so a bug here silently weakens
every downstream measurement (a leaked label, a duplicated holdout row, a ledger that certifies
labels nobody reviewed). These tests pin the four behaviours that make it trustworthy against a
real catalog: slots are deterministic and label-complete, the validator refuses prose that names
its own label, the leakage gate catches a reworded copy, and certification fails closed on an
unreviewed label or a verdict whose label changed since review.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from src.decisions import seam_prompts
from src.patterns import loader as pattern_loader
from src.style_affinity import STYLE_AFFINITY
from tests.benchmark.harness import draft as draft_mod
from tests.unit.conftest import REPO_ROOT

SCRIPTS = REPO_ROOT / "scripts" / "decision_models"
RUBRIC = REPO_ROOT / "qa" / "typed" / "style-rubric.md"


def _load(name: str, filename: str) -> types.ModuleType:
    """Import one script by path (registered in sys.modules so dataclasses resolve)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gap() -> types.ModuleType:
    """The pool/validation/leakage/apply tooling."""
    return _load("style_label_gap", "style_label_gap.py")


@pytest.fixture(scope="module")
def review() -> types.ModuleType:
    """The blind-pack/agreement/certification tooling."""
    _load("style_label_gap", "style_label_gap.py")
    return _load("style_label_review", "style_label_review.py")


@pytest.fixture(scope="module")
def catalog() -> Any:
    """The repository catalog through the production loader."""
    return pattern_loader.load_catalog(REPO_ROOT / "pattern", REPO_ROOT / "docs")


@pytest.fixture(scope="module")
def corpus() -> list[dict[str, Any]]:
    """The real benchmark corpus (169 scenarios before the gap expansion)."""
    payload = json.loads((REPO_ROOT / "tests" / "benchmark" / "scenarios" / "full.json").read_text(encoding="utf-8"))
    return list(payload["scenarios"])


# ---------------------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------------------


def test_rubric_glosses_match_the_served_prompt() -> None:
    """The annotation sheet must quote the served vocabulary, or annotators label a different seam."""
    text = RUBRIC.read_text(encoding="utf-8")
    bullets = {
        line.split("`")[1]: line.split(" — ", 1)[1].strip()
        for line in text.splitlines()
        if line.startswith("- `") and " — " in line
    }
    expected = {key: seam_prompts.STYLE_GLOSSES[key] for key in STYLE_AFFINITY}
    expected[seam_prompts.STYLE_NONE_KEY] = seam_prompts.STYLE_NONE_GLOSS
    assert bullets == expected


def test_rubric_covers_every_leak_token_owner() -> None:
    """A style whose everyday name is unlisted in the rubric invites a description that names it."""
    text = RUBRIC.read_text(encoding="utf-8")
    missing = [style for style in STYLE_AFFINITY if f"`{style}`" not in text]
    assert missing == []


# ---------------------------------------------------------------------------------------
# Brief planning
# ---------------------------------------------------------------------------------------


def test_slot_plan_is_deterministic_and_label_complete(gap: types.ModuleType, catalog: Any, corpus: list[Any]) -> None:
    """Two builds produce identical slots, and every served label gets train and holdout slots."""
    first = gap.build_slots(catalog, corpus)
    second = gap.build_slots(catalog, corpus)
    assert [slot.brief() for slot in first] == [slot.brief() for slot in second]
    labels = set(gap.POOL_LABELS)
    assert labels - {slot.style for slot in first if slot.split == "train"} == set()
    assert labels - {slot.style for slot in first if slot.split == "holdout"} == set()
    assert len(first) == gap.TRAIN_POOL_TOTAL + gap.HOLDOUT_POOL_TOTAL
    train_organic = sum(1 for slot in first if slot.split == "train" and slot.kind != "synthetic")
    assert train_organic / gap.TRAIN_POOL_TOTAL >= gap.ORGANIC_SHARE_FLOOR
    assert all(slot.kind != "synthetic" for slot in first if slot.split == "holdout")


def test_slot_families_never_cross_the_split(gap: types.ModuleType, catalog: Any, corpus: list[Any]) -> None:
    """A train slot in a holdout family would put holdout families into the weights."""
    slots = gap.build_slots(catalog, corpus)
    families = gap.corpus_families(corpus)
    train = {slot.family for slot in slots if slot.split == "train"}
    holdout = {slot.family for slot in slots if slot.split == "holdout"}
    assert train <= families["train"]
    assert holdout <= families["holdout"]
    assert not train & holdout


def test_holdout_targets_floor_every_label(gap: types.ModuleType) -> None:
    """The holdout is an instrument: every class needs rows or it contributes nothing."""
    counts = gap.holdout_targets()
    assert min(counts.values()) >= gap.HOLDOUT_STYLE_FLOOR
    assert sum(counts.values()) == gap.HOLDOUT_POOL_TOTAL


def test_api_gateway_is_over_sampled_on_purpose(gap: types.ModuleType) -> None:
    """The audit left api-gateway with no train positive, so the pool over-samples it."""
    targets = gap.train_targets({})
    assert targets["api-gateway"][0] == gap.API_GATEWAY_TRAIN_TARGET
    assert targets["api-gateway"][0] > targets["none"][0]


def test_scenario_id_keeps_the_style_prefix_inside_the_cap(gap: types.ModuleType) -> None:
    """The id schema caps 80 chars; the style prefix must survive truncation."""
    long_topic = "a-very-long-topic-name-that-keeps-going-with-many-words-to-overflow-the-limit"
    scenario_id = gap.compose_scenario_id("api-gateway", long_topic)
    assert scenario_id.startswith("api-gateway-")
    assert len(scenario_id) <= 80
    assert gap.compose_scenario_id("saga", "trip-holds") == "saga-trip-holds"
    assert gap.compose_scenario_id("saga", "saga-trip-holds") == "saga-trip-holds"


# ---------------------------------------------------------------------------------------
# Pool validation
# ---------------------------------------------------------------------------------------


def _brief(catalog: Any, style: str = "monolithic") -> dict[str, Any]:
    """A minimal valid brief over a real catalog family with a sibling pair."""
    families = {subgroup: names for subgroup, names in gap_family_index(catalog).items() if len(names) >= 2}
    subgroup, names = next((sub, members) for sub, members in sorted(families.items()))
    primary, sibling = names[0], names[1]
    return {
        "slot_id": "tr-doc-0001",
        "split": "train",
        "kind": "organic-doc",
        "style": style,
        "family": subgroup,
        "primary": primary,
        "acceptable_primary": [primary, sibling],
        "acceptable_supporting": [],
        "category": catalog.records[primary].category,
        "language": None,
        "intent_hint": catalog.records[primary].intent,
        "domain_hint": "freight forwarding",
    }


def gap_family_index(catalog: Any) -> dict[str, tuple[str, ...]]:
    """Local copy of the family index (the tooling's is under test elsewhere)."""
    grouped: dict[str, list[str]] = {}
    for name in catalog.order:
        grouped.setdefault(catalog.records[name].subgroup, []).append(name)
    return {subgroup: tuple(names) for subgroup, names in grouped.items()}


def _authored(description: str, *, slot_id: str = "tr-doc-0001", style: str = "monolithic") -> dict[str, Any]:
    """One authored row in the shape the authoring sheet demands."""
    del style
    return {
        "slot_id": slot_id,
        "topic": "settlement-batch",
        "description": description,
        "decision_statement": "How should the settlement job structure its monthly run so a re-run cannot double-post?",
        "notes": "The system is a single deployable; expected primary fits its batch decision.",
        "rationale": "One jar deployed weekly behind a single scheduler, with no other structural evidence",
        "provenance": "organic-doc",
        "source_url": "https://example.invalid/docs",
    }


DESCRIPTION = (
    "Ledgerline is a settlement service that ships as a single deployable: one application archive "
    "built on a scheduled trigger, deployed to two hosts behind a load balancer, with a single "
    "database it alone owns. Operations teams restart it as a whole and scale it by resizing the "
    "hosts, because no part of it can run on its own. Inside, request handling, rule evaluation and "
    "persistence are interleaved in the same runtime rather than being separable units. The monthly "
    "settlement run must re-price every open invoice for the period, and a re-run after an operator "
    "fixes a currency table must not double-post a single line. The decision to make: how should the "
    "run structure its work so that a mid-run failure resumes without re-posting already-settled "
    "invoices, given that the deployable is restarted as a whole and operators have no way to pause "
    "part of it? Volumes are 40 thousand invoices a night and the finance close has a four hour "
    "window, so a full restart-from-zero is not acceptable."
)


def test_validate_accepts_a_clean_row(gap: types.ModuleType, catalog: Any, corpus: list[Any]) -> None:
    """A structurally described row with no leaks is accepted and carries its brief's labels."""
    brief = _brief(catalog)
    scenarios, issues = gap.validate_pool(
        [_authored(DESCRIPTION)],
        {brief["slot_id"]: brief},
        catalog,
        corpus,
        style_leaks=draft_mod.style_leaks,
        style_target=draft_mod.style_target,
    )
    assert issues == []
    assert len(scenarios) == 1
    row = scenarios[0]
    assert row["scenario_id"] == "monolithic-settlement-batch"
    assert row["family"] == brief["family"]
    assert row["expected"]["primary"] == brief["primary"]
    assert row["_pool"]["style"] == "monolithic"


def test_validate_refuses_a_description_that_names_its_style(
    gap: types.ModuleType, catalog: Any, corpus: list[Any]
) -> None:
    """Naming the style teaches string matching; the row is refused with the leak as the reason."""
    brief = _brief(catalog)
    leaked = DESCRIPTION.replace("ships as a single deployable", "is a monolithic deployment")
    _scenarios, issues = gap.validate_pool(
        [_authored(leaked)],
        {brief["slot_id"]: brief},
        catalog,
        corpus,
        style_leaks=draft_mod.style_leaks,
        style_target=draft_mod.style_target,
    )
    assert len(issues) == 1
    assert any("monolithic" in reason for reason in issues[0].reasons)


def test_validate_refuses_an_organic_row_without_its_source(
    gap: types.ModuleType, catalog: Any, corpus: list[Any]
) -> None:
    """The organic share is a claim about grounding: a row without its page is not checkable."""
    brief = _brief(catalog)
    unsourced = {**_authored(DESCRIPTION), "source_url": ""}
    scenarios, issues = gap.validate_pool(
        [unsourced],
        {brief["slot_id"]: brief},
        catalog,
        corpus,
        style_leaks=draft_mod.style_leaks,
        style_target=draft_mod.style_target,
    )
    assert scenarios == []
    assert any("requires the source_url" in reason for reason in issues[0].reasons)
    synthetic = {**_authored(DESCRIPTION), "provenance": "synthetic"}
    synthetic.pop("source_url")  # a synthetic row has no documentation page to cite
    accepted, issues = gap.validate_pool(
        [synthetic],
        {brief["slot_id"]: {**brief, "kind": "synthetic"}},
        catalog,
        corpus,
        style_leaks=draft_mod.style_leaks,
        style_target=draft_mod.style_target,
    )
    assert issues == [] and len(accepted) == 1


def test_validate_refuses_thin_text_and_duplicate_ids(gap: types.ModuleType, catalog: Any, corpus: list[Any]) -> None:
    """A short description carries no evidence, and one topic answered twice collides on id."""
    brief = _brief(catalog)
    briefs = {brief["slot_id"]: brief, "tr-doc-0002": {**brief, "slot_id": "tr-doc-0002"}}
    scenarios, issues = gap.validate_pool(
        [_authored("Too short to show a system.", slot_id="tr-doc-0001")],
        briefs,
        catalog,
        corpus,
        style_leaks=draft_mod.style_leaks,
        style_target=draft_mod.style_target,
    )
    assert scenarios == []
    assert any("too short" in reason for reason in issues[0].reasons)
    repeated = _authored(DESCRIPTION)
    scenarios, issues = gap.validate_pool(
        [repeated, {**repeated, "slot_id": "tr-doc-0002"}],
        briefs,
        catalog,
        corpus,
        style_leaks=draft_mod.style_leaks,
        style_target=draft_mod.style_target,
    )
    assert len(scenarios) == 1
    assert any("repeats inside the pool" in reason for reason in issues[0].reasons)


def test_validate_refuses_an_answer_to_an_unknown_slot(gap: types.ModuleType, catalog: Any, corpus: list[Any]) -> None:
    """Rows are only valid answers to planned slots; an unplanned row cannot enter the pool."""
    _scenarios, issues = gap.validate_pool(
        [{**_authored(DESCRIPTION), "slot_id": "tr-doc-9999"}],
        {},
        catalog,
        corpus,
        style_leaks=draft_mod.style_leaks,
        style_target=draft_mod.style_target,
    )
    assert issues and "not in the plan" in issues[0].reasons[0]


# ---------------------------------------------------------------------------------------
# Leakage gate
# ---------------------------------------------------------------------------------------


def test_leakage_gate_catches_a_reworded_copy(gap: types.ModuleType) -> None:
    """A holdout row that reuses a train row's text must be caught before the split freezes."""
    original = " ".join(f"word{index}" for index in range(80))
    copy = original.replace("word10", "token10")
    train = [("train-row", original)]
    assert gap.near_duplicates(train, [("copy-row", copy)]) != []
    assert gap.near_duplicates(train, [("distinct-row", " ".join(f"other{index}" for index in range(80)))]) == []


def test_dedup_command_reads_each_side_by_its_own_split(gap: types.ModuleType, tmp_path: Path) -> None:
    """A pool carries both splits; the gate must compare its holdout rows against train rows only.

    Regression: treating every pool row as a holdout row made each holdout family overlap itself and
    every row a near-duplicate of itself, so the gate failed on a pool that was clean.
    """
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "_meta": {},
                "scenarios": [
                    {"scenario_id": "t1", "family": "train family", "split": "train", "description": "a b c d e"},
                    {"scenario_id": "h1", "family": "holdout family", "split": "holdout", "description": "q w e r t"},
                ],
            }
        ),
        encoding="utf-8",
    )
    pool = _write(
        tmp_path / "pool.jsonl",
        [
            {"scenario_id": "t2", "family": "train family", "split": "train", "description": "z x c v b"},
            {"scenario_id": "h2", "family": "holdout family", "split": "holdout", "description": "n m b v c"},
        ],
    )
    code = gap.main(
        [
            "dedup",
            "--corpus", str(corpus),
            "--holdout", str(pool),
            "--train-pool", str(pool),
            "--report", str(tmp_path / "dedup.json"),
        ]
    )
    assert code == 0
    report = json.loads((tmp_path / "dedup.json").read_text(encoding="utf-8"))
    assert report["holdout_rows"] == 1
    assert report["family_overlap"] == []
    assert report["near_duplicate_pairs"] == []


def test_leakage_gate_flags_a_long_shared_run(gap: types.ModuleType) -> None:
    """A 50-token shared substring is copied text even when the surrounding rows differ."""
    shared = " ".join(f"shared{index}" for index in range(60))
    train = [("train-row", f"prefix {' '.join(f'a{i}' for i in range(60))} {shared}")]
    holdout = [("holdout-row", f"{shared} suffix {' '.join(f'b{i}' for i in range(60))}")]
    pairs = gap.near_duplicates(train, holdout)
    assert pairs and pairs[0].shared_run > 0


# ---------------------------------------------------------------------------------------
# Agreement and certification
# ---------------------------------------------------------------------------------------


def _write(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Write JSONL rows."""
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_kappa_is_one_for_perfect_agreement_and_zero_at_chance(review: types.ModuleType) -> None:
    """kappa must actually measure chance-corrected agreement, not restate raw agreement."""
    labels = {f"s{index}": "saga" for index in range(10)}
    other = {f"s{index}": "broker" for index in range(10)}
    assert review.cohen_kappa(labels, labels) == pytest.approx(1.0)
    assert review.cohen_kappa(labels, other) == pytest.approx(0.0)


def test_score_batch_accepts_agreement_and_requires_adjudication(review: types.ModuleType, tmp_path: Path) -> None:
    """A disputed row blocks the batch until it is adjudicated, then carries the adjudicated label."""
    pool = [
        {"scenario_id": "s1", "split": "train", "description": "x", "_pool": {"style": "saga", "rationale": "r1"}},
        {"scenario_id": "s2", "split": "train", "description": "y", "_pool": {"style": "broker", "rationale": "r2"}},
    ]
    pack = _write(
        tmp_path / "batch-00.blind.jsonl", [{"scenario_id": "s1", "state": "x"}, {"scenario_id": "s2", "state": "y"}]
    )
    a = _write(tmp_path / "a.jsonl", [{"scenario_id": "s1", "label": "saga"}, {"scenario_id": "s2", "label": "broker"}])
    b = _write(tmp_path / "b.jsonl", [{"scenario_id": "s1", "label": "saga"}, {"scenario_id": "s2", "label": "saga"}])
    ledger, accepted = review.score_batch(pool, pack, a, b)
    assert ledger["raw_agreement"] == pytest.approx(0.5)
    assert ledger["accepted"] is False
    assert accepted == []
    assert ledger["unresolved"] == ["s2"]
    adjudication = _write(
        tmp_path / "adj.jsonl",
        [{"scenario_id": "s2", "label": "broker", "reason": "a central broker mediates every interaction"}],
    )
    ledger, accepted = review.score_batch(pool, pack, a, b, adjudication=adjudication)
    assert ledger["accepted"] is False  # raw agreement 0.5 is below the floor even when adjudicated
    assert [row["scenario_id"] for row in accepted] == []


def test_score_batch_uses_the_adjudicated_label_in_accepted_rows(review: types.ModuleType, tmp_path: Path) -> None:
    """One dispute in a batch of eight keeps raw agreement at 0.875, so the batch is accepted."""
    ids = [f"s{index}" for index in range(8)]
    pool = [
        {
            "scenario_id": sid,
            "split": "train",
            "description": "d",
            # s3's brief asked for a different style than the annotators settle on, so the
            # adjudicator's reason has to become the recorded rationale.
            "_pool": {"style": ("saga" if sid == "s3" else "broker"), "rationale": "r"},
        }
        for sid in ids
    ]
    pack = _write(tmp_path / "batch-01.blind.jsonl", [{"scenario_id": sid, "state": "d"} for sid in ids])
    a = _write(tmp_path / "a.jsonl", [{"scenario_id": sid, "label": "broker"} for sid in ids])
    b = _write(
        tmp_path / "b.jsonl",
        [{"scenario_id": sid, "label": ("saga" if sid == "s3" else "broker")} for sid in ids],
    )
    adjudication = _write(
        tmp_path / "adj.jsonl",
        [{"scenario_id": "s3", "label": "broker", "reason": "no compensating steps are described, only mediation"}],
    )
    ledger, accepted = review.score_batch(pool, pack, a, b, adjudication=adjudication)
    assert ledger["accepted"] is True
    assert ledger["raw_agreement"] == pytest.approx(0.875)
    assert len(accepted) == 8
    disputed = next(row for row in accepted if row["scenario_id"] == "s3")
    assert disputed["_pool"]["style"] == "broker"
    assert disputed["_pool"]["rationale"].startswith("no compensating steps")


def test_disputed_row_keeps_the_adjudicators_reason_even_when_the_label_matches_the_brief(
    review: types.ModuleType, tmp_path: Path
) -> None:
    """The reason explains a disagreement, so it must survive a final label the brief also wanted."""
    ids = [f"s{index}" for index in range(8)]
    pool = [
        {"scenario_id": sid, "split": "train", "description": "d", "_pool": {"style": "broker", "rationale": "authored"}}
        for sid in ids
    ]
    pack = _write(tmp_path / "batch-02.blind.jsonl", [{"scenario_id": sid, "state": "d"} for sid in ids])
    a = _write(tmp_path / "a.jsonl", [{"scenario_id": sid, "label": "broker"} for sid in ids])
    b = _write(
        tmp_path / "b.jsonl",
        [{"scenario_id": sid, "label": ("saga" if sid == "s4" else "broker")} for sid in ids],
    )
    adjudication = _write(
        tmp_path / "adj.jsonl",
        [{"scenario_id": "s4", "label": "broker", "reason": "the text describes mediation, not compensation"}],
    )
    ledger, accepted = review.score_batch(pool, pack, a, b, adjudication=adjudication)
    assert ledger["accepted"] is True
    disputed = next(row for row in accepted if row["scenario_id"] == "s4")
    assert disputed["_pool"]["rationale"] == "authored"
    assert disputed["_pool"]["adjudication_reason"] == "the text describes mediation, not compensation"


def test_certify_fails_closed_on_an_unreviewed_label(review: types.ModuleType, tmp_path: Path) -> None:
    """Certification must refuse a label source it cannot cover completely."""
    labels = tmp_path / "labels.py"
    labels.write_text('STYLE_LABELS: dict[str, tuple[str, str]] = {\n    "a": ("saga", "r"),\n}\n', encoding="utf-8")
    rubric = tmp_path / "rubric.md"
    rubric.write_text("# rubric\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="unreviewed"):
        review.certify(
            labels_path=labels,
            review_path=tmp_path / "review.json",
            rubric=rubric,
            accepted_files=[],
            carry_forward=None,
            method="test",
            reviewer="test",
            human_review=None,
        )


def test_certify_drops_a_carried_verdict_whose_label_changed(review: types.ModuleType, tmp_path: Path) -> None:
    """A carried entry is a claim about the label: if the label changed, the verdict is void."""
    labels = tmp_path / "labels.py"
    labels.write_text('STYLE_LABELS: dict[str, tuple[str, str]] = {\n    "a": ("broker", "r"),\n}\n', encoding="utf-8")
    rubric = tmp_path / "rubric.md"
    rubric.write_text("# rubric\n", encoding="utf-8")
    previous = tmp_path / "previous.json"
    previous.write_text(
        json.dumps(
            {
                "reviews": {
                    "style": [
                        {
                            "scenario_id": "a",
                            "label": "saga",
                            "labels_sha256": "old",
                            "reviewer": "auditor",
                            "rationale_ok": True,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="unreviewed"):
        review.certify(
            labels_path=labels,
            review_path=tmp_path / "review.json",
            rubric=rubric,
            accepted_files=[],
            carry_forward=previous,
            method="test",
            reviewer="test",
            human_review=None,
        )
    accepted = _write(
        tmp_path / "batch-00.accepted.jsonl",
        [{"scenario_id": "a", "_pool": {"style": "broker", "annotator_a": "broker", "annotator_b": "broker"}}],
    )
    ledger = review.certify(
        labels_path=labels,
        review_path=tmp_path / "review.json",
        rubric=rubric,
        accepted_files=[accepted],
        carry_forward=previous,
        method="test",
        reviewer="test",
        human_review=None,
    )
    assert ledger["status"] == "reviewed"
    assert ledger["dropped_carry_forward"] == ["a"]
    assert ledger["labels_sha256"] == hashlib.sha256(labels.read_bytes()).hexdigest()
    assert [entry["scenario_id"] for entry in ledger["reviews"]["style"]] == ["a"]


# ---------------------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------------------


def test_insert_style_labels_preserves_every_existing_byte(tmp_path: Path) -> None:
    """The label source is a reviewed artifact: an append must not rewrite the prior entries."""
    source = 'STYLE_LABELS: dict[str, tuple[str, str]] = {\n    "a": (\n        "saga",\n        "why",\n    ),\n}\n\n\ndef f() -> None:\n    """doc"""\n'
    updated = _gap_insert(source, {"b": ("broker", 'a "quoted" rationale')})
    assert updated.startswith(source.split("}\n", 1)[0])
    assert '"b": (\n        "broker",\n        "a \\"quoted\\" rationale",\n    ),\n}' in updated
    assert updated.endswith('\n\n\ndef f() -> None:\n    """doc"""\n')


def _gap_insert(source: str, entries: dict[str, tuple[str, str]]) -> str:
    """Local wrapper so the module fixture is not needed by a pure-string test."""
    module = _load("style_label_gap", "style_label_gap.py")
    return module.insert_style_labels(source, entries, "test block")


def test_corpus_serialization_round_trips() -> None:
    """Applying a pool rewrites the corpus, so the serializer must reproduce the stored bytes."""
    path = REPO_ROOT / "tests" / "benchmark" / "scenarios" / "seed.json"
    raw = path.read_text(encoding="utf-8")
    module = _load("style_label_gap", "style_label_gap.py")
    assert module.corpus_text(json.loads(raw)) == raw


def test_choice_rows_mirror_the_corpus_categories() -> None:
    """The choice mirror is derived, so its labels must be the corpus categories verbatim."""
    module = _load("style_label_gap", "style_label_gap.py")
    corpus = {
        "_meta": {},
        "scenarios": [
            {"scenario_id": "s1", "description": "d1", "category": "behaviour-selection", "split": "train"},
            {"scenario_id": "s2", "description": "d2", "category": "domain-modeling", "split": "holdout"},
        ],
    }
    rows = module.choice_rows(corpus, {"a": "gloss"}, "Which?", "deadbeef1234")
    assert [row["label"] for row in rows] == ["behaviour-selection", "domain-modeling"]
    assert [row["source"]["split"] for row in rows] == ["train", "holdout"]
    assert all(row["source"]["corpus_sha12"] == "deadbeef1234" for row in rows)


def test_blind_packs_carry_no_labels_and_no_style_in_the_id(review: types.ModuleType, tmp_path: Path) -> None:
    """A pack that leaks the label destroys the measurement it exists to make.

    Scenario ids are ``<style>-<topic>``, so the pack must speak opaque tokens: a pack carrying the
    real ids hands the annotator the answer, which is how the first attempt at this corpus went.
    """
    rows = [
        {"scenario_id": "saga-trip-holds", "split": "train", "description": "d2", "_pool": {"style": "saga"}},
        {"scenario_id": "saga-trip-releases", "split": "train", "description": "d1", "_pool": {"style": "saga"}},
    ]
    packs = review.build_packs(rows, tmp_path, batch_size=1, prefix="b")
    assert [path.name for path in packs] == ["b-00.blind.jsonl", "b-01.blind.jsonl"]
    pack_rows = [json.loads(line) for pack in packs for line in pack.read_text(encoding="utf-8").splitlines()]
    assert all(set(row) == {"scenario_id", "state"} for row in pack_rows)
    tokens = {row["scenario_id"] for row in pack_rows}
    assert tokens.isdisjoint({"saga-trip-holds", "saga-trip-releases"})
    assert all(token.startswith("s") and len(token) == 13 for token in tokens)
    mapped = {}
    for pack in packs:
        mapped.update(review.id_map_for(pack))
    assert set(mapped) == tokens
    assert set(mapped.values()) == {"saga-trip-holds", "saga-trip-releases"}


def test_coverage_failures_name_what_is_missing(gap: types.ModuleType) -> None:
    """The build contract is machine-checked: counts, organic share, and per-label presence."""
    report = {
        "train": 10,
        "holdout": 2,
        "train_organic_share": 0.5,
        "train_missing_labels": ["saga"],
        "holdout_missing_labels": ["broker"],
    }
    failures = gap.coverage_failures(report, expected={"train": 20, "holdout": 3})
    assert any("train pool 10" in failure for failure in failures)
    assert any("holdout pool 2" in failure for failure in failures)
    assert any("organic share" in failure for failure in failures)
    assert any("saga" in failure for failure in failures)


def test_style_labels_reader_matches_the_repository_source(gap: types.ModuleType) -> None:
    """The ledger's coverage check reads STYLE_LABELS by AST; it must see every real entry."""
    labels = gap.read_style_labels(SCRIPTS / "derive_style_labels.py")
    counts = Counter(label for label, _rationale in labels.values())
    assert set(counts) - {seam_prompts.STYLE_NONE_KEY} <= set(seam_prompts.STYLE_VOCABULARY)
    assert labels["algorithms-traversal-variation-pricing-rules"][0] == "pipe-and-filter"
