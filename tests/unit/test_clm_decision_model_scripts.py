"""Build-time decision-model scripts: train/serve prompt parity and the export contract.

These scripts run inside the sidecar image builds (``docker/Dockerfile.tacit``,
``docker/Dockerfile.clm``), where a wiring bug only surfaces after an hour of state
generation or a 16 GB encoder download. Everything expensive is stubbed here; what is
asserted is the *wiring*: the trainer asks exactly the questions the server asks
(``src/decisions/seam_prompts.py``), its ``main()`` walks generate → problems → teacher →
train → save, and the CLM exporter emits the row shape the pinned ``train/adapters.py``
reads.

The scripts are loaded by path (neither is an importable package), the same way
``test_client_examples.py`` loads the example clients.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from src.decisions import seam_prompts
from tests.unit.conftest import REPO_ROOT


SCRIPTS = REPO_ROOT / "scripts" / "decision_models"


def _load(name: str, filename: str) -> types.ModuleType:
    """Import one script by path (registered in sys.modules so dataclasses resolve)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter() -> types.ModuleType:
    return _load("export_clm_dataset", "export_clm_dataset.py")


def test_export_row_matches_the_pinned_clm_adapter_contract(exporter: types.ModuleType) -> None:
    """``typed_decision_examples`` reads {id, state, questions, gold}; so must we write it.

    The row layout was verified against the upstream adapter at one CLM revision:
    the exporter's default pin, the Dockerfile's ``CLM_REPO_PIN`` default, and
    this test's constant must all name that same revision, so bumping the pin in
    either place re-opens the contract check instead of silently invalidating it.
    """
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.clm").read_text(encoding="utf-8")
    assert "\nARG CLM_REPO_PIN=d5f9ef0\n" in dockerfile
    assert exporter.parse_args(["--out", "/tmp/x"]).clm_pin == "d5f9ef0"
    row = exporter.to_typed_row(
        {
            "id": "scenario:category",
            "state": "System goal: x\nDecision part: y",
            "instructions": "Which category?",
            "criteria": {"domain-modeling": "entities and invariants", "behavior-selection": "dispatch"},
            "label": "behavior-selection",
            "source": {"split": "train"},
        },
        exporter.CATEGORY_QUESTION_ID,
    )
    assert row["id"] == "scenario:category"
    questions = json.loads(row["questions"])
    assert questions["category"]["type"] == "choice"
    assert questions["category"]["instructions"] == "Which category?"
    assert set(questions["category"]["criteria"]) == {"domain-modeling", "behavior-selection"}
    assert json.loads(row["gold"]) == {"category": {"label": "behavior-selection"}}


def test_export_refuses_a_label_outside_the_criteria(exporter: types.ModuleType) -> None:
    with pytest.raises(SystemExit):
        exporter.to_typed_row(
            {
                "id": "bad",
                "state": "s",
                "criteria": {"domain-modeling": "x"},
                "label": "security-verification",
                "source": {"split": "train"},
            },
            exporter.CATEGORY_QUESTION_ID,
        )


def test_export_filters_by_source_split(exporter: types.ModuleType, tmp_path: Path) -> None:
    """Holdout families must never reach the weights."""
    path = tmp_path / "choice.jsonl"
    rows = [
        {
            "id": "a",
            "state": "s",
            "criteria": {"domain-modeling": "x"},
            "label": "domain-modeling",
            "source": {"split": "train"},
        },
        {
            "id": "b",
            "state": "s",
            "criteria": {"domain-modeling": "x"},
            "label": "domain-modeling",
            "source": {"split": "holdout"},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    exported = exporter.load_rows(path, "train")
    assert [row["id"] for row in exported] == ["a"]


def test_style_row_asks_the_served_question(exporter: types.ModuleType) -> None:
    """The style export must use the Ch2 qid, not the category one."""
    row = exporter.to_typed_row(
        {
            "id": "scenario:style",
            "state": "A checkout service with flash sales.",
            "instructions": "Which style?",
            "criteria": {"event-driven": "events", "none": "nothing applies"},
            "label": "event-driven",
            "source": {"split": "train"},
        },
        exporter.STYLE_QUESTION_ID,
    )
    questions = json.loads(row["questions"])
    assert list(questions) == ["architecture_style"]
    assert questions["architecture_style"]["type"] == "choice"
    assert questions["architecture_style"]["instructions"] == "Which style?"
    assert json.loads(row["gold"]) == {"architecture_style": {"label": "event-driven"}}


def test_style_corpus_matches_the_served_prompt_bytes() -> None:
    """Trainer asks exactly what the server asks: qid, instructions, criteria, state.

    ``criteria`` is compared as an ORDERED mapping (``list(row) == list(criteria)``), not
    just as a dict: ``==`` on dicts ignores insertion order, so a re-ordering would pass
    dict equality while serving the CLM head a criteria serialisation it never trained on.
    """
    from src.decisions.client import STYLE_QUESTION_ID, build_style_criteria

    style_path = REPO_ROOT / "qa" / "typed" / "style.jsonl"
    criteria = build_style_criteria()
    rows = [json.loads(line) for line in style_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "qa/typed/style.jsonl must not be empty"
    corpus = json.loads((REPO_ROOT / "tests" / "benchmark" / "scenarios" / "full.json").read_text(encoding="utf-8"))
    scenarios = {item["scenario_id"]: item for item in corpus.get("scenarios", corpus)}
    for row in rows:
        scenario_id = row["source"]["scenario_id"]
        assert row["id"] == f"{scenario_id}:style"
        assert row["state"] == scenarios[scenario_id]["description"]
        assert row["instructions"] == seam_prompts.STYLE_INSTRUCTIONS
        assert list(row["criteria"]) == list(criteria)
        assert row["criteria"] == criteria
        assert row["label"] in criteria
        assert row["source"]["split"] == scenarios[scenario_id]["split"]
    assert {row["source"]["scenario_id"] for row in rows} == set(scenarios)
    assert STYLE_QUESTION_ID == "architecture_style"


@pytest.fixture(scope="module")
def eval_gate() -> types.ModuleType:
    return _load("clm_eval_gate", "clm_eval_gate.py")


def _gate_manifest(**overrides: Any) -> dict[str, Any]:
    """A manifest shaped like ``export_clm_dataset.py`` v2: 15 holdout rows per question."""
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "splits": {
            "train": {
                "source_split": "train",
                "rows": 70,
                "per_question": {"architecture_style": 70, "category": 70},
                "label_coverage": {
                    # R6-neutral default: train supports every test label, so tests that do
                    # not target R6 keep their expectations (R6 tests mutate this block).
                    "architecture_style": {
                        "labels": {
                            "monolithic": 20,
                            "none": 11,
                            "broker": 2,
                            "saga": 1,
                            "api-gateway": 2,
                            "sse": 1,
                            "x": 1,
                        },
                        "uncovered": ["hexagonal"],
                    },
                    "category": {
                        "labels": {"domain-modeling": 12, "behavior-selection": 2, "persistence-state": 14},
                        "uncovered": [],
                    },
                },
            },
            "test": {
                "source_split": "holdout",
                "rows": 30,
                "per_question": {"architecture_style": 15, "category": 15},
                "label_coverage": {
                    "architecture_style": {
                        "labels": {
                            "monolithic": 4,
                            "none": 3,
                            "broker": 2,
                            "saga": 2,
                            "api-gateway": 2,
                            "sse": 1,
                            "x": 1,
                        },
                        "uncovered": ["saga"],
                    },
                    "category": {
                        "labels": {"domain-modeling": 5, "behavior-selection": 4, "persistence-state": 3},
                        "uncovered": [],
                    },
                },
            },
        },
    }
    manifest.update(overrides)
    return manifest


def _gate_summary(**overrides: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "best_epoch": 3,
        "val_acc": 0.8,
        "test": {"acc": 0.75, "soft_ce": 0.5, "per_question": {"architecture_style": 0.7, "category": 0.8}},
        "init_test": {
            "acc": 0.6,
            "soft_ce": 0.7,
            "per_question": {"architecture_style": 0.4, "category": 0.5},
        },
        "majority_test": 0.3,
        "train_questions": 70,
    }
    summary.update(overrides)
    return summary


@lru_cache(maxsize=1)
def _gate_module() -> types.ModuleType:
    return _load("clm_eval_gate", "clm_eval_gate.py")


def _served_gate(
    summary: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    questions: tuple[str, ...] = ("architecture_style",),
    advisory: tuple[str, ...] = ("category",),
    ledger: dict[str, Any] | None = None,
    source_digests: dict[str, str | None] | None = None,
) -> Any:
    """Run the gate with the served question binding and a v2 manifest.

    R7-neutral by default: the fixture's ``uncovered`` criteria arrive
    acknowledged at the fixture digest, so tests that do not target R7 keep
    their expectations (pass ``ledger={}`` to exercise the unacknowledged path).
    """
    resolved = _gate_manifest() if manifest is None else manifest
    digests: dict[str, str | None] = (
        {"choice": "fixture-choice", "style": "fixture-style"} if source_digests is None else source_digests
    )
    active: dict[str, Any] | None
    if ledger is None:
        train_cov: dict[str, Any] = ((resolved.get("splits") or {}).get("train") or {}).get("label_coverage") or {}
        style_cov: dict[str, Any] = train_cov.get("architecture_style") or {}
        uncovered: list[str] = style_cov.get("uncovered", [])
        active = {
            "coverage_ack": {
                "architecture_style": [
                    {
                        "criterion": key,
                        "reviewer": "fixture",
                        "note": "R7-neutral default",
                        "source_sha256": digests["style"],
                    }
                    for key in uncovered
                ]
            },
            "reviews": {"choice": [], "style": []},
        }
    else:
        active = ledger
    return _gate_module().check(
        summary, resolved, list(questions), list(advisory), active, digests if active is not None else None
    )


def test_eval_gate_passes_an_improved_head() -> None:
    assert _served_gate(_gate_summary()).failures == []


def test_eval_gate_passes_an_equal_head() -> None:
    """No regression = PASS once some epoch improved val (equal test, best_epoch > 0)."""
    summary = _gate_summary()
    summary["test"]["acc"] = summary["init_test"]["acc"]
    summary["test"]["per_question"] = dict(summary["init_test"]["per_question"])
    assert _served_gate(summary).failures == []


def test_eval_gate_fails_a_no_improvement_run() -> None:
    """best_epoch 0 ships the released weights — never as ``runtime-finetuned``."""
    summary = _gate_summary()
    summary["best_epoch"] = 0
    summary["test"]["acc"] = summary["init_test"]["acc"]
    summary["test"]["per_question"] = dict(summary["init_test"]["per_question"])
    failures = _served_gate(summary).failures
    assert len(failures) == 1
    assert "best_epoch is 0" in failures[0]


def test_eval_gate_fails_a_regressed_head() -> None:
    summary = _gate_summary()
    summary["test"]["acc"] = 0.55
    failures = _served_gate(summary).failures
    assert any("regressed below init" in failure for failure in failures)


def test_eval_gate_tolerates_exactly_one_row_of_regression() -> None:
    """15 rows → one flipped row (1/15) is inside the measurement's granularity."""
    summary = _gate_summary()
    summary["test"]["per_question"]["architecture_style"] = 7 / 15
    summary["init_test"]["per_question"]["architecture_style"] = 8 / 15
    assert _served_gate(summary).failures == []
    summary["test"]["per_question"]["architecture_style"] = 6 / 15
    assert any("more than one row" in failure for failure in _served_gate(summary).failures)


def test_eval_gate_fails_closed_without_a_per_question_baseline() -> None:
    """A missing init baseline must fail, never silently disable the regression rule."""
    summary = _gate_summary()
    summary["init_test"] = {"acc": 0.6, "soft_ce": 0.7}
    assert any("no per-question baseline" in failure for failure in _served_gate(summary).failures)


def test_eval_gate_fails_when_the_head_only_learned_the_prior() -> None:
    """14/15 labels monolithic on the test shard → the head must beat 0.9333."""
    manifest = _gate_manifest()
    manifest["splits"]["test"]["label_coverage"]["architecture_style"]["labels"] = {"monolithic": 14, "none": 1}
    failures = _served_gate(_gate_summary(), manifest).failures
    assert len(failures) == 1
    assert "majority-label share" in failures[0]


def test_eval_gate_skips_the_prior_rule_on_a_single_label_shard() -> None:
    """A one-label shard makes the prior rule vacuous: SKIP, and never a quiet pass."""
    manifest = _gate_manifest()
    manifest["splits"]["test"]["label_coverage"]["architecture_style"]["labels"] = {"monolithic": 15}
    result = _served_gate(_gate_summary(), manifest)
    assert result.failures == []
    assert [entry["rule"] for entry in result.skipped] == ["R5-prior"]


def test_eval_gate_fails_closed_without_a_manifest() -> None:
    """No manifest (absent file, or a layout the gate does not know) blocks the build."""
    module = _gate_module()
    assert any(
        "schema_version" in failure
        for failure in module.check(_gate_summary(), None, ["architecture_style"], ["category"]).failures
    )
    assert any(
        "no rows for question ids" in failure
        for failure in module.check(_gate_summary(), {}, ["architecture_style"], ["category"]).failures
    )


def test_eval_gate_load_manifest_rejects_an_unknown_schema(tmp_path: Path) -> None:
    module = _gate_module()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "splits": {}}), encoding="utf-8")
    assert module.load_manifest(path) is None
    path.write_text(json.dumps(_gate_manifest()), encoding="utf-8")
    assert module.load_manifest(path) is not None


def test_eval_gate_reports_the_unserved_question_without_gating_it() -> None:
    """No CLM client sends the category question: a regression there is advisory."""
    summary = _gate_summary()
    summary["test"]["per_question"]["category"] = 0.1
    result = _served_gate(summary)
    assert result.failures == []
    assert any(check["rule"] == "R3-advisory" and "test 0.1000" in check["detail"] for check in result.checks)


def test_eval_gate_fails_a_missing_question() -> None:
    """Every manifest question the gate reasons about must have test rows."""
    manifest = _gate_manifest()
    del manifest["splits"]["test"]["per_question"]["category"]
    failures = _served_gate(_gate_summary(), manifest).failures
    assert any("no rows for question ids" in failure for failure in failures)


def test_eval_gate_fails_without_accuracies() -> None:
    assert _served_gate({}).failures != []


def test_eval_gate_fails_a_test_label_without_train_support() -> None:
    """R6: the fixture's train block lacks saga — a saga test label blocks the build."""
    manifest = _gate_manifest()
    manifest["splits"]["train"]["label_coverage"]["architecture_style"]["labels"] = {
        "monolithic": 20,
        "none": 11,
    }
    result = _served_gate(_gate_summary(), manifest)
    assert any("R6-support" in failure and "saga" in failure for failure in result.failures)


def test_eval_gate_passes_r6_when_train_supports_every_test_label() -> None:
    """R6 is silent when the train histogram covers every test label."""
    manifest = _gate_manifest()
    test_labels = manifest["splits"]["test"]["label_coverage"]["architecture_style"]["labels"]
    manifest["splits"]["train"]["label_coverage"]["architecture_style"]["labels"] = dict(test_labels)
    result = _served_gate(_gate_summary(), manifest)
    assert result.failures == []
    assert any(check["rule"] == "R6-support" and check["status"] == "pass" for check in result.checks)


def test_eval_gate_fails_unacknowledged_train_uncovered_criteria() -> None:
    """R7: the fixture's uncovered ``hexagonal`` blocks the build with an empty ledger."""
    result = _served_gate(_gate_summary(), ledger={})
    assert any("R7-coverage" in failure and "hexagonal" in failure for failure in result.failures)


def test_eval_gate_passes_r7_when_coverage_is_acknowledged() -> None:
    """R7 passes when the ledger acks every uncovered key at the source digest."""
    result = _served_gate(_gate_summary())
    assert result.failures == []
    assert any(check["rule"] == "R7-coverage" and check["status"] == "pass" for check in result.checks)


def test_eval_gate_voids_coverage_ack_on_stale_digest() -> None:
    """An ack recorded against other bytes does not certify the current export."""
    ledger: dict[str, Any] = {
        "coverage_ack": {
            "architecture_style": [
                {
                    "criterion": "hexagonal",
                    "reviewer": "ada",
                    "note": "reviewed last quarter",
                    "source_sha256": "stale-digest",
                }
            ]
        },
        "reviews": {"choice": [], "style": []},
    }
    result = _served_gate(_gate_summary(), ledger=ledger)
    assert any("R7-coverage" in failure for failure in result.failures)


def test_eval_gate_reports_unreviewed_category_without_blocking() -> None:
    """R8: unreviewed category gold is a visible caveat, never a failure."""
    result = _served_gate(_gate_summary())
    assert result.failures == []
    notes = [check for check in result.checks if check["rule"] == "R8-choice-review"]
    assert len(notes) == 1
    assert "stays unreviewed" in notes[0]["detail"]


def test_eval_gate_reports_reviewed_category_when_certified() -> None:
    """R8 flips to reviewed once 100 choice rows are certified at the digest."""
    ledger: dict[str, Any] = {
        "coverage_ack": {
            "architecture_style": [
                {
                    "criterion": "hexagonal",
                    "reviewer": "ada",
                    "note": "ok",
                    "source_sha256": "fixture-style",
                }
            ]
        },
        "reviews": {
            "choice": [
                {
                    "scenario_id": f"s{index}",
                    "label": "domain-modeling",
                    "reviewer": "ada",
                    "label_ok": True,
                    "source_sha256": "fixture-choice",
                }
                for index in range(100)
            ],
            "style": [],
        },
    }
    result = _served_gate(_gate_summary(), ledger=ledger)
    assert result.failures == []
    notes = [check for check in result.checks if check["rule"] == "R8-choice-review"]
    assert "category gold reviewed" in notes[0]["detail"]


def test_eval_gate_main_writes_the_runbook_line_and_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    eval_gate = _load("clm_eval_gate", "clm_eval_gate.py")
    summary_path = tmp_path / "finetune_summary.json"
    summary_path.write_text(json.dumps(_gate_summary()), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_gate_manifest()), encoding="utf-8")
    choice_path = tmp_path / "choice.jsonl"
    choice_path.write_text("{}\n", encoding="utf-8")
    style_path = tmp_path / "style.jsonl"
    style_bytes = b"{}\n"
    style_path.write_bytes(style_bytes)
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "coverage_ack": {
                    "architecture_style": [
                        {
                            "criterion": "hexagonal",
                            "reviewer": "fixture",
                            "note": "R7-neutral default",
                            "source_sha256": hashlib.sha256(style_bytes).hexdigest(),
                        }
                    ]
                },
                "reviews": {"choice": [], "style": []},
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "gate.json"
    args = [
        "--summary",
        str(summary_path),
        "--manifest",
        str(manifest_path),
        "--questions",
        "architecture_style",
        "--advisory-questions",
        "category",
        "--choice",
        str(choice_path),
        "--style",
        str(style_path),
        "--review",
        str(review_path),
        "--report",
        str(report_path),
    ]
    assert eval_gate.main(args) == 0
    assert "clm-eval-gate: PASS" in capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["questions"] == ["architecture_style"]
    assert report["advisory_questions"] == ["category"]
    with pytest.raises(SystemExit, match="does not exist"):
        eval_gate.main(["--summary", str(tmp_path / "absent.json")])


def test_eval_gate_main_fails_and_still_writes_a_report(tmp_path: Path) -> None:
    eval_gate = _load("clm_eval_gate", "clm_eval_gate.py")
    summary = _gate_summary()
    summary["best_epoch"] = 0
    summary_path = tmp_path / "finetune_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_gate_manifest()), encoding="utf-8")
    report_path = tmp_path / "gate.json"
    code = eval_gate.main(
        ["--summary", str(summary_path), "--manifest", str(manifest_path), "--report", str(report_path)]
    )
    assert code == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "fail"


@pytest.fixture(scope="module")
def rank_parity() -> types.ModuleType:
    return _load("clm_rank_parity", "clm_rank_parity.py")


def _payload_row(
    scenario_id: str, answers: list[str], question: str | None = None, relevance: list[int] | None = None
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "family": "family-a",
        "context": "",
        "question": question or f"decide {scenario_id}",
        "answers": answers,
        "answer_names": [f"pattern-{index}" for index in range(len(answers))],
        "relevance": relevance if relevance is not None else [3] + [0] * (len(answers) - 1),
    }


def _write_payloads(path: Path, payloads: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(payload) for payload in payloads), encoding="utf-8")
    return path


def test_rank_payloads_load_in_order(rank_parity: types.ModuleType, tmp_path: Path) -> None:
    """The artifact is replayed as written: one served-shape payload per scenario."""
    path = _write_payloads(
        tmp_path / "payloads.jsonl",
        [_payload_row("s2", ["a", "b"]), _payload_row("s1", ["a", "b"])],
    )
    payloads = rank_parity.load_payloads(path)
    assert [payload["scenario_id"] for payload in payloads] == ["s2", "s1"]
    assert payloads[0]["answers"] == ["a", "b"]


def test_rank_payloads_refuse_a_single_answer_payload(rank_parity: types.ModuleType, tmp_path: Path) -> None:
    """A one-answer payload has no ranking to measure — refuse rather than score it."""
    path = _write_payloads(tmp_path / "payloads.jsonl", [_payload_row("s1", ["only"])])
    with pytest.raises(SystemExit, match="fewer than two answers"):
        rank_parity.load_payloads(path)


def test_rank_payloads_refuse_an_empty_file(rank_parity: types.ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "payloads.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="no payloads"):
        rank_parity.load_payloads(path)


def test_rank_payloads_refuse_a_payload_without_relevance(rank_parity: types.ModuleType, tmp_path: Path) -> None:
    """nDCG is a gate now: a payload without aligned grades stops the build."""
    row = _payload_row("s1", ["a", "b"])
    del row["relevance"]
    path = _write_payloads(tmp_path / "payloads.jsonl", [row])
    with pytest.raises(SystemExit, match="no relevance grades"):
        rank_parity.load_payloads(path)


def test_ndcg_scores_graded_rankings(rank_parity: types.ModuleType) -> None:
    """Ideal ordering is 1.0; burying the primary below a negative costs DCG."""
    answers = ["best", "mid", "junk"]
    relevance = [3.0, 1.0, 0.0]
    assert rank_parity.ndcg_at(["best", "mid", "junk"], answers, relevance, 10) == pytest.approx(1.0)
    buried = 1.0 + 3.0 / 1.584962500721156  # gain-1 at rank 1, gain-3 demoted to rank 2 (log2(3))
    ideal = 3.0 + 1.0 / 1.584962500721156
    assert rank_parity.ndcg_at(["mid", "best", "junk"], answers, relevance, 10) == pytest.approx(buried / ideal)
    assert rank_parity.ndcg_at(["junk", "junk"], answers, relevance, 10) == pytest.approx(0.0)
    # All-negative payload: there is nothing to rank, so any ordering is perfect.
    assert rank_parity.ndcg_at(["junk"], ["junk"], [0.0], 10) == pytest.approx(1.0)


def test_top5_overlap_measures_head_survival(rank_parity: types.ModuleType) -> None:
    """Identity is 1.0, a disjoint head is 0.0, and only the first five count."""
    head = ["a", "b", "c", "d", "e"]
    assert rank_parity.top5_overlap(head, head) == pytest.approx(1.0)
    assert rank_parity.top5_overlap(head, ["f", "g"]) == pytest.approx(0.0)
    assert rank_parity.top5_overlap(head, ["b", "a"]) == pytest.approx(0.4)
    assert rank_parity.top5_overlap(["a"], ["b"]) == pytest.approx(0.0)


def test_rank_gate_requires_an_explicit_budget(rank_parity: types.ModuleType, tmp_path: Path) -> None:
    """The budget is a reviewed constant at the call site, never a hidden default."""
    path = _write_payloads(tmp_path / "payloads.jsonl", [_payload_row("s1", ["a", "b"])])
    with pytest.raises(SystemExit, match="--max-flip-rate is required"):
        rank_parity.main(
            [
                "--payloads",
                str(path),
                "--released",
                "r.pt",
                "--trained",
                "t.pt",
                "--clm-src",
                str(tmp_path),
                "--emb-url",
                "http://127.0.0.1:8090/v1/embeddings",
            ]
        )


def test_rank_gate_counts_flips_and_writes_the_report(
    rank_parity: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate compares released vs trained through the served rank call and reports."""
    path = _write_payloads(
        tmp_path / "payloads.jsonl",
        # Uniform gains: nDCG is 1.0 under any ordering, so this test isolates the flip budget.
        [
            _payload_row(f"s{index}", ["p0", "p1", "p2", "p3", "p4", "p5"], relevance=[1, 1, 1, 1, 1, 1])
            for index in range(4)
        ],
    )

    class _StubEngine:
        def rank(self, context: str, candidates: list[str], instructions: Any, model: str) -> list[dict[str, Any]]:
            assert context == ""
            order = candidates if model == "released" or instructions != "decide s0" else candidates[::-1]
            return [{"rank": index + 1, "candidate": text, "prob": 0.9} for index, text in enumerate(order)]

    class _StubEmbedder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    stub_engine = types.ModuleType("clm.engine")
    stub_engine.Engine = lambda *args, **kwargs: _StubEngine()  # type: ignore[attr-defined]
    stub_embedder = types.ModuleType("clm.embedder")
    stub_embedder.Embedder = _StubEmbedder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "clm.engine", stub_engine)
    monkeypatch.setitem(sys.modules, "clm.embedder", stub_embedder)
    report = tmp_path / "parity.json"
    base = [
        "--payloads",
        str(path),
        "--released",
        "released.pt",
        "--trained",
        "trained.pt",
        "--clm-src",
        str(tmp_path),
        "--emb-url",
        "http://127.0.0.1:8090/v1/embeddings",
        "--report",
        str(report),
    ]
    assert rank_parity.main([*base, "--max-flip-rate", "0.25"]) == 0
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["status"] == "pass"
    assert written["flip_rate"] == pytest.approx(0.25)
    assert [flip["scenario_id"] for flip in written["flips"]] == ["s0"]
    # Names, not answer texts: document_text runs to kilobytes and the report is sha-pinned.
    assert written["flips"][0] == {"scenario_id": "s0", "released": "pattern-0", "trained": "pattern-5"}
    # Reversing a six-answer list keeps four of the five released head entries.
    assert written["mean_top5_overlap"] == pytest.approx((3 * 1.0 + 0.8) / 4)
    assert written["mean_trained_ndcg10"] == pytest.approx(1.0)
    assert written["mean_released_ndcg10"] == pytest.approx(1.0)
    assert rank_parity.main([*base, "--max-flip-rate", "0.24"]) == 1


def test_rank_gate_fails_a_rank_regression_within_flip_budget(
    rank_parity: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flip toward a worse ranking fails even when the flip count is inside budget."""
    path = _write_payloads(
        tmp_path / "payloads.jsonl",
        [_payload_row("s0", ["best", "junk"], relevance=[3, 0])],
    )

    class _StubEngine:
        def rank(self, context: str, candidates: list[str], instructions: Any, model: str) -> list[dict[str, Any]]:
            del context, instructions
            order = candidates if model == "released" else candidates[::-1]
            return [{"rank": index + 1, "candidate": text, "prob": 0.9} for index, text in enumerate(order)]

    class _StubEmbedder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    stub_engine = types.ModuleType("clm.engine")
    stub_engine.Engine = lambda *args, **kwargs: _StubEngine()  # type: ignore[attr-defined]
    stub_embedder = types.ModuleType("clm.embedder")
    stub_embedder.Embedder = _StubEmbedder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "clm.engine", stub_engine)
    monkeypatch.setitem(sys.modules, "clm.embedder", stub_embedder)
    report = tmp_path / "parity.json"
    code = rank_parity.main(
        [
            "--payloads",
            str(path),
            "--released",
            "released.pt",
            "--trained",
            "trained.pt",
            "--clm-src",
            str(tmp_path),
            "--emb-url",
            "http://127.0.0.1:8090/v1/embeddings",
            "--max-flip-rate",
            "1.0",
            "--report",
            str(report),
        ]
    )
    assert code == 1
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["status"] == "fail"
    assert written["mean_released_ndcg10"] == pytest.approx(1.0)
    # Reversed: the gain-3 answer sits at rank 2, so nDCG = (3/log2(3)) / 3.
    assert written["mean_trained_ndcg10"] == pytest.approx(1 / 1.584962500721156)


@pytest.fixture(scope="module")
def calibrator() -> types.ModuleType:
    return _load("clm_calibrate", "clm_calibrate.py")


def test_calibration_metrics_match_definitions(calibrator: types.ModuleType) -> None:
    """ECE = binned |acc − conf|, Brier = mean (conf − outcome)²."""
    pairs = [(0.9, True), (0.9, False), (0.1, False), (0.1, False)]
    assert calibrator.brier_score(pairs) == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)
    ece = calibrator.expected_calibration_error(pairs, 10)
    assert ece == pytest.approx(0.5 * abs(0.5 - 0.9) + 0.5 * abs(0.0 - 0.1))


def test_calibration_table_reports_override_accuracy(calibrator: types.ModuleType) -> None:
    """Acted accuracy is over overrides only; confident ``none`` answers are stratified out."""
    pairs = [
        (0.9, True, "hexagonal"),
        (0.8, True, "hexagonal"),
        (0.4, False, "hexagonal"),
        (0.2, False, "hexagonal"),
    ]
    report = calibrator.summarize(pairs, [0.5, 0.85], 10)
    assert report["n"] == 4
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["override_rate"] == pytest.approx(1.0)
    by_floor = {row["floor"]: row for row in report["floors"]}
    assert by_floor[0.5]["acted"] == 2
    assert by_floor[0.5]["acted_accuracy"] == pytest.approx(1.0)
    assert by_floor[0.85]["acted"] == 1
    assert report["confidence"]["min"] == pytest.approx(0.2)
    assert report["confidence"]["max"] == pytest.approx(0.9)


def test_calibration_table_stratifies_abstentions(calibrator: types.ModuleType) -> None:
    """A confident ``none`` is a seam no-op: counted separately, never in the enable bar."""
    pairs = [(0.9, True, "hexagonal"), (0.9, True, "none"), (0.9, False, "none")]
    by_floor = {row["floor"]: row for row in calibrator.summarize(pairs, [0.5], 10)["floors"]}
    row = by_floor[0.5]
    assert row["acted"] == 1
    assert row["acted_rate"] == pytest.approx(1 / 3)
    assert row["acted_accuracy"] == pytest.approx(1.0)
    assert row["none_acted"] == 2
    assert row["none_accuracy"] == pytest.approx(0.5)
    low, high = row["acted_accuracy_ci95"]
    assert low <= 1.0 <= high


def test_calibration_table_bands_small_samples(calibrator: types.ModuleType) -> None:
    """Every floor row carries a 95% Wilson interval; an empty override set spans [0, 1]."""
    pairs = [(0.9, True, "none")]
    by_floor = {row["floor"]: row for row in calibrator.summarize(pairs, [0.5], 10)["floors"]}
    row = by_floor[0.5]
    assert row["acted"] == 0
    assert row["acted_accuracy"] is None
    assert row["acted_accuracy_ci95"] == [0.0, 1.0]
    low, high = calibrator.wilson_interval(24, 30)
    assert 0.0 < low < 0.8 < high < 1.0


def test_calibration_metric_selects_the_floored_number(calibrator: types.ModuleType) -> None:
    answer = {"choice": "x", "confidence": 0.3, "probabilities": {"x": 0.6, "y": 0.4}}
    assert calibrator.metric_of(answer, "confidence") == pytest.approx(0.3)
    assert calibrator.metric_of(answer, "top_prob") == pytest.approx(0.6)


def test_calibration_replays_holdout_only_by_default(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay asks the served question shape and scores choice vs label."""
    seen: list[dict[str, Any]] = []
    path = tmp_path / "style.jsonl"
    rows = [
        {
            "id": "s1:style",
            "state": "system one",
            "instructions": "Which style?",
            "criteria": {"a": "A", "b": "B"},
            "label": "a",
            "source": {"split": "holdout", "scenario_id": "s1"},
        },
        {
            "id": "s2:style",
            "state": "system two",
            "instructions": "Which style?",
            "criteria": {"a": "A", "b": "B"},
            "label": "b",
            "source": {"split": "train", "scenario_id": "s2"},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def _fake_ask(
        base_url: str,
        model: str,
        row: dict[str, Any],
        question_id: str,
        timeout: float,
        criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        seen.append({"model": model, "state": row["state"], "qid": question_id, "criteria": criteria})
        return {"choice": "a", "confidence": 0.7, "probabilities": {"a": 0.7, "b": 0.3}}

    monkeypatch.setattr(calibrator, "ask", _fake_ask)
    assert calibrator.main(["--style", str(path), "--base-url", "http://x/v1", "--floors", "0.5"]) == 0
    assert [entry["state"] for entry in seen] == ["system one"]
    assert seen[0]["qid"] == "architecture_style"


def test_recommend_picks_the_lowest_qualifying_floor(calibrator: types.ModuleType) -> None:
    """The recommendation is the lowest floor clearing both bars, not the most accurate one."""
    pairs = [
        (0.9, True, "hexagonal"),
        (0.8, True, "hexagonal"),
        (0.7, False, "hexagonal"),
        (0.6, False, "hexagonal"),
        (0.2, False, "hexagonal"),
    ]
    table = calibrator.summarize(pairs, [0.5, 0.65, 0.85], 10)["floors"]
    # 0.5 acts on four rows at 0.50 accuracy (fails the 0.6 bar); 0.65 acts on two of
    # three at 0.667 — so the recommendation is the lowest floor that clears both.
    # min_acted=1: this test targets floor selection, not the small-n guard.
    recommended = calibrator.recommend(table, 0.6, 0.4, 1)
    assert recommended["floor"] == pytest.approx(0.65)
    assert "acted accuracy" in recommended["reason"]


def test_recommend_refuses_a_floor_below_the_min_acted_count(calibrator: types.ModuleType) -> None:
    """A perfect floor on too few overrides never qualifies, however accurate."""
    pairs = [(0.9, True, "hexagonal"), (0.9, True, "hexagonal")]
    table = calibrator.summarize(pairs, [0.5], 10)["floors"]
    assert calibrator.recommend(table, 0.8, 0.1, 10)["floor"] is None
    assert calibrator.recommend(table, 0.8, 0.1, 2)["floor"] == pytest.approx(0.5)
    qualified = calibrator.recommend(table, 0.8, 0.1, 2)
    assert qualified["acted"] == 2
    assert qualified["acted_accuracy_ci95"][0] <= 1.0 <= qualified["acted_accuracy_ci95"][1]


def test_recommend_reports_nothing_when_no_floor_qualifies(calibrator: types.ModuleType) -> None:
    pairs = [(0.9, False, "hexagonal"), (0.8, False, "hexagonal"), (0.7, False, "hexagonal")]
    table = calibrator.summarize(pairs, [0.5, 0.85], 10)["floors"]
    recommended = calibrator.recommend(table, 0.9, 0.3)
    assert recommended["floor"] is None
    assert "cannot be enabled safely" in recommended["reason"]


def test_recommend_reports_only_without_thresholds(calibrator: types.ModuleType) -> None:
    table = calibrator.summarize([(0.9, True, "hexagonal")], [0.5], 10)["floors"]
    assert calibrator.recommend(table, None, None)["floor"] is None
    assert calibrator.recommend(table, 0.5, None)["floor"] is None


def test_calibration_main_fails_when_no_floor_qualifies(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corpus whose head is never right must not yield an enableable floor."""
    path = tmp_path / "style.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "s1:style",
                "state": "system one",
                "instructions": "Which style?",
                "criteria": {"a": "A", "b": "B"},
                "label": "a",
                "source": {"split": "holdout", "scenario_id": "s1"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        calibrator,
        "ask",
        lambda *args, **kwargs: {"choice": "b", "confidence": 0.9, "probabilities": {"a": 0.1, "b": 0.9}},
    )
    report = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(path),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--min-acted-accuracy",
            "0.8",
            "--min-acted-rate",
            "0.1",
            "--report",
            str(report),
        ]
    )
    assert code == 2
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["recommended"]["floor"] is None
    assert written["floors"][0]["acted_accuracy"] == pytest.approx(0.0)


def _write_labels_source(path: Path) -> str:
    """A minimal STYLE_LABELS module; returns its sha256 for the test ledger."""
    source = 'STYLE_LABELS: dict[str, tuple[str, str]] = {\n    "s1": ("a", "rationale one"),\n}\n'
    path.write_text(source, encoding="utf-8")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _write_style_row(path: Path, rows: int = 1) -> None:
    """One acted override per row (all choice ``a``); ``rows`` clears the min-acted bar."""
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"s{index}:style",
                    "state": f"system {index}",
                    "instructions": "Which style?",
                    "criteria": {"a": "A", "b": "B"},
                    "label": "a",
                    "source": {"split": "holdout", "scenario_id": f"s{index}"},
                }
            )
            for index in range(1, rows + 1)
        ),
        encoding="utf-8",
    )


def test_calibration_main_refuses_to_recommend_without_human_review(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The table prints, but an unreviewed ledger blocks the floor and fails the run."""
    style = tmp_path / "style.jsonl"
    _write_style_row(style, rows=10)
    digest = _write_labels_source(tmp_path / "labels.py")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"status": "unreviewed", "reviews": {"style": []}}), encoding="utf-8")
    monkeypatch.setattr(
        calibrator,
        "ask",
        lambda *args, **kwargs: {"choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}},
    )
    report = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(style),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--min-acted-accuracy",
            "0.8",
            "--min-acted-rate",
            "0.1",
            "--labels",
            str(tmp_path / "labels.py"),
            "--review",
            str(review),
            "--report",
            str(report),
        ]
    )
    assert code == 2
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["label_review"]["certified"] is False
    assert written["recommended"]["floor"] is None
    assert "not 'reviewed'" in written["recommended"]["reason"]
    assert written["floors"][0]["acted_accuracy"] == pytest.approx(1.0)
    assert digest  # the ledger below pins these exact bytes


def test_calibration_main_recommends_once_the_ledger_certifies(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full coverage at the recorded sha unlocks the recommendation the table supports."""
    style = tmp_path / "style.jsonl"
    _write_style_row(style, rows=10)
    digest = _write_labels_source(tmp_path / "labels.py")
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "status": "reviewed",
                "reviewers": ["ada"],
                "reviews": {
                    "style": [
                        {
                            "scenario_id": "s1",
                            "label": "a",
                            "reviewer": "ada",
                            "rationale_ok": True,
                            "labels_sha256": digest,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        calibrator,
        "ask",
        lambda *args, **kwargs: {"choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}},
    )
    report = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(style),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--min-acted-accuracy",
            "0.8",
            "--min-acted-rate",
            "0.1",
            "--labels",
            str(tmp_path / "labels.py"),
            "--review",
            str(review),
            "--report",
            str(report),
        ]
    )
    assert code == 0
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["label_review"]["certified"] is True
    assert written["recommended"]["floor"] == pytest.approx(0.5)


def test_calibration_main_voids_coverage_when_labels_change(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A label edit after review changes the bytes, so the recorded sha no longer certifies."""
    style = tmp_path / "style.jsonl"
    _write_style_row(style, rows=10)
    digest = _write_labels_source(tmp_path / "labels.py")
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "status": "reviewed",
                "reviews": {
                    "style": [
                        {
                            "scenario_id": "s1",
                            "label": "a",
                            "reviewer": "ada",
                            "rationale_ok": True,
                            "labels_sha256": digest,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    with (tmp_path / "labels.py").open("a", encoding="utf-8") as handle:
        handle.write("# post-review tweak\n")
    monkeypatch.setattr(
        calibrator,
        "ask",
        lambda *args, **kwargs: {"choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}},
    )
    report = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(style),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--min-acted-accuracy",
            "0.8",
            "--min-acted-rate",
            "0.1",
            "--labels",
            str(tmp_path / "labels.py"),
            "--review",
            str(review),
            "--report",
            str(report),
        ]
    )
    assert code == 2
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["recommended"]["floor"] is None


def test_label_coverage_names_histograms_and_gaps(exporter: types.ModuleType) -> None:
    """The manifest records per-QUESTION label counts and the criteria keys with zero rows."""
    rows = [
        {
            "id": "s1:category",
            "state": "s",
            "questions": json.dumps({"category": {"criteria": {"a": "A", "b": "B"}}}),
            "gold": json.dumps({"category": {"label": "a"}}),
        },
        {
            "id": "s2:category",
            "state": "s",
            "questions": json.dumps({"category": {"criteria": {"a": "A", "b": "B"}}}),
            "gold": json.dumps({"category": {"label": "a"}}),
        },
        {
            "id": "s1:style",
            "state": "s",
            "questions": json.dumps({"architecture_style": {"criteria": {"x": "X", "y": "Y"}}}),
            "gold": json.dumps({"architecture_style": {"label": "y"}}),
        },
    ]
    coverage = exporter.label_coverage(rows)
    assert coverage["category"] == {"labels": {"a": 2}, "uncovered": ["b"]}
    assert coverage["architecture_style"] == {"labels": {"y": 1}, "uncovered": ["x"]}
    assert exporter.per_question_counts(rows) == {"architecture_style": 1, "category": 2}


def test_manifest_schema_matches_the_gate(exporter: types.ModuleType) -> None:
    """Exporter and gate must agree on the manifest layout, or the gate fails closed."""
    assert exporter.MANIFEST_SCHEMA_VERSION == _gate_module().SUPPORTED_MANIFEST_SCHEMA


def test_manifest_records_the_clm_pin_the_exporter_assumes(exporter: types.ModuleType) -> None:
    """The pin selects which upstream adapter revision the row layout must match."""
    args = exporter.parse_args(["--out", "/tmp/x", "--clm-pin", "abc1234"])
    assert args.clm_pin == "abc1234"
    assert exporter.parse_args(["--out", "/tmp/x"]).clm_pin == "d5f9ef0"


def test_calibrator_none_choice_matches_the_served_vocabulary(calibrator: types.ModuleType) -> None:
    """The stratified ``none`` must be the seam's own abstention key, not a lookalike."""
    assert calibrator.NONE_CHOICE == seam_prompts.STYLE_NONE_KEY


def test_split_summary_carries_counts_and_coverage(exporter: types.ModuleType) -> None:
    rows = [
        {
            "id": "s1:style",
            "state": "s",
            "questions": json.dumps({"architecture_style": {"criteria": {"x": "X", "y": "Y"}}}),
            "gold": json.dumps({"architecture_style": {"label": "x"}}),
        }
    ]
    summary = exporter.split_summary(rows, "holdout")
    assert summary["source_split"] == "holdout"
    assert summary["rows"] == 1
    assert summary["per_question"] == {"architecture_style": 1}
    assert summary["label_coverage"]["architecture_style"]["uncovered"] == ["y"]


def _typed_source_row(row_id: str, split: str, label: str = "a") -> dict[str, Any]:
    return {
        "id": row_id,
        "state": f"state for {row_id}",
        "type": "choice",
        "criteria": {"a": "A", "b": "B"},
        "label": label,
        "source": {"split": split, "scenario_id": row_id},
    }


def _stub_parquet(monkeypatch: pytest.MonkeyPatch, exporter: types.ModuleType) -> list[list[str]]:
    """Record the shard each row lands in without needing pyarrow."""
    shards: list[list[str]] = []

    def _write(rows: list[dict[str, Any]], path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        shards.append([str(row["id"]) for row in rows])
        return f"sha-{path.name}-{len(rows)}"

    monkeypatch.setattr(exporter, "write_parquet", _write)
    return shards


def test_export_routes_the_holdout_split_to_the_test_shard(
    exporter: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate's evidence is the family-disjoint holdout shard, not a train carve-out."""
    choice = tmp_path / "choice.jsonl"
    choice.write_text(
        "\n".join(
            json.dumps(row) for row in [_typed_source_row("s1", "train"), _typed_source_row("s2", "holdout", label="b")]
        ),
        encoding="utf-8",
    )
    style = tmp_path / "style.jsonl"
    style.write_text(
        "\n".join(
            json.dumps(row) for row in [_typed_source_row("s3", "train", label="b"), _typed_source_row("s4", "holdout")]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "data"
    shards = _stub_parquet(monkeypatch, exporter)
    assert exporter.main(["--choice", str(choice), "--style", str(style), "--out", str(out), "--workflow", "all"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == exporter.MANIFEST_SCHEMA_VERSION
    assert manifest["splits"]["train"]["source_split"] == "train"
    assert manifest["splits"]["test"]["source_split"] == "holdout"
    assert manifest["splits"]["train"]["per_question"] == {"architecture_style": 1, "category": 1}
    assert manifest["splits"]["test"]["per_question"] == {"architecture_style": 1, "category": 1}
    assert shards == [["s1", "s3"], ["s2", "s4"]]


def test_export_refuses_the_same_id_in_both_shards(
    exporter: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train and eval must never share a row; a corpus bug must stop the build."""
    choice = tmp_path / "choice.jsonl"
    choice.write_text(
        "\n".join(json.dumps(row) for row in [_typed_source_row("dup", "train"), _typed_source_row("dup", "holdout")]),
        encoding="utf-8",
    )
    style = tmp_path / "style.jsonl"
    style.write_text(
        "\n".join(
            json.dumps(row) for row in [_typed_source_row("s1", "train"), _typed_source_row("s2", "holdout", label="b")]
        ),
        encoding="utf-8",
    )
    _stub_parquet(monkeypatch, exporter)
    with pytest.raises(SystemExit, match="both splits"):
        exporter.main(["--choice", str(choice), "--style", str(style), "--out", str(tmp_path / "data")])


def test_rank_payload_artifact_is_served_shaped() -> None:
    """The committed artifact carries served bytes: document_text answers, holdout only.

    The derived set is exactly the HOLDOUT scenarios that have noul candidates — the
    deriver refuses a train row and skips a scenario with no candidates, so both halves of
    the expectation are recomputed here from the two inputs instead of pinned as a count.
    """
    payloads = [
        json.loads(line)
        for line in (REPO_ROOT / "qa" / "typed" / "rank_payloads.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corpus = json.loads((REPO_ROOT / "tests" / "benchmark" / "scenarios" / "full.json").read_text(encoding="utf-8"))
    scenarios = {item["scenario_id"]: item for item in corpus["scenarios"]}
    holdout = {sid for sid, item in scenarios.items() if item["split"] == "holdout"}
    noul = [
        json.loads(line)
        for line in (REPO_ROOT / "qa" / "typed" / "noul.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with_candidates = {(row.get("source") or {}).get("scenario_id") for row in noul} & holdout
    assert payloads, "qa/typed/rank_payloads.jsonl must not be empty"
    assert {payload["scenario_id"] for payload in payloads} == with_candidates
    for payload in payloads:
        assert payload["context"] == ""
        assert payload["question"] == scenarios[payload["scenario_id"]]["decision_statement"]
        assert len(payload["answers"]) == len(payload["answer_names"]) == 20
        # document_text, not the noul "name: intent" short string.
        assert all(len(answer) > 1000 and "\n" in answer for answer in payload["answers"])
        assert all(isinstance(name, str) and name for name in payload["answer_names"])
        # Graded relevance travels with the payload: aligned gains, at least one positive.
        assert len(payload["relevance"]) == 20
        assert all(gain in (0, 1, 2, 3) for gain in payload["relevance"])
        assert any(gain > 0 for gain in payload["relevance"])


@pytest.fixture(scope="module")
def preflight() -> types.ModuleType:
    return _load("clm_preflight", "clm_preflight.py")


def test_preflight_pin_matches_and_rejects_a_mismatch(preflight: types.ModuleType, tmp_path: Path) -> None:
    """The pin check is what stops a dataset exported for another CLM revision."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"clm_pin": "d5f9ef0"}), encoding="utf-8")
    assert "d5f9ef0" in preflight.check_pin(manifest, "d5f9ef0")
    with pytest.raises(SystemExit, match="adapter layout may have moved"):
        preflight.check_pin(manifest, "deadbee")
    manifest.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(SystemExit, match="dataset exported for CLM None"):
        preflight.check_pin(manifest, "d5f9ef0")


def test_preflight_refuses_a_served_alias_instead_of_a_snapshot(preflight: types.ModuleType, tmp_path: Path) -> None:
    """A bare alias is the 401 bug: it is not a local directory, so the build must stop."""
    with pytest.raises(SystemExit, match="not a local directory"):
        preflight.check_entity(tmp_path / "qwen3-8b")


def test_preflight_requires_consistent_arguments(preflight: types.ModuleType) -> None:
    """Neither flag alone may silently no-op a check."""
    with pytest.raises(SystemExit):
        preflight.parse_args([])
    with pytest.raises(SystemExit):
        preflight.parse_args(["--manifest", "m.json"])
    with pytest.raises(SystemExit):
        preflight.parse_args(["--expect-pin", "d5f9ef0"])


def test_repeat_consistency_counts_flips_and_acted_floor_crossings(calibrator: types.ModuleType) -> None:
    """A choice that moves across identical replays is a determinism defect, not noise.

    The probe reports it three ways because they fail differently: a flip at all
    (``style_choice_repeat_agreement``), the confidence spread (a head whose softmax is
    near-tied), and per-floor acted-set flips (a row that crosses the flap the floor sits on).
    """
    stable = {"scenario_id": "s1", "label": "a", "choice": "a", "confidence": 0.9}
    flips = {"scenario_id": "s2", "label": "b", "choice": "b", "confidence": 0.55}
    flips_again = {**flips, "choice": "a", "confidence": 0.75}
    straddle = {"scenario_id": "s3", "label": "a", "choice": "a", "confidence": 0.45}
    straddle_high = {**straddle, "confidence": 0.65}
    report = calibrator.consistency(
        [[stable, flips, straddle], [stable, flips_again, straddle_high]],
        [0.5, 0.7],
    )
    assert report is not None
    assert report["replays"] == 2
    # s1 and s3 keep their choice; only s2 flips — s3 moves on confidence alone, which the
    # acted-flip columns catch at the floors it straddles.
    assert report["stable_rows"] == 2
    assert report["style_choice_repeat_agreement"] == pytest.approx(2 / 3)
    assert report["unstable_rows"] == ["s2"]
    assert report["acted_flips"] == {"0.5": 1, "0.7": 1}
    assert report["max_confidence_spread"] == pytest.approx(0.2)


def test_repeat_consistency_needs_two_replays(calibrator: types.ModuleType) -> None:
    """One replay measures nothing; the block is None rather than a perfect score."""
    row = {"scenario_id": "s1", "label": "a", "choice": "a", "confidence": 0.9}
    assert calibrator.consistency([[row]], [0.5]) is None


def test_permute_criteria_keeps_the_answer_set_and_only_reorders_it(calibrator: types.ModuleType) -> None:
    """A permutation must preserve every item: reordering an answer set may not edit it."""
    import random

    criteria = {f"style-{index}": f"gloss {index}" for index in range(36)}
    shuffled = calibrator.permute_criteria(criteria, random.Random(7))
    assert list(shuffled) != list(criteria)
    assert dict(shuffled) == dict(criteria)
    assert list(shuffled) == list(calibrator.permute_criteria(criteria, random.Random(7)))


def test_order_invariance_separates_flips_from_canonical_drift(calibrator: types.ModuleType) -> None:
    """Order agreement is the gated number; canonical drift is reported beside it.

    A head can be perfectly order-stable yet disagree with the canonical run (all its
    permutations answer the same wrong way), which the agreement number alone would hide.
    """
    run_a = [
        {"scenario_id": "s1", "choice": "a", "confidence": 0.9},
        {"scenario_id": "s2", "choice": "a", "confidence": 0.8},
    ]
    run_b = [
        {"scenario_id": "s1", "choice": "a", "confidence": 0.9},
        {"scenario_id": "s2", "choice": "b", "confidence": 0.7},
    ]
    drift = [
        {"scenario_id": "s1", "choice": "b", "confidence": 0.6},
        {"scenario_id": "s2", "choice": "b", "confidence": 0.7},
    ]
    report = calibrator.order_invariance([run_a, run_a], ["a", "a"])
    assert report is not None
    assert report["style_choice_order_agreement"] == pytest.approx(1.0)
    assert report["canonical_agreement"] == pytest.approx(1.0)
    unstable = calibrator.order_invariance([run_a, run_b], ["a", "a"])
    assert unstable is not None
    assert unstable["style_choice_order_agreement"] == pytest.approx(0.5)
    assert unstable["unstable_rows"] == ["s2"]
    drifted = calibrator.order_invariance([drift, drift], ["a", "a"])
    assert drifted is not None
    assert drifted["style_choice_order_agreement"] == pytest.approx(1.0)
    assert drifted["canonical_agreement"] == pytest.approx(0.0)
    assert calibrator.order_invariance([], ["a"]) is None


def test_calibration_main_records_both_probes_in_the_report(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--repeats``/``--permutations`` write the two consistency blocks into the artifact."""
    style = tmp_path / "style.jsonl"
    _write_style_row(style, rows=4)
    _write_labels_source(tmp_path / "labels.py")
    seen_criteria: list[dict[str, Any] | None] = []

    def _ask(base_url: str, model: str, row: dict[str, Any], question_id: str, timeout: float, criteria: Any = None) -> dict[str, Any]:
        seen_criteria.append(criteria)
        return {"choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}}

    monkeypatch.setattr(calibrator, "ask", _ask)
    report_path = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(style),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--repeats",
            "3",
            "--permutations",
            "2",
            "--report",
            str(report_path),
        ]
    )
    assert code == 0
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["consistency"]["replays"] == 3
    assert written["consistency"]["style_choice_repeat_agreement"] == pytest.approx(1.0)
    assert written["order_invariance"]["orderings"] == 2
    assert written["order_invariance"]["style_choice_order_agreement"] == pytest.approx(1.0)
    # 4 rows x (3 repeats + 2 permutations) = 20 calls, and only the permuted runs override.
    assert len(seen_criteria) == 20
    assert sum(1 for criteria in seen_criteria if criteria is not None) == 8
    assert all(sorted(criteria) == ["a", "b"] for criteria in seen_criteria if criteria is not None)  # type: ignore[arg-type]


def test_calibration_main_refuses_a_floor_on_a_nondeterministic_head(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A floor fitted on a head that flips on identical bytes is a threshold on noise."""
    style = tmp_path / "style.jsonl"
    rows = 10  # clears the default --min-acted=10 guard, so only the repeat probe can block
    _write_style_row(style, rows=rows)
    digest = _write_labels_source(tmp_path / "labels.py")
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "status": "reviewed",
                "reviewers": ["ada"],
                "reviews": {
                    "style": [
                        {
                            "scenario_id": "s1",
                            "label": "a",
                            "reviewer": "ada",
                            "rationale_ok": True,
                            "labels_sha256": digest,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    calls = {"n": 0}

    def _flipping(base_url: str, model: str, row: dict[str, Any], question_id: str, timeout: float, criteria: Any = None) -> dict[str, Any]:
        # First replay all correct (so the floor table alone clears every bar); the second
        # replay flips every row. Only the repeat probe may then block the recommendation.
        calls["n"] += 1
        return {"choice": "a" if calls["n"] <= rows else "b", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}}

    monkeypatch.setattr(calibrator, "ask", _flipping)
    report_path = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(style),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--min-acted-accuracy",
            "0.8",
            "--min-acted-rate",
            "0.1",
            "--labels",
            str(tmp_path / "labels.py"),
            "--review",
            str(review),
            "--repeats",
            "2",
            "--report",
            str(report_path),
        ]
    )
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["recommended"]["floor"] is None
    assert "flip choice across identical replays" in written["recommended"]["reason"]
    assert written["consistency"]["stable_rows"] < written["consistency"]["rows"]
    assert code == 2
