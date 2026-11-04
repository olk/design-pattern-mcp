"""Build-time decision-model scripts: train/serve prompt parity and the seam gates.

These scripts run inside the sidecar image build (``docker/Dockerfile.tacit``), where a
wiring bug only surfaces after hours of state generation and teacher answering. Everything
expensive is stubbed here; what is asserted is the *wiring*: the trainer asks exactly the
questions the server asks (``src/decisions/seam_prompts.py``) at every seam, its ``main()``
walks generate → problems → sharded teacher → train → save, the style/rerank seams cannot
train on nothing, and the per-seam fidelity gates can fail the build. The calibration
replay is covered against a stubbed gateway.

The scripts are loaded by path (neither is an importable package), the same way
``test_client_examples.py`` loads the example clients.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from src.config import DecisionSeamConfig
from src.decisions import seam_prompts
from src.decisions.tacit import category_options, gate_state
from src.pipeline import (
    _gate_payload_analyze,
    _gate_payload_evaluate,
    _gate_payload_generate,
    _gate_payload_weights,
)
from src.reasoning.config import ReasoningConfig
from src.schemas.enums import QUALITY_AXES
from src.style_affinity import STYLE_AFFINITY
from tests.unit.conftest import REPO_ROOT

SCRIPTS = REPO_ROOT / "scripts" / "decision_models"
SEAM_PROMPTS = REPO_ROOT / "src" / "decisions" / "seam_prompts.py"


def _load(name: str, filename: str) -> types.ModuleType:
    """Import one script by path (registered in sys.modules so dataclasses resolve)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer() -> Iterator[types.ModuleType]:
    """The Tacit trainer, with stub ``anyjev``/``torch`` so importing it needs no GPU."""
    stub = types.ModuleType("anyjev")

    class _StubTacit:
        batch_size = 2

        def __init__(self) -> None:
            engine = types.SimpleNamespace(model=types.SimpleNamespace(device="cuda:0"))
            self.engine = engine  # noqa: F841 - read by main()'s teacher-device check

        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> _StubTacit:
            return cls()

    stub.Tacit = _StubTacit  # type: ignore[attr-defined]

    # ``teacher_examples`` imports these at call time, exactly as in the image.
    readout = types.ModuleType("anyjev.readout")
    readout.DEFAULT_SYSTEM = "system"  # type: ignore[attr-defined]
    readout.build_prompt = lambda state, question, permutation, system, labels: (  # type: ignore[attr-defined]
        state,
        question,
        tuple(permutation),
        system,
        tuple(labels),
    )
    readout.label_ids_for_perm = lambda question, ids, permutation: list(ids)  # type: ignore[attr-defined]
    readout.render_chat = lambda tok, spec: f"{spec[0]} || {spec[1].text}"  # type: ignore[attr-defined]
    readout.resolve_labels = lambda tok, question: (["label"], [7])  # type: ignore[attr-defined]
    tacit_module = types.ModuleType("anyjev.tacit")
    tacit_module.make_question = lambda text, options, kind: types.SimpleNamespace(  # type: ignore[attr-defined]
        text=text, options=list(options), kind=kind, k=len(options)
    )
    stub.readout = readout  # type: ignore[attr-defined]
    stub.tacit = tacit_module  # type: ignore[attr-defined]

    torch_stub = types.ModuleType("torch")
    cuda_stub = types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda _i: "stub-cuda:0"
    )
    torch_stub.cuda = cuda_stub  # type: ignore[attr-defined]
    torch_stub.manual_seed = lambda _seed: None  # type: ignore[attr-defined]
    torch_stub.cuda.manual_seed_all = lambda _seed: None  # type: ignore[attr-defined]
    installed = ("anyjev", "anyjev.readout", "anyjev.tacit", "torch")
    previous = {name: sys.modules.get(name) for name in installed}
    sys.modules["anyjev"] = stub
    sys.modules["anyjev.readout"] = readout
    sys.modules["anyjev.tacit"] = tacit_module
    sys.modules["torch"] = torch_stub
    yield _load("tacit_self_distill", "tacit_self_distill.py")
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_trainer_asks_the_questions_the_server_asks(trainer: types.ModuleType) -> None:
    """The student is trained on the serving bytes: same question text, same options."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    state = "System goal: goal\nDecision part: part"
    gate_state = "phase: analyze\ndescription: build a checkout"
    problems = trainer.build_problems([state], seams, [gate_state])
    assert len(problems) == 10
    assert [problem["kind"] for problem in problems] == ["choice"] + ["score"] * 8 + ["yes_no"]
    assert problems[0]["question"] == seam_prompts.category_question()
    assert problems[0]["options"] == category_options()
    for index, axis in enumerate(trainer.QUALITY_AXES):
        score = problems[1 + index]
        assert score["question"] == seam_prompts.axis_question(axis)
        assert score["options"] == list(seam_prompts.WEIGHT_LEVELS)
    assert problems[9]["question"] == seam_prompts.gate_question()
    assert problems[9]["options"] == ["Yes", "No"]
    assert problems[9]["state"] == gate_state


def test_load_topic_hints_filters_holdout(trainer: types.ModuleType, tmp_path: Path) -> None:
    """Holdout rows must never become topic hints (qa/README train-split-only)."""
    seed = tmp_path / "qa.jsonl"
    seed.write_text(
        "\n".join(
            [
                json.dumps({"split": "train", "question": "train q"}),
                json.dumps({"split": "holdout", "question": "hold q"}),
                json.dumps({"question": "no split"}),
            ]
        ),
        encoding="utf-8",
    )
    assert trainer.load_topic_hints(seed, 10) == ["train q"]


def test_normalise_state_keeps_serving_shape(trainer: types.ModuleType) -> None:
    """Scope/Constraints kept, junk dropped, goal line required, cap mirrored exactly."""
    cap = seam_prompts.DECISION_STATE_MAX_CHARS
    text = "\n".join(
        [
            "System goal: achieve checkout",
            "Components: cart, payment",
            "Decision part: choose the payment flow",
            "Scope: online orders only",
            "Constraints: pci-dss; idempotent",
            "Extra: junk line",
        ]
    )
    assert trainer._normalise_state(text, cap) == "\n".join(text.splitlines()[:5])
    assert trainer._normalise_state("System goal: g\nComponents: a", cap) is None
    # Serving always leads with the goal line (pipeline._decision_part_state); a state
    # without it never ships, so it must never become a training row.
    assert trainer._normalise_state("Decision part: choose a payment flow " + "x" * 20, cap) is None
    # Over-cap states truncate exactly where serving truncates (mid-line included).
    long_state = "System goal: g\nDecision part: p\nConstraints: " + "c" * 2000
    assert trainer._normalise_state(long_state, cap) == long_state[:cap]


def test_normalise_gate_state(trainer: types.ModuleType) -> None:
    """Gate states must be a serving shape: the declared key PREFIX, in order, capped."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    cap = seam_prompts.GATE_STATE_MAX_CHARS
    assert (
        trainer._normalise_gate_state("phase: analyze\ndescription: Build a checkout", seams)
        == "phase: analyze\ndescription: Build a checkout"
    )
    assert trainer._normalise_gate_state("phase: unknown\ndescription: d", seams) is None
    # ``analyze`` has ONE free-form field, so a multi-line description (real user text
    # contains newlines) is legal and must round-trip untouched.
    multiline = "phase: analyze\ndescription: reconcile the ledger\nunder retry: partially"
    assert trainer._normalise_gate_state(multiline, seams) == multiline
    # Undeclared keys, gaps and reordering are shapes the server never sends.
    assert trainer._normalise_gate_state("phase: analyze", seams) is None
    assert trainer._normalise_gate_state("phase: weights\ndescription: d\nfoo: bar", seams) is None
    assert trainer._normalise_gate_state("phase: weights\nrestated_goal: g", seams) is None
    assert (
        trainer._normalise_gate_state("phase: weights\nrestated_goal: g\ndescription: d\ndecision_parts: p", seams)
        is None
    )
    # A SHORT prefix is legal only when the text sits exactly at the cap: that is how a
    # long description truncates trailing keys, and it is what serving really sends.
    prefix = "phase: weights\ndescription: "
    assert trainer._normalise_gate_state(prefix + "x" * 200, seams) is None
    truncated = prefix + "x" * (cap - len(prefix))
    assert trainer._normalise_gate_state(truncated, seams) == truncated
    # A complete payload that also hits the cap keeps every key.
    full = "phase: analyze\ndescription: " + "x" * (cap - len("phase: analyze\ndescription: "))
    assert trainer._normalise_gate_state(full, seams) == full


def test_gate_states_are_the_bytes_serving_sends(trainer: types.ModuleType) -> None:
    """Built gate states round-trip through the server's own gate_state, cap included."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    state = "System goal: check out a cart\nComponents: Cart, Pricing\nDecision part: choose a pricing strategy"
    hints = ["A commerce platform recalculates cart totals per request. " * 40]  # over the cap on purpose
    gate_states = trainer.build_gate_states([state], hints, seams, len(seam_prompts.GATE_PHASES) * 2)
    assert len(gate_states) == len(seam_prompts.GATE_PHASES) * 2
    assert {text.splitlines()[0].removeprefix("phase: ") for text in gate_states} == set(seam_prompts.GATE_PHASES)
    for text in gate_states:
        phase = text.splitlines()[0].removeprefix("phase: ")
        required = seam_prompts.GATE_PHASE_PAYLOAD_KEYS[phase]
        payload: dict[str, str] = {}
        for line in text.splitlines()[1:]:
            key, separator, value = line.partition(": ")
            if separator and value:
                payload[key] = value
        # A prefix: all keys when the payload fits, fewer when the cap truncated a tail.
        assert tuple(payload) == required[: len(payload)]
        if "decision_parts" in payload:
            # Serving part ids are slug-shaped ("p-delivery"), never enumerated ("p1").
            assert all(segment.startswith("p-") for segment in payload["decision_parts"].split("; "))
        if len(payload) < len(required):
            assert len(text) == seam_prompts.GATE_STATE_MAX_CHARS
        # The server composes this string from the same payload (never a longer one).
        assert text.startswith(gate_state(phase, payload))
        assert trainer._normalise_gate_state(text, seams) == text


def test_gate_part_ids_are_serving_shaped(trainer: types.ModuleType) -> None:
    """Composed parts/applications use serving id shapes (p-<slug>), never p1/a1."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    state = "System goal: serve ads\nComponents: AdServer, Bidder\nDecision part: choose the ad rotation strategy"
    gate_states = trainer.build_gate_states([state], [], seams, len(seam_prompts.GATE_PHASES) * 2)

    def field(text: str, key: str) -> str:
        payload = dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)
        return payload[key]

    weights = next(text for text in gate_states if text.startswith("phase: weights"))
    assert all(segment.startswith("p-") for segment in field(weights, "decision_parts").split("; "))
    generate = next(text for text in gate_states if text.startswith("phase: generate"))
    assert all(segment.split(":")[0].startswith("p-") for segment in field(generate, "applications").split("; "))
    evaluate = next(text for text in gate_states if text.startswith("phase: evaluate"))
    assert all(segment.startswith("p-") for segment in field(evaluate, "parts").split("; "))


def test_long_aggregate_payloads_truncate_like_serving(trainer: types.ModuleType) -> None:
    """Phases with no description (generate/evaluate) overflow too: truncate, never crash."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    statement = "choose the reconciliation strategy under retry and partial failure " * 10
    component = "ReconciliationCoordinator" * 8
    states = [
        f"System goal: reconcile ledgers\nComponents: {component}, Ledger\nDecision part: {statement}{i}"
        for i in range(4)
    ]
    # 16 states cycle every phase through all four part spreads, so the biggest aggregates
    # (generate/evaluate) are built from four long parts and overflow the cap.
    gate_states = trainer.build_gate_states(states, [], seams, 16)
    assert len(gate_states) == 16  # nothing is dropped for being long
    capped = [text for text in gate_states if len(text) == seam_prompts.GATE_STATE_MAX_CHARS]
    assert capped, "the long aggregates must actually reach the cap"
    assert any(not text.startswith("phase: analyze") for text in capped)
    for text in gate_states:
        assert len(text) <= seam_prompts.GATE_STATE_MAX_CHARS
        assert trainer._normalise_gate_state(text, seams) == text


def test_gate_phases_match_the_pipeline_phases() -> None:
    """The trainer's phase list is the pipeline's, so no phase is trained that never runs."""
    assert tuple(ReasoningConfig().per_phase) == seam_prompts.GATE_PHASES


def test_gate_payload_keys_match_the_pipeline_builders() -> None:
    """GATE_PHASE_PAYLOAD_KEYS is the pipeline's payload table, not a parallel copy."""
    assert tuple(_gate_payload_analyze("d")) == seam_prompts.GATE_PHASE_PAYLOAD_KEYS["analyze"]
    assert tuple(_gate_payload_weights("d", "g", "p")) == seam_prompts.GATE_PHASE_PAYLOAD_KEYS["weights"]
    assert tuple(_gate_payload_generate("g", "a")) == seam_prompts.GATE_PHASE_PAYLOAD_KEYS["generate"]
    assert tuple(_gate_payload_evaluate("p", "a", "p")) == seam_prompts.GATE_PHASE_PAYLOAD_KEYS["evaluate"]
    assert set(seam_prompts.GATE_PHASE_PAYLOAD_KEYS) == set(seam_prompts.GATE_PHASES)


def test_confidence_floor_matches_the_server_default(trainer: types.ModuleType) -> None:
    """Training keeps rows the seam would act on: its floor defaults to the server's."""
    assert DecisionSeamConfig().confidence_floor == trainer._DEFAULT_CONFIDENCE_FLOOR


def test_quality_axes_match_the_enum(trainer: types.ModuleType) -> None:
    """The trainer's axis list is the canonical tuple, in order (same drift class as above)."""
    assert QUALITY_AXES == trainer.QUALITY_AXES


def test_retention_floor_is_kind_aware(trainer: types.ModuleType) -> None:
    """Score rows keep graded mass; choice rows clear the category seam's floor."""
    assert trainer.retention_floor("score", 0.5, 0.0) == 0.0
    assert trainer.retention_floor("choice", 0.5, 0.0) == 0.5
    assert trainer.retention_floor("choice", 0.9, 0.0) == 0.9
    # The gate needs its boundary rows: no max floor binds yes_no at any setting (a
    # raised --confidence-floor must not drop near-boundary calibration rows).
    assert trainer.retention_floor("yes_no", 0.9, 0.0) == 0.0


def test_fidelity_metrics_hand_computed(trainer: types.ModuleType) -> None:
    """Per SEAM: agreement, Brier, floor accuracy, and expected-level MAE follow serving."""
    category = trainer._fidelity_metrics(
        [([0.9, 0.1], [1.0, 0.0], "choice", "category"), ([0.4, 0.6], [0.5, 0.5], "choice", "category")],
        0.5,
        0.5,
    )["category"]
    assert category["n"] == 2
    assert category["agreement"] == 0.5
    assert abs(category["brier"] - 0.02) < 1e-9
    assert category["floor_accuracy"] == 0.5
    # The same kind under two seams stays two entries: a seam is the reporting unit now.
    split = trainer._fidelity_metrics(
        [([0.9, 0.1], [1.0, 0.0], "choice", "style-stage1"), ([0.9, 0.1], [1.0, 0.0], "choice", "style-stage2")],
        0.5,
        0.5,
    )
    assert set(split) == {"style-stage1", "style-stage2"}
    teacher = [0.1, 0.3, 0.4, 0.15, 0.05]
    student = [0.0, 0.0, 1.0, 0.0, 0.0]
    expected_teacher = sum(index * value for index, value in enumerate(teacher)) / 4
    expected_student = sum(index * value for index, value in enumerate(student)) / 4
    weights = trainer._fidelity_metrics([(student, teacher, "score", "weights")], 0.5, 0.5)["weights"]
    assert abs(weights["expected_level_mae"] - abs(expected_student - expected_teacher)) < 1e-9
    # The gate seam's floor accuracy is ACT-decision agreement (it compares p(Yes) with its
    # floor): a student that would not act where the teacher would is a miss.
    gate = trainer._fidelity_metrics(
        [([0.9, 0.1], [1.0, 0.0], "yes_no", "gate"), ([0.4, 0.6], [0.55, 0.45], "yes_no", "gate")],
        0.5,
        0.5,
    )["gate"]
    assert gate["floor_accuracy"] == 0.5
    # At a raised gate floor the act rule bites: argmax agreement (both "Yes") must NOT
    # count as correct when the student fires (0.8 >= 0.7) and the teacher would not.
    raised = trainer._fidelity_metrics([([0.8, 0.2], [0.6, 0.4], "yes_no", "gate")], 0.5, 0.7)["gate"]
    assert raised["floor_accuracy"] == 0.0
    # The rerank seam reports agreement alone: serving consumes the whole p(Yes) distribution.
    rerank = trainer._fidelity_metrics([([0.8, 0.2], [0.6, 0.4], "yes_no", "rerank")], 0.5, 0.7)["rerank"]
    assert "floor_accuracy" not in rerank
    assert rerank["agreement"] == 1.0
    # The acting floor is the served one, not a literal: raising it drops the acted cases.
    assert (
        trainer._fidelity_metrics([([0.9, 0.1], [1.0, 0.0], "choice", "category")], 0.95, 0.5)["category"][
            "floor_accuracy"
        ]
        == 0.0
    )
    assert trainer._fidelity_metrics([], 0.5, 0.5) == {}


def test_seam_gate_failures_certify_only_sampled_seams(trainer: types.ModuleType) -> None:
    """Certified seams are gated; thin or missing seams are skipped, not failed."""
    clean = {
        "category": {"n": 50, "agreement": 0.9},
        "style-stage1": {"n": 50, "agreement": 0.9},
        "style-stage2": {"n": 50, "agreement": 0.9},
        "rerank": {"n": 50, "agreement": 0.9},
        "gate": {"n": 50, "agreement": 0.9},
        "weights": {"n": 50, "agreement": 0.9, "expected_level_mae": 0.1},
    }
    assert trainer.seam_gate_failures(clean, 0.6, 0.2) == []
    weak = dict(clean)
    weak["rerank"] = {"n": 50, "agreement": 0.3}
    weak["weights"] = {"n": 50, "agreement": 0.9, "expected_level_mae": 0.4}
    failures = trainer.seam_gate_failures(weak, 0.6, 0.2)
    assert any("'rerank'" in failure and "agreement" in failure for failure in failures)
    assert any("expected-level MAE" in failure for failure in failures)
    # A thin sample cannot certify: 2 rows at 0.5 is noise, not an under-trained seam.
    thin = dict(clean)
    thin["gate"] = {"n": 2, "agreement": 0.5}
    assert trainer.seam_gate_failures(thin, 0.6, 0.2) == []
    # A seam absent from the eval split is skipped for the same reason.
    missing = {name: entry for name, entry in clean.items() if name != "style-stage2"}
    assert trainer.seam_gate_failures(missing, 0.6, 0.2) == []
    # No eval rows at all (stub/quick-smoke run): nothing is certified, nothing fails.
    assert trainer.seam_gate_failures({}, 0.6, 0.2) == []
    # The evidence threshold is a parameter, so a caller can certify at a smaller sample.
    assert trainer.seam_gate_failures(thin, 0.6, 0.2, min_eval_rows=2) != []


def test_split_examples_stratifies_by_seam(trainer: types.ModuleType) -> None:
    """Every certifiable seam keeps eval rows; thin seams and tiny corpora stay in training."""

    def rows(seam: str, count: int) -> list[Any]:
        return [
            trainer.Example(prompt=f"{seam}-{index}", kind="choice", label_ids=[1, 2], target=[0.5, 0.5], seam=seam)
            for index in range(count)
        ]

    examples = rows("category", 100) + rows("rerank", 100) + rows("gate", 5) + rows("weights", 100)
    train, eval_rows = trainer._split_examples(examples, 0.1, 13)
    assert len(train) + len(eval_rows) == len(examples)
    # gate has 5 rows — too thin to certify, so it must not lose training rows to eval.
    assert {row.seam for row in eval_rows} == {"category", "rerank", "weights"}
    # Deterministic for a fixed seed.
    _, again = trainer._split_examples(examples, 0.1, 13)
    assert [row.prompt for row in eval_rows] == [row.prompt for row in again]
    # A corpus under 20 rows keeps every row for training.
    tiny = rows("gate", 19)
    assert trainer._split_examples(tiny, 0.1, 13) == (tiny, [])


def test_false_act_rate_counts_clearing_the_floor(trainer: types.ModuleType) -> None:
    """The probe metric is P(max(student) >= floor) on below-floor rows, else None."""
    assert trainer.false_act_rate([[0.9, 0.05, 0.05], [0.4, 0.3, 0.3]], 0.5) == 0.5
    assert trainer.false_act_rate([[0.51, 0.49]], 0.5) == 1.0
    assert trainer.false_act_rate([], 0.5) is None


def test_build_problems_windows_eight_axes_across_states(trainer: types.ModuleType) -> None:
    """A narrow axis window still covers every axis across the corpus, not just the first."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    states = [f"System goal: g{i}\nDecision part: p{i}" for i in range(len(trainer.QUALITY_AXES))]
    problems = trainer.build_problems(states, seams, [], axes_per_state=3)
    score_axes = [problem["question"] for problem in problems if problem["kind"] == "score"]
    assert len(score_axes) == 3 * len(states)
    assert set(score_axes) == {seams.axis_question(axis) for axis in trainer.QUALITY_AXES}
    assert len(trainer.build_problems(states, seams, [])) == len(states) * (len(trainer.QUALITY_AXES) + 1)


class _FakeTeacher:
    """Stands in for anyjev's Tacit: ``decide_batch`` keyed by the problem's state."""

    batch_size = 2
    tok = object()

    def __init__(self, results: dict[str, dict[str, Any]]) -> None:
        self._results = results
        self.sent: list[dict[str, Any]] = []

    def decide_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.sent.extend(items)
        return [self._results[item["state"]] for item in items]


def test_answer_and_examples_split_reproduces_retention(trainer: types.ModuleType) -> None:
    """Retention is per kind, and a result missing an option is dropped, never rescaled.

    ``answer_problems`` is the shardable answer phase (wire keys only, order preserved);
    ``examples_from_results`` is the unchanged retention/example loop, now tagging each row
    with its seam so the build can gate per seam.
    """
    labels = list(seam_prompts.CATEGORY_SLUGS)
    unsure = dict.fromkeys(labels, 0.1)
    unsure[labels[0]] = 0.2  # renormalized max 0.2: below the acting floor
    sure = dict.fromkeys(labels, 0.05)
    sure[labels[1]] = 0.6  # clears the acting floor
    results = {
        "diffuse": {
            "route": "cot",
            "probs": {"none": 0.1, "low": 0.3, "moderate": 0.4, "high": 0.15, "critical": 0.05},
        },
        "unsure": {"route": "cot", "probs": unsure},
        "sure": {"route": "cot", "probs": sure},
        "partial": {"route": "cot", "probs": {"Yes": 1.0}},  # "No" missing
        "unrouted": {"route": "one_forward", "probs": {"Yes": 0.9, "No": 0.1}},
    }
    fake = _FakeTeacher(results)
    problems = [
        {
            "state": "diffuse",
            "question": "q",
            "options": list(seam_prompts.WEIGHT_LEVELS),
            "kind": "score",
            "seam": "weights",
        },
        {"state": "unsure", "question": "q", "options": labels, "kind": "choice", "seam": "style-stage2"},
        {"state": "sure", "question": "q", "options": labels, "kind": "choice", "seam": "style-stage2"},
        {"state": "partial", "question": "q", "options": ["Yes", "No"], "kind": "yes_no", "seam": "gate"},
        {"state": "unrouted", "question": "q", "options": ["Yes", "No"], "kind": "yes_no", "seam": "rerank"},
    ]
    answered = trainer.answer_problems(fake, problems)
    assert answered == [results[problem["state"]] for problem in problems]
    # Only the four wire keys cross the gateway: ``seam`` is trainer bookkeeping.
    assert fake.sent == [
        {key: problem[key] for key in ("state", "question", "options", "kind")} for problem in problems
    ]
    examples, dropped, probe = trainer.examples_from_results(fake, problems, answered, 0.5, 0.0)
    assert [example.kind for example in examples] == ["score", "choice"]
    assert [example.seam for example in examples] == ["weights", "style-stage2"]
    # Diffuse score rows survive: serving consumes the graded expected level.
    assert abs(sum(examples[0].target) - 1.0) < 1e-9
    assert abs(examples[0].target[2] - 0.4) < 1e-9
    # The unsure choice is filtered; the partial yes_no is rejected, not renormalized to [1, 0].
    assert dropped == {
        "choice": {"unrouted": 0, "shape": 0, "unsharp": 1},
        "score": {"unrouted": 0, "shape": 0, "unsharp": 0},
        "yes_no": {"unrouted": 1, "shape": 1, "unsharp": 0},
    }
    assert examples[0].label_ids == [7]
    assert examples[1].prompt.startswith("sure || ")  # the rebuilt serving prompt, not the raw state
    # The below-floor choice row survives as the false-act probe, on the same prompt bytes.
    assert [example.kind for example in probe] == ["choice"]
    assert probe[0].prompt.startswith("unsure || ")
    assert abs(max(probe[0].target) - 0.2) < 1e-9


def test_build_style_problems_train_both_stages_for_every_shortlist(trainer: types.ModuleType) -> None:
    """Two questions per state, every category shortlist, always inside the option cap."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    states = [f"description {index}" for index in range(len(seams.CATEGORY_SLUGS))]
    problems = trainer.build_style_problems(states, seams)
    assert len(problems) == 2 * len(states)
    stage1 = problems[0::2]
    stage2 = problems[1::2]
    for problem in stage1:
        assert problem["seam"] == "style-stage1"
        assert problem["question"] == seams.category_question()
        assert problem["options"] == list(seams.CATEGORY_SLUGS)
        assert problem["kind"] == "choice"
    # The shortlist rotates over ALL nine categories, so every stage-2 option set trains.
    covered = {problem["options"][0] for problem in stage2}
    assert covered == {seams.styles_for_category(slug)[0] for slug in seams.CATEGORY_SLUGS}
    for problem, slug in zip(stage2, seams.CATEGORY_SLUGS, strict=True):
        styles = seams.styles_for_category(slug)
        assert problem["seam"] == "style-stage2"
        assert problem["question"] == seams.style_question(styles)
        assert problem["options"] == [*styles, seams.STYLE_NONE_KEY]
        assert len(problem["options"]) <= 26


def test_build_style_problems_caps_each_state(trainer: types.ModuleType) -> None:
    """Serving truncates the description, so the trained bytes are the truncated bytes."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    cap = seams.DECISION_STATE_MAX_CHARS
    problems = trainer.build_style_problems(["d" * (cap + 100)], seams)
    assert [problem["state"] for problem in problems] == ["d" * cap, "d" * cap]


def test_load_rerank_pairs_keeps_train_rows_only(trainer: types.ModuleType, tmp_path: Path) -> None:
    """Holdout pairs are eval material: they must never reach the training weights."""
    path = tmp_path / "rerank_pairs.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"split": "train", "statement": "s1", "document": "d1"}),
                json.dumps({"split": "holdout", "statement": "s2", "document": "d2"}),
                json.dumps({"split": "train", "statement": "", "document": "d3"}),
                "not json",
            ]
        ),
        encoding="utf-8",
    )
    assert trainer.load_rerank_pairs(path) == [{"statement": "s1", "document": "d1"}]


def test_build_rerank_problems_use_serving_bytes_and_stride_a_budget(trainer: types.ModuleType) -> None:
    """One ``yes_no`` per pair with the served state/question; a positive budget strides."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    pairs = [{"statement": f"statement {index}", "document": f"document {index}"} for index in range(10)]
    full = trainer.build_rerank_problems(pairs, seams, 0)
    assert len(full) == 10  # 0 = all: the max-effort default
    assert full[0] == {
        "state": seams.rerank_state("statement 0", "document 0"),
        "question": seams.RERANK_RELEVANCE_QUESTION,
        "options": ["Yes", "No"],
        "kind": "yes_no",
        "seam": "rerank",
    }
    strided = trainer.build_rerank_problems(pairs, seams, 4)
    assert len(strided) == 4
    # A stride, not a prefix: a reduced run still spreads across the corpus.
    assert strided[0]["state"] == seams.rerank_state("statement 0", "document 0")
    assert strided[1]["state"] != seams.rerank_state("statement 1", "document 1")
    assert len({problem["state"] for problem in strided}) == 4


def test_shard_command_parses_with_the_real_cli(trainer: types.ModuleType, tmp_path: Path) -> None:
    """The child argv must satisfy ``parse_args`` — a required flag missing here kills sharding.

    This is the failure mode the fake-Popen tests cannot see: the child re-parses its own
    argv, so a newly required flag (``--seam-prompts`` did exactly this) makes every shard
    exit 2 in argparse and the parent abort with "a teacher shard failed" — after the
    hours-long teacher pass has already started.
    """
    args = types.SimpleNamespace(
        init_dir=tmp_path / "init",
        out=tmp_path / "out",
        seam_prompts=SEAM_PROMPTS,
        repo="morriszjm/Tacit-4B",
        cot_max_tokens=512,
        batch=2,
    )
    command = trainer._shard_command(
        args, shard=1, total_shards=3, problems_path=tmp_path / "p.json", out_path=tmp_path / "s.jsonl"
    )
    # Exactly what the child gets: interpreter, script path, then the flags.
    parsed = trainer.parse_args(command[2:])
    assert parsed._answer_shard == 1
    assert parsed._shard_total == 3
    assert parsed._shard_problems == str(tmp_path / "p.json")
    assert parsed._shard_out == str(tmp_path / "s.jsonl")
    assert parsed.init_dir == str(tmp_path / "init")
    assert parsed.out == str(tmp_path / "out")
    assert parsed.seam_prompts == str(SEAM_PROMPTS)
    assert parsed.cot_max_tokens == 512
    assert parsed.batch == 2


def test_answer_shard_writes_globally_indexed_rows(
    trainer: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Shard ``i`` of ``n`` answers ``problems[i::n]`` and tags each row with its global index."""
    problems = [{"state": f"s{index}", "question": "q", "options": ["a"], "kind": "choice"} for index in range(7)]
    problems_path = tmp_path / "problems.json"
    problems_path.write_text(json.dumps(problems), encoding="utf-8")
    monkeypatch.setattr(trainer, "_load_teacher", lambda _args: object())
    monkeypatch.setattr(
        trainer,
        "answer_problems",
        lambda _teacher, sliced: [{"route": "cot", "state": item["state"]} for item in sliced],
    )
    seen: dict[int, int] = {}
    for shard, total in ((0, 3), (1, 3), (2, 3)):
        out = tmp_path / f"shard-{shard}.jsonl"
        args = types.SimpleNamespace(
            _answer_shard=shard, _shard_total=total, _shard_problems=str(problems_path), _shard_out=str(out)
        )
        assert trainer._answer_shard(args) == 0
        for line in out.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            assert entry["result"]["state"] == problems[entry["index"]]["state"]
            seen[entry["index"]] = seen.get(entry["index"], 0) + 1
    assert seen == dict.fromkeys(range(7), 1)  # full coverage, no overlap


def test_teacher_answers_merges_shards_in_order(trainer: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent merges by index and refuses a partial or failed shard set."""

    class _FakeChild:
        def __init__(
            self, command: list[str], env: dict[str, str] | None = None, *, code: int = 0, drop_last: bool = False
        ) -> None:
            del env  # the child pins CUDA_VISIBLE_DEVICES; the fake only needs the argv
            out = Path(command[command.index("--_shard-out") + 1])
            count = int(command[command.index("--_shard-total") + 1])
            shard = int(command[command.index("--_answer-shard") + 1])
            problems = json.loads(Path(command[command.index("--_shard-problems") + 1]).read_text(encoding="utf-8"))
            sliced = problems[shard::count]
            if drop_last:
                sliced = sliced[:-1]
            out.write_text(
                "\n".join(
                    json.dumps({"index": shard + offset * count, "result": {"state": item["state"]}})
                    for offset, item in enumerate(sliced)
                ),
                encoding="utf-8",
            )
            self._code = code

        def wait(self) -> int:
            return self._code

    problems = [{"state": f"s{index}", "question": "q", "options": ["a"], "kind": "choice"} for index in range(6)]
    args = types.SimpleNamespace(
        teacher_shards=2,
        seam_prompts=SEAM_PROMPTS,
        init_dir=Path("/models/tacit-4b"),
        out=Path("/out/tacit-4b"),
        repo="morriszjm/Tacit-4B",
        cot_max_tokens=2048,
        batch=4,
    )
    monkeypatch.setattr(trainer.subprocess, "Popen", _FakeChild)
    merged, shards = trainer.teacher_answers(object(), args, problems)
    assert shards == 2
    assert merged == [{"state": problem["state"]} for problem in problems]

    monkeypatch.setattr(trainer.subprocess, "Popen", lambda command, env=None: _FakeChild(command, env, code=1))
    with pytest.raises(SystemExit, match="teacher shard failed"):
        trainer.teacher_answers(object(), args, problems)

    monkeypatch.setattr(trainer.subprocess, "Popen", lambda command, env=None: _FakeChild(command, env, drop_last=True))
    with pytest.raises(SystemExit, match="covered"):
        trainer.teacher_answers(object(), args, problems)


def test_assemble_problems_prefers_hints_then_generated_descriptions(
    trainer: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ch2 states are real serving hint bytes first; generated prose fills the rest."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    monkeypatch.setattr(
        trainer,
        "generate_descriptions",
        lambda _tacit, _hints, count, _max_new, _cap: [f"generated {index}" for index in range(count)],
    )
    args = types.SimpleNamespace(
        problems=36,
        style_states=0,
        rerank_pairs=str(_rerank_pairs_file(tmp_path)),
        rerank_problems=0,
        max_new_tokens=256,
    )
    problems, style_states, style_problems, rerank_problems = trainer._assemble_problems(
        object(),
        ["hint one", "hint two"],
        seams,
        args,
        ["System goal: g\nDecision part: p"],
        ["phase: analyze\ndescription: d"],
        8,
    )
    assert style_states == ["hint one", "hint two", "generated 0"]  # 36 // 12 = 3 states
    assert len(style_problems) == 6
    assert len(rerank_problems) == 2
    assert len(problems) == len(style_problems) + len(rerank_problems) + (1 + 8 + 1)
    # When the hint corpus already fills the budget, no teacher time is spent generating
    # descriptions that would be truncated away; every hint is still kept.
    args.style_states = 1
    monkeypatch.setattr(
        trainer,
        "generate_descriptions",
        lambda *_args: (_ for _ in ()).throw(AssertionError("generation must be skipped")),
    )
    _, hint_only, _, _ = trainer._assemble_problems(
        object(),
        ["hint one", "hint two"],
        seams,
        args,
        ["System goal: g\nDecision part: p"],
        [],
        8,
    )
    assert hint_only == ["hint one", "hint two"]


def test_assemble_problems_refuses_an_untrained_seam(
    trainer: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Style and rerank must both train: an empty seam aborts the build unless --skip."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    # No generated prose: the hint text alone carries the style seam in the passing cases.
    monkeypatch.setattr(trainer, "generate_descriptions", lambda *_args: [])
    holdout_only = tmp_path / "holdout_only.jsonl"
    holdout_only.write_text(json.dumps({"split": "holdout", "statement": "s", "document": "d"}), encoding="utf-8")

    def assemble(pairs: str) -> Any:
        args = types.SimpleNamespace(
            problems=12, style_states=0, rerank_pairs=pairs, rerank_problems=0, max_new_tokens=256
        )
        return trainer._assemble_problems(object(), ["hint"], seams, args, ["System goal: g\nDecision part: p"], [], 8)

    with pytest.raises(SystemExit, match="--rerank-pairs"):
        assemble("")
    with pytest.raises(SystemExit, match="0 pairs"):
        assemble(str(holdout_only))
    # No hints and no generated descriptions: nothing to train the two-stage style seam on.
    no_hints = types.SimpleNamespace(
        problems=12,
        style_states=0,
        rerank_pairs=str(_rerank_pairs_file(tmp_path)),
        rerank_problems=0,
        max_new_tokens=256,
    )
    with pytest.raises(SystemExit, match="0 description-shaped states"):
        trainer._assemble_problems(object(), [], seams, no_hints, ["System goal: g\nDecision part: p"], [], 8)
    # ``--skip`` needs no clause here: main returns from skip_training before this point.
    assert "skip" not in trainer._assemble_problems.__code__.co_names


def _main_fakes(
    trainer: types.ModuleType,
    seen: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    *,
    kinds: tuple[str, ...] = ("choice", "score", "yes_no"),
    descriptions: tuple[str, ...] = ("generated description",),
) -> None:
    """Patch the expensive stages of ``main()`` so the wiring is what gets asserted."""

    def fake_states(
        _tacit: Any, _hints: Any, problems: int, _seed: int, _max_new: int, max_state_chars: int
    ) -> list[str]:
        seen["state_cap"] = max_state_chars
        return [f"System goal: g{i}\nDecision part: p{i}" for i in range(problems)]

    def fake_gate_states(states: list[str], hints: list[str], _seams: Any, count: int) -> list[str]:
        seen["gate_sources"] = len(states)
        seen["gate_hints"] = list(hints)
        return [f"phase: analyze\ndescription: g{i}" for i in range(count)]

    def fake_descriptions(_tacit: Any, _hints: Any, count: int, _max_new: int, max_state_chars: int) -> list[str]:
        seen["description_cap"] = max_state_chars
        seen["description_count"] = count
        return [f"{text} {index}" for index, text in enumerate(descriptions)][:count]

    def fake_teacher(_tacit: Any, args: Any, problems: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        seen["kinds"] = [problem["kind"] for problem in problems]
        seen["seams"] = [problem.get("seam") for problem in problems]
        seen["teacher_shards_arg"] = args.teacher_shards
        return [{"route": "cot"} for _ in problems], 1

    def fake_examples(
        _tacit: Any,
        problems: list[dict[str, Any]],
        _results: Any,
        floor: float,
        score_floor: float,
    ) -> tuple[list[Any], dict[str, dict[str, int]], list[Any]]:
        seen["floors"] = (floor, score_floor)
        seen["problem_count"] = len(problems)
        seam_for_kind = {
            "choice": ("category", "style-stage1", "style-stage2"),
            "score": ("weights",),
            "yes_no": ("gate", "rerank"),
        }
        rows = [
            trainer.Example(
                prompt=f"prompt-{kind}-{repeat}",
                kind=kind,
                label_ids=[1, 2],
                target=[0.6, 0.4],
                seam=seam_for_kind[kind][repeat % len(seam_for_kind[kind])],
            )
            for repeat in range(7)  # 21 rows: past the >=20 bar, so the eval split is real
            for kind in kinds
        ]
        probe = [trainer.Example(prompt="probe-choice", kind="choice", label_ids=[1, 2], target=[0.55, 0.45])]
        dropped_counts = {kind: {"unrouted": 0, "shape": 0, "unsharp": 0} for kind in ("choice", "score", "yes_no")}
        return rows, dropped_counts, probe

    def fake_train(_init: Path, examples: list[Any], args: Any) -> object:
        seen["examples"] = len(examples)
        seen["epochs"] = args.epochs
        return object()

    def fake_save(_model: Any, _init: Path, out_dir: Path) -> object:
        out_dir.mkdir(parents=True, exist_ok=True)
        seen["saved_model"] = object()  # sentinel: the merged artifact main() must evaluate
        return seen["saved_model"]

    def fake_eval(
        _model: Any, _init: Path, _rows: Any, _batch: int, choice_floor: float, gate_floor: float
    ) -> dict[str, Any]:
        seen["eval_model"] = _model
        seen["eval_floors"] = (choice_floor, gate_floor)
        return {seam: {"n": 3, "agreement": 0.9, "expected_level_mae": 0.1} for seam in trainer._GATED_SEAMS} | {
            "weights": {"n": 3, "agreement": 0.9, "expected_level_mae": 0.1}
        }

    def fake_probe(_model: Any, _init: Path, rows: Any, _batch: int, floor: float) -> float:
        seen["probe_model"] = _model
        seen["probe_floor"] = floor
        seen["probe_rows"] = len(rows)
        return 0.25

    monkeypatch.setattr(trainer, "generate_states", fake_states)
    monkeypatch.setattr(trainer, "build_gate_states", fake_gate_states)
    monkeypatch.setattr(trainer, "generate_descriptions", fake_descriptions)
    monkeypatch.setattr(trainer, "teacher_answers", fake_teacher)
    monkeypatch.setattr(trainer, "examples_from_results", fake_examples)
    monkeypatch.setattr(trainer, "train_student", fake_train)
    monkeypatch.setattr(trainer, "save_checkpoint", fake_save)
    monkeypatch.setattr(trainer, "evaluate_student", fake_eval)
    monkeypatch.setattr(trainer, "probe_false_act_rate", fake_probe)


def _rerank_pairs_file(tmp_path: Path, rows: int = 2) -> Path:
    """A tiny rerank-pairs artifact: train rows the trainer may use, plus one holdout."""
    path = tmp_path / "rerank_pairs.jsonl"
    path.write_text(
        "\n".join(
            [
                *[
                    json.dumps({"split": "train", "statement": f"s{index}", "document": f"d{index}"})
                    for index in range(rows)
                ],
                json.dumps({"split": "holdout", "statement": "hs", "document": "hd"}),
            ]
        ),
        encoding="utf-8",
    )
    return path


def _main_args(init_dir: Path, out_dir: Path, pairs: Path, *extra: str) -> list[str]:
    """The mandatory ``main`` argv, with the J1 pairs the seam requires."""
    return [
        "--init-dir",
        str(init_dir),
        "--out",
        str(out_dir),
        "--seam-prompts",
        str(SEAM_PROMPTS),
        "--problems",
        "10",
        "--rerank-pairs",
        str(pairs),
        *extra,
    ]


def test_trainer_main_walks_generate_to_save(
    trainer: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main()`` must thread states → problems → teacher → train → save, not unpack them."""
    seen: dict[str, Any] = {}
    _main_fakes(trainer, seen, monkeypatch)

    init_dir = tmp_path / "init"
    init_dir.mkdir()
    (init_dir / "config.json").write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "out"
    code = trainer.main(_main_args(init_dir, out_dir, _rerank_pairs_file(tmp_path)))
    assert code == 0
    # 10 design problems (category + 8 axes + gate), 2 style questions, 2 train rerank pairs.
    assert seen["kinds"] == ["choice"] + ["score"] * 8 + ["yes_no"] + ["choice"] * 2 + ["yes_no"] * 2
    assert seen["seams"] == [
        "category",
        "weights",
        "weights",
        "weights",
        "weights",
        "weights",
        "weights",
        "weights",
        "weights",
        "gate",
        "style-stage1",
        "style-stage2",
        "rerank",
        "rerank",
    ]
    assert seen["floors"] == (0.5, 0.0)  # the answer floor defaults to the confidence floor
    assert seen["examples"] == 19  # 21 kept rows minus the 10% eval carve-out (2)
    assert seen["epochs"] == 1
    assert seen["gate_sources"] == 1  # gate states are built FROM the design states
    assert seen["state_cap"] == seam_prompts.DECISION_STATE_MAX_CHARS
    assert seen["description_cap"] == seam_prompts.DECISION_STATE_MAX_CHARS
    assert seen["description_count"] == 1  # derived default: problems // 12
    assert seen["teacher_shards_arg"] == 0  # auto: one shard per visible GPU
    assert seen["eval_floors"] == (0.5, 0.5)  # the gate floor defaults to the confidence floor
    assert seen["probe_floor"] == 0.5
    assert seen["probe_rows"] == 1
    assert seen["eval_model"] is seen["saved_model"]  # fidelity scores the MERGED artifact
    assert seen["probe_model"] is seen["saved_model"]  # and so does the false-act probe
    manifest = json.loads((out_dir / "self-distill.json").read_text(encoding="utf-8"))
    assert manifest["problems"] == 14
    assert manifest["examples"] == 19
    assert manifest["kept"] == {"choice": 7, "score": 7, "yes_no": 7}
    assert manifest["repo"] == "morriszjm/Tacit-4B"
    assert manifest["confidence_floor"] == 0.5
    assert manifest["gate_confidence_floor"] == 0.5
    assert manifest["answer_mass_floor"] == 0.5
    assert manifest["score_mass_floor"] == 0.0
    assert manifest["eval_frac"] == 0.1
    assert manifest["states"] == 1
    assert manifest["gate_states"] == 1
    assert manifest["axes_per_state"] == 8
    assert manifest["style_states"] == 1
    assert manifest["style_problems"] == 2
    assert manifest["rerank_problems"] == 2  # train rows only; the holdout pair never trains
    assert manifest["teacher_shards"] == 1
    assert manifest["min_seam_agreement"] == 0.6
    assert manifest["max_score_mae"] == 0.2
    # Every seam's kept count and its eval fidelity land in one block.
    assert set(manifest["seams"]) == {"category", "style-stage1", "style-stage2", "gate", "rerank", "weights"}
    assert manifest["seams"]["weights"]["expected_level_mae"] == pytest.approx(0.1)
    # ``yes_no`` rows alternate gate/rerank: 7 rows → 4 gate + 3 rerank.
    assert manifest["seams"]["rerank"]["kept"] == 3
    assert manifest["seams"]["gate"]["kept"] == 4
    assert "n" not in manifest["seams"]["category"]
    assert manifest["false_act"] == {"n": 1, "rate": 0.25}


def test_trainer_main_honours_a_raised_confidence_floor(
    trainer: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Floors flow from one knob: answer, gate and eval floors follow --confidence-floor."""
    seen: dict[str, Any] = {}
    _main_fakes(trainer, seen, monkeypatch)
    init_dir = tmp_path / "init"
    init_dir.mkdir()
    (init_dir / "config.json").write_text("{}", encoding="utf-8")
    code = trainer.main(
        _main_args(init_dir, tmp_path / "out", _rerank_pairs_file(tmp_path), "--confidence-floor", "0.7")
    )
    assert code == 0
    assert seen["floors"] == (0.7, 0.0)
    assert seen["eval_floors"] == (0.7, 0.7)
    assert seen["probe_floor"] == 0.7
    manifest = json.loads((tmp_path / "out" / "self-distill.json").read_text(encoding="utf-8"))
    assert manifest["confidence_floor"] == 0.7
    assert manifest["gate_confidence_floor"] == 0.7
    assert manifest["answer_mass_floor"] == 0.7


def test_trainer_main_honours_a_distinct_gate_floor(
    trainer: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate seam's floor is its own knob: choice retention stays, act metrics follow it."""
    seen: dict[str, Any] = {}
    _main_fakes(trainer, seen, monkeypatch)
    init_dir = tmp_path / "init"
    init_dir.mkdir()
    (init_dir / "config.json").write_text("{}", encoding="utf-8")
    code = trainer.main(
        _main_args(
            init_dir,
            tmp_path / "out",
            _rerank_pairs_file(tmp_path),
            "--confidence-floor",
            "0.5",
            "--gate-confidence-floor",
            "0.8",
        )
    )
    assert code == 0
    assert seen["floors"] == (0.5, 0.0)  # choice retention is unchanged
    assert seen["eval_floors"] == (0.5, 0.8)  # yes_no act metrics use the gate floor
    assert seen["probe_floor"] == 0.5
    manifest = json.loads((tmp_path / "out" / "self-distill.json").read_text(encoding="utf-8"))
    assert manifest["gate_confidence_floor"] == 0.8


def test_trainer_main_refuses_a_kind_with_no_rows(
    trainer: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An under-trained seam must fail the build, not ship: every kind needs rows."""
    seen: dict[str, Any] = {}
    _main_fakes(trainer, seen, monkeypatch, kinds=("choice", "score"))
    init_dir = tmp_path / "init"
    init_dir.mkdir()
    (init_dir / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="too few usable teacher rows"):
        trainer.main(_main_args(init_dir, tmp_path / "out", _rerank_pairs_file(tmp_path)))


def test_trainer_main_fails_an_under_trained_seam(
    trainer: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seam whose student-teacher agreement misses the bar must fail the build."""
    seen: dict[str, Any] = {}
    _main_fakes(trainer, seen, monkeypatch)
    # Certify at 3 eval rows so the fake metrics exercise the failure path, not the
    # thin-sample skip (production certifies at _MIN_EVAL_ROWS_PER_SEAM).
    monkeypatch.setattr(trainer, "_MIN_EVAL_ROWS_PER_SEAM", 1)
    metrics = {seam: {"n": 3, "agreement": 0.9, "expected_level_mae": 0.1} for seam in trainer._GATED_SEAMS}
    metrics["rerank"] = {"n": 3, "agreement": 0.2, "expected_level_mae": 0.1}
    monkeypatch.setattr(trainer, "evaluate_student", lambda *_args, **_kwargs: metrics)
    init_dir = tmp_path / "init"
    init_dir.mkdir()
    (init_dir / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="under-trained seam"):
        trainer.main(_main_args(init_dir, tmp_path / "out", _rerank_pairs_file(tmp_path)))


def test_trainer_skip_copies_the_released_checkpoint(trainer: types.ModuleType, tmp_path: Path) -> None:
    init_dir = tmp_path / "init"
    init_dir.mkdir()
    (init_dir / "config.json").write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "copy"
    code = trainer.main(
        ["--init-dir", str(init_dir), "--out", str(out_dir), "--seam-prompts", str(SEAM_PROMPTS), "--skip"]
    )
    assert code == 0
    assert (out_dir / "config.json").is_file()


def test_style_corpus_matches_the_served_vocabulary() -> None:
    """Every style row's criteria are the served gloss table, in affinity order + ``none``."""
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "qa" / "typed" / "style.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    for row in rows:
        assert row["instructions"] == seam_prompts.STYLE_INSTRUCTIONS
        assert list(row["criteria"]) == [*list(STYLE_AFFINITY), seam_prompts.STYLE_NONE_KEY]
        assert set(row["criteria"]) == set(seam_prompts.STYLE_GLOSSES) | {seam_prompts.STYLE_NONE_KEY}
        # The label is the stage-2 answer the seam can actually return.
        assert row["label"] in seam_prompts.STYLE_GLOSSES or row["label"] == seam_prompts.STYLE_NONE_KEY


def test_style_corpus_states_are_raw_descriptions() -> None:
    """Ch2 receives the raw scenario description, so the gold state must be exactly that."""
    corpus = json.loads((REPO_ROOT / "tests" / "benchmark" / "scenarios" / "full.json").read_text(encoding="utf-8"))
    scenarios = {item["scenario_id"]: item for item in corpus["scenarios"]}
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "qa" / "typed" / "style.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        scenario = scenarios[row["source"]["scenario_id"]]
        assert row["state"] == scenario["description"]
        assert row["source"]["split"] == scenario["split"]


@pytest.fixture(scope="module")
def calibrator() -> types.ModuleType:
    return _load("tacit_calibrate", "tacit_calibrate.py")


def test_calibration_metrics_match_definitions(calibrator: types.ModuleType) -> None:
    """ECE = binned |acc − conf|, Brier = mean (conf − outcome)²."""
    pairs = [(0.9, True), (0.9, False), (0.1, False), (0.1, False)]
    assert calibrator.brier_score(pairs) == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)
    ece = calibrator.expected_calibration_error(pairs, 10)
    assert ece == pytest.approx(0.5 * abs(0.5 - 0.9) + 0.5 * abs(0.0 - 0.1))


def test_calibration_table_reports_override_accuracy(calibrator: types.ModuleType) -> None:
    """Acted accuracy is over overrides only; confident ``none`` answers are stratified out.

    Each pair is ``(stage-2 confidence, acted confidence, correct, choice)``: the acted
    number is ``min(stage1, stage2)``, the value the two-stage seam compares with its floor.
    """
    pairs = [
        (0.9, 0.9, True, "hexagonal"),
        (0.8, 0.8, True, "hexagonal"),
        (0.4, 0.4, False, "hexagonal"),
        (0.2, 0.2, False, "hexagonal"),
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
    pairs = [(0.9, 0.9, True, "hexagonal"), (0.9, 0.9, True, "none"), (0.9, 0.9, False, "none")]
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
    pairs = [(0.9, 0.9, True, "none")]
    by_floor = {row["floor"]: row for row in calibrator.summarize(pairs, [0.5], 10)["floors"]}
    row = by_floor[0.5]
    assert row["acted"] == 0
    assert row["acted_accuracy"] is None
    assert row["acted_accuracy_ci95"] == [0.0, 1.0]
    low, high = calibrator.wilson_interval(24, 30)
    assert 0.0 < low < 0.8 < high < 1.0


def test_calibration_acted_rule_needs_both_stages(calibrator: types.ModuleType) -> None:
    """A confident stage 2 behind a weak stage 1 is NOT acted: the seam would not override."""
    pairs = [
        (0.95, 0.95, True, "hexagonal"),  # both stages confident → acted and right
        (0.95, 0.30, False, "hexagonal"),  # stage 1 below the floor → never acted
    ]
    row = next(row for row in calibrator.summarize(pairs, [0.5], 10)["floors"] if row["floor"] == 0.5)
    assert row["acted"] == 1
    assert row["acted_accuracy"] == pytest.approx(1.0)
    assert row["acted_rate"] == pytest.approx(0.5)
    # ECE/Brier stay over the stage-2 probability of every row, not the acted number.
    summary = calibrator.summarize(pairs, [0.5], 10)
    assert summary["n"] == 2
    assert summary["replay_failures"] == 0
    # A broken replay (no usable answer) is reported as a FAILURE at every floor: a failed
    # row is never "acted", so a per-floor count would read 0 and look like abstention.
    broken = calibrator.summarize([*pairs, (0.0, 0.0, False, None)], [0.0, 0.5], 10)
    assert broken["replay_failures"] == 1
    assert all("replay_failures" not in row for row in broken["floors"])


def test_calibration_ask_replays_the_served_two_stage_bytes(calibrator: types.ModuleType) -> None:
    """Stage 1 is the category question/cap; stage 2 the shortlist question; both cap the state."""
    seams = seam_prompts
    calls: list[dict[str, Any]] = []

    def _fake_decide(base_url: str, item: dict[str, Any], timeout: float) -> dict[str, Any]:
        calls.append(item)
        if len(calls) == 1:
            return {"answer": "domain-modeling", "probs": {"domain-modeling": 0.8, "saga": 0.2}}
        options = [str(option) for option in item["options"]]
        return {
            "answer": options[0],
            "probs": dict.fromkeys(options, 0.5 / max(1, len(options) - 1)) | {options[0]: 0.5},
        }

    original = calibrator._decide
    calibrator._decide = _fake_decide  # type: ignore[assignment]
    try:
        answer = calibrator.ask("http://x:18100", {"state": "y" * (seams.DECISION_STATE_MAX_CHARS + 10)}, seams, 5.0)
    finally:
        calibrator._decide = original  # type: ignore[assignment]
    assert answer["stage1"] == pytest.approx(0.8)
    assert answer["choice"] == seams.styles_for_category("domain-modeling")[0]
    assert answer["confidence"] == pytest.approx(0.5)
    assert [call["kind"] for call in calls] == ["choice", "choice"]
    assert calls[0]["question"] == seams.category_question()
    assert calls[0]["options"] == list(seams.CATEGORY_SLUGS)
    assert calls[1]["question"] == seams.style_question(seams.styles_for_category("domain-modeling"))
    assert calls[1]["options"] == [*seams.styles_for_category("domain-modeling"), seams.STYLE_NONE_KEY]
    assert len(calls[1]["options"]) <= 26
    # Serving truncates the description at the shared cap before asking either question.
    assert all(call["state"] == "y" * seams.DECISION_STATE_MAX_CHARS for call in calls)


def test_calibration_ask_never_acts_on_a_broken_stage_one(calibrator: types.ModuleType) -> None:
    """An off-taxonomy stage-1 answer yields a never-acted row instead of aborting the replay."""
    original = calibrator._decide
    calibrator._decide = lambda base_url, item, timeout: {  # type: ignore[assignment]
        "answer": "not-a-category",
        "probs": {"not-a-category": 1.0},
    }
    try:
        answer = calibrator.ask("http://x", {"state": "s"}, seam_prompts, 5.0)
    finally:
        calibrator._decide = original  # type: ignore[assignment]
    assert answer == {"choice": None, "probabilities": {}, "confidence": 0.0, "stage1": 0.0}


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

    def _fake_ask(base_url: str, row: dict[str, Any], seams: Any, timeout: float) -> dict[str, Any]:
        seen.append({"base_url": base_url, "state": row["state"], "category": seams.CATEGORY_SLUGS[0]})
        return {"choice": "a", "confidence": 0.7, "probabilities": {"a": 0.7, "b": 0.3}, "stage1": 0.7}

    monkeypatch.setattr(calibrator, "ask", _fake_ask)
    assert calibrator.main(["--style", str(path), "--base-url", "http://x:18100", "--floors", "0.5"]) == 0
    # Holdout only, and the replay gets the SEAM vocabulary module (serving bytes).
    assert [entry["state"] for entry in seen] == ["system one"]
    assert seen[0]["base_url"] == "http://x:18100"
    assert seen[0]["category"] == "domain-modeling"


def test_recommend_picks_the_lowest_qualifying_floor(calibrator: types.ModuleType) -> None:
    """The recommendation is the lowest floor clearing both bars, not the most accurate one."""
    pairs = [
        (0.9, 0.9, True, "hexagonal"),
        (0.8, 0.8, True, "hexagonal"),
        (0.7, 0.7, False, "hexagonal"),
        (0.6, 0.6, False, "hexagonal"),
        (0.2, 0.2, False, "hexagonal"),
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
    pairs = [(0.9, 0.9, True, "hexagonal"), (0.9, 0.9, True, "hexagonal")]
    table = calibrator.summarize(pairs, [0.5], 10)["floors"]
    assert calibrator.recommend(table, 0.8, 0.1, 10)["floor"] is None
    assert calibrator.recommend(table, 0.8, 0.1, 2)["floor"] == pytest.approx(0.5)
    qualified = calibrator.recommend(table, 0.8, 0.1, 2)
    assert qualified["acted"] == 2
    assert qualified["acted_accuracy_ci95"][0] <= 1.0 <= qualified["acted_accuracy_ci95"][1]


def test_recommend_reports_nothing_when_no_floor_qualifies(calibrator: types.ModuleType) -> None:
    pairs = [
        (0.9, 0.9, False, "hexagonal"),
        (0.8, 0.8, False, "hexagonal"),
        (0.7, 0.7, False, "hexagonal"),
    ]
    table = calibrator.summarize(pairs, [0.5, 0.85], 10)["floors"]
    recommended = calibrator.recommend(table, 0.9, 0.3)
    assert recommended["floor"] is None
    assert "cannot be enabled safely" in recommended["reason"]


def test_recommend_reports_only_without_thresholds(calibrator: types.ModuleType) -> None:
    table = calibrator.summarize([(0.9, 0.9, True, "hexagonal")], [0.5], 10)["floors"]
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
        lambda *args, **kwargs: {
            "choice": "b",
            "confidence": 0.9,
            "probabilities": {"a": 0.1, "b": 0.9},
            "stage1": 0.9,
        },
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
        lambda *args, **kwargs: {
            "choice": "a",
            "confidence": 0.9,
            "probabilities": {"a": 0.9, "b": 0.1},
            "stage1": 0.9,
        },
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
        lambda *args, **kwargs: {
            "choice": "a",
            "confidence": 0.9,
            "probabilities": {"a": 0.9, "b": 0.1},
            "stage1": 0.9,
        },
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
        lambda *args, **kwargs: {
            "choice": "a",
            "confidence": 0.9,
            "probabilities": {"a": 0.9, "b": 0.1},
            "stage1": 0.9,
        },
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


def test_calibrator_none_choice_matches_the_served_vocabulary(calibrator: types.ModuleType) -> None:
    """The stratified ``none`` must be the seam's own abstention key, not a lookalike."""
    assert calibrator.NONE_CHOICE == seam_prompts.STYLE_NONE_KEY


@pytest.fixture(scope="module")
def shell_check() -> types.ModuleType:
    return _load("check_dockerfile_shell", "check_dockerfile_shell.py")


def test_shell_check_flags_an_embedded_single_quote(shell_check: types.ModuleType) -> None:
    """The exact defect that produced 'Syntax error: \"(\" unexpected'."""

    class _P:
        line = 1
        instruction = "RUN"
        command = "sh -c 'set -eu; python3 -c \"x('y')\"'"
        payload = "set -eu; python3 -c \"x('y')\""

    found = shell_check.problems(_P())
    assert any("single quote inside the sh -c payload" in problem for problem in found)


def test_shell_check_flags_a_comment_that_swallows_later_commands(shell_check: types.ModuleType) -> None:
    """A '#' outside quotes silently skipped the rank gate; it must fail the check."""

    class _P:
        line = 1
        instruction = "RUN"
        command = "sh -c 'a; # note; b'"
        payload = "a; # note; b"

    assert any("outside quotes" in problem for problem in shell_check.problems(_P()))


def test_shell_check_passes_a_clean_payload(shell_check: types.ModuleType) -> None:
    """The current Dockerfile form: no embedded quotes, no bare '#'."""

    class _P:
        line = 1
        instruction = "RUN"
        command = 'sh -c \'set -eu; echo "a (b)"; python3 -u /build/x.py --pin "${P}"\''
        payload = 'set -eu; echo "a (b)"; python3 -u /build/x.py --pin "${P}"'

    assert shell_check.problems(_P()) == []
    assert shell_check._unquoted_hash('echo "a # b"') is False


def test_shell_check_reads_the_real_dockerfile(shell_check: types.ModuleType) -> None:
    """The fat self-distill RUN (the ``ldconfig && exec`` shape) must carry no hazard."""
    text = (REPO_ROOT / "docker" / "Dockerfile.tacit").read_text(encoding="utf-8")
    fat = [p for p in shell_check.payloads(text) if "tacit_self_distill.py" in p.command]
    assert fat, "the self-distill RUN must be discoverable"
    for payload in fat:
        assert shell_check.problems(payload) == []
