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

import importlib.util
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from src.decisions import seam_prompts
from src.decisions.tacit import category_options
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
    torch_stub = types.ModuleType("torch")
    cuda_stub = types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda _i: "stub-cuda:0"
    )
    torch_stub.cuda = cuda_stub  # type: ignore[attr-defined]
    torch_stub.manual_seed = lambda _seed: None  # type: ignore[attr-defined]
    torch_stub.cuda.manual_seed_all = lambda _seed: None  # type: ignore[attr-defined]
    previous = sys.modules.get("anyjev")
    previous_torch = sys.modules.get("torch")
    sys.modules["anyjev"] = stub
    sys.modules["torch"] = torch_stub
    yield _load("tacit_self_distill", "tacit_self_distill.py")
    if previous is None:
        sys.modules.pop("anyjev", None)
    else:
        sys.modules["anyjev"] = previous
    if previous_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = previous_torch


def test_trainer_asks_the_questions_the_server_asks(trainer: types.ModuleType) -> None:
    """The student is trained on the serving bytes: same question text, same options."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    states = [f"System goal: goal {index}\nDecision part: part {index}" for index in range(3)]
    problems = trainer.build_problems(states, seams)
    assert [problem["kind"] for problem in problems] == ["choice", "score", "yes_no"]
    assert problems[0]["question"] == seam_prompts.category_question()
    assert problems[0]["options"] == category_options()
    assert problems[1]["question"] == seam_prompts.axis_question(trainer.QUALITY_AXES[0])
    assert problems[1]["options"] == list(seam_prompts.WEIGHT_LEVELS)
    assert problems[2]["question"] == seam_prompts.gate_question()
    assert problems[2]["options"] == ["Yes", "No"]


def test_trainer_main_walks_generate_to_save(
    trainer: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main()`` must thread states → problems → teacher → train → save, not unpack them."""
    seen: dict[str, Any] = {}

    def fake_states(_tacit: Any, _hints: Any, problems: int, _seed: int, _max_new: int) -> list[str]:
        return [f"System goal: g{i}\nDecision part: p{i}" for i in range(problems)]

    def fake_teacher(_tacit: Any, problems: list[dict[str, Any]], _floor: float) -> tuple[list[Any], dict[str, int]]:
        seen["kinds"] = [problem["kind"] for problem in problems]
        return [trainer.Example(prompt="prompt", label_ids=[1, 2], target=[0.6, 0.4])], {"unsharp": 0}

    def fake_train(_init: Path, examples: list[Any], args: Any) -> object:
        seen["examples"] = len(examples)
        seen["epochs"] = args.epochs
        return object()

    def fake_save(_model: Any, _init: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(trainer, "generate_states", fake_states)
    monkeypatch.setattr(trainer, "teacher_examples", fake_teacher)
    monkeypatch.setattr(trainer, "train_student", fake_train)
    monkeypatch.setattr(trainer, "save_checkpoint", fake_save)

    init_dir = tmp_path / "init"
    init_dir.mkdir()
    (init_dir / "config.json").write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "out"
    code = trainer.main(
        [
            "--init-dir",
            str(init_dir),
            "--out",
            str(out_dir),
            "--seam-prompts",
            str(SEAM_PROMPTS),
            "--problems",
            "3",
        ]
    )
    assert code == 0
    assert seen == {"kinds": ["choice", "score", "yes_no"], "examples": 1, "epochs": 1}
    manifest = json.loads((out_dir / "self-distill.json").read_text(encoding="utf-8"))
    assert manifest["problems"] == 3
    assert manifest["examples"] == 1
    assert manifest["repo"] == "morriszjm/Tacit-4B"


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


@pytest.fixture(scope="module")
def exporter() -> types.ModuleType:
    return _load("export_clm_dataset", "export_clm_dataset.py")


def test_export_row_matches_the_pinned_clm_adapter_contract(exporter: types.ModuleType) -> None:
    """``typed_decision_examples`` reads {id, state, questions, gold}; so must we write it."""
    row = exporter.to_typed_row(
        {
            "id": "scenario:category",
            "state": "System goal: x\nDecision part: y",
            "instructions": "Which category?",
            "criteria": {"domain-modeling": "entities and invariants", "behavior-selection": "dispatch"},
            "label": "behavior-selection",
            "source": {"split": "train"},
        }
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
            }
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
