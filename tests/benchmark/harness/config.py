"""Harness configuration: server config load, scenario corpus load, run provenance (plan §4).

The harness never mutates the repository configuration: it loads ``config/config.json`` through
the production :class:`ConfigManager` (honoring ``CONFIG_PATH``), applies the
documented per-mode overrides only, and pins the effective ``ServerConfig`` into every run
manifest so two arms can be compared for configuration drift (plan §6, "provider drift").

Secrets never enter a manifest: :func:`mask_secrets` strips every value behind a
key/token/secret/password/credential-shaped name, and :func:`masked_env_snapshot` records only
the environment the deployment actually reads — every name ``config/config.json`` declares
through a ``{env:...}`` placeholder, the ``REASONING_*`` namespace, and the provider keys in
``_ENV_EXTRA_KEYS``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from src.config import ConfigManager, ServerConfig
from src.patterns.loader import PatternLoader, load_catalog

#: Harness contract version — bump on any change to record/summary/manifest shapes.
HARNESS_VERSION = "0.1.0"

#: Repository root (``tests/benchmark/harness/config.py`` → three parents up).
REPO_ROOT = Path(__file__).resolve().parents[3]

BENCHMARK_DIR = REPO_ROOT / "tests" / "benchmark"
SCENARIOS_DIR = BENCHMARK_DIR / "scenarios"
DEFAULT_SCENARIOS = SCENARIOS_DIR / "seed.json"
SCENARIO_SCHEMA = SCENARIOS_DIR / "scenario.schema.json"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "benchmark-runs"

MODES: tuple[str, ...] = ("offline", "live", "e2e")
SPLITS: tuple[str, ...] = ("train", "holdout")

_SECRET_KEY_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)
#: Namespace whose every name belongs to the deployment regardless of the config file.
_ENV_PREFIXES: tuple[str, ...] = ("REASONING_",)
#: Deployment-read names that bypass the ``{env:...}`` placeholders for discovery.
_ENV_EXTRA_KEYS: tuple[str, ...] = (
    "CONFIG_PATH",
    "JOBS_DB",
    "MINIMAXAI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)
_CONFIG_ENV_RE = re.compile(r"\{env:([A-Z_][A-Z0-9_]*)")


def deployment_env_names() -> frozenset[str]:
    """Env names the deployment reads: config ``{env:...}`` placeholders plus ``_ENV_EXTRA_KEYS``."""
    try:
        raw = resolve_config_path().read_text(encoding="utf-8")
    except OSError:
        return frozenset(_ENV_EXTRA_KEYS)
    return frozenset(_CONFIG_ENV_RE.findall(raw)) | frozenset(_ENV_EXTRA_KEYS)


def utc_now_iso() -> str:
    """Current UTC timestamp in ISO-8601 (seconds resolution)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes (scenario-corpus and artifact pinning)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as deterministic pretty JSON (UTF-8, trailing newline), creating dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path``."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def mask_secrets(value: Any) -> Any:
    """Recursively replace values behind secret-shaped keys with ``"***set***"``/``"***empty***"``.

    Any value type behind a secret-shaped key is masked (not only strings), so a
    non-string credential cannot slip into a manifest unmasked.
    """
    if isinstance(value, dict):
        return {
            key: (
                f"***{'set' if item else 'empty'}***"
                if _SECRET_KEY_RE.search(str(key)) and not isinstance(item, (dict, list))
                else mask_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [mask_secrets(item) for item in value]
    return value


def masked_env_snapshot() -> dict[str, str]:
    """The deployment-relevant environment, secrets masked (drift attribution between arms)."""
    snapshot: dict[str, str] = {}
    names = deployment_env_names()
    for key in sorted(os.environ):
        if key.startswith(_ENV_PREFIXES) or key in names:
            value = os.environ[key]
            snapshot[key] = f"***{'set' if value else 'empty'}***" if _SECRET_KEY_RE.search(key) else value
    return snapshot


def _git(*args: str) -> str:
    """Run a read-only git command in the repository root; empty string when git fails."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def git_snapshot() -> dict[str, Any]:
    """HEAD, dirty flag and a hash over the working-tree diff (replay provenance)."""
    status = _git("status", "--porcelain")
    diff = _git("diff", "HEAD")
    combined = (status + diff).encode("utf-8")
    return {
        "head": _git("rev-parse", "HEAD").strip() or None,
        "dirty": bool(status.strip()),
        "status_lines": status.splitlines()[:50],
        "worktree_sha256": hashlib.sha256(combined).hexdigest(),
    }


# --------------------------------------------------------------------------------------
# Server configuration
# --------------------------------------------------------------------------------------


def resolve_config_path() -> Path:
    """``CONFIG_PATH`` > repository ``config/config.json`` (never cwd-relative)."""
    configured = os.environ.get("CONFIG_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return REPO_ROOT / "config" / "config.json"


def load_server_config(mode: str) -> ServerConfig:
    """Load the effective ``ServerConfig`` for ``mode`` through the production loader.

    Only ``offline`` changes anything: its scripted Stage-C hits always clear the §8.1 dual gate,
    so the two thresholds are pinned to 0 there (see ``README.md`` "Offline mode"). Live and e2e
    run the shipped configuration unmodified.
    """
    if mode not in MODES:
        raise ValueError(f"unknown benchmark mode {mode!r}; expected one of {MODES}")
    raw = ConfigManager.load_config(str(resolve_config_path()))
    if mode == "offline":
        raw = dict(raw)
        raw["selection"] = {
            **raw.get("selection", {}),
            "primary_score_threshold": 0.0,
            "supporting_score_threshold": 0.0,
        }
        raw["retrieval"] = {**raw.get("retrieval", {}), "min_fusion_score": 0.0}
    return ServerConfig.model_validate(raw)


def resolve_catalog_paths(config: ServerConfig) -> tuple[Path, Path]:
    """Catalog ``pattern``/``docs`` directories, resolved against the repository root."""
    pattern_dir = Path(config.paths.pattern_directory)
    docs_dir = Path(config.paths.docs_directory)
    if not pattern_dir.is_absolute():
        pattern_dir = REPO_ROOT / pattern_dir
    if not docs_dir.is_absolute():
        docs_dir = REPO_ROOT / docs_dir
    return pattern_dir, docs_dir


def load_repo_catalog(config: ServerConfig) -> PatternLoader:
    """Load the repository catalog through the production loader (integrity gate included)."""
    pattern_dir, docs_dir = resolve_catalog_paths(config)
    return load_catalog(pattern_dir, docs_dir)


def dense_cache_ready(path: str) -> bool:
    """True when ``data/dense-cache/{manifest.json,vectors.npy}`` both exist at ``path``."""
    directory = Path(path).expanduser()
    return (directory / "manifest.json").is_file() and (directory / "vectors.npy").is_file()


# --------------------------------------------------------------------------------------
# Scenario corpus
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One benchmark scenario: the request, its expert label and its split membership."""

    scenario_id: str
    family: str
    category: str
    split: str
    description: str
    decision_statement: str
    language: str | None = None
    architecture_style: str | None = None
    paradigm: str | None = None
    focus: str | None = None
    acceptable_primary: tuple[str, ...] = ()
    acceptable_supporting: tuple[str, ...] = ()
    notes: str | None = None
    source_path: Path | None = None

    @property
    def primary(self) -> str:
        """The single expected (expert-chosen) primary pattern."""
        return self.acceptable_primary[0]

    @property
    def acceptable(self) -> tuple[str, ...]:
        """Union of every acceptable label id (primary + supporting)."""
        return self.acceptable_primary + self.acceptable_supporting

    def record_identity(self) -> dict[str, Any]:
        """The label/identity block echoed into every run record (replay without the corpus)."""
        return {
            "scenario_id": self.scenario_id,
            "family": self.family,
            "category": self.category,
            "split": self.split,
            "language": self.language,
            "architecture_style": self.architecture_style,
            "paradigm": self.paradigm,
            "focus": self.focus,
            "expected": {
                "primary": self.primary,
                "acceptable_primary": list(self.acceptable_primary),
                "acceptable_supporting": list(self.acceptable_supporting),
            },
            "description": self.description,
            "decision_statement": self.decision_statement,
            "notes": self.notes,
        }


def _scenario_schema_validator() -> Draft7Validator:
    """Draft-07 validator over ``scenarios/scenario.schema.json``."""
    return Draft7Validator(read_json(SCENARIO_SCHEMA))


def load_scenarios(path: Path, catalog: PatternLoader | None = None) -> list[Scenario]:
    """Validate and load a scenario corpus; fails loudly on any label the catalog cannot resolve.

    Checks (plan §4/§5.7): schema conformance, unique ids, split coverage, catalog-resolvable
    pattern ids, ``primary ∈ acceptable_primary``, at least two acceptable primaries and at
    least one acceptable primary that is a *sibling* (same catalog subgroup) of the expected one
    — the near-miss discrimination the corpus is built for.
    """
    payload = read_json(path)
    entries = payload.get("scenarios")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: 'scenarios' must be a non-empty list")
    validator = _scenario_schema_validator()
    scenarios: list[Scenario] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        errors = sorted(validator.iter_errors(entry), key=lambda error: list(error.absolute_path))
        if errors:
            details = "; ".join(
                f"{'/'.join(str(part) for part in error.absolute_path)}: {error.message}" for error in errors
            )
            raise ValueError(f"{path}: scenario #{index} violates scenario.schema.json: {details}")
        scenario = _scenario_from_entry(entry, path)
        if scenario.scenario_id in seen_ids:
            raise ValueError(f"{path}: duplicate scenario_id {scenario.scenario_id!r}")
        seen_ids.add(scenario.scenario_id)
        scenarios.append(scenario)
    if catalog is not None:
        for scenario in scenarios:
            _assert_labels_resolvable(scenario, catalog)
        _assert_family_disjoint(scenarios)
    return scenarios


def _scenario_from_entry(entry: dict[str, Any], path: Path) -> Scenario:
    """Build a :class:`Scenario` from one schema-validated JSON object."""
    hints: dict[str, Any] = entry.get("hints") or {}
    expected = entry["expected"]
    return Scenario(
        scenario_id=entry["scenario_id"],
        family=entry["family"],
        category=entry["category"],
        split=entry["split"],
        description=entry["description"],
        decision_statement=entry["decision_statement"],
        language=entry.get("language"),
        architecture_style=hints.get("architecture_style"),
        paradigm=hints.get("paradigm"),
        focus=hints.get("focus"),
        acceptable_primary=tuple(expected["acceptable_primary"]),
        acceptable_supporting=tuple(expected.get("acceptable_supporting") or ()),
        notes=entry.get("notes"),
        source_path=path,
    )


def _assert_family_disjoint(scenarios: Sequence[Scenario]) -> None:
    """The train/holdout split must be family-disjoint (plan §11)."""
    split_of: dict[str, str] = {}
    conflicts: list[str] = []
    for scenario in scenarios:
        known = split_of.setdefault(scenario.family, scenario.split)
        if known != scenario.split:
            conflicts.append(f"{scenario.family!r} appears in both {known} and {scenario.split}")
    if conflicts:
        raise ValueError("scenario split is not family-disjoint: " + "; ".join(sorted(set(conflicts))))


def _assert_labels_resolvable(scenario: Scenario, catalog: PatternLoader) -> None:
    """Fail loudly when a label id has no ``pattern/<id>.json`` record (plan §5.7)."""
    records = catalog.records
    unknown = [name for name in scenario.acceptable if name not in records]
    if unknown:
        raise ValueError(
            f"{scenario.source_path}: scenario {scenario.scenario_id!r} labels unresolvable in the catalog: {unknown}"
        )
    if scenario.category not in catalog.enums.categories:
        raise ValueError(
            f"{scenario.source_path}: scenario {scenario.scenario_id!r} category {scenario.category!r} "
            f"is not a catalog category"
        )
    if scenario.language is not None and scenario.language not in catalog.enums.languages:
        raise ValueError(
            f"{scenario.source_path}: scenario {scenario.scenario_id!r} language {scenario.language!r} "
            f"is not a catalog language"
        )
    if scenario.primary not in scenario.acceptable_primary:
        raise ValueError(
            f"{scenario.source_path}: scenario {scenario.scenario_id!r} primary {scenario.primary!r} "
            "is not listed in acceptable_primary"
        )
    if len(set(scenario.acceptable_primary)) < 2:
        raise ValueError(
            f"{scenario.source_path}: scenario {scenario.scenario_id!r} needs at least two distinct "
            "acceptable_primary ids (near-miss discrimination)"
        )
    primary_subgroup = records[scenario.primary].subgroup
    if scenario.family != primary_subgroup:
        raise ValueError(
            f"{scenario.source_path}: scenario {scenario.scenario_id!r} family {scenario.family!r} is not the "
            f"catalog ### subgroup of its primary {scenario.primary!r} ({primary_subgroup!r})"
        )
    siblings = [
        name
        for name in scenario.acceptable_primary
        if name != scenario.primary and records[name].subgroup == primary_subgroup
    ]
    if not siblings:
        raise ValueError(
            f"{scenario.source_path}: scenario {scenario.scenario_id!r} has no sibling of {scenario.primary!r} "
            f"in subgroup {primary_subgroup!r} among its acceptable_primary ids"
        )


def select_scenarios(scenarios: list[Scenario], *, split: str, limit: int | None) -> list[Scenario]:
    """Filter by ``split`` (``train``/``holdout``/``all``) preserving corpus order, then cut."""
    if split not in (*SPLITS, "all"):
        raise ValueError(f"unknown split {split!r}; expected one of {(*SPLITS, 'all')}")
    selected = [scenario for scenario in scenarios if split in ("all", scenario.split)]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        selected = selected[:limit]
    return selected


@dataclass
class RunPaths:
    """Resolved run-directory layout (``data/benchmark-runs/<run-id>/``)."""

    root: Path
    manifest: Path
    summary: Path
    report: Path
    scenarios_dir: Path
    extra: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def for_run(cls, out_dir: Path) -> RunPaths:
        """Build the layout for a run directory created at ``out_dir``."""
        return cls(
            root=out_dir,
            manifest=out_dir / "manifest.json",
            summary=out_dir / "summary.json",
            report=out_dir / "report.md",
            scenarios_dir=out_dir / "scenarios",
        )


def default_run_id(mode: str, *, now: datetime | None = None) -> str:
    """Timestamped run id: ``<mode>-<UTC yyyymmdd-HHMMSS>``."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{mode}-{stamp}"
