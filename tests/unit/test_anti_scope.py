"""§5 rule 5 — the anti-scope guard (E9): architecture-level parts are dropped, implementable ones kept.

The term set is parsed from the catalog document's Background section; "catalogued locally" is
decided by record existence, never by the prose example list.
"""

from __future__ import annotations

from src.anti_scope import AntiScopeGuard
from src.patterns.loader import (
    CATALOG_DOC_FILE,
    parse_locally_implementable_names,
    parse_out_of_scope_terms,
)
from src.schemas import DecisionPart
from tests.unit.conftest import DOCS_DIR


def _guard(catalog) -> AntiScopeGuard:
    doc = DOCS_DIR / CATALOG_DOC_FILE
    return AntiScopeGuard(
        parse_out_of_scope_terms(doc), catalog, parse_locally_implementable_names(doc)
    )


def _part(statement: str, *, component_refs: list[str] | None = None) -> DecisionPart:
    return DecisionPart(
        id="some-part",
        statement=statement,
        component_refs=component_refs or [],
        scope="module",
        priority=1.0,
    )


def test_architecture_restatement_is_dropped(catalog) -> None:
    """A part that only restates an architecture-level choice is architecture-only (E9 case)."""
    guard = _guard(catalog)
    assert guard.is_architecture_only(_part("use microservices for the ingest pipeline"), []) is True
    assert guard.is_architecture_only(_part("adopt a clean onion architecture"), []) is True


def test_implementable_statement_is_kept(catalog) -> None:
    """No out-of-scope term → the guard never interferes."""
    guard = _guard(catalog)
    assert guard.is_architecture_only(_part("bound the synchronous lookup with a timeout"), []) is False


def test_word_boundary_matching_does_not_fire_on_longer_words(catalog) -> None:
    """``clean`` must not match ``cleanup`` and ``monolith`` must not match ``monolithic``."""
    guard = _guard(catalog)
    assert guard.matched_terms("run the cleanup job after each batch") == ()
    assert guard.matched_terms("keep the ingest pipeline clean") != ()
    assert "monolith" not in guard.matched_terms("a monolithic deployment is out of scope")


def test_component_scoped_statement_survives(catalog) -> None:
    """A term plus real component work is an implementation decision, not a restatement."""
    guard = _guard(catalog)
    statement = "keep the ingest pipeline clean while batching the writes"
    assert guard.is_architecture_only(_part(statement, component_refs=["IngestPipeline"]), []) is False
    assert guard.is_architecture_only(_part(statement), ["IngestPipeline"]) is False


def test_locally_catalogued_pattern_is_kept_with_its_category(catalog) -> None:
    """§5 rule 5: a named locally implementable part keeps the part and sets its hypothesis."""
    guard = _guard(catalog)
    part = _part("use a hexagonal layout with an anti-corruption layer at the seam")
    assert guard.hypothesis_for(part) == catalog.get_by_name("anti-corruption-layer").category
    assert guard.is_architecture_only(part, []) is False
    part_without_acl = _part("use a hexagonal layout")
    assert guard.hypothesis_for(part_without_acl) is None
    assert guard.is_architecture_only(part_without_acl, []) is True


def test_hypothesis_is_only_derived_when_a_term_matched(catalog) -> None:
    """The hypothesis path exists solely for architecture-level statements naming a local record."""
    guard = _guard(catalog)
    assert guard.hypothesis_for(_part("apply the version number convention to the payload")) is None


def test_real_catalog_exposes_the_keep_list(catalog) -> None:
    """Record existence — not prose — decides what is locally implementable (§5 rule 5)."""
    for name in ("anti-corruption-layer", "presentation-model", "poison-pill", "backpressure", "version-number"):
        assert catalog.get_by_name(name) is not None
    for excluded in ("microservices", "hexagonal", "cqrs", "event-sourcing"):
        assert catalog.get_by_name(excluded) is None
