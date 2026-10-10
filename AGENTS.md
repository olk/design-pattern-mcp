# Agent Rules — design-pattern-mcp

## Static typing (MANDATORY)
- `src/` is mypy strict. `uv run mypy --strict` MUST exit 0 before finishing any task.
- Annotate every function signature; parameterize all generics (`dict[str, X]`, not `dict`).
- No bare `Any` in public signatures.
- Pydantic: use `Field(default=...)` keyword form, never positional defaults.
- `# type: ignore[<code>]` ONLY where third-party stubs are provably wrong; always with a
  justification comment. No bare `type: ignore`, no blanket ignores.
- Scope: `src/` only (`tests/`, `examples/` currently unchecked).

## Quality gates
- `make check-lint` (ruff), `make check-static-typing` (mypy), `make test-unit`,
  `make check-deadcode`, `make check-depcheck` — all must pass before task completion
  (`make check-all` runs the four check-* gates in one go).
- Runtime behavior of existing code must not change while fixing type errors; `tests/unit/`
  is the behavioral oracle.

## Domain contract
- `docs/implementation-guide.md` is normative for the pipeline (§-numbers referenced from the
  code), `docs/result-schema.json` is the wire contract of `DesignResult`,
  `docs/pattern-schema.json` is the catalog record contract. Changing a schema means changing
  the document and the code in the same commit; `src/patterns/loader.py::assert_enums_in_sync`
  fails startup when the `Literal` enums drift from the schema.

## Verification program
- Changing job/pipeline control flow: update the matching `.fizz` model in `verify/fizz/` in the
  same PR, prove it with `make verify-fizz`; every assertion must kill at least one spec-garden
  mutant and appear in the `verify/fizz/README.md` ledger.
- Every `.fizz` `always` / `always eventually` assertion must kill at least one spec-garden
  mutant or carry a `# spec-explains:` justification — vacuous assertions are rejected.
- The executable layers (L1 unit tests, L2 Hypothesis oracles, L3 gardens, L4 FizzBee, L6 DST)
  are the verification authority; a change to a decision module (see
  `verify/mutmut-scope.md`) must keep `tests/unit/` and `tests/verification/` green.
- L3 scope changes (any edit to `[tool.mutmut]` in `pyproject.toml`) must update the
  `verify/mutmut-scope.md` ledger row in the same commit. `make test-mutations` enforces a
  ratchet against the committed `verify/mutmut-baseline.json` (new survivors and kill-ratio
  drops fail the gate; deliberate refresh via `make regen-mutmut-baseline`; acknowledged
  equivalents go to `verify/mutmut-equivalents.md`).

## Reasoning layer
- `REASONING_ENABLED` defaults to true; the client degrades silently per call and reports its
  health at startup. A reasoning outage must never fail a tool call — the in-prompt scaffold
  is the documented fallback.
