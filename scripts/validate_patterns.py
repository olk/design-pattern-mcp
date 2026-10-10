#!/usr/bin/env python3
"""
Validate all pattern JSON files against their schemas.

pattern/*.json          -> docs/pattern-schema.json
pattern/*.<lang>.json   -> docs/pattern-language-schema.json
pattern/*.generic.json  -> docs/pattern-language-schema.json (generic realization, layout V8)

Exit code 0 on clean, 1 on errors, 2 on schema/usage errors.
Stored main records must not carry the key language_implementations (layout V8: the
generic realization lives in <name>.generic.json). The root schema already rejects it
(additionalProperties: false); the structural check below keeps the targeted message.
Language files carrying variants are additionally checked against guide §2.3 rule 5: the
file's flat fields must equal the default variant's, default_variant must resolve and the
variant_id slugs must be unique (the server's startup gate enforces the same rule).
language_files map values that do not resolve to a file on disk are reported as
non-fatal WARNINGs on stderr.
"""
import json
import pathlib
import sys
import argparse
from jsonschema import Draft7Validator

# File-suffix gate for language files: the middle dot-segment of pattern/<name>.<lang>.json
# must be one of these. "generic" is the layout-V8 generic realization
# (pattern/<name>.generic.json); generic files validate against the language schema
# like any other language file.
LANGS = {
    "generic",
    "c", "cplusplus", "csharp", "go", "java", "javascript",
    "kotlin", "python", "ruby", "rust", "scala", "swift", "typescript",
}


class _StructuralError:
    """Non-schema violation surfaced through the same (name, errs) channel."""

    def __init__(self, message):
        self.message = message
        self.path = []


def resolve(node, base, ps):
    """
    Recursively inline all $ref in a schema dict.
    Handles:
      - pattern-schema.json#/*  -> look up in ps (pattern-schema.json)
      - pattern-language-schema.json#/* -> look up in base (pattern-language-schema.json)
    """
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            doc, frag = (ps, ref.split("#", 1)[1]) if ref.startswith("pattern-schema.json#") \
                else (base, ref.split("#", 1)[1])
            t = doc
            for part in frag.strip("/").split("/"):
                t = t[part]
            merged = dict(resolve(t, doc, ps))
            merged.update({k: v for k, v in node.items() if k != "$ref"})
            return merged
        return {k: resolve(v, base, ps) for k, v in node.items()}
    return [resolve(v, base, ps) for v in node] if isinstance(node, list) else node


def load_schemas():
    docs = pathlib.Path(__file__).resolve().parent.parent / "docs"
    ps = json.loads((docs / "pattern-schema.json").read_text())
    ls = json.loads((docs / "pattern-language-schema.json").read_text())
    return ps, ls


def validate_patterns(ps, ls, verbose=False, pattern_dir=None):
    """Validate all pattern/*.json (main records) against pattern-schema.json."""
    if pattern_dir is None:
        pattern_dir = pathlib.Path(__file__).resolve().parent.parent / "pattern"
    errors = []
    warnings = []
    checked = 0

    MV = Draft7Validator(resolve(ps, ps, ps))

    for main_file in sorted(pattern_dir.glob("*.json")):
        # main record: exactly one dot (name.json); language files: two dots (name.lang.json)
        if main_file.name.count(".") != 1:
            continue
        name = main_file.name
        rec = json.loads(main_file.read_text())
        # layout V8: the stored record must not carry the inline realization field
        # (the schema rejects it as an unknown property; this keeps the targeted message)
        structural = "language_implementations" in rec
        if structural:
            errors.append((name, [_StructuralError(
                "language_implementations must be absent from stored records "
                "(layout V8; the generic realization lives in <name>.generic.json)")]))
        # non-fatal: language_files values that point nowhere on disk (visible every run,
        # exit code unchanged - see guide §2.3 rule 5 for the eventual CI gate)
        for key, value in sorted(rec.get("language_files", {}).items()):
            if not (pattern_dir / value).exists():
                warnings.append(f"{name}: language_files[{key}] -> {value} missing")
        errs = sorted(MV.iter_errors(rec), key=lambda e: list(e.path))
        if errs:
            errors.append((name, errs))
        if errs or structural:
            continue
        checked += 1
        if verbose:
            print(f"  OK  {name}")

    return errors, checked, warnings


def _variant_problems(rec):
    """Layout V8 / guide §2.3 rule 5: the default variant's flat fields MUST equal the file's.

    Returns a list of _StructuralError items (empty when the file carries no variants).
    """
    variants = rec.get("variants")
    if not variants:
        return []
    problems = []
    ids = [v.get("variant_id") for v in variants]
    duplicates = sorted({vid for vid in ids if ids.count(vid) > 1})
    if duplicates:
        problems.append(_StructuralError(f"duplicate variant_id(s): {duplicates}"))
    default = rec.get("default_variant")
    if default not in ids:
        problems.append(_StructuralError(
            f"default_variant {default!r} matches no variant_id {ids}"))
        return problems
    default_variant = next(v for v in variants if v.get("variant_id") == default)
    for key in ("idiom_name", "implementation_notes", "code_sketch", "pitfalls", "sources"):
        flat = rec.get(key)
        variant_value = default_variant.get(key)
        if key in ("pitfalls", "sources"):
            flat = list(flat or [])
            variant_value = list(variant_value or [])
        if flat != variant_value:
            problems.append(_StructuralError(
                f"flat field {key!r} differs from the default variant's "
                f"({str(flat)[:80]!r} != {str(variant_value)[:80]!r})"))
    return problems


def validate_language_files(ps, ls, verbose=False, pattern_dir=None):
    """Validate all pattern/*.<lang>.json against pattern-language-schema.json."""
    if pattern_dir is None:
        pattern_dir = pathlib.Path(__file__).resolve().parent.parent / "pattern"
    errors = []
    checked = 0

    LV = Draft7Validator(resolve(ls, ls, ps))

    for lang_file in sorted(pattern_dir.glob("*.*.json")):
        # language file: exactly two dots (name.lang.json); main record: one dot (name.json)
        if lang_file.name.count(".") != 2:
            continue
        parts = lang_file.name.rsplit(".", 2)   # [name, lang, json]
        if len(parts) != 3 or parts[1] not in LANGS:
            continue
        rec = json.loads(lang_file.read_text())
        errs = sorted(LV.iter_errors(rec), key=lambda e: list(e.path))
        errs = list(errs) + _variant_problems(rec)
        if errs:
            errors.append((lang_file.name, errs))
        else:
            checked += 1
            if verbose:
                print(f"  OK  {lang_file.name}")

    return errors, checked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every validated file")
    ap.add_argument("--pattern-schema", metavar="FILE",
                    default=None,
                    help="path to pattern-schema.json (default: <repo>/docs/pattern-schema.json)")
    ap.add_argument("--language-schema", metavar="FILE",
                    default=None,
                    help="path to pattern-language-schema.json")
    ap.add_argument("--pattern-dir", metavar="DIR",
                    default=None,
                    help="path to pattern/ directory (default: <repo>/pattern/)")
    args = ap.parse_args()

    # resolve paths
    repo = pathlib.Path(__file__).resolve().parent.parent
    ps_path = pathlib.Path(args.pattern_schema) if args.pattern_schema \
              else repo / "docs" / "pattern-schema.json"
    ls_path = pathlib.Path(args.language_schema) if args.language_schema \
              else repo / "docs" / "pattern-language-schema.json"
    pattern_dir = pathlib.Path(args.pattern_dir) if args.pattern_dir \
                  else repo / "pattern"

    if not ps_path.exists():
        sys.exit(f"pattern-schema.json not found: {ps_path}")
    if not ls_path.exists():
        sys.exit(f"pattern-language-schema.json not found: {ls_path}")
    if not pattern_dir.is_dir():
        sys.exit(f"pattern/ directory not found: {pattern_dir}")

    ps = json.loads(ps_path.read_text())
    ls = json.loads(ls_path.read_text())

    print("=== Main records (pattern-schema.json) ===")
    main_errors, main_ok, warnings = validate_patterns(ps, ls, args.verbose, pattern_dir=pattern_dir)
    print(f"checked: {main_ok}, errors: {len(main_errors)}")

    print("\n=== Language files (pattern-language-schema.json) ===")
    lang_errors, lang_ok = validate_language_files(ps, ls, args.verbose, pattern_dir=pattern_dir)
    print(f"checked: {lang_ok}, errors: {len(lang_errors)}")

    if warnings:
        print(f"WARNING: {len(warnings)} language_files entries reference files missing on disk",
              file=sys.stderr)
        for w in warnings:
            print(f"WARNING: {w}", file=sys.stderr)

    total_errors = len(main_errors) + len(lang_errors)
    if total_errors == 0:
        print(f"\nALL VALID ({main_ok} records + {lang_ok} language files)")
        sys.exit(0)
    else:
        print(f"\n{total_errors} FILES WITH ERRORS:")
        for fname, errs in main_errors + lang_errors:
            print(f"\n  {fname}:")
            for e in errs[:5]:
                print(f"    {e.message}  @ {'.'.join(str(p) for p in e.path)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
