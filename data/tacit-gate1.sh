#!/bin/sh
# Gate 1 of the 2026-11 retrain: did gold supervision move, and is the artifact final?
cd ~/design-pattern-mcp || exit 1
echo "== build:"; if pgrep -f "make docker-build-tacit" >/dev/null; then
  echo "  running ($(ps -o etime= -p $(pgrep -f 'make docker-build-tacit' | head -1) | tr -d ' '))"
else
  echo "  not running"
fi
echo "== host artifact dir:"
ls -la models/tacit-4b/self-distill.json models/tacit-4b/.extract-complete 2>/dev/null | sed 's/  */ /g'
echo "== marker verifies against current train inputs:"
if sha256sum -c models/tacit-4b/.extract-complete >/dev/null 2>&1; then echo "  OK"; else echo "  STALE/MISSING"; fi
echo "== gate 1 (style-gold fidelity, was 0.333):"
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("models/tacit-4b/self-distill.json")
if not p.is_file():
    print("  self-distill.json absent (build not finished)"); raise SystemExit
d = json.loads(p.read_text())
seams = d.get("seams", {})
for name in ("style-gold", "style-gold-none", "style-stage2", "style"):
    if name in seams:
        v = seams[name]
        print(f"  {name}: {json.dumps(v)[:200]}")
print("  manifest keys:", sorted(d.keys())[:12])
PY
echo "== images:"; docker images --format "{{.Repository}}:{{.Tag}} {{.ID}} {{.CreatedSince}}" | grep tacit | head -4
