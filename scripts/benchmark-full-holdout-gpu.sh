#!/usr/bin/env bash
# Full-holdout base-vs-shipped-stack e2e A/B on the GPU host (138 scenarios).
# Runs ON 192.168.1.153 (4x RTX 3090). Two arms on identical corpus/split/repeat:
#   full-base     TEI-only (base files; every seam off, rerank tei)
#   full-bundle   GPU decision overlay at SHIPPED DEFAULTS: TEI rerank (the overlay
#                 default since the J1 verdict 20261124-173228 — NO explicit
#                 RERANKER_PROVIDER, so this arm exercises the default itself),
#                 linear style (0.6) + Tacit weights/category/gate.
# Purpose: quality-operator measurement at full-holdout power (seed n=18/12 CIs of
# ±0.29/±0.34 cannot resolve small effects; n=276 can). Verdicts are DATA: exits 0
# once both arms and the compare ran, nonzero only on arm/assert failures.
#
# Knobs (env): STAMP BENCH_SCENARIOS SPLIT LIMIT REPEAT BENCH_RUNS MCP_IMAGE
#   TEI_IMAGE TEI_RERANK_IMAGE TACIT_IMAGE STYLE_HEAD_IMAGE
#   BENCH_STACK_TIMEOUT BENCH_READY_TIMEOUT.
#
# Each arm tears its compose stack down afterwards. Runtime ~22 h at
# REPEAT=2 (276 calls/arm at ~2.5 min/call). Launch detached ON the host, e.g.:
#   S=$(date -u +%Y%m%d-%H%M%S); STAMP=$S nohup bash scripts/benchmark-full-holdout-gpu.sh \
#     > data/benchmark-runs/$S-full-driver.log 2>&1 & echo $! > data/benchmark-runs/$S-full-driver.pid

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
SCENARIOS="${BENCH_SCENARIOS:-tests/benchmark/scenarios/full.json}"
SPLIT="${SPLIT:-holdout}"
LIMIT="${LIMIT:-}"
REPEAT="${REPEAT:-2}"
RUNS="${BENCH_RUNS:-data/benchmark-runs}"
OVERLAY="docker/docker-compose.decision-models.yml"
STYLE_CALIBRATION="models/style-head-calibration.json"
BASE_ID="$STAMP-full-base"
BUNDLE_ID="$STAMP-full-bundle"

# Pinned images: both arms run the same bytes (provenance, no :latest drift).
# TACIT_IMAGE (not TACIT_CPU_IMAGE): the GPU overlay's variable.
export MCP_IMAGE="${MCP_IMAGE:-design-pattern-mcp:1.0.1}"
export TEI_IMAGE="${TEI_IMAGE:-design-pattern-tei-embed:1.0.1}"
export TEI_RERANK_IMAGE="${TEI_RERANK_IMAGE:-design-pattern-tei-rerank:1.0.1}"
export TACIT_IMAGE="${TACIT_IMAGE:-design-pattern-tacit:1.0.1}"
export STYLE_HEAD_IMAGE="${STYLE_HEAD_IMAGE:-design-pattern-style-head:1.0.1}"
# Image pins as MAKE ARGS (not exports): a Makefile `:=` shadows same-named env
# vars in recipes (proven 2026-11-23) — CLI args beat file assignments. Only names
# the Makefile defines need this; MCP_IMAGE/TEI_IMAGE flow via export.
PIN_ARGS=("TACIT_IMAGE=$TACIT_IMAGE" "TEI_RERANK_IMAGE=$TEI_RERANK_IMAGE" "STYLE_HEAD_IMAGE=$STYLE_HEAD_IMAGE")
export BENCH_STACK_TIMEOUT="${BENCH_STACK_TIMEOUT:-3600}"
export BENCH_READY_TIMEOUT="${BENCH_READY_TIMEOUT:-600}"

die() { echo "full-holdout-gpu: $*" >&2; exit 1; }
log() { echo ">> $(date -u +%FT%TZ) $*"; }

# --- seam-usage watch + ledger capture (shared shape with the GPU driver) ------
TACIT_STATS_URL="http://127.0.0.1:18100/v1/stats"
TACIT_WATCH_FILE="" # set per arm; the EXIT trap stops whatever is polling
LEDGER_FILE=""      # set per arm alongside the watch file

stop_tacit_watch() {
  if [ -n "$TACIT_WATCH_FILE" ] && [ -f "$TACIT_WATCH_FILE.pid" ]; then
    kill "$(cat "$TACIT_WATCH_FILE.pid")" 2>/dev/null || true
    rm -f "$TACIT_WATCH_FILE.pid"
  fi
  TACIT_WATCH_FILE=""
  LEDGER_FILE=""
}
trap stop_tacit_watch EXIT

start_tacit_watch() { # $1=stats outfile $2=ledger outfile
  stop_tacit_watch
  TACIT_WATCH_FILE="$1"
  LEDGER_FILE="$2"
  # Direct backgrounding: $! is only valid in the shell that spawned the job.
  # Loop-body names must be GLOBALS ($2 would be unset after this returns).
  while true; do
    echo "=== $(date -u +%FT%TZ) ==="
    curl -fsS -m 10 "$TACIT_STATS_URL" 2>/dev/null || echo "STATS_UNREACHABLE"
    docker logs --tail 30 design-pattern-mcp-dev 2>&1 | grep -iE 'decision degraded|choice degraded' || true
    docker logs --since 70s design-pattern-mcp-dev >> "$LEDGER_FILE" 2>&1 || true
    sleep 60
  done > "$1" 2>&1 & echo $! > "$1.pid"
  log "watch polling -> $1 + ledger $2"
}

tacit_decisions_max() { # $1=watchfile; max decisions counter seen (empty if none)
  grep -o '"decisions": *[0-9]*' "$1" 2>/dev/null | grep -o '[0-9]*' | sort -n | tail -1 || true
}

ledger_count() { # $1=ledger path, $2+=grep args; match count (empty when unreadable)
  local file="$1"; shift
  grep -c "$@" "$file" 2>/dev/null || true
}

# Served-byte prover: benchmark-docker owns the stack lifecycle (up…run…down),
# so no in-driver hook runs while containers are up. This one-shot backgrounds
# alongside the watch, waits for the healthy MCP, records served Config.Image +
# image IDs once, and exits. A pin that silently drifted to :latest fails loud.
prove_served() { # $1=outfile
  (
    for _ in $(seq 1 100); do
      if docker inspect design-pattern-mcp-dev --format '{{.State.Health.Status}}' 2>/dev/null | grep -qx healthy; then
        break
      fi
      sleep 30
    done
    for c in design-pattern-mcp-dev design-pattern-tacit-dev; do
      docker inspect "$c" --format "$c Config.Image={{.Config.Image}} Id={{.Image}}" 2>&1 || echo "$c absent"
    done
  ) > "$1" 2>&1 &
}

# --- preflight (fail loud before warming anything) ---------------------------
[ -f "$SCENARIOS" ] || die "scenario corpus not found: $SCENARIOS"
[ -f "$OVERLAY" ] || die "overlay not found: $OVERLAY"
for img in "$MCP_IMAGE" "$TEI_IMAGE" "$TEI_RERANK_IMAGE" "$TACIT_IMAGE" "$STYLE_HEAD_IMAGE"; do
  docker image inspect "$img" >/dev/null 2>&1 || die "image missing: $img"
done
log "images pinned: $MCP_IMAGE $TEI_IMAGE $TEI_RERANK_IMAGE $TACIT_IMAGE $STYLE_HEAD_IMAGE"

# Ambient seam vars would leak through compose ${VAR:-default} interpolation and
# silently rewire an arm. The driver sets every seam var per arm itself — except
# RERANKER_PROVIDER on the bundle arm, which must come from the overlay DEFAULT.
leaked_decisions="$(compgen -A export | grep -E '^DECISIONS_' || true)"
[ -z "$leaked_decisions" ] || die "ambient DECISIONS_* would leak into an arm: $leaked_decisions — unset first"
for leaked in RERANKER_PROVIDER TASKS_DEADLINE_SECONDS REASONING_ENABLED BENCH_DECISION_OVERLAY; do
  [ -z "${!leaked:-}" ] || die "ambient $leaked=${!leaked} would leak into an arm — unset it first"
done

# The style floor must come from the calibration artifact, never hand-copied.
python3 - "$STYLE_CALIBRATION" <<'EOF' || die "floor 0.6 not certified by $STYLE_CALIBRATION"
import json, sys
d = json.load(open(sys.argv[1]))
floors = [f.get("floor", f.get("threshold")) for f in d.get("floors", [])]
sys.exit(0 if 0.6 in floors else 1)
EOF
log "style floor 0.6 certified by $STYLE_CALIBRATION"

# The bundle arm exercises the overlay RERANKER default (no explicit provider):
# assert it renders tei with the var unset (J1 verdict), and that an explicit
# tacit still passes through (opt-in intact).
# Capture-then-grep (never `producer | grep -q`): with `set -o pipefail`, grep -q's
# early exit can SIGPIPE the producer into a flaky false failure (hit 1/5 preflights).
default_provider="$(env -u RERANKER_PROVIDER docker compose -f docker/docker-compose.yml \
  -f docker/docker-compose.benchmark.yml -f "$OVERLAY" config 2>/dev/null \
  | grep -E 'RERANKER_PROVIDER' | head -1)" || die "compose config failed"
grep -q 'tei' <<<"$default_provider" || die "overlay rerank default is not tei: $default_provider"
explicit_provider="$(RERANKER_PROVIDER=tacit docker compose -f docker/docker-compose.yml \
  -f docker/docker-compose.benchmark.yml -f "$OVERLAY" config 2>/dev/null \
  | grep -E 'RERANKER_PROVIDER' | head -1)" || die "compose config failed"
grep -q 'tacit' <<<"$explicit_provider" || die "explicit RERANKER_PROVIDER=tacit lost: $explicit_provider"
log "overlay rerank default renders tei; explicit tacit passes through"

# An arm whose generator key renders empty still "completes" down the degraded
# paths and measures noise — assert the rendered value in BOTH file sets.
# (Shape check only: never print the key — transcripts persist.)
for files in "-f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml" \
  "-f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml -f $OVERLAY"; do
  # shellcheck disable=SC2086
  key_len="$(docker compose $files config 2>/dev/null | grep -E 'GENERATOR_API_KEY' | head -1 | sed -E 's/.*KEY:? *//' | tr -d ' "'\'' ' | wc -c)"
  [ "$key_len" -gt 10 ] || die "GENERATOR_API_KEY renders empty in: $files"
done
log "GENERATOR_API_KEY renders non-empty in both file sets"

# The candidate file set must actually carry the Tacit service; the baseline set
# must not (a silent no-op arm would compare TEI against TEI).
rendered_services="$(docker compose -f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml \
  -f "$OVERLAY" config --services 2>/dev/null)" || die "compose config failed for the candidate file set"
grep -qx "design-pattern-tacit" <<<"$rendered_services" \
  || die "overlay does not render a design-pattern-tacit service"
rendered_base="$(docker compose -f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml \
  config --services 2>/dev/null)" || die "compose config failed for the baseline file set"
if grep -qx "design-pattern-tacit" <<<"$rendered_base"; then
  die "baseline file set renders design-pattern-tacit — the arms would not differ"
fi
log "overlay renders design-pattern-tacit; baseline does not"

for dir in "$RUNS/$BASE_ID" "$RUNS/$BUNDLE_ID"; do
  [ -e "$dir" ] && die "run dir already exists: $dir (refusing to overwrite)"
done
running_containers="$(docker ps --format '{{.Names}}' 2>/dev/null)"
if grep -qx "design-pattern-mcp-dev-benchmark-run-.*" <<<"$running_containers"; then
  die "another benchmark run container is active"
fi
log "preflight green: corpus=$SCENARIOS split=$SPLIT limit=${LIMIT:-all} repeat=$REPEAT"

# --- arm runner: clean slate + one retry --------------------------------------
BASE_FILES="-f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml"
CAND_FILES="$BASE_FILES -f $OVERLAY"

style_pre() {
  # The style head is NOT in MCP depends_on (its seam degrades silently), so
  # bring it up explicitly on the arm's own file set.
  log "style head up (explicit: not a compose dependency)"
  # shellcheck disable=SC2086
  docker compose $CAND_FILES up -d --wait design-pattern-style-head
}

run_arm() { # $1=tag $2=files $3=RUN_ID $4=pre-hook-fn $5...=extra make args
  local tag="$1" files="$2" run_id="$3" pre="$4"; shift 4
  local attempt
  for attempt in 1 2; do
    log "$tag: clean slate (attempt $attempt)"
    # shellcheck disable=SC2086
    docker compose $files down --remove-orphans >/dev/null 2>&1 || true
    if ! "$pre"; then log "$tag: pre-hook failed"; continue; fi
    if make benchmark-e2e BENCH_SCENARIOS="$SCENARIOS" SPLIT="$SPLIT" \
      LIMIT="$LIMIT" REPEAT="$REPEAT" RUN_ID="$run_id" "$@" "${PIN_ARGS[@]}"; then
      log "$tag complete: $RUNS/$run_id"
      return 0
    fi
    log "$tag: attempt $attempt failed"
  done
  die "$tag failed twice — not comparing against a partial arm"
}

# Engagement markers (ledger greps). Asserts prove WIRED+SERVED; firing counts are
# MEASUREMENTS (abstention is data).
STYLE_HIT='Architecture style resolved by decision model'
NARROW_HIT='Category narrowing (applied|kept nothing)'
DMW_HIT='"weights_source": ?"decision-model"'
DEGRADED_HIT='decision degraded|choice degraded'

check_tally() { # $1=arm $2=style_n $3=narrow_n $4=dmw_n $5=tacit $6=want $7=seam_ok $8=degraded
  local arm="$1" style_n="$2" narrow_n="$3" dmw_n="$4" tacit="$5" want="$6" ok="$7" degraded="$8"
  log "$arm engagement: style_resolved=$style_n narrowing=$narrow_n dm_weights=$dmw_n tacit_max=$tacit degraded=$degraded"
  case "$arm" in
    full-base)   [ "$style_n" = "0" ] || die "$arm: style fired $style_n× (must be 0)"
                [ "$narrow_n" = "0" ] || die "$arm: narrowing fired $narrow_n× (must be 0)"
                [ "$dmw_n" = "0" ] || die "$arm: dm-weights fired $dmw_n× (must be 0)"
                [ "$tacit" = "0" ] || die "$arm: shows $tacit Tacit decisions (must be 0)"
                [ "$degraded" = "0" ] || die "$arm: $degraded degraded lines with no seams (contamination)" ;;
    full-bundle) [ "$ok" = "1" ] || die "$arm: a seam is not wired (startup lacks an ok marker) — void arm"
                [ "$tacit" -gt "0" ] || die "$arm: ZERO Tacit decisions — measured the fallback path"
                [ "$degraded" = "0" ] || die "$arm: $degraded degraded lines — the full stack did not fully serve" ;;
  esac
  log "$arm asserts green"
}

tally_arm() { # $1=arm-tag $2=want-tacit; gathers watch + ledger counts and checks them
  local arm="$1" want="$2"
  local watch="$RUNS/$STAMP-$arm-watch.log" ledger="$RUNS/$STAMP-$arm-ledger.log"
  local tacit style_n narrow_n dmw_n degraded seam_ok
  tacit=$(tacit_decisions_max "$watch"); tacit=${tacit:-0}
  style_n=$(ledger_count "$ledger" -F "$STYLE_HIT"); style_n=${style_n:-0}
  narrow_n=$(ledger_count "$ledger" -E "$NARROW_HIT"); narrow_n=${narrow_n:-0}
  dmw_n=$(ledger_count "$ledger" -E "$DMW_HIT"); dmw_n=${dmw_n:-0}
  degraded=$(ledger_count "$ledger" -iE "$DEGRADED_HIT"); degraded=${degraded:-0}
  case "$arm" in
    full-base) seam_ok=1 ;;
    full-bundle)
      if grep -qF 'style=ok' "$ledger" 2>/dev/null \
        && grep -qF 'weights=ok' "$ledger" 2>/dev/null \
        && grep -qF 'category=ok' "$ledger" 2>/dev/null \
        && grep -qF 'reranker=tei' "$ledger" 2>/dev/null; then seam_ok=1; else seam_ok=0; fi ;;
  esac
  check_tally "$arm" "$style_n" "$narrow_n" "$dmw_n" "$tacit" "$want" "$seam_ok" "$degraded"
  if [ "$degraded" -gt "0" ]; then
    log "degraded lines in $arm ledger ($degraded; precise bigram grep; duplicates from overlapping windows):"
    grep -hiE "$DEGRADED_HIT" "$ledger" 2>/dev/null | cut -c1-250 | sort | uniq -c | sort -rn | head -10 || true
  fi
}

# --- arm 1/2: TEI-only baseline -----------------------------------------------
# shellcheck disable=SC2086
docker compose $BASE_FILES down --remove-orphans >/dev/null 2>&1 || true
start_tacit_watch "$RUNS/$STAMP-full-base-watch.log" "$RUNS/$STAMP-full-base-ledger.log"
prove_served "$RUNS/$STAMP-full-base-served.log"
run_arm "arm 1/2 full-base (TEI-only)" "$BASE_FILES" "$BASE_ID" true TASKS_DEADLINE_SECONDS=3600
stop_tacit_watch
tally_arm "full-base" 0

# --- arm 2/2: shipped stack (overlay defaults incl. TEI rerank) ----------------
# NOTE: no explicit RERANKER_PROVIDER — this arm exercises the overlay default.
# shellcheck disable=SC2086
docker compose $CAND_FILES down --remove-orphans >/dev/null 2>&1 || true
start_tacit_watch "$RUNS/$STAMP-full-bundle-watch.log" "$RUNS/$STAMP-full-bundle-ledger.log"
prove_served "$RUNS/$STAMP-full-bundle-served.log"
run_arm "arm 2/2 full-bundle" "$CAND_FILES" "$BUNDLE_ID" style_pre \
  BENCH_DECISION_OVERLAY="$OVERLAY" \
  DECISIONS_STYLE_ENABLED=true DECISIONS_STYLE_CONFIDENCE_FLOOR=0.6 \
  DECISIONS_STYLE_PROVIDER=linear \
  DECISIONS_STYLE_BASE_URL=http://design-pattern-style-head:8800/v1 \
  DECISIONS_STYLE_MODEL=style-linear \
  DECISIONS_WEIGHTS_ENABLED=true \
  DECISIONS_CATEGORY_ENABLED=true \
  DECISIONS_REASONING_GATE_ENABLED=true \
  TASKS_DEADLINE_SECONDS=3600
stop_tacit_watch
tally_arm "full-bundle" 1

# --- compare (verdicts are DATA) ------------------------------------------------
log "compare full-base vs full-bundle (pre-registered rule + repeat-stability gate)"
set +e
make benchmark-compare A="$RUNS/$BASE_ID" B="$RUNS/$BUNDLE_ID" REQUIRE_REPEAT_STABILITY=1
rc=$?
set -e
log "compare exit=$rc (nonzero = candidate-loss; verdicts are data here)"
python3 -c "
import json
v = json.load(open('$RUNS/$BUNDLE_ID/comparison/comparison.json'))['verdict']
print('>> verdict full-base-vs-full-bundle:', v['outcome'], '|', '; '.join(v['reasons']))"
log "done: base=$RUNS/$BASE_ID bundle=$RUNS/$BUNDLE_ID — restart the dev stack when finished"
exit 0
