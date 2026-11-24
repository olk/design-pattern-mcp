#!/usr/bin/env bash
# Seam ablation: which decision seam moves selection? Follow-up to 20261122-185306,
# where the full CPU decision stack lost to TEI-only on latency, hits and repeat
# agreement. Four e2e arms on identical corpus/split/repeat, then three pairwise
# base-vs-seam compares with REQUIRE_REPEAT_STABILITY=1:
#   abl-base     TEI-only (base files; every seam off)
#   abl-style    base + linear style head (floor 0.6); Tacit never started
#   abl-weights  overlay + Tacit weights (900 s); style/category/gate off
#   abl-category overlay + Tacit category (120 s); style/weights/gate off
# The J5 gate is off in ALL arms (proven inert with reasoning off: its closure is
# only built inside ReasoningClient construction — src/server.py — unreachable with
# REASONING_ENABLED=false, and the constructed-but-uncalled client makes no network
# calls). Rerank is TEI in all arms (CPU lane; J1 Tacit infeasible at ~55 s/item).
# Deadline 3600 and reasoning off everywhere. Single-seam-on FIRST: if no single
# seam reproduces the bundle harm, the effect is an interaction → leave-one-out
# arms come next (not this driver).
#
# Per-arm engagement asserts (host-side /v1/stats poller + server-ledger capture):
# each arm must show its own seam firing AND every other seam silent — a
# wrongly-wired arm dies loud instead of comparing noise. The ledger (docker logs
# --since sampled per minute) also carries per-part weights vectors, candidate
# scores and selections for the attribution analysis. Verdicts are DATA here (harm
# is expected somewhere): the driver exits 0 once all arms and compares ran, and
# nonzero only on arm/assert failures.
#
# Knobs (env): STAMP BENCH_SCENARIOS SPLIT LIMIT REPEAT BENCH_RUNS MCP_IMAGE
#   TEI_IMAGE TEI_RERANK_IMAGE TACIT_CPU_IMAGE STYLE_HEAD_IMAGE
#   BENCH_STACK_TIMEOUT BENCH_READY_TIMEOUT.
# Resume: SKIP_BASE=1 SKIP_STYLE=1 BASE_ID=<old> BASE_STAMP=<old>
#   STYLE_ID=<old> STYLE_STAMP=<old> reuses complete arm dirs (re-tallied).
#
# Each arm tears its compose stack down afterwards; restart the dev stack when the
# driver finishes. Runtime ~5 h at LIMIT=6/REPEAT=2. Launch detached, e.g.:
#   S=$(date -u +%Y%m%d-%H%M%S); STAMP=$S nohup bash scripts/benchmark-seam-ablation.sh \
#     > data/benchmark-runs/$S-ablation-driver.log 2>&1 & echo $! > data/benchmark-runs/$S-ablation-driver.pid

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
SCENARIOS="${BENCH_SCENARIOS:-tests/benchmark/scenarios/seed.json}"
SPLIT="${SPLIT:-holdout}"
LIMIT="${LIMIT:-6}"
REPEAT="${REPEAT:-2}"
RUNS="${BENCH_RUNS:-data/benchmark-runs}"
OVERLAY="docker/docker-compose.benchmark-tacit-cpu.yml"
STYLE_CALIBRATION="models/style-head-calibration.json"
BASE_ID="${BASE_ID:-$STAMP-abl-base}"
STYLE_ID="${STYLE_ID:-$STAMP-abl-style}"
WEIGHTS_ID="${WEIGHTS_ID:-$STAMP-abl-weights}"
CATEGORY_ID="$STAMP-abl-category"
BASE_STAMP="${BASE_STAMP:-$STAMP}"
STYLE_STAMP="${STYLE_STAMP:-$STAMP}"
WEIGHTS_STAMP="${WEIGHTS_STAMP:-$STAMP}"

# Pinned images: all arms run the same bytes (provenance, no :latest drift).
export MCP_IMAGE="${MCP_IMAGE:-design-pattern-mcp:1.0.1}"
export TEI_IMAGE="${TEI_IMAGE:-design-pattern-tei-embed:1.0.1}"
export TEI_RERANK_IMAGE="${TEI_RERANK_IMAGE:-design-pattern-tei-rerank:1.0.1}"
export TACIT_CPU_IMAGE="${TACIT_CPU_IMAGE:-design-pattern-tacit-cpu:1.0.1}"
export STYLE_HEAD_IMAGE="${STYLE_HEAD_IMAGE:-design-pattern-style-head:1.0.1}"
# Image pins as MAKE ARGS (not exports): a Makefile `:=` shadows same-named env
# vars in recipes (proven 2026-11-23) — CLI args beat file assignments. Only names
# the Makefile defines need this; MCP_IMAGE/TEI_IMAGE flow via export.
PIN_ARGS=("TACIT_CPU_IMAGE=$TACIT_CPU_IMAGE" "TEI_RERANK_IMAGE=$TEI_RERANK_IMAGE" "STYLE_HEAD_IMAGE=$STYLE_HEAD_IMAGE")
export BENCH_STACK_TIMEOUT="${BENCH_STACK_TIMEOUT:-3600}"
export BENCH_READY_TIMEOUT="${BENCH_READY_TIMEOUT:-600}"

die() { echo "seam-ablation: $*" >&2; exit 1; }
log() { echo ">> $(date -u +%FT%TZ) $*"; }

# --- seam-usage watch + ledger capture ------------------------------------------
# e2e's wire contract hides fallbacks and per-part selections, and a dead/slow seam
# degrades with only a log warning — an arm can "complete" while measuring nothing.
# A host-side poller therefore samples per arm, once a minute: the sidecar's
# /v1/stats decisions counter, any "degraded" server-log lines, and the last ~70 s
# of MCP server logs (stderr JSON: "Part weights extracted" with weights_source,
# "Candidate scored", "Part selected", "Architecture style resolved by decision
# model", "Category narrowing applied"). The stats feed the void-state asserts;
# the ledger feeds the attribution analysis (dedup by record timestamp: the 70 s
# window overlaps the 60 s period at the edges).
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
  # Direct backgrounding: $! is only valid in the shell that spawned the job, so a
  # `( … & )` wrapper would capture the SUBSHELL's job id, unkillable from here.
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

# --- preflight (fail loud before warming anything) ---------------------------
[ -f "$SCENARIOS" ] || die "scenario corpus not found: $SCENARIOS"
[ -f "$OVERLAY" ] || die "overlay not found: $OVERLAY"
for img in "$MCP_IMAGE" "$TEI_IMAGE" "$TEI_RERANK_IMAGE" "$TACIT_CPU_IMAGE" "$STYLE_HEAD_IMAGE"; do
  docker image inspect "$img" >/dev/null 2>&1 || die "image missing: $img"
done
log "images pinned: $MCP_IMAGE $TEI_IMAGE $TEI_RERANK_IMAGE $TACIT_CPU_IMAGE $STYLE_HEAD_IMAGE"

# Ambient seam vars would leak through compose ${VAR:-default} interpolation and
# silently rewire an arm. The driver sets every seam var per arm itself (and sweeps
# the whole DECISIONS_* namespace, so a stray floor/provider leaks loud too).
leaked_decisions="$(compgen -A export | grep -E '^DECISIONS_' || true)"
[ -z "$leaked_decisions" ] || die "ambient DECISIONS_* would leak into an arm: $leaked_decisions — unset first"
for leaked in RERANKER_PROVIDER TASKS_DEADLINE_SECONDS REASONING_ENABLED BENCH_DECISION_OVERLAY; do
  [ -z "${!leaked:-}" ] || die "ambient $leaked=${!leaked} would leak into an arm — unset it first"
done

# The style floor must come from the calibration artifact, never hand-copied.
python3 - "$STYLE_CALIBRATION" <<'EOF' || die "floor 0.6 not certified by $STYLE_CALIBRATION"
import json, sys
floors = {r.get("floor") for r in json.load(open(sys.argv[1])).get("floors", [])}
sys.exit(0 if 0.6 in floors else 1)
EOF
log "style floor 0.6 certified by $STYLE_CALIBRATION"

# An arm whose generator key renders empty still "completes" down the degraded
# paths and measures noise — assert the rendered value in BOTH file sets.
for files in "-f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml" \
  "-f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml -f $OVERLAY"; do
  # shellcheck disable=SC2086
  rendered_key="$(docker compose $files config 2>/dev/null \
    | sed -n 's/^ *GENERATOR_API_KEY: *//p' | head -1)"
  case "$rendered_key" in
    "" | "''" | '""') die "GENERATOR_API_KEY renders EMPTY under [$files] — export OPENCODE_API_KEY" ;;
  esac
done
log "GENERATOR_API_KEY renders non-empty in both file sets"

# The candidate file set must actually carry the Tacit service; the baseline set
# must not (a silent no-op arm would compare TEI against TEI).
# Capture-then-grep (never `producer | grep -q`): with `set -o pipefail`, grep -q's
# early exit can SIGPIPE the producer into a flaky false failure (hit 1/5 preflights).
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

CHECK_DIRS=()
[ "${SKIP_BASE:-0}" = "1" ] || CHECK_DIRS+=("$RUNS/$BASE_ID")
[ "${SKIP_STYLE:-0}" = "1" ] || CHECK_DIRS+=("$RUNS/$STYLE_ID")
[ "${SKIP_WEIGHTS:-0}" = "1" ] || CHECK_DIRS+=("$RUNS/$WEIGHTS_ID")
CHECK_DIRS+=("$RUNS/$CATEGORY_ID")
for dir in "${CHECK_DIRS[@]}"; do
  [ ! -e "$dir" ] || [ -z "$(ls -A "$dir" 2>/dev/null)" ] \
    || die "run dir not empty (stale results would mix): $dir"
done
running_containers="$(docker ps --format '{{.Names}}' 2>/dev/null)"
if grep -qx "design-pattern-mcp-dev-benchmark-run-.*" <<<"$running_containers"; then
  die "another benchmark run container is active"
fi
log "preflight green: corpus=$SCENARIOS split=$SPLIT limit=$LIMIT repeat=$REPEAT"

# --- arm runner: clean slate + one retry --------------------------------------
BASE_FILES="-f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml"
CAND_FILES="$BASE_FILES -f $OVERLAY"

style_pre() {
  # The style head is NOT in MCP depends_on (its seam degrades silently), so
  # bring it up explicitly on the arm's own file set.
  log "style head up (explicit: not a compose dependency)"
  # shellcheck disable=SC2086
  docker compose $BASE_FILES up -d --wait design-pattern-style-head
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
# MEASUREMENTS (abstention is data: 20261123-070305 abl-style fired 0/11 with a
# healthy head and zero degraded lines — below-floor abstention, which exonerates
# Ch2 on this corpus instead of voiding the arm).
STYLE_HIT='Architecture style resolved by decision model'
NARROW_HIT='Category narrowing (applied|kept nothing)'
DMW_HIT='"weights_source": ?"decision-model"'
DEGRADED_HIT='decision degraded|choice degraded'

check_tally() { # $1=arm $2=style_n $3=narrow_n $4=dmw_n $5=tacit $6=want $7=seam_ok $8=degraded
  local arm="$1" style_n="$2" narrow_n="$3" dmw_n="$4" tacit="$5" want="$6" ok="$7" degraded="$8"
  local fired
  log "$arm engagement: style_resolved=$style_n narrowing=$narrow_n dm_weights=$dmw_n tacit_max=$tacit degraded=$degraded"
  case "$arm" in
    abl-base)   [ "$style_n" = "0" ] || die "$arm: style fired $style_n× (must be 0)"
                [ "$narrow_n" = "0" ] || die "$arm: narrowing fired $narrow_n× (must be 0)"
                [ "$dmw_n" = "0" ] || die "$arm: dm-weights fired $dmw_n× (must be 0)"
                [ "$tacit" = "0" ] || die "$arm: shows $tacit Tacit decisions (must be 0)"
                [ "$degraded" = "0" ] || die "$arm: $degraded degraded lines with no seams (contamination)" ;;
    abl-style)  fired="$style_n" ;;
    abl-weights) fired="$dmw_n" ;;
    abl-category) fired="$narrow_n" ;;
  esac
  if [ "$arm" != "abl-base" ]; then
    [ "$ok" = "1" ] || die "$arm: seam not wired (startup lacks ok marker) — void arm"
    if [ "$want" = "1" ]; then
      [ "$tacit" -gt "0" ] || die "$arm: ZERO Tacit decisions — seam never served"
    else
      [ "$tacit" = "0" ] || die "$arm: shows $tacit Tacit decisions (must be 0)"
    fi
    if [ "$fired" = "0" ] && [ "$degraded" -gt "0" ]; then
      die "$arm: seam never engaged ($degraded degraded, 0 fired) — outage, not abstention"
    fi
    case "$arm" in
      abl-style) [ "$narrow_n" = "0" ] || die "$arm: narrowing fired (must be 0)"
                 [ "$dmw_n" = "0" ] || die "$arm: dm-weights fired (must be 0)" ;;
      abl-weights) [ "$style_n" = "0" ] || die "$arm: style fired (must be 0)"
                 [ "$narrow_n" = "0" ] || die "$arm: narrowing fired (must be 0)" ;;
      abl-category) [ "$style_n" = "0" ] || die "$arm: style fired (must be 0)"
                 [ "$dmw_n" = "0" ] || die "$arm: dm-weights fired (must be 0)" ;;
    esac
    if [ "$fired" = "0" ]; then
      log "$arm: seam abstained on the whole corpus (0 fired, $degraded degraded) — reported as data"
    fi
  fi
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
    abl-base) seam_ok=1 ;;
    abl-style) grep -qF 'style=ok' "$ledger" 2>/dev/null && seam_ok=1 || seam_ok=0 ;;
    abl-weights) grep -qF 'weights=ok' "$ledger" 2>/dev/null && seam_ok=1 || seam_ok=0 ;;
    abl-category) grep -qF 'category=ok' "$ledger" 2>/dev/null && seam_ok=1 || seam_ok=0 ;;
  esac
  check_tally "$arm" "$style_n" "$narrow_n" "$dmw_n" "$tacit" "$want" "$seam_ok" "$degraded"
  if [ "$degraded" -gt "0" ]; then
    log "degraded lines in $arm ledger ($degraded; precise bigram grep; duplicates from overlapping windows):"
    grep -hiE "$DEGRADED_HIT" "$ledger" 2>/dev/null | cut -c1-250 | sort | uniq -c | sort -rn | head -10 || true
  fi
}

# --- arm 1/4: TEI-only baseline -----------------------------------------------
if [ "${SKIP_BASE:-0}" = "1" ]; then
  [ -f "$RUNS/$BASE_ID/summary.json" ] || die "resume target $BASE_ID has no summary.json"
  log "arm 1/4 abl-base skipped — reusing $RUNS/$BASE_ID (re-tallying)"
  ( STAMP="$BASE_STAMP" tally_arm "abl-base" 0 )
else
# shellcheck disable=SC2086
docker compose $BASE_FILES down --remove-orphans >/dev/null 2>&1 || true
start_tacit_watch "$RUNS/$STAMP-abl-base-watch.log" "$RUNS/$STAMP-abl-base-ledger.log"
run_arm "arm 1/4 abl-base (TEI-only)" "$BASE_FILES" "$BASE_ID" true TASKS_DEADLINE_SECONDS=3600
stop_tacit_watch
tally_arm "abl-base" 0
fi

# --- arm 2/4: style only (base files; Tacit never started) --------------------
if [ "${SKIP_STYLE:-0}" = "1" ]; then
  [ -f "$RUNS/$STYLE_ID/summary.json" ] || die "resume target $STYLE_ID has no summary.json"
  log "arm 2/4 abl-style skipped — reusing $RUNS/$STYLE_ID (re-tallying)"
  ( STAMP="$STYLE_STAMP" tally_arm "abl-style" 0 )
else
# shellcheck disable=SC2086
docker compose $BASE_FILES down --remove-orphans >/dev/null 2>&1 || true
start_tacit_watch "$RUNS/$STAMP-abl-style-watch.log" "$RUNS/$STAMP-abl-style-ledger.log"
run_arm "arm 2/4 abl-style" "$BASE_FILES" "$STYLE_ID" style_pre \
  DECISIONS_STYLE_ENABLED=true DECISIONS_STYLE_CONFIDENCE_FLOOR=0.6 \
  DECISIONS_STYLE_PROVIDER=linear \
  DECISIONS_STYLE_BASE_URL=http://design-pattern-style-head:8800/v1 \
  DECISIONS_STYLE_MODEL=style-linear \
  TASKS_DEADLINE_SECONDS=3600
stop_tacit_watch
tally_arm "abl-style" 0
fi

# --- arm 3/4: weights only (overlay; style/category/gate explicitly off) ------
if [ "${SKIP_WEIGHTS:-0}" = "1" ]; then
  [ -f "$RUNS/$WEIGHTS_ID/summary.json" ] || die "resume target $WEIGHTS_ID has no summary.json"
  log "arm 3/4 abl-weights skipped — reusing $RUNS/$WEIGHTS_ID (re-tallying)"
  ( STAMP="$WEIGHTS_STAMP" tally_arm "abl-weights" 1 )
else
# shellcheck disable=SC2086
docker compose $CAND_FILES down --remove-orphans >/dev/null 2>&1 || true
start_tacit_watch "$RUNS/$STAMP-abl-weights-watch.log" "$RUNS/$STAMP-abl-weights-ledger.log"
run_arm "arm 3/4 abl-weights" "$CAND_FILES" "$WEIGHTS_ID" true \
  BENCH_DECISION_OVERLAY="$OVERLAY" \
  DECISIONS_STYLE_ENABLED=false \
  DECISIONS_WEIGHTS_ENABLED=true DECISIONS_WEIGHTS_TIMEOUT_SECONDS=900 \
  DECISIONS_CATEGORY_ENABLED=false DECISIONS_REASONING_GATE_ENABLED=false \
  TASKS_DEADLINE_SECONDS=3600
stop_tacit_watch
tally_arm "abl-weights" 1
fi

# --- arm 4/4: category only (overlay; style/weights/gate explicitly off) ------
# shellcheck disable=SC2086
docker compose $CAND_FILES down --remove-orphans >/dev/null 2>&1 || true
start_tacit_watch "$RUNS/$STAMP-abl-category-watch.log" "$RUNS/$STAMP-abl-category-ledger.log"
run_arm "arm 4/4 abl-category" "$CAND_FILES" "$CATEGORY_ID" true \
  BENCH_DECISION_OVERLAY="$OVERLAY" \
  DECISIONS_STYLE_ENABLED=false DECISIONS_WEIGHTS_ENABLED=false \
  DECISIONS_CATEGORY_ENABLED=true DECISIONS_CATEGORY_TIMEOUT_SECONDS=120 \
  DECISIONS_REASONING_GATE_ENABLED=false \
  TASKS_DEADLINE_SECONDS=3600
stop_tacit_watch
tally_arm "abl-category" 1

# --- three pairwise compares (verdicts are DATA: harm is expected somewhere) --
for seam in style weights category; do
  case "$seam" in
    style) BID="$STYLE_ID" ;;
    weights) BID="$WEIGHTS_ID" ;;
    *) BID="$STAMP-abl-$seam" ;;
  esac
  log "compare abl-base vs abl-$seam: A=$RUNS/$BASE_ID B=$RUNS/$BID (pre-registered rule + repeat-stability gate)"
  set +e
  make benchmark-compare A="$RUNS/$BASE_ID" B="$RUNS/$BID" REQUIRE_REPEAT_STABILITY=1
  rc=$?
  set -e
  log "compare base-vs-$seam exit=$rc (nonzero = candidate-loss; verdicts are data here)"
  python3 -c "
import json
v = json.load(open('$RUNS/$BID/comparison/comparison.json'))['verdict']
print('>> verdict base-vs-$seam:', v['outcome'], '|', '; '.join(v['reasons']))"
done
log "done: base=$RUNS/$BASE_ID + 3 seam arms — restart the dev stack (e.g. make docker-up-decisions-cpu)"
exit 0
