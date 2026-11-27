# =============================================================================
# Makefile for design-pattern-mcp
# =============================================================================
# Build automation for the MCP server: quality gates, the layered verification
# program (verify/), the demo clients and the Docker delivery stack.
#
# Docker files: Dockerfile, docker/docker-compose.yml
# =============================================================================

.PHONY: help install install-mcps \
	check-lint check-static-typing check-deadcode check-depcheck check-all check \
	test-unit test-oracles test-mutations regen-mutmut-baseline test-all \
	verify-fizz verify-fizz-simulation verify-fizz-garden regen-fizz-traces \
	verify-ledger verify-cross-consistency verify-patterns verify-all \
	health client-analyze client-design client-async \
	benchmark-selfcheck benchmark-offline benchmark-live benchmark-analyze benchmark-e2e benchmark-compare \
	benchmark-sidecars-up benchmark-draft benchmark-style-ab \
	docker-build docker-build-all docker-build-base docker-build-cached docker-build-nocache \
docker-build-bootstrap docker-build-tei docker-build-tei-embed-gpu docker-build-tei-embed-cpu \
	docker-build-tei-rerank-gpu docker-build-tei-rerank-cpu docker-build-tei-cpu dense-cache \
	docker-build-style-head \
	decisions-smoke \
	style-head-calibrate \
	docker-publish docker-publish-tei docker-publish-style-head docker-publish-all \
docker-up-gpu docker-up-cpu docker-down docker-logs docker-logs-follow docker-rm clean \
	embed-cache embed-cache-cpu

COMPOSE := docker compose -f docker/docker-compose.yml
UV ?= uv

DOCKER_IMAGE    := design-pattern-mcp
DOCKER_TAG      := $(shell grep -m 1 '^version' pyproject.toml | sed -E 's/.*"([^"]+)".*/\1/')
DOCKER_HUB_REPO := olkowa/design-pattern-mcp
GHCR_REPO       := ghcr.io/olk/design-pattern-mcp

# Local TEI image names: the lane is part of the NAME, never the tag. The -gpu
# pair (CUDA) and the -cpu pair (candle) both carry :$(DOCKER_TAG) and :latest;
# the compose files and scripts/benchmark-docker.sh default to :latest. Only
# the GPU pair is published, under the same $(DOCKER_TAG) as the server.
TEI_EMBED_IMAGE      := design-pattern-tei-embed-gpu
TEI_RERANK_IMAGE     := design-pattern-tei-rerank-gpu
TEI_EMBED_IMAGE_CPU  := design-pattern-tei-embed-cpu
TEI_RERANK_IMAGE_CPU := design-pattern-tei-rerank-cpu
TEI_EMBED_HUB_REPO   := olkowa/pattern-tei-embed-gpu
TEI_EMBED_GHCR_REPO  := ghcr.io/olk/pattern-tei-embed-gpu
TEI_RERANK_HUB_REPO  := olkowa/pattern-tei-rerank-gpu
TEI_RERANK_GHCR_REPO := ghcr.io/olk/pattern-tei-rerank-gpu

EMBED_MODEL_REPO   ?= Qwen/Qwen3-Embedding-0.6B
EMBED_POOLING      ?= last-token
EMBED_DTYPE        ?= float32
EMBED_TEI_BASE     ?= ghcr.io/huggingface/text-embeddings-inference:cuda-1.9
EMBED_TEI_BASE_CPU ?= ghcr.io/huggingface/text-embeddings-inference:cpu-1.9
CUDA_COMPUTE_CAP   ?= 61   # minimum arch served by the -gpu pair (sm >= 61 via PTX JIT)
DENSE_CACHE_DIR    ?= data/dense-cache
# Extra compose file(s) for the cache-builder's sidecar start (e.g.
# `-f docker/docker-compose.cpu.yml` via `embed-cache-cpu` on GPU-less hosts).
DENSE_CACHE_OVERLAY ?=

# Linear style head (the Ch2 seam's only provider): a TF-IDF + calibrated-SVM head
# fitted in its own image build, whose training RUN is also the accuracy gate.
STYLE_HEAD_IMAGE := design-pattern-style-head
# Registry pair for the style head, published by docker-publish-style-head.
STYLE_HEAD_HUB_REPO  := olkowa/design-pattern-style-head
STYLE_HEAD_GHCR_REPO := ghcr.io/olk/design-pattern-style-head
# The Ch2 style seam's one switch: it always names the linear head, so the endpoint,
# model name and wire shape move together (the config validator refuses any other
# provider, and this keeps a mismatch unexpressible).
STYLE_PROVIDER ?= linear
STYLE_PROVIDER_ENV = DECISIONS_STYLE_PROVIDER=linear \
	DECISIONS_STYLE_BASE_URL=http://design-pattern-style-head:8800/v1 \
	DECISIONS_STYLE_MODEL=style-linear

# HF Hub token for authenticated model downloads (higher rate limits, gated repos).
# File holds the raw token only, no key=value. Empty/missing file = anonymous
# download (fine for the pinned public repos, just the huggingface_hub warning).
# Every HF-fetching Dockerfile (tei-embed, tei-rerank) reads it via
# `RUN --mount=type=secret,id=hf_token`; the secret is mounted per-RUN, never baked
# into a layer.
comma := ,
# BuildKit progress mode for the style-head build. `plain` streams RUN
# stdout (the trainer's gate line) to the terminal and `tee`d logs;
# the default `tty` collapses it. Override per-call: `make docker-build-style-head
# DOCKER_BUILD_PROGRESS=tty`.
DOCKER_BUILD_PROGRESS ?= plain
HF_TOKEN_FILE ?= $(HOME)/.cache/hf_token
# Resolve to --secret when the file exists and is non-empty. A missing default
# is fine (anonymous download); but an explicitly-set missing path is a typo,
# so fail instead of silently going anonymous. Recursive (=, not :=) so the
# error fires only when a docker-build recipe expands HF_SECRET_ARG.
HF_SECRET_ARG = $(if $(shell test -s "$(HF_TOKEN_FILE)" && echo yes),--secret id=hf_token$(comma)src=$(HF_TOKEN_FILE),$(if $(filter-out file,$(origin HF_TOKEN_FILE)),$(error HF_TOKEN_FILE="$(HF_TOKEN_FILE)" is missing or empty)))

.DEFAULT_GOAL := help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

install: ## Sync dependencies into .venv (dev group included by default)
	$(UV) sync

install-mcps: ## Install reasoning MCP servers globally (local dev; Docker embeds them at build time)
	npm install -g server-shannon-thinking@0.1.1 @mettamatt/code-reasoning@0.8.1

##@ Quality
check-lint: ## Run ruff check
	$(UV) run ruff check .

# mypy strict check on src/. --strict is redundant with [tool.mypy] strict=true —
# kept explicit so the gate survives future config edits.
check-static-typing: install ## Run mypy --strict static type check on src/
	$(UV) run mypy --strict

# Decorators that register callables with a framework (FastMCP tools/prompts/
# resources, pydantic validators): the decorated symbol is live without a
# static call site. Keep in sync with whitelist.py.
VULTURE_IGNORE_DECORATORS := "@field_validator,@model_validator,@*.resource,@*.prompt,@*.tool"

check-deadcode: ## Run vulture dead-code scan on src/ and examples/
	$(UV) run vulture src whitelist.py --min-confidence 80 \
		--ignore-decorators $(VULTURE_IGNORE_DECORATORS)

check-depcheck: ## Run deptry dependency-hygiene scan
	$(UV) run deptry .

# Side-effect-free gate aggregator — auto-fix/format targets must never join:
# a gate must not mutate the working tree.
check-all: ## Run every check-* gate (CI entry point)
	@echo "==> [1/4] check-lint";          $(MAKE) --no-print-directory check-lint
	@echo "==> [2/4] check-static-typing"; $(MAKE) --no-print-directory check-static-typing
	@echo "==> [3/4] check-deadcode";      $(MAKE) --no-print-directory check-deadcode
	@echo "==> [4/4] check-depcheck";      $(MAKE) --no-print-directory check-depcheck

##@ Tests
test-unit: install ## Run unit tests with uv (tests/unit/)
	$(UV) run pytest tests/unit/ -v

test-oracles: install ## Run executable oracles (tests/verification/): L1 canary, L2 PBT oracles, spec-garden replay
	$(UV) run pytest tests/verification/ -v

# Hypothesis isolation: the gate re-runs the same tests under many executors;
# a per-run database keeps the shared repo database free of foreign executors.
#
# mutmut is version-pinned (mutmut==3.7.*) in lockstep with the source-string
# guards in verify/mutmut/mutmut_compat.py (TCB discipline): the shim fails
# loudly when the guarded upstream code changes, so a version bump must update
# the shim and this pin in the same change.
#
# Cleanup hygiene: the whole recipe is one shell with an EXIT trap, so the
# per-run Hypothesis database is removed even when a gate step fails.
#
# Pytest tmp isolation: every pytest process under this gate pins its own
# per-PID --basetemp under MUTMUT_PYTEST_TMP (consumed by
# verify/mutmut/mutmut_compat.py's pytest_configure; same EXIT-trap lifecycle
# as HYPO_DIR). Without it the parallel children race on the pytest-current
# symlink at teardown and cleanup failures are recorded as "killed".
#
# Enforcement: `mutmut run` exits 0 even with survivors, so the gate only means
# something because of the final step — scripts/mutmut_baseline.py distills
# mutants/ (gitignored scratch) into verify/mutmut-baseline.json and FAILS on a
# survivor/kill-ratio regression against the committed baseline (ratchet:
# refresh deliberately via `make regen-mutmut-baseline`).
MUTMUT_PIN := 'mutmut==3.7.*'
test-mutations: ## L3: mutmut over the decision modules + gardens (manual/nightly; ephemeral install via uv)
	@bash -o pipefail -c '\
	HYPO_DIR=$$(mktemp -d); \
	export HYPOTHESIS_STORAGE_DIRECTORY="$$HYPO_DIR"; \
	export MUTMUT_PYTEST_TMP="$$HYPO_DIR/pytest-tmp"; \
	mkdir -p "$$MUTMUT_PYTEST_TMP"; \
	trap '"'"'rm -rf "$$HYPO_DIR"'"'"' EXIT; \
	$(UV) run --with $(MUTMUT_PIN) python -c "import verify.mutmut.mutmut_compat as compat; compat.apply(); from mutmut.__main__ import cli; raise SystemExit(cli())" run || exit 1; \
	$(UV) run --with $(MUTMUT_PIN) mutmut results || exit 1; \
	$(UV) run python scripts/mutmut_baseline.py --gate'

# Deliberate baseline refresh: records the CURRENT mutation run as the new
# ratchet floor. Only run after inspecting the survivor list — a new survivor
# is either a test-suite gap to fix or a documented equivalent
# (verify/mutmut-equivalents.md) to acknowledge.
regen-mutmut-baseline: ## L3: rewrite verify/mutmut-baseline.json from the last test-mutations run
	$(UV) run python scripts/mutmut_baseline.py --write verify/mutmut-baseline.json

# Fast-path aggregator: unit + oracles only. test-mutations is deliberately
# outside — mutmut is a manual/nightly gate (ephemeral toolchain, long runtime).
test-all: ## Run the fast test gates (PR-time; test-mutations is nightly-only)
	@echo "==> [1/2] test-unit";    $(MAKE) --no-print-directory test-unit
	@echo "==> [2/2] test-oracles"; $(MAKE) --no-print-directory test-oracles

# fizz exits 0 even on invariant failure (only panics exit non-zero), so
# `|| exit 1` alone is a vacuous gate — every run must also print the PASSED
# line. scripts/fizz-check.sh checks both conditions per spec.
#
# Isolation: each spec runs through scripts/fizz-check.sh, which copies the
# spec directory into a private temp dir (compiled .json ASTs and out/ graphs
# are written there and discarded).
FIZZ ?= fizz
FIZZ_CHECK := scripts/fizz-check.sh
SPEC_DIR := verify/fizz

verify-fizz: ## L4: exhaustive FizzBee model checks over verify/fizz/
	@if ! command -v $(FIZZ) >/dev/null 2>&1; then \
		echo "fizz binary not found: pip install fizzbee, or set FIZZ=scripts/fizz-docker.sh"; exit 1; \
	fi
	@set -e; for spec in $(SPEC_DIR)/*.fizz; do \
		echo ">> $$(date +%H:%M:%S) exhaustive check: $$spec"; \
		FIZZ_BIN=$(FIZZ) $(FIZZ_CHECK) "$$spec" exhaustive; \
	done; \
	echo "verify-fizz: all specs green (exhaustive, bounds per fizz.yaml)"

verify-fizz-simulation: ## L4: seeded parallel FizzBee simulation (nightly relief valve)
	@if ! command -v $(FIZZ) >/dev/null 2>&1; then \
		echo "fizz binary not found: pip install fizzbee, or set FIZZ=scripts/fizz-docker.sh"; exit 1; \
	fi
	@set -e; seed=$$(date +%s); workers=$$(nproc); \
	echo ">> seeded simulation seed=$$seed workers=$$workers"; \
	for spec in $(SPEC_DIR)/*.fizz; do \
		echo ">> $$(date +%H:%M:%S) simulation: $$spec"; \
		FIZZ_BIN=$(FIZZ) $(FIZZ_CHECK) "$$spec" simulation -x --seed $$seed --parallel $$workers; \
	done

# Mechanical revalidation of the .fizz spec garden (verify/fizz/garden.toml
# <-> the README garden table): every FG-* mutation is applied to a hermetic
# temp copy and must produce its documented outcome — an expect-fail mutant
# that survives is a vacuous assertion, a budget-stopped run is inconclusive
# (both fail the gate). Mutations never touch the working tree. The
# toolchain-free subset rides every `make test-oracles` pass via
# tests/verification/test_fizz_garden.py.
verify-fizz-garden: ## L3/L4: re-run the FG-* spec garden against the fizz toolchain
	@if ! command -v $(FIZZ) >/dev/null 2>&1; then \
		echo "fizz binary not found: pip install fizzbee, or set FIZZ=scripts/fizz-docker.sh"; exit 1; \
	fi
	@$(UV) run python scripts/fizz_garden.py --specs $(SPEC_DIR) --fizz-bin $(FIZZ)

# Specs covered by the L4 conformance corpus (tests/verification/fizz_corpus/).
# A spec change must regenerate its corpus in the same PR. 2026-10-07: the corpus
# JSONs are NOT checked in and the conformance replay tests do not exist, so this
# target is aspirational until the corpus is regenerated (see verify/fizz/README.md).
CONFORMANCE_SPECS := jobs_protocol retrieval_fusion

regen-fizz-traces: ## L4: regenerate the conformance corpora (requires the fizz toolchain)
	@if ! command -v $(FIZZ) >/dev/null 2>&1; then \
		echo "fizz binary not found: pip install fizzbee, or set FIZZ=scripts/fizz-docker.sh"; exit 1; \
	fi
	@set -e; for spec in $(CONFORMANCE_SPECS); do \
		echo ">> extracting state graph: $$spec"; \
		$(UV) run python scripts/fizz_traces.py $(SPEC_DIR)/$$spec.fizz \
			--out tests/verification/fizz_corpus/$$spec.json --fizz-bin $(FIZZ); \
	done; \
	echo "regen-fizz-traces: corpora up to date"

verify-ledger: ## L8: property-ID ledger consistency (verify/fizz/README.md <-> artifacts)
	@$(UV) run python scripts/verify_ledger.py

# Advisory NL-Doc cross-consistency gate (mechanical subset; the LLM comparison
# activates via DESIGN_CONSISTENCY_MODEL, checker identity pinned per
# run). Findings are leads to triage, not failures.
verify-cross-consistency: ## L8: NL-Doc/docstring consistency over the decision modules (advisory)
	@$(UV) run python scripts/cross_consistency.py

# ajv-cli is an npm toolchain (not uv-managed): needs `npm install -g ajv-cli`.
verify-patterns: ## Verify pattern/*.json against docs/pattern-schema.json + language schema via ajv-cli
	bash scripts/verify-patterns.sh

# Nightly aggregator over the formal/audit verify-* targets.
verify-all: ## Run every verify-* target (nightly entry point)
	@echo "==> [1/6] verify-fizz";              $(MAKE) --no-print-directory verify-fizz
	@echo "==> [2/6] verify-fizz-simulation";   $(MAKE) --no-print-directory verify-fizz-simulation
	@echo "==> [3/6] verify-fizz-garden";       $(MAKE) --no-print-directory verify-fizz-garden
	@echo "==> [4/6] verify-ledger";            $(MAKE) --no-print-directory verify-ledger
	@echo "==> [5/6] verify-cross-consistency"; $(MAKE) --no-print-directory verify-cross-consistency
	@echo "==> [6/6] verify-patterns";          $(MAKE) --no-print-directory verify-patterns

##@ Demo
health: ## Run the offline readiness check (prints OK)
	@$(UV) run python -m src --health

client-analyze: ## Run the analyze_design MCP client demo (synchronous)
	@CLIENT_URL=http://localhost:8822/mcp $(UV) run python examples/design_pattern_client.py --tool analyze

client-design: ## Run the select_design_patterns MCP client demo (synchronous, long-running)
	@CLIENT_URL=http://localhost:8822/mcp $(UV) run python examples/design_pattern_client.py --tool design

client-async: ## Run the async job trio demo (submit_design_job + get_design_status)
	@CLIENT_URL=http://localhost:8822/mcp $(UV) run python examples/design_pattern_client_async.py

##@ Docker
# The dense cache is a BUILD INPUT of the server image (Dockerfile COPYs
# $(DENSE_CACHE_DIR)/), so `docker-build` generates it when absent — a built
# image must not need a 19-minute self-embed at first start. Generating it needs
# two images that do not exist yet on a fresh clone: the TEI embed sidecar (the
# embedder) and the server image itself (the builder runs inside it, because the
# sidecars publish no host ports and are reachable only on the compose network).
# Hence the two-pass shape: bootstrap the server image, generate the cache, then
# rebuild with the cache baked in. With a cache already on disk the parse-time
# wildcard check keeps `docker-build` a single pass.
DENSE_CACHE_MANIFEST := $(DENSE_CACHE_DIR)/manifest.json
DENSE_CACHE_VECTORS := $(DENSE_CACHE_DIR)/vectors.npy
# Both artifacts or neither: a lone manifest would bake a cache whose vectors are
# missing (the runtime then reports vectors-absent and re-embeds).
DENSE_CACHE_PREREQ := $(if $(filter 1,$(SKIP_DENSE_CACHE)),,$(if $(and $(wildcard $(DENSE_CACHE_MANIFEST)),$(wildcard $(DENSE_CACHE_VECTORS))),,dense-cache))

docker-build: $(DENSE_CACHE_PREREQ) ## Build the MCP server image with data/dense-cache/ baked in
	@if [ ! -f "$(DENSE_CACHE_MANIFEST)" ] || [ ! -f "$(DENSE_CACHE_VECTORS)" ]; then \
		echo "WARNING: $(DENSE_CACHE_DIR)/ is incomplete - the image will embed the catalog at first start (~19 min on CPU)."; \
		echo "         Re-run without SKIP_DENSE_CACHE=1 to generate and bake it."; \
	fi
	docker build --target production -f Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker build --target production -f Dockerfile -t $(DOCKER_IMAGE):latest .

docker-build-bootstrap: ## [internal] Build the server image that runs the cache builder (no cache baked)
	docker build --target production -f Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker build --target production -f Dockerfile -t $(DOCKER_IMAGE):latest .

docker-build-tei-embed-gpu: ## Build the GPU TEI embed sidecar image (design-pattern-tei-embed-gpu)
	docker build -f docker/Dockerfile.tei-embed-gpu $(HF_SECRET_ARG) \
		--build-arg EMBED_MODEL_REPO=$(EMBED_MODEL_REPO) \
		--build-arg EMBED_POOLING=$(EMBED_POOLING) \
		--build-arg EMBED_DTYPE=$(EMBED_DTYPE) \
		--build-arg TEI_BASE_IMAGE=$(EMBED_TEI_BASE) \
		--build-arg CUDA_COMPUTE_CAP=$(CUDA_COMPUTE_CAP) \
		-t $(TEI_EMBED_IMAGE):$(DOCKER_TAG) -t $(TEI_EMBED_IMAGE):latest .

docker-build-tei-embed-cpu: ## Build the CPU TEI embed sidecar image (design-pattern-tei-embed-cpu; same model repo, shared cache)
	docker build -f docker/Dockerfile.tei-embed-cpu $(HF_SECRET_ARG) \
		--build-arg EMBED_MODEL_REPO=$(EMBED_MODEL_REPO) \
		--build-arg EMBED_POOLING=$(EMBED_POOLING) \
		--build-arg EMBED_DTYPE=$(EMBED_DTYPE) \
		--build-arg TEI_BASE_IMAGE=$(EMBED_TEI_BASE_CPU) \
		-t $(TEI_EMBED_IMAGE_CPU):$(DOCKER_TAG) -t $(TEI_EMBED_IMAGE_CPU):latest .

docker-build-tei-rerank-gpu: ## Build the GPU TEI rerank sidecar image (design-pattern-tei-rerank-gpu)
	docker build -f docker/Dockerfile.tei-rerank-gpu $(HF_SECRET_ARG) \
		--build-arg TEI_BASE_IMAGE=$(EMBED_TEI_BASE) \
		--build-arg CUDA_COMPUTE_CAP=$(CUDA_COMPUTE_CAP) \
		-t $(TEI_RERANK_IMAGE):$(DOCKER_TAG) -t $(TEI_RERANK_IMAGE):latest .

docker-build-tei-rerank-cpu: ## Build the CPU TEI rerank sidecar image (design-pattern-tei-rerank-cpu)
	docker build -f docker/Dockerfile.tei-rerank-cpu $(HF_SECRET_ARG) \
		--build-arg TEI_BASE_IMAGE=$(EMBED_TEI_BASE_CPU) \
		-t $(TEI_RERANK_IMAGE_CPU):$(DOCKER_TAG) -t $(TEI_RERANK_IMAGE_CPU):latest .

docker-build-tei: docker-build-tei-embed-gpu docker-build-tei-rerank-gpu ## Build both TEI sidecar images (CUDA pair, the default)

docker-build-tei-cpu: docker-build-tei-embed-cpu docker-build-tei-rerank-cpu ## Build both CPU TEI sidecar images (design-pattern-tei-embed-cpu, design-pattern-tei-rerank-cpu)

# The runnable base stack in one command (server + TEI pair + style head): everything
# `docker-up-gpu` starts. The default TEI pair needs an NVIDIA GPU at run time (but
# not at build time — nvcc cross-compiles); the -cpu pair (docker-build-tei-cpu,
# docker-up-cpu) runs anywhere. `docker-build`
# stays server-only on purpose — it feeds `docker-publish`, which must not drag in
# sidecars.
docker-build-base: docker-build docker-build-tei docker-build-style-head ## Build the base stack images (server + GPU TEI pair + style head)

docker-build-all: docker-build docker-build-tei docker-build-tei-cpu docker-build-style-head ## Build every image: server (+ dense cache), TEI pair, style head

docker-build-cached: embed-cache docker-build-all ## Force-regenerate the dense cache, then build every image

##@ Decision models (linear style head)

# The linear Ch2 style head: CPU only, ~1 min, and the training RUN inside the build is also
# the accuracy gate (a head below 0.65 holdout accuracy exits 2 and fails the build). There
# is no cached/host-artifact path: the fit is cheap enough that a stale-head cache would cost
# more attention than it saves.
docker-build-style-head: ## Build the linear style head sidecar (CPU; the build gates the head)
	docker build --progress=$(DOCKER_BUILD_PROGRESS) -f docker/Dockerfile.style-head \
		-t $(STYLE_HEAD_IMAGE):$(DOCKER_TAG) -t $(STYLE_HEAD_IMAGE):latest .

decisions-smoke: ## Smoke-test the style head's wire contract (needs docker-up-gpu)
	@# The linear head is criteria-blind, so this payload deliberately carries NO criteria: any
	@# criteria map it is sent must equal its 36 trained labels exactly, and a partial one is
	@# refused with 422 (that refusal is the drift guard, exercised by test_style_head.py).
	@echo "==> linear head /v1/systemone"
	@curl -fsS http://127.0.0.1:18800/v1/systemone -H 'Content-Type: application/json' \
		-d '{"model":"style-linear","state":"A checkout service with flash sales.","questions":{"architecture_style":{"type":"choice","instructions":"Which architecture style best characterizes the system described in the state?"}}}' \
		| python3 -c "import json,sys; a=json.load(sys.stdin)['answers']['architecture_style']; print(f\"choice={a['choice']} confidence={a['confidence']:.4f} classes={len(a['probabilities'])}\")"

# Floor calibration for the Ch2 style seam (maintainer tool for corpus/head changes). The
# seam ships at the A/B-winning 0.6, not at the report's `recommended.floor` of 0.1 — that
# point lost run 1 on repeat stability, so re-shipping still means re-winning the A/B.
STYLE_HEAD_BASE_URL            ?= http://127.0.0.1:18800/v1
STYLE_HEAD_CAL_FLOORS          ?= 0.1 0.2 0.3 0.4 0.5 0.6 0.7
STYLE_HEAD_CAL_MIN_ACTED_ACCURACY ?= 0.8
STYLE_HEAD_CAL_MIN_ACTED_RATE  ?= 0.3
STYLE_HEAD_CAL_MIN_ACTED       ?= 10
# Replays per held-out row: >1 measures repeat consistency and refuses a floor that is not
# stable (serving is deterministic per question, so identical requests must agree).
STYLE_HEAD_CAL_REPEATS         ?= 5
# Seeded criteria-map permutations per row: >1 measures order invariance (the served map is
# one ordering of an unordered answer set, so the argmax must not depend on it).
STYLE_HEAD_CAL_PERMUTATIONS    ?= 5
# The container whose identity the calibration report records.
STYLE_HEAD_SERVED_CONTAINER    ?= design-pattern-style-head-dev
STYLE_HEAD_CAL_REPORT          := models/style-head-calibration.json
# Provenance of the head: it is fitted inside the image from this corpus, so there is no
# upstream repo to name — the image tag is captured by --served-image-tag beside this.
STYLE_HEAD_SERVED_REPO         ?= built-in-image (qa/typed/style.jsonl)
# Floor the style A/B runs with: the shipped operating point (0.6, the run-2 winner from
# docs/linear-style-head.md §6) — guarded by the artifact's floors table, so an arm cannot
# run a threshold the measurements did not produce. The artifact's own `recommended.floor`
# is 0.1 (the lowest bar-clearing floor); that point LOST run 1 on repeat stability, so the
# A/B must not default to it. Override only to re-test a value:
# `make benchmark-style-ab LINEAR_STYLE_FLOOR=0.5`.
LINEAR_STYLE_FLOOR         ?= $(shell python3 -c "import json;d=json.load(open('$(STYLE_HEAD_CAL_REPORT)'));\
floors={r.get('floor') for r in d.get('floors',[])};print('0.6' if 0.6 in floors else '')" 2>/dev/null)

style-head-calibrate: ## Calibrate decisions.style.confidence_floor against a running linear style head
	uv run python scripts/decision_models/style_head_calibrate.py \
		--style qa/typed/style.jsonl --base-url $(STYLE_HEAD_BASE_URL) --model style-linear \
		--floors $(STYLE_HEAD_CAL_FLOORS) --metric confidence \
		--min-acted-accuracy $(STYLE_HEAD_CAL_MIN_ACTED_ACCURACY) \
		--min-acted-rate $(STYLE_HEAD_CAL_MIN_ACTED_RATE) \
		--min-acted $(STYLE_HEAD_CAL_MIN_ACTED) \
		--repeats $(STYLE_HEAD_CAL_REPEATS) \
		--permutations $(STYLE_HEAD_CAL_PERMUTATIONS) \
		--report $(STYLE_HEAD_CAL_REPORT) \
		--served-container "$$(docker inspect -f '{{.Name}}' $(STYLE_HEAD_SERVED_CONTAINER) 2>/dev/null | sed 's|^/||')" \
		--served-image "$$(docker inspect -f '{{.Image}}' $(STYLE_HEAD_SERVED_CONTAINER) 2>/dev/null || true)" \
		--served-image-tag "$$(docker inspect -f '{{.Config.Image}}' $(STYLE_HEAD_SERVED_CONTAINER) 2>/dev/null || true)" \
		--served-container-started-at "$$(docker inspect -f '{{.State.StartedAt}}' $(STYLE_HEAD_SERVED_CONTAINER) 2>/dev/null || true)" \
		--served-model-repo "$(STYLE_HEAD_SERVED_REPO)"

# Tag and push a local image to both Docker Hub and GHCR.
# Args: local_image hub_repo ghcr_repo
# Usage: $(call publish-image,$(DOCKER_IMAGE),$(DOCKER_HUB_REPO),$(GHCR_REPO))
publish-image = \
	docker tag $(1):$(DOCKER_TAG) $(2):$(DOCKER_TAG) && \
	docker tag $(1):latest $(2):latest && \
	docker tag $(1):$(DOCKER_TAG) $(3):$(DOCKER_TAG) && \
	docker tag $(1):latest $(3):latest && \
	docker push $(2):$(DOCKER_TAG) && \
	docker push $(2):latest && \
	docker push $(3):$(DOCKER_TAG) && \
	docker push $(3):latest

docker-publish: docker-build ## Push the MCP server image to Docker Hub + GHCR + git tag v$(DOCKER_TAG)
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "ERROR: uncommitted changes - commit before publishing:"; \
		git status --short; exit 1; \
	fi
	docker login -u olkowa
	docker login ghcr.io -u olk
	$(call publish-image,$(DOCKER_IMAGE),$(DOCKER_HUB_REPO),$(GHCR_REPO))
	@echo "MCP image published. Logging out of ghcr.io."
	docker logout ghcr.io
	@if git rev-parse -q --verify "refs/tags/v$(DOCKER_TAG)" >/dev/null; then \
		echo "git tag v$(DOCKER_TAG) already exists - skipping creation"; \
	else \
		echo "Creating annotated git tag v$(DOCKER_TAG)"; \
		git tag -a "v$(DOCKER_TAG)" -m "Release v$(DOCKER_TAG)"; \
	fi
	git push origin "v$(DOCKER_TAG)"
	mcp-publisher validate server.json
	mcp-publisher login github
	mcp-publisher publish

docker-publish-tei: docker-build-tei ## Push both TEI images to Docker Hub + GHCR
	docker login -u olkowa
	docker login ghcr.io -u olk
	$(call publish-image,$(TEI_EMBED_IMAGE),$(TEI_EMBED_HUB_REPO),$(TEI_EMBED_GHCR_REPO))
	$(call publish-image,$(TEI_RERANK_IMAGE),$(TEI_RERANK_HUB_REPO),$(TEI_RERANK_GHCR_REPO))
	docker logout ghcr.io

# TEI shape: build first (the training RUN inside the build is the accuracy gate, so a
# built image is always a gated head), then tag + push both registries.
docker-publish-style-head: docker-build-style-head ## Push the style head image to Docker Hub + GHCR
	docker login -u olkowa
	docker login ghcr.io -u olk
	$(call publish-image,$(STYLE_HEAD_IMAGE),$(STYLE_HEAD_HUB_REPO),$(STYLE_HEAD_GHCR_REPO))
	docker logout ghcr.io

docker-publish-all: docker-publish docker-publish-tei docker-publish-style-head ## Build and push every image: server, TEI pair, style head

# File target, not phony: the cache is expensive (one TEI round-trip per
# embedder batch, ~5 min for the 486-record catalog through the GPU sidecar,
# ~15 min through the CPU sidecar) and is only rebuilt when the artifact is
# gone — `embed-cache` is the explicit force path.
# The image prerequisites are ORDER-ONLY on purpose: they are phony (always
# "remade"), so ordinary prerequisites would mark this target out of date on
# every run and re-embed the whole catalog 5 min at a time.
$(DENSE_CACHE_MANIFEST) $(DENSE_CACHE_VECTORS) &: | docker-build-tei-embed-gpu docker-build-bootstrap ## Generate data/dense-cache/ from the catalog (starts the TEI embed sidecar)
	@$(COMPOSE) $(DENSE_CACHE_OVERLAY) up -d design-pattern-tei-embed
	@printf ">> waiting for design-pattern-tei-embed-dev"; \
	for _ in $$(seq 1 60); do \
		if [ "$$(docker inspect -f '{{.State.Health.Status}}' design-pattern-tei-embed-dev 2>/dev/null)" = "healthy" ]; then echo " healthy"; break; fi; \
		printf "."; sleep 5; \
	done
	@mkdir -p $(DENSE_CACHE_DIR)
	$(COMPOSE) run --rm --no-deps -T \
		-v "$(CURDIR)/$(DENSE_CACHE_DIR):/out" \
		-e EMBED_MODEL_REPO=$(EMBED_MODEL_REPO) \
		--entrypoint python design-pattern-mcp \
		-m src.patterns.dense_cache build --out /out
	@ls -l $(DENSE_CACHE_DIR)

dense-cache: $(DENSE_CACHE_MANIFEST) ## Generate data/dense-cache/ if missing (no-op when present)

embed-cache: ## Force-regenerate data/dense-cache/ (starts the TEI embed sidecar)
	@rm -f $(DENSE_CACHE_MANIFEST) $(DENSE_CACHE_DIR)/vectors.npy
	@$(MAKE) --no-print-directory dense-cache

embed-cache-cpu: ## Force-regenerate data/dense-cache/ via the CPU sidecar (no GPU needed)
	@rm -f $(DENSE_CACHE_MANIFEST) $(DENSE_CACHE_DIR)/vectors.npy
	@$(MAKE) --no-print-directory dense-cache DENSE_CACHE_OVERLAY="-f docker/docker-compose.cpu.yml"

docker-up-gpu: ## Start the dev stack on the GPU pair (requires make docker-build-all once)
	EMBED_MODEL_REPO=$(EMBED_MODEL_REPO) $(COMPOSE) up -d --build

docker-up-cpu: ## Start the dev stack on the CPU pair (-cpu images, no GPU needed)
	EMBED_MODEL_REPO=$(EMBED_MODEL_REPO) $(COMPOSE) -f docker/docker-compose.cpu.yml up -d --build

docker-down: ## Stop the dev stack
	$(COMPOSE) down

docker-logs: ## Show the MCP server logs (one-time)
	$(COMPOSE) logs design-pattern-mcp

docker-logs-follow: ## Show and follow the MCP server logs
	$(COMPOSE) logs -f design-pattern-mcp

docker-rm: ## Remove the local server image
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG)

##@ Benchmark (Stage 0 Jev instrument)
# The harness lives in tests/benchmark/ (no pytest collection) and never edits src/: the pipeline
# is constructor-injected, so the runners wrap the seams they hand in. See tests/benchmark/README.md
# for the mode matrix, the metric definitions and the pre-registered A/B decision rule.
#
# Every target executes the harness inside Docker via scripts/benchmark-docker.sh
# (docker compose -f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml run --rm
# benchmark): the modules, corpora, metrics and verdicts are unchanged — the script owns image
# preflight, stack lifecycle (up --wait → run → down), the MCP readiness probe and exit-code
# propagation. live/analyze need a warm data/dense-cache (make dense-cache) plus the generator
# key from config/.env; e2e runs the full stack under DENSE_CACHE_MODE=off (cold start re-embeds
# the catalog, ~20 min on CPU) and tears the stack down afterwards (restart dev stack:
# make docker-up-gpu).
BENCH_RUNS      ?= data/benchmark-runs
BENCH_SCENARIOS ?= tests/benchmark/scenarios/seed.json
BENCH_COMPOSE   := docker compose -f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml
BENCH_DOCKER    := bash scripts/benchmark-docker.sh
BENCH_ENV := BENCH_SCENARIOS="$(BENCH_SCENARIOS)" OUT="$(OUT)" RUN_ID="$(RUN_ID)" LIMIT="$(LIMIT)" \
	SPLIT="$(SPLIT)" REPEAT="$(REPEAT)" NO_WARMUP="$(NO_WARMUP)" LOG_LEVEL="$(LOG_LEVEL)" \
	FLIP="$(FLIP)" CLIENT_URL="$(CLIENT_URL)" CALL_TIMEOUT="$(CALL_TIMEOUT)" A="$(A)" B="$(B)" \
	ALLOW_MISMATCH="$(ALLOW_MISMATCH)" DRY_RUN="$(DRY_RUN)" PER_FAMILY="$(PER_FAMILY)" \
	REQUIRE_REPEAT_STABILITY="$(REQUIRE_REPEAT_STABILITY)" \
	FAMILIES="$(FAMILIES)" DRAFT_OUT="$(DRAFT_OUT)" \
	DECISIONS_STYLE_ENABLED="$(if $(DECISIONS_STYLE_ENABLED),$(DECISIONS_STYLE_ENABLED),false)" \
	$(if $(DECISIONS_STYLE_CONFIDENCE_FLOOR),DECISIONS_STYLE_CONFIDENCE_FLOOR="$(DECISIONS_STYLE_CONFIDENCE_FLOOR)")
# `DECISIONS_STYLE_ENABLED` defaults to false (not unset) on purpose: the shipped
# config.json carries the seam ON, but no non-style target starts the head, so an absent
# var would make every scenario log a degradation warning and report style=wired against
# a dead endpoint. The floor forward below stays CONDITIONAL — an unset make variable
# exports the name with an EMPTY value, and the benchmark container's ServerConfig load
# rejects '' for a float. The arms that DO want a value pass it explicitly after BENCH_ENV,
# which still wins on the command line.

benchmark-selfcheck: ## Benchmark: metric self-check, hand-computed anchors (in Docker; no stack)
	@$(BENCH_ENV) $(BENCH_DOCKER) selfcheck

benchmark-offline: ## Benchmark: deterministic offline run (in Docker; FLIP=<scenario_id> = flip arm)
	@$(BENCH_ENV) $(BENCH_DOCKER) offline

benchmark-sidecars-up: ## Benchmark: publish the TEI sidecars on loopback (manual host-native escape hatch)
	$(BENCH_COMPOSE) up -d design-pattern-tei-embed design-pattern-tei-rerank

benchmark-live: ## Benchmark: in-process live run (in Docker; starts/tears down the TEI sidecars)
	@$(BENCH_ENV) $(BENCH_DOCKER) live

benchmark-analyze: ## Benchmark: in-process analyze-only arm, hit/recall (in Docker; sidecars managed)
	@$(BENCH_ENV) $(BENCH_DOCKER) analyze

benchmark-e2e: ## Benchmark: black-box run through docker compose (full stack, readiness-gated, torn down)
	@$(BENCH_ENV) $(BENCH_DOCKER) e2e

benchmark-draft: ## Benchmark: draft full.json candidates from the catalog (in Docker; user review gate)
	@$(BENCH_ENV) $(BENCH_DOCKER) draft

# Exits 1 on a candidate-loss verdict (pre-registered rule in tests/benchmark/README.md).
# Pre-registered A/B of the Ch2 style seam with the linear head in the loop. The style head
# lives in the base file, so the base stack alone is the CPU-only instrument; the seam is off
# in the baseline arm and on in the candidate arm, and everything else (corpus, split, repeat,
# image) is identical, which is what `benchmark-compare`'s +15% p95 rule requires.
#
# The linear style head serves the seam, and its calibration artifact supplies the floor.
# The floor is always read from the artifact of the provider actually under test, never
# hand-copied from another.
STYLE_AB_SCENARIOS ?= tests/benchmark/scenarios/full.json
STYLE_AB_REPEAT    ?= 3
STYLE_AB_STAMP     ?= $(shell date -u +%Y%m%d-%H%M%S)
STYLE_AB_FLOOR      = $(LINEAR_STYLE_FLOOR)
STYLE_AB_FLOOR_FILE = $(STYLE_HEAD_CAL_REPORT)
STYLE_CALIBRATE_TARGET = style-head-calibrate
# Bring the linear head up inside the A/B project. The harness tears its whole compose project
# down at the end of EVERY arm, so this runs before each arm — starting it once would leave
# arm 2 with a dead seam (and a dead seam is silent: the LLM style stands, the arm still
# completes, and the comparison would quietly measure nothing).
STYLE_HEAD_UP = STYLE_HEAD_IMAGE=$(STYLE_HEAD_IMAGE):$(DOCKER_TAG) \
	docker compose -f docker/docker-compose.yml up -d --wait design-pattern-style-head

benchmark-style-ab: ## Benchmark: pre-registered Ch2 style-seam A/B (needs the sidecar + GPU)
	@test -n "$(STYLE_AB_FLOOR)" || { \
		echo "ERROR: no usable floor in $(STYLE_AB_FLOOR_FILE) for provider=$(STYLE_PROVIDER)"; \
		echo "       run 'make $(STYLE_CALIBRATE_TARGET)' first (or pass a floor to re-test a threshold)"; exit 1; }
	@# BOTH arms carry the floor, not just the candidate: BENCH_ENV forwards
	@# DECISIONS_STYLE_CONFIDENCE_FLOOR, and an arm that leaves it unset exports it EMPTY —
	@# which the benchmark container's ServerConfig load rejects ('input_value=""'), failing
	@# the arm after the whole stack is already warm. Passing the same value in both also makes
	@# the arms differ in exactly one variable, which is what the rule requires.
	@echo "==> arm 1/2: DECISIONS_STYLE_ENABLED=false (the LLM-sampled style stands)"
	@$(STYLE_HEAD_UP)
	@$(BENCH_ENV) BENCH_SCENARIOS=$(STYLE_AB_SCENARIOS) SPLIT=holdout REPEAT=$(STYLE_AB_REPEAT) \
		RUN_ID=$(STYLE_AB_STAMP)-style-off \
		$(STYLE_PROVIDER_ENV) DECISIONS_STYLE_ENABLED=false DECISIONS_STYLE_CONFIDENCE_FLOOR=$(STYLE_AB_FLOOR) $(BENCH_DOCKER) e2e
	@echo "==> arm 2/2: DECISIONS_STYLE_ENABLED=true provider=$(STYLE_PROVIDER) (floor from $(STYLE_AB_FLOOR_FILE))"
	@$(STYLE_HEAD_UP)
	@$(BENCH_ENV) BENCH_SCENARIOS=$(STYLE_AB_SCENARIOS) SPLIT=holdout REPEAT=$(STYLE_AB_REPEAT) \
		RUN_ID=$(STYLE_AB_STAMP)-style-on \
		$(STYLE_PROVIDER_ENV) DECISIONS_STYLE_ENABLED=true DECISIONS_STYLE_CONFIDENCE_FLOOR=$(STYLE_AB_FLOOR) $(BENCH_DOCKER) e2e
	@echo "==> compare against the pre-registered rule (holdout p95 >= +15% AND repeat non-inferiority)"
	@$(BENCH_ENV) REQUIRE_REPEAT_STABILITY=1 A=$(BENCH_RUNS)/$(STYLE_AB_STAMP)-style-off \
		B=$(BENCH_RUNS)/$(STYLE_AB_STAMP)-style-on $(BENCH_DOCKER) compare

benchmark-compare: ## Benchmark: compare two run dirs (A=<baseline> B=<candidate>), in Docker
	@[ -n "$(A)" ] && [ -n "$(B)" ] || { echo "usage: make benchmark-compare A=<baseline-dir> B=<candidate-dir>"; exit 2; }
	@$(BENCH_ENV) $(BENCH_DOCKER) compare

##@ Maintenance
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache dist build mutants
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
