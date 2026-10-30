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
	benchmark-sidecars-up benchmark-draft \
	docker-build docker-build-all docker-build-cached docker-build-nocache \
	docker-build-bootstrap docker-build-tei docker-build-tei-embed docker-build-tei-embed-gpu \
	docker-build-tei-rerank dense-cache \
	docker-build-tacit docker-build-tacit-base docker-build-clm docker-build-clm-base \
	docker-build-decision-models docker-up-decisions docker-down-decisions decisions-smoke \
	docker-publish docker-publish-tei docker-publish-all \
	docker-up docker-up-gpu docker-down docker-logs docker-logs-follow docker-rm clean \
	embed-cache

COMPOSE := docker compose -f docker/docker-compose.yml
UV ?= uv

DOCKER_IMAGE    := design-pattern-mcp
DOCKER_TAG      := $(shell grep -m 1 '^version' pyproject.toml | sed -E 's/.*"([^"]+)".*/\1/')
DOCKER_HUB_REPO := olkowa/design-pattern-mcp
GHCR_REPO       := ghcr.io/olk/design-pattern-mcp

TEI_EMBED_IMAGE      := design-pattern-tei-embed
TEI_RERANK_IMAGE     := design-pattern-tei-rerank
TEI_EMBED_HUB_REPO   := olkowa/pattern-tei-embed
TEI_EMBED_GHCR_REPO  := ghcr.io/olk/pattern-tei-embed
TEI_RERANK_HUB_REPO  := olkowa/pattern-tei-rerank
TEI_RERANK_GHCR_REPO := ghcr.io/olk/pattern-tei-rerank

EMBED_MODEL_REPO   ?= onnx-community/Qwen3-Embedding-0.6B-ONNX
EMBED_ONNX_VARIANT ?= fp32
EMBED_POOLING      ?= last-token
EMBED_DTYPE        ?= float32
EMBED_TEI_BASE     ?= ghcr.io/huggingface/text-embeddings-inference:cpu-1.9
EMBED_TEI_BASE_GPU ?= ghcr.io/huggingface/text-embeddings-inference:cuda-1.9
DENSE_CACHE_DIR    ?= data/dense-cache

# Decision-model sidecars (docker/docker-compose.decision-models.yml). Both are GPU-only
# at run time; the *-base targets build the released-weights variant and need no GPU.
TACIT_IMAGE      := design-pattern-tacit
CLM_IMAGE        := design-pattern-clm
TACIT_MODEL_REPO ?= morriszjm/Tacit-4B
# Self-distillation knobs. The teacher pass is one sequential generate per problem, so
# build time scales with TACIT_PROBLEMS x TACIT_COT_MAX_TOKENS (defaults: single-digit
# hours on one 3090). `make docker-build-tacit TACIT_PROBLEMS=200 TACIT_COT_MAX_TOKENS=512`
# is the quick-smoke shape.
TACIT_PROBLEMS       ?= 2000
TACIT_EPOCHS         ?= 1
TACIT_LR             ?= 1e-4
TACIT_BATCH          ?= 4
TACIT_ACCUM          ?= 8
TACIT_COT_MAX_TOKENS ?= 2048
TACIT_MAX_NEW_TOKENS ?= 256
CLM_ENCODER_REPO ?= Qwen/Qwen3-8B
CLM_HEADS_REPO   ?= Contrastive-LM/CLM-v0.1-8B
CLM_REPO_PIN     ?= d5f9ef0
VLLM_IMAGE_TAG   ?= v0.17.1
COMPOSE_DM := docker compose -f docker/docker-compose.yml -f docker/docker-compose.decision-models.yml

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

docker-build-nocache: ## Build the MCP server image without a dense cache (first start embeds, ~19 min)
	@$(MAKE) --no-print-directory docker-build SKIP_DENSE_CACHE=1

docker-build-bootstrap: ## [internal] Build the server image that runs the cache builder (no cache baked)
	docker build --target production -f Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker build --target production -f Dockerfile -t $(DOCKER_IMAGE):latest .

docker-build-tei-embed: ## Build the TEI embed sidecar (EMBED_MODEL_REPO=…, EMBED_ONNX_VARIANT=fp32|int8)
	docker build -f docker/Dockerfile.tei-embed \
		--build-arg EMBED_MODEL_REPO=$(EMBED_MODEL_REPO) \
		--build-arg EMBED_ONNX_VARIANT=$(EMBED_ONNX_VARIANT) \
		--build-arg EMBED_POOLING=$(EMBED_POOLING) \
		--build-arg EMBED_DTYPE=$(EMBED_DTYPE) \
		--build-arg TEI_BASE_IMAGE=$(EMBED_TEI_BASE) \
		-t $(TEI_EMBED_IMAGE):$(DOCKER_TAG) -t $(TEI_EMBED_IMAGE):latest .

docker-build-tei-embed-gpu: ## Build the TEI embed sidecar on the CUDA base (needs an NVIDIA GPU at run time)
	$(MAKE) --no-print-directory docker-build-tei-embed EMBED_TEI_BASE=$(EMBED_TEI_BASE_GPU)

docker-build-tei-rerank: ## Build the TEI rerank sidecar image
	docker build -f docker/Dockerfile.tei-rerank \
		-t $(TEI_RERANK_IMAGE):$(DOCKER_TAG) -t $(TEI_RERANK_IMAGE):latest .

docker-build-tei: docker-build-tei-embed docker-build-tei-rerank ## Build both TEI sidecar images

docker-build-all: docker-build docker-build-tei ## Build all three images, dense cache generated and baked in

docker-build-cached: embed-cache docker-build-all ## Force-regenerate the dense cache, then build all three images

##@ Decision models (CLM-8B + AnyJev Tacit-4B)

docker-build-tacit: ## Build the Tacit sidecar, self-distilled in-image (needs a GPU + CDI on the builder)
	docker build --allow device=nvidia.com/gpu=all \
		-f docker/Dockerfile.tacit --target runtime-distilled \
		--build-arg VLLM_IMAGE_TAG=$(VLLM_IMAGE_TAG) \
		--build-arg TACIT_MODEL_REPO=$(TACIT_MODEL_REPO) \
		--build-arg TACIT_PROBLEMS=$(TACIT_PROBLEMS) \
		--build-arg TACIT_EPOCHS=$(TACIT_EPOCHS) \
		--build-arg TACIT_LR=$(TACIT_LR) \
		--build-arg TACIT_BATCH=$(TACIT_BATCH) \
		--build-arg TACIT_ACCUM=$(TACIT_ACCUM) \
		--build-arg TACIT_COT_MAX_TOKENS=$(TACIT_COT_MAX_TOKENS) \
		--build-arg TACIT_MAX_NEW_TOKENS=$(TACIT_MAX_NEW_TOKENS) \
		-t $(TACIT_IMAGE):$(DOCKER_TAG) -t $(TACIT_IMAGE):latest .

docker-build-tacit-base: ## Build the Tacit sidecar with the released weights (no GPU needed; tag :base)
	docker build -f docker/Dockerfile.tacit --target runtime-released \
		--build-arg VLLM_IMAGE_TAG=$(VLLM_IMAGE_TAG) \
		--build-arg TACIT_MODEL_REPO=$(TACIT_MODEL_REPO) \
		-t $(TACIT_IMAGE):$(DOCKER_TAG)-base -t $(TACIT_IMAGE):base .

docker-build-clm: ## Build the CLM sidecar, heads fine-tuned in-image (needs a GPU + CDI on the builder)
	docker build --allow device=nvidia.com/gpu=all \
		-f docker/Dockerfile.clm --target runtime-finetuned \
		--build-arg VLLM_IMAGE_TAG=$(VLLM_IMAGE_TAG) \
		--build-arg CLM_ENCODER_REPO=$(CLM_ENCODER_REPO) \
		--build-arg CLM_HEADS_REPO=$(CLM_HEADS_REPO) \
		--build-arg CLM_REPO_PIN=$(CLM_REPO_PIN) \
		-t $(CLM_IMAGE):$(DOCKER_TAG) -t $(CLM_IMAGE):latest .

docker-build-clm-base: ## Build the CLM sidecar with the released heads (no GPU needed; tag :base)
	docker build -f docker/Dockerfile.clm --target runtime-released \
		--build-arg VLLM_IMAGE_TAG=$(VLLM_IMAGE_TAG) \
		--build-arg CLM_ENCODER_REPO=$(CLM_ENCODER_REPO) \
		--build-arg CLM_HEADS_REPO=$(CLM_HEADS_REPO) \
		--build-arg CLM_REPO_PIN=$(CLM_REPO_PIN) \
		-t $(CLM_IMAGE):$(DOCKER_TAG)-base -t $(CLM_IMAGE):base .

docker-build-decision-models: docker-build-tacit docker-build-clm ## Build both decision-model sidecars, trained in-image

docker-up-decisions: ## Start the dev stack with both decision-model sidecars (one GPU each)
	TACIT_IMAGE=$(TACIT_IMAGE):$(DOCKER_TAG) CLM_IMAGE=$(CLM_IMAGE):$(DOCKER_TAG) $(COMPOSE_DM) up -d

docker-down-decisions: ## Stop the whole dev stack including the decision-model sidecars
	TACIT_IMAGE=$(TACIT_IMAGE):$(DOCKER_TAG) CLM_IMAGE=$(CLM_IMAGE):$(DOCKER_TAG) $(COMPOSE_DM) down

decisions-smoke: ## Smoke-test the running sidecars' wire contracts (needs docker-up-decisions)
	@echo "==> Tacit /v1/decide"
	@curl -fsS http://127.0.0.1:18100/v1/decide -H 'Content-Type: application/json' \
		-d '{"state":"System goal: serve order views\nDecision part: invalidate cached views","question":"Which design-pattern category best describes the concern of this decision part?","options":["persistence-state","behavior-selection"],"kind":"choice"}' \
		| python3 -m json.tool
	@echo "==> CLM /v1/systemone"
	@curl -fsS http://127.0.0.1:18700/v1/systemone -H 'Content-Type: application/json' \
		-d '{"model":"clm-latest","state":"A checkout service with flash sales.","questions":{"style":{"type":"choice","instructions":"Which style fits?","criteria":{"event-driven":"Event flow and reactions","layered":"Tiered separation"}}}}' \
		| python3 -m json.tool
	@echo "==> CLM /v1/rank"
	@curl -fsS http://127.0.0.1:18700/v1/rank -H 'Content-Type: application/json' \
		-d '{"context":"","question":"Wrap external payment providers.","answers":["Adapter","Observer","Memento"]}' \
		| python3 -m json.tool

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

docker-publish-all: docker-publish docker-publish-tei ## Build and push all three images

# File target, not phony: the cache is expensive (one TEI round-trip per
# embedder batch, ~15 min for the 486-record catalog on CPU) and is only rebuilt
# when the artifact is gone — `embed-cache` is the explicit force path.
# The image prerequisites are ORDER-ONLY on purpose: they are phony (always
# "remade"), so ordinary prerequisites would mark this target out of date on
# every run and re-embed the whole catalog 15 min at a time.
$(DENSE_CACHE_MANIFEST) $(DENSE_CACHE_VECTORS) &: | docker-build-tei-embed docker-build-bootstrap ## Generate data/dense-cache/ from the catalog (starts the TEI embed sidecar)
	@$(COMPOSE) up -d design-pattern-tei-embed
	@printf ">> waiting for design-pattern-tei-embed-dev"; \
	for _ in $$(seq 1 60); do \
		if [ "$$(docker inspect -f '{{.State.Health.Status}}' design-pattern-tei-embed-dev 2>/dev/null)" = "healthy" ]; then echo " healthy"; break; fi; \
		printf "."; sleep 5; \
	done
	@mkdir -p $(DENSE_CACHE_DIR)
	$(COMPOSE) run --rm --no-deps -T \
		-v "$(CURDIR)/$(DENSE_CACHE_DIR):/out" \
		-e EMBED_MODEL_REPO=$(EMBED_MODEL_REPO) \
		-e EMBED_ONNX_VARIANT=$(EMBED_ONNX_VARIANT) \
		--entrypoint python design-pattern-mcp \
		-m src.patterns.dense_cache build --out /out
	@ls -l $(DENSE_CACHE_DIR)

dense-cache: $(DENSE_CACHE_MANIFEST) ## Generate data/dense-cache/ if missing (no-op when present)

embed-cache: ## Force-regenerate data/dense-cache/ (starts the TEI embed sidecar)
	@rm -f $(DENSE_CACHE_MANIFEST) $(DENSE_CACHE_DIR)/vectors.npy
	@$(MAKE) --no-print-directory dense-cache

docker-up: ## Start the dev stack (requires make docker-build-all once)
	EMBED_MODEL_REPO=$(EMBED_MODEL_REPO) EMBED_ONNX_VARIANT=$(EMBED_ONNX_VARIANT) $(COMPOSE) up -d --build

docker-up-gpu: ## Start the dev stack with the CUDA embed sidecar (needs an NVIDIA GPU)
	EMBED_MODEL_REPO=$(EMBED_MODEL_REPO) EMBED_ONNX_VARIANT=$(EMBED_ONNX_VARIANT) \
	$(COMPOSE) -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up -d --build

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
# make docker-up).
BENCH_RUNS      ?= data/benchmark-runs
BENCH_SCENARIOS ?= tests/benchmark/scenarios/seed.json
BENCH_COMPOSE   := docker compose -f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml
BENCH_DOCKER    := bash scripts/benchmark-docker.sh
BENCH_ENV := BENCH_SCENARIOS="$(BENCH_SCENARIOS)" OUT="$(OUT)" RUN_ID="$(RUN_ID)" LIMIT="$(LIMIT)" \
	SPLIT="$(SPLIT)" REPEAT="$(REPEAT)" NO_WARMUP="$(NO_WARMUP)" LOG_LEVEL="$(LOG_LEVEL)" \
	FLIP="$(FLIP)" CLIENT_URL="$(CLIENT_URL)" CALL_TIMEOUT="$(CALL_TIMEOUT)" A="$(A)" B="$(B)" \
	ALLOW_MISMATCH="$(ALLOW_MISMATCH)" DRY_RUN="$(DRY_RUN)" PER_FAMILY="$(PER_FAMILY)" \
	FAMILIES="$(FAMILIES)" DRAFT_OUT="$(DRAFT_OUT)"

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
benchmark-compare: ## Benchmark: compare two run dirs (A=<baseline> B=<candidate>), in Docker
	@[ -n "$(A)" ] && [ -n "$(B)" ] || { echo "usage: make benchmark-compare A=<baseline-dir> B=<candidate-dir>"; exit 2; }
	@$(BENCH_ENV) $(BENCH_DOCKER) compare

##@ Maintenance
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache dist build mutants
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
