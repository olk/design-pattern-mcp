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
	benchmark-sidecars-up benchmark-draft benchmark-style-ab benchmark-rerank-ab \
	docker-build docker-build-all docker-build-cached docker-build-nocache \
	docker-build-bootstrap docker-build-tei docker-build-tei-embed docker-build-tei-embed-gpu \
	docker-build-tei-rerank dense-cache \
	docker-build-tacit docker-build-tacit-base docker-build-tacit-tunnel \
	docker-build-style-head \
	docker-build-decision docker-up-decisions docker-down-decisions decisions-smoke \
	tacit-payloads tacit-payloads-check \
	style-head-calibrate check-dockerfile-shell \
	docker-publish docker-publish-tei docker-publish-tacit docker-publish-style-head docker-publish-all \
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

# Decision-model sidecar (docker/docker-compose.decision-models.yml). It is GPU-only at
# run time; the *-base target builds the released-weights variant and needs no GPU.
TACIT_IMAGE      := design-pattern-tacit
TACIT_MODEL_REPO ?= morriszjm/Tacit-4B
# SSH-tunnel sidecar for the systemd stack (alpine + openssh-client, no model):
# it forwards the GPU host's loopback-bound gateway onto the shared network, so the
# production container — whose 127.0.0.1 is its own — can reach it as tacit-tunnel:8100.
# Local-only image, never published: it carries no artifact, only the ssh client.
TACIT_TUNNEL_IMAGE := design-pattern-tacit-tunnel
# Registry pair for the sidecar. Ships in docker-publish-all (and on demand via
# docker-publish-tacit): the image is ~29 GB, so budget time/bandwidth for the upload.
TACIT_HUB_REPO   := olkowa/design-pattern-tacit
TACIT_GHCR_REPO  := ghcr.io/olk/design-pattern-tacit
# Publish refuses weights whose manifest does not certify every seam (>= 10 eval rows each):
# the build gate lets a quick-smoke run through with warnings, and the marker hashes inputs,
# not knobs, so this is the last check that a smoke artifact cannot ship by accident.
TACIT_ALLOW_UNCERTIFIED ?= 0
# Self-distillation knobs, sized to MAX EFFORT: the fine-tune's aim is pattern-selection
# consistency and quality, so the defaults spend the available GPU envelope on volume and
# coverage (all train rerank pairs) and keep the recipe's single epoch.
# The teacher pass shards across the visible GPUs (TACIT_TEACHER_SHARDS=0 = one per device).
# `make docker-build-tacit TACIT_PROBLEMS=200 TACIT_RERANK_PROBLEMS=100 TACIT_COT_MAX_TOKENS=512`
# is the quick-smoke shape.
TACIT_PROBLEMS       ?= 6000
# Design states to write. 0 = derive from TACIT_PROBLEMS (questions per state shrink as
# this grows, so the teacher budget holds while state diversity rises).
TACIT_STATES         ?= 0
# J1 rerank question budget; 0 = ALL train pairs in qa/typed/rerank_pairs.jsonl.
TACIT_RERANK_PROBLEMS ?= 0
# Teacher answer shards; 0 = auto (one per visible CUDA device), 1 = in-process.
TACIT_TEACHER_SHARDS ?= 0
# Build fails when an answer seam's student-vs-teacher eval agreement drops below this.
TACIT_MIN_SEAM_AGREEMENT ?= 0.6
# Must equal the served seam floor (DESIGN_PATTERN_DECISIONS_*_CONFIDENCE_FLOOR).
TACIT_CONFIDENCE_FLOOR ?= 0.5
# The reasoning-gate seam's own floor, when the deployment raises it alone.
TACIT_GATE_CONFIDENCE_FLOOR ?= $(TACIT_CONFIDENCE_FLOOR)
# Build fails when a serving kind (choice/score/yes_no) keeps fewer rows than this.
TACIT_MIN_KIND_ROWS  ?= 1
TACIT_EPOCHS         ?= 1
TACIT_LR             ?= 1e-4
TACIT_BATCH          ?= 4
TACIT_ACCUM          ?= 8
TACIT_COT_MAX_TOKENS ?= 2048
TACIT_MAX_NEW_TOKENS ?= 384
VLLM_IMAGE_TAG   ?= v0.17.1
# Linear style head (the Ch2 seam's only provider): a TF-IDF + calibrated-SVM head
# fitted in its own image build, whose training RUN is also the accuracy gate.
STYLE_HEAD_IMAGE := design-pattern-style-head
# Registry pair for the style head, published by docker-publish-style-head.
STYLE_HEAD_HUB_REPO  := olkowa/design-pattern-style-head
STYLE_HEAD_GHCR_REPO := ghcr.io/olk/design-pattern-style-head
# The Ch2 style seam's one switch: it always names the linear head, so the endpoint,
# model name and wire shape move together and a style run cannot inherit a Tacit
# gateway URL (the config validator rejects that pairing outright, and this keeps it
# unexpressible).
STYLE_PROVIDER ?= linear
STYLE_PROVIDER_ENV = DECISIONS_STYLE_PROVIDER=linear \
	DECISIONS_STYLE_BASE_URL=http://design-pattern-style-head:8800/v1 \
	DECISIONS_STYLE_MODEL=style-linear

# Host-persisted trained artifacts (extract-then-rebuild).
# First run of docker-build-tacit trains on the GPU builder,
# tags the image as usual, then extracts the shipped weights to the host dir via
# `docker create` + `docker cp` (the container is never started, no GPU needed).
# A later run finds the sentinels plus the .extract-complete marker and builds
# the *-cached Dockerfile stage instead, which COPYs the host dir into the image
# as the fine-tuned model — same image names, tags, and build args, no GPU, no
# retrain. The marker is written LAST, only after a successful copy with
# non-empty sentinels, so a partial copy never leaves a reusable dir; it is
# committed with the artifacts so a fresh clone still reuses.
#   Tacit: models/tacit-4b  <- /models/tacit (full HF checkpoint: config.json,
#          model.safetensors shards, tokenizer files, self-distill.json)
TACIT_HOST_DIR      := models/tacit-4b
TACIT_HOST_CONFIG   := $(TACIT_HOST_DIR)/config.json
TACIT_HOST_MANIFEST := $(TACIT_HOST_DIR)/self-distill.json
TACIT_HOST_COMPLETE := $(TACIT_HOST_DIR)/.extract-complete
# Parse-time target switch (same shape as DENSE_CACHE_PREREQ): sentinels plus
# the completion marker, or the train path — a dir without the marker is not
# proven complete output, even if a header file survived a partial copy.
TACIT_TRAIN_TARGET  := runtime-distilled
TACIT_CACHED_TARGET := runtime-distilled-cached
TACIT_BUILD_TARGET  := $(if $(and $(wildcard $(TACIT_HOST_CONFIG)),$(wildcard $(TACIT_HOST_MANIFEST)),$(wildcard $(TACIT_HOST_COMPLETE))),$(TACIT_CACHED_TARGET),$(TACIT_TRAIN_TARGET))
# Content provenance for the cached distilled weights: the files whose bytes the
# self-distillation consumed. The train path records their shas in the completion marker;
# the cached path refuses to build when the marker's shas no longer match, so a stale
# host dir can never be re-baked against a changed prompt vocabulary or pair artifact.
TACIT_TRAIN_INPUTS  := scripts/decision_models/tacit_self_distill.py \
	src/decisions/seam_prompts.py qa/qa.jsonl qa/typed/rerank_pairs.jsonl \
	docker/Dockerfile.tacit
# HF Hub token for authenticated model downloads (higher rate limits, gated repos).
# File holds the raw token only, no key=value. Empty/missing file = anonymous
# download (fine for the pinned public repos, just the huggingface_hub warning).
# Every HF-fetching Dockerfile (tacit, tei-embed, tei-rerank) reads it via
# `RUN --mount=type=secret,id=hf_token`; the secret is mounted per-RUN, never baked
# into a layer.
comma := ,
# BuildKit progress mode for the decision-model builds. `plain` streams RUN
# stdout (trainer phases, per-shard progress) to the terminal and `tee`d logs;
# the default `tty` collapses it. Override per-call: `make docker-build-tacit
# DOCKER_BUILD_PROGRESS=tty`.
DOCKER_BUILD_PROGRESS ?= plain
HF_TOKEN_FILE ?= $(HOME)/.cache/hf_token
# Resolve to --secret when the file exists and is non-empty. A missing default
# is fine (anonymous download); but an explicitly-set missing path is a typo,
# so fail instead of silently going anonymous. Recursive (=, not :=) so the
# error fires only when a docker-build recipe expands HF_SECRET_ARG.
HF_SECRET_ARG = $(if $(shell test -s "$(HF_TOKEN_FILE)" && echo yes),--secret id=hf_token$(comma)src=$(HF_TOKEN_FILE),$(if $(filter-out file,$(origin HF_TOKEN_FILE)),$(error HF_TOKEN_FILE="$(HF_TOKEN_FILE)" is missing or empty)))
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

docker-build-bootstrap: ## [internal] Build the server image that runs the cache builder (no cache baked)
	docker build --target production -f Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker build --target production -f Dockerfile -t $(DOCKER_IMAGE):latest .

docker-build-tei-embed: ## Build the TEI embed sidecar (EMBED_MODEL_REPO=…, EMBED_ONNX_VARIANT=fp32|int8)
	docker build -f docker/Dockerfile.tei-embed $(HF_SECRET_ARG) \
		--build-arg EMBED_MODEL_REPO=$(EMBED_MODEL_REPO) \
		--build-arg EMBED_ONNX_VARIANT=$(EMBED_ONNX_VARIANT) \
		--build-arg EMBED_POOLING=$(EMBED_POOLING) \
		--build-arg EMBED_DTYPE=$(EMBED_DTYPE) \
		--build-arg TEI_BASE_IMAGE=$(EMBED_TEI_BASE) \
		-t $(TEI_EMBED_IMAGE):$(DOCKER_TAG) -t $(TEI_EMBED_IMAGE):latest .

docker-build-tei-embed-gpu: ## Build the TEI embed sidecar on the CUDA base (needs an NVIDIA GPU at run time)
	$(MAKE) --no-print-directory docker-build-tei-embed EMBED_TEI_BASE=$(EMBED_TEI_BASE_GPU)

docker-build-tei-rerank: ## Build the TEI rerank sidecar image
	docker build -f docker/Dockerfile.tei-rerank $(HF_SECRET_ARG) \
		-t $(TEI_RERANK_IMAGE):$(DOCKER_TAG) -t $(TEI_RERANK_IMAGE):latest .

docker-build-tei: docker-build-tei-embed docker-build-tei-rerank ## Build both TEI sidecar images

# Sequential prerequisites on purpose: docker-build-tacit holds the GPU for its
# self-distill RUN, so parallel make must not overlap it with another GPU build.
docker-build-all: docker-build docker-build-tei docker-build-decision ## Build every image: server (+dense cache), TEI pair, Tacit, style head

docker-build-cached: embed-cache docker-build-all ## Force-regenerate the dense cache, then build every image

##@ Decision models (AnyJev Tacit-4B)

docker-build-tacit: tacit-payloads-check check-dockerfile-shell ## Fine-tune Tacit in the image build (GPU self-distill once; later runs reuse models/tacit-4b, no GPU)
	@# Cached path first: refuse a stale host dir whose marker shas no longer match
	@# the current train inputs — BEFORE docker build, so a stale cache never
	@# re-tags the image and the failure is loud, not a silent old-weights rebuild.
	@if [ "$(TACIT_BUILD_TARGET)" = "$(TACIT_CACHED_TARGET)" ]; then \
		if ! ls -A "$(TACIT_HOST_DIR)"/*.safetensors >/dev/null 2>&1; then \
			echo "ERROR: $(TACIT_HOST_DIR)/ holds the COMMITTED sentinels but no weight shards."; \
			echo "       A fresh clone looks exactly like this: config.json + self-distill.json are in git,"; \
			echo "       the multi-GB weights are not. The cached stage would COPY a dir vLLM cannot load."; \
			echo "Run 'rm -rf $(TACIT_HOST_DIR)' so the train path rebuilds the weights."; \
			exit 1; \
		fi; \
		if ! (cd "$(CURDIR)" && sha256sum -c "$(TACIT_HOST_COMPLETE)" --quiet --status 2>/dev/null); then \
			echo "ERROR: $(TACIT_HOST_DIR)/ is stale — a train input changed since the weights were extracted."; \
			echo "Run 'rm -rf $(TACIT_HOST_DIR)' and rebuild to retrain, or restore the inputs."; \
			(cd "$(CURDIR)" && sha256sum -c "$(TACIT_HOST_COMPLETE)" 2>&1 | grep -v ': OK' || true); \
			exit 1; \
		fi; \
	fi
	docker build --progress=$(DOCKER_BUILD_PROGRESS) --allow device=nvidia.com/gpu=all $(HF_SECRET_ARG) \
		-f docker/Dockerfile.tacit --target $(TACIT_BUILD_TARGET) \
		--build-arg VLLM_IMAGE_TAG=$(VLLM_IMAGE_TAG) \
		--build-arg TACIT_MODEL_REPO=$(TACIT_MODEL_REPO) \
		--build-arg TACIT_PROBLEMS=$(TACIT_PROBLEMS) \
		--build-arg TACIT_STATES=$(TACIT_STATES) \
		--build-arg TACIT_RERANK_PROBLEMS=$(TACIT_RERANK_PROBLEMS) \
		--build-arg TACIT_TEACHER_SHARDS=$(TACIT_TEACHER_SHARDS) \
		--build-arg TACIT_MIN_SEAM_AGREEMENT=$(TACIT_MIN_SEAM_AGREEMENT) \
		--build-arg TACIT_CONFIDENCE_FLOOR=$(TACIT_CONFIDENCE_FLOOR) \
		--build-arg TACIT_GATE_CONFIDENCE_FLOOR=$(TACIT_GATE_CONFIDENCE_FLOOR) \
		--build-arg TACIT_MIN_KIND_ROWS=$(TACIT_MIN_KIND_ROWS) \
		--build-arg TACIT_EPOCHS=$(TACIT_EPOCHS) \
		--build-arg TACIT_LR=$(TACIT_LR) \
		--build-arg TACIT_BATCH=$(TACIT_BATCH) \
		--build-arg TACIT_ACCUM=$(TACIT_ACCUM) \
		--build-arg TACIT_COT_MAX_TOKENS=$(TACIT_COT_MAX_TOKENS) \
		--build-arg TACIT_MAX_NEW_TOKENS=$(TACIT_MAX_NEW_TOKENS) \
		-t $(TACIT_IMAGE):$(DOCKER_TAG) -t $(TACIT_IMAGE):latest .
	@# Train path only: extract the shipped weights to $(TACIT_HOST_DIR)/ via
	@# `docker create` (never started) + `docker cp`; the next run builds cached.
	@# The .extract-complete marker is written LAST, only after a successful copy
	@# with non-empty sentinels — a partial copy never leaves a reusable dir. It
	@# carries the sha256 of every train input so the check above can refuse a stale dir.
	@if [ "$(TACIT_BUILD_TARGET)" = "$(TACIT_TRAIN_TARGET)" ]; then \
		mkdir -p "$(TACIT_HOST_DIR)"; \
		cid=$$(docker create "$(TACIT_IMAGE):$(DOCKER_TAG)"); \
		docker cp "$$cid:/models/tacit/." "$(TACIT_HOST_DIR)/"; rc=$$?; \
		docker rm "$$cid" >/dev/null; \
		if [ $$rc -ne 0 ]; then rm -f "$(TACIT_HOST_COMPLETE)"; exit $$rc; fi; \
		test -s "$(TACIT_HOST_CONFIG)" && test -s "$(TACIT_HOST_MANIFEST)" || exit 1; \
		{ echo "# $(TACIT_IMAGE):$(DOCKER_TAG) $$(date -u +%FT%TZ)"; \
		  sha256sum $(TACIT_TRAIN_INPUTS); } > "$(TACIT_HOST_COMPLETE)"; \
	fi
	@test -s "$(TACIT_HOST_CONFIG)" && test -s "$(TACIT_HOST_MANIFEST)" && test -f "$(TACIT_HOST_COMPLETE)"

docker-build-tacit-base: ## Build the Tacit sidecar with the released weights (no GPU needed; tag :base)
	docker build --progress=$(DOCKER_BUILD_PROGRESS) -f docker/Dockerfile.tacit --target runtime-released $(HF_SECRET_ARG) \
		--build-arg VLLM_IMAGE_TAG=$(VLLM_IMAGE_TAG) \
		--build-arg TACIT_MODEL_REPO=$(TACIT_MODEL_REPO) \
		-t $(TACIT_IMAGE):$(DOCKER_TAG)-base -t $(TACIT_IMAGE):base .

# The linear Ch2 style head: CPU only, ~1 min, and the training RUN inside the build is also
# the accuracy gate (a head below 0.65 holdout accuracy exits 2 and fails the build). There
# is no cached/host-artifact path: the fit is cheap enough that a stale-head cache would cost
# more attention than it saves.
docker-build-style-head: ## Build the linear style head sidecar (CPU; the build gates the head)
	docker build --progress=$(DOCKER_BUILD_PROGRESS) -f docker/Dockerfile.style-head \
		-t $(STYLE_HEAD_IMAGE):$(DOCKER_TAG) -t $(STYLE_HEAD_IMAGE):latest .

docker-build-tacit-tunnel: ## Build the Tacit SSH-tunnel sidecar (alpine + openssh-client)
	docker build --progress=$(DOCKER_BUILD_PROGRESS) -f docker/Dockerfile.tacit-tunnel \
		-t $(TACIT_TUNNEL_IMAGE):$(DOCKER_TAG) -t $(TACIT_TUNNEL_IMAGE):latest .

docker-build-decision: docker-build-tacit docker-build-style-head ## Build the Tacit + style-head sidecars, in-image

docker-up-decisions: ## Start the dev stack with the decision-model sidecars (one GPU, one CPU)
	$(STYLE_PROVIDER_ENV) TACIT_IMAGE=$(TACIT_IMAGE):$(DOCKER_TAG) \
		STYLE_HEAD_IMAGE=$(STYLE_HEAD_IMAGE):$(DOCKER_TAG) $(COMPOSE_DM) up -d

docker-down-decisions: ## Stop the whole dev stack including the decision-model sidecars
	TACIT_IMAGE=$(TACIT_IMAGE):$(DOCKER_TAG) \
		STYLE_HEAD_IMAGE=$(STYLE_HEAD_IMAGE):$(DOCKER_TAG) $(COMPOSE_DM) down

decisions-smoke: ## Smoke-test the running sidecars' wire contracts (needs docker-up-decisions)
	@echo "==> Tacit /v1/decide"
	@uv run python scripts/decision_models/decision_smoke.py --base-url $(TACIT_BASE_URL)
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

# Served rerank pairs (statement x candidate document) for the Ch2/J1 training seams.
# Re-run after any corpus, noul or catalog change: the artifact carries the candidates'
# document_text, which the Tacit image cannot derive (it has no pattern catalog).
TACIT_PAYLOAD_CHECK  := /tmp/tacit-rerank-pairs.check.jsonl

tacit-payloads: ## Regenerate qa/typed/rerank_pairs.jsonl from the corpus + catalog
	uv run python scripts/decision_models/derive_rerank_pairs.py

# The catalog is not hashed in TACIT_TRAIN_INPUTS (7290 files), so a catalog edit that was
# never re-derived would otherwise leave a byte-identical pair file and a valid cache:
# training (and the eval replay) would score document_text the server no longer sends.
# Regenerate to a temp path and byte-compare instead of trusting the marker.
tacit-payloads-check: ## Fail when qa/typed/rerank_pairs.jsonl drifts from the corpus + catalog
	@uv run python scripts/decision_models/derive_rerank_pairs.py --out $(TACIT_PAYLOAD_CHECK) >/dev/null
	@if ! cmp -s $(TACIT_PAYLOAD_CHECK) qa/typed/rerank_pairs.jsonl; then \
		echo "ERROR: qa/typed/rerank_pairs.jsonl is stale — the corpus, noul or catalog changed since it was derived."; \
		echo "       Run 'make tacit-payloads' and commit the new artifact (the trainer consumes these bytes)."; \
		rm -f $(TACIT_PAYLOAD_CHECK); exit 1; \
	fi
	@rm -f $(TACIT_PAYLOAD_CHECK)

# `docker build --check` does not evaluate shell syntax, and Dockerfile shells are dash
# even when the developer's /bin/sh is bash. Two real defects reached a build this way: a
# stray quote that produced `Syntax error: "(" unexpected`, and a '#' that commented out a
# gate command. Validate every shell payload before spending an hour on the image; this
# runs from the build targets so it cannot regress silently.
DOCKERFILE_SHELL_CHECK := $(UV) run python scripts/decision_models/check_dockerfile_shell.py

check-dockerfile-shell: ## Validate Dockerfile shell payloads (dash) before a build
	@$(DOCKERFILE_SHELL_CHECK) docker/Dockerfile.tacit

# Live gateway root for the decisions smoke probe (decisions-smoke).
TACIT_BASE_URL             ?= http://127.0.0.1:18100

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

# Ships from docker-publish-all (and on demand via this target). A release from a machine
# without the trained weights aborts in docker-build-tacit on the committed sentinels
# ("holds the COMMITTED sentinels but no weight shards … Run rm -rf"), and only after
# that does the GPU self-distill run; the cert guard below catches a different failure
# (a smoke build that slipped through with warnings), not this one.
# It depends on the build rule on purpose — that is what pins the pushed weights to
# TACIT_TRAIN_INPUTS through models/tacit-4b/.extract-complete. No clean-tree gate
# (unlike docker-publish): the trained checkpoint is deliberately untracked, so a real
# train output would always trip it. The manifest is echoed as the provenance record.
# Publish also refuses a manifest that does not certify every seam: the trainer records
# the certification threshold (min_eval_rows, 20) in self-distill.json and the guard
# below reads it from there, so the two can never drift. The build gate lets a
# quick-smoke run through with warnings, and the marker hashes inputs, not knobs — this
# is the last check that a smoke artifact cannot ship by accident
# (TACIT_ALLOW_UNCERTIFIED=1 overrides).
docker-publish-tacit: docker-build-tacit ## Push the Tacit sidecar image to Docker Hub + GHCR
	@test -s "$(TACIT_HOST_MANIFEST)" || { \
		echo "ERROR: $(TACIT_HOST_MANIFEST) is missing - the image has no train provenance"; exit 1; }
	@echo "Publishing $(TACIT_IMAGE):$(DOCKER_TAG) distilled from:"
	@python3 -c "import json;d=json.load(open('$(TACIT_HOST_MANIFEST)'));\
print({k: d.get(k) for k in ('repo','problems','examples','kept','epochs','lr')})"
	@python3 -c 'import json,sys;m=json.load(open("$(TACIT_HOST_MANIFEST)"));g=("category","rerank","gate","weights");e=m.get("eval") or {};t=m.get("min_eval_rows");thin=g if t is None else [s for s in g if int((e.get(s) or {}).get("n") or 0)<int(t)];allow="$(TACIT_ALLOW_UNCERTIFIED)"=="1";print("ERROR: manifest records no min_eval_rows - legacy uncertified artifact; retrain or pass TACIT_ALLOW_UNCERTIFIED=1") if t is None and not allow else (print("ERROR: uncertified seams",thin,"- fewer than",t,"eval rows each; run the full build or pass TACIT_ALLOW_UNCERTIFIED=1") if thin and not allow else (print("WARNING: publishing uncertified seams",thin) if thin else None));sys.exit(1 if thin and not allow else 0)'
	docker login -u olkowa
	docker login ghcr.io -u olk
	$(call publish-image,$(TACIT_IMAGE),$(TACIT_HUB_REPO),$(TACIT_GHCR_REPO))
	docker logout ghcr.io

# TEI shape: build first (the training RUN inside the build is the accuracy gate, so a
# built image is always a gated head), then tag + push both registries.
docker-publish-style-head: docker-build-style-head ## Push the style head image to Docker Hub + GHCR
	docker login -u olkowa
	docker login ghcr.io -u olk
	$(call publish-image,$(STYLE_HEAD_IMAGE),$(STYLE_HEAD_HUB_REPO),$(STYLE_HEAD_GHCR_REPO))
	docker logout ghcr.io

docker-publish-all: docker-publish docker-publish-tei docker-publish-tacit docker-publish-style-head ## Build and push every image: server, TEI pair, Tacit, style head

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
	REQUIRE_REPEAT_STABILITY="$(REQUIRE_REPEAT_STABILITY)" \
	FAMILIES="$(FAMILIES)" DRAFT_OUT="$(DRAFT_OUT)" BENCH_DECISION_OVERLAY="$(BENCH_DECISION_OVERLAY)" \
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
# Pre-registered A/B of the Ch2 style seam with a decision-model sidecar in the loop. The
# decision overlay renders without a GPU (`docker compose ... config`), but the runs need the
# card and the trained image (`make docker-build-decision`). The seam is off in the
# baseline arm and on in the candidate arm; everything else (corpus, split, repeat, image) is
# identical, which is what `benchmark-compare`'s +15% p95 rule requires.
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
# Which overlay the A/B stack carries. The default is the CPU-only style-seam overlay,
# because the GPU overlay's design-pattern-mcp depends_on the Tacit gateway and therefore
# cannot come up on a host whose card predates Ampere. On a GPU host, run the full stack
# instead by overriding the overlay:
#   make benchmark-style-ab \
#     STYLE_AB_OVERLAY=docker/docker-compose.decision-models.yml
STYLE_HEAD_OVERLAY  := docker/docker-compose.style-head.yml
STYLE_AB_OVERLAY     = $(STYLE_HEAD_OVERLAY)
# Bring the linear head up inside the A/B project. The harness tears its whole compose project
# down at the end of EVERY arm, so this runs before each arm — starting it once would leave
# arm 2 with a dead seam (and a dead seam is silent: the LLM style stands, the arm still
# completes, and the comparison would quietly measure nothing). It uses the SAME overlay as the
# arm, so the sidecar is part of the project's file set either way.
STYLE_HEAD_UP = STYLE_HEAD_IMAGE=$(STYLE_HEAD_IMAGE):$(DOCKER_TAG) \
	docker compose -f docker/docker-compose.yml -f $(STYLE_AB_OVERLAY) up -d --wait design-pattern-style-head

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
		RUN_ID=$(STYLE_AB_STAMP)-style-off BENCH_DECISION_OVERLAY=$(STYLE_AB_OVERLAY) \
		$(STYLE_PROVIDER_ENV) DECISIONS_STYLE_ENABLED=false DECISIONS_STYLE_CONFIDENCE_FLOOR=$(STYLE_AB_FLOOR) $(BENCH_DOCKER) e2e
	@echo "==> arm 2/2: DECISIONS_STYLE_ENABLED=true provider=$(STYLE_PROVIDER) (floor from $(STYLE_AB_FLOOR_FILE))"
	@$(STYLE_HEAD_UP)
	@$(BENCH_ENV) BENCH_SCENARIOS=$(STYLE_AB_SCENARIOS) SPLIT=holdout REPEAT=$(STYLE_AB_REPEAT) \
		RUN_ID=$(STYLE_AB_STAMP)-style-on BENCH_DECISION_OVERLAY=$(STYLE_AB_OVERLAY) \
		$(STYLE_PROVIDER_ENV) DECISIONS_STYLE_ENABLED=true DECISIONS_STYLE_CONFIDENCE_FLOOR=$(STYLE_AB_FLOOR) $(BENCH_DOCKER) e2e
	@echo "==> compare against the pre-registered rule (holdout p95 >= +15% AND repeat non-inferiority)"
	@$(BENCH_ENV) REQUIRE_REPEAT_STABILITY=1 A=$(BENCH_RUNS)/$(STYLE_AB_STAMP)-style-off \
		B=$(BENCH_RUNS)/$(STYLE_AB_STAMP)-style-on $(BENCH_DOCKER) compare

# Selection-level acceptance for the J1 rerank leg: same overlay stack, corpus, split and
# repeat, only RERANKER_PROVIDER differs. Keep the Tacit leg as the overlay default only if
# the candidate arm is non-inferior on holdout hit rate and recall@3, with zero veto/fallback
# regressions and e2e p95 within +15% — `benchmark-compare` exits 1 on a candidate-loss
# verdict, which is a verdict to report (fall back to RERANKER_PROVIDER=tei), not a crash.
RERANK_AB_SCENARIOS ?= tests/benchmark/scenarios/full.json
RERANK_AB_REPEAT    ?= 3
RERANK_AB_STAMP     ?= $(shell date -u +%Y%m%d-%H%M%S)

benchmark-rerank-ab: ## Benchmark: pre-registered J1 rerank A/B, TEI vs Tacit (needs the sidecar + GPU)
	@echo "==> arm 1/2: RERANKER_PROVIDER=tei (the cross-encoder baseline)"
	@$(BENCH_ENV) BENCH_SCENARIOS=$(RERANK_AB_SCENARIOS) SPLIT=holdout REPEAT=$(RERANK_AB_REPEAT) \
		RUN_ID=$(RERANK_AB_STAMP)-rerank-tei BENCH_DECISION_OVERLAY=docker/docker-compose.decision-models.yml \
		RERANKER_PROVIDER=tei $(BENCH_DOCKER) e2e
	@echo "==> arm 2/2: RERANKER_PROVIDER=tacit (per-candidate yes_no leg)"
	@$(BENCH_ENV) BENCH_SCENARIOS=$(RERANK_AB_SCENARIOS) SPLIT=holdout REPEAT=$(RERANK_AB_REPEAT) \
		RUN_ID=$(RERANK_AB_STAMP)-rerank-tacit BENCH_DECISION_OVERLAY=docker/docker-compose.decision-models.yml \
		RERANKER_PROVIDER=tacit $(BENCH_DOCKER) e2e
	@echo "==> compare against the pre-registered rule (non-inferior; holdout p95 within +15%)"
	@$(BENCH_ENV) A=$(BENCH_RUNS)/$(RERANK_AB_STAMP)-rerank-tei B=$(BENCH_RUNS)/$(RERANK_AB_STAMP)-rerank-tacit \
		$(BENCH_DOCKER) compare

benchmark-compare: ## Benchmark: compare two run dirs (A=<baseline> B=<candidate>), in Docker
	@[ -n "$(A)" ] && [ -n "$(B)" ] || { echo "usage: make benchmark-compare A=<baseline-dir> B=<candidate-dir>"; exit 2; }
	@$(BENCH_ENV) $(BENCH_DOCKER) compare

##@ Maintenance
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache dist build mutants
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
