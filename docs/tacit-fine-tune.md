# Fine-Tuning Tacit-4B for the Decision Seams

How to self-distil the AnyJev Tacit-4B decision model, bake the result into
`design-pattern-tacit`, verify it, and publish it.

The entry point is one make rule — `make docker-build-tacit`. Training runs inside the
image build (BuildKit `RUN --device`), not on the host; the shipped image only serves.
`Makefile:386` drives `docker/Dockerfile.tacit`, whose `self-distill` stage runs
`scripts/decision_models/tacit_self_distill.py`.

Related documents: `docs/trainer-build-plan.md` (Proxmox LXC runbook, host prep),
`docs/trainer-build-plan-baremetal.md`, `docs/JEV_DECISION_MODEL_ANALYSIS_PLAN.md` (§7
outage invariant), `README.md` §Decision models.

## 1. What the fine-tune is

The released `morriszjm/Tacit-4B` (Apache-2.0) is the **teacher**. It writes decision
states in this server's own state shape, answers them with its own chain of thought, and a
QLoRA **student** (4-bit bitsandbytes + LoRA via `peft`) is trained to reproduce that
reasoning-conditioned label distribution in a single forward pass.

The questions come from `src/decisions/seam_prompts.py` — the same module the server asks
with at serving time, loaded by path so the trainer sees the exact production bytes
(`tacit_self_distill.py:391`). A prompt-vocabulary change therefore invalidates the cached
weights (it is one of `TACIT_TRAIN_INPUTS`).

Phases, all in one process (`tacit_self_distill.py:1492-1616`):

1. **Phase 1/3 — write states.** `problems // 10` design states plus gate states
   (`_GATE_SHARE = 10`), and `problems // 12` description-shaped style states
   (`_STYLE_SHARE = 12`; corpus train hints first, generated descriptions fill the rest).
2. **Teach.** One sequential CoT `generate` per problem, sharded across the visible GPUs
   (`teacher_answers`; `TACIT_TEACHER_SHARDS=0` = one shard per device). This is the
   dominant cost.
3. **Filter.** Rows whose teacher answer is unrouted, shape-invalid (an option omitted — never
   padded and rescaled) or below `--answer-mass-floor` (default: `--confidence-floor`) /
   `--score-mass-floor` (default `0.0`, graded targets are not filtered) are dropped per
   serving kind; a kind left with fewer than `TACIT_MIN_KIND_ROWS` rows aborts the build.
4. **Phase 3/3 — QLoRA.** Single epoch over the kept rows, then the LoRA is merged and the
   merged checkpoint is what ships (fidelity is measured on the merged weights, not on the
   adapter-wrapped student).
5. **Gate + manifest.** Per-seam student-vs-teacher agreement, weights expected-level MAE,
   and a below-floor false-act probe; results go to `self-distill.json` next to the weights.

### Seams trained

| Seam label | Kind | Served by |
|---|---|---|
| `category` | `choice` | J2 category filter on part states — the same question as `style-stage1`, on a different state distribution |
| `weights` | `score` | J4 quality-axis weights |
| `gate` | `yes_no` | J5 reasoning gate |
| `style-stage1` | `choice` | Ch2 style stage 1: the category question on description states |
| `style-stage2` | `choice` | Ch2 style stage 2: ≤26-style shortlist + `none` |
| `rerank` | `yes_no` | J1 rerank stage 2: one relevance question per candidate (stage 1 is the deterministic fused pool) |

Gated seams (`_GATED_SEAMS`, `tacit_self_distill.py:1204`) are `category`, `style-stage1`,
`style-stage2`, `rerank`, `gate`; `weights` is judged by MAE instead of agreement.

## 2. Prerequisites (GPU builder)

| Requirement | Check |
|---|---|
| NVIDIA GPU, compute capability ≥ 7.5 (vLLM ≥ 7.0, 4-bit QLoRA ≥ 7.5 + bfloat16) | `nvidia-smi --query-gpu=name,compute_cap --format=csv` |
| Host driver ≥ 575.x for the pinned `vllm/vllm-openai:v0.17.1` (CUDA 12.9.1) | `cat /proc/driver/nvidia/version` |
| Docker ≥ 25 with the `dockerfile:1.14-labs` frontend (`RUN --device` does not parse on 1.7–1.13) | `docker version` |
| CDI specs generated | `ls -l /etc/cdi/nvidia.yaml` |
| `uv` on PATH for non-interactive shells (the rule shells out to `uv run` for the payload check) | `uv --version` |
| Disk: ~400 GB (vLLM layer set, two checkpoints, build cache, ~29 GB final image) | `df -h` |
| Optional HF token, raw token only, for rate limits / gated repos | `cat ~/.cache/hf_token` |

Generate the CDI spec when missing:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
docker info --format '{{json .CDISpecDirs}}'
```

Over SSH, wrap in a login shell — non-interactive shells lack `uv`:

```bash
ssh <trainer> 'bash -lc "cd ~/design-pattern-mcp && make -n docker-build-tacit | head"'
```

## 3. Knobs

All are `?=` variables on the make rule and are forwarded as Docker build args.

| Variable | Default | Meaning |
|---|---|---|
| `TACIT_MODEL_REPO` | `morriszjm/Tacit-4B` | Teacher/base checkpoint |
| `VLLM_IMAGE_TAG` | `v0.17.1` | vLLM base image tag |
| `TACIT_PROBLEMS` | `6000` | Teacher question budget; state, gate and style counts derive from it |
| `TACIT_STATES` | `0` | Design states; 0 = `problems // 10` |
| `TACIT_STYLE_STATES` | `0` | Ch2 description states; 0 = `problems // 12` |
| `TACIT_RERANK_PROBLEMS` | `0` | J1 questions; 0 = **all** train pairs in `qa/typed/rerank_pairs.jsonl` |
| `TACIT_TEACHER_SHARDS` | `0` | 0 = one shard per visible CUDA device (capped at 8); 1 = in-process |
| `TACIT_EPOCHS` / `TACIT_LR` / `TACIT_BATCH` / `TACIT_ACCUM` | `1` / `1e-4` / `4` / `8` | QLoRA recipe |
| `TACIT_COT_MAX_TOKENS` | `2048` | Teacher CoT bound; cost scales ≈ questions × this |
| `TACIT_MAX_NEW_TOKENS` | `256` | Bound for the state/description writing passes only |
| `TACIT_MIN_SEAM_AGREEMENT` | `0.6` | Build fails when a *certified* seam (≥10 eval rows) drops below this |
| `TACIT_CONFIDENCE_FLOOR` | `0.5` | Must equal the served `DESIGN_PATTERN_DECISIONS_*_CONFIDENCE_FLOOR` |
| `TACIT_GATE_CONFIDENCE_FLOOR` | `$(TACIT_CONFIDENCE_FLOOR)` | The reasoning gate's own floor |
| `TACIT_MIN_KIND_ROWS` | `1` | Build fails when a serving kind keeps fewer rows |
| `HF_TOKEN_FILE` | `$(HOME)/.cache/hf_token` | Mounted as a BuildKit secret; never baked |

Defaults are MAX EFFORT: ~8–9k teacher questions (category/weights/gate + ~1k style + all
~1.4k rerank pairs). Expect roughly 20–30 h on one 3090 and 5–8 h on a 4-GPU host.

## 4. Procedure

### 4.1 Smoke first

```bash
make docker-build-tacit \
  TACIT_PROBLEMS=200 TACIT_RERANK_PROBLEMS=100 TACIT_COT_MAX_TOKENS=512
```

This exercises the entitlement, the CDI/`ldconfig` handshake, the teacher pass, QLoRA, the
gates and the extraction marker. Expect `too thin to certify [...]` warnings for the
low-frequency seams — a smoke build cannot certify them, and that is not a failure. Only the
full build produces a certifiable artifact.

### 4.2 Drop the smoke artifact — mandatory

The cache marker (`.extract-complete`) hashes **the train inputs only** — trainer script,
`src/decisions/seam_prompts.py`, `qa/qa.jsonl`, `qa/typed/rerank_pairs.jsonl`,
`docker/Dockerfile.tacit` (`Makefile:117-119`). It does **not** hash the knob values.
A smoke build therefore leaves a marker that validates, and the next run takes the
`runtime-distilled-cached` path and silently ships the 200-problem weights.

```bash
rm -rf models/tacit-4b
```

### 4.3 Full fine-tune

```bash
set -o pipefail
make docker-build-tacit 2>&1 | tee build-tacit.log
grep -E "teacher kept .* of .* problems|wrote distilled checkpoint" build-tacit.log
```

`DOCKER_BUILD_PROGRESS` already defaults to `plain` (`Makefile:131`), so the trainer's
`logger.info` lines land in the log. `set -o pipefail` is required: without it `tee`'s exit
status masks a failed multi-hour build, and the `grep` is the proof-of-training gate — a
cached rebuild never prints either line.

What the rule does, in order:

1. `tacit-payloads-check` — regenerates `qa/typed/rerank_pairs.jsonl` to a temp path and
   byte-compares. A drift means the corpus, the noul or the catalog changed since the pairs
   were derived: run `make tacit-payloads` and commit the new artifact.
2. `check-dockerfile-shell` — dash-syntax validation of the Dockerfile payloads.
3. Cache switch — `models/tacit-4b/{config.json,self-distill.json,.extract-complete}` all
   present ⇒ `--target runtime-distilled-cached` (no GPU); otherwise
   `--target runtime-distilled` (trains).
4. `docker build --allow device=nvidia.com/gpu=all --progress=plain` with the knobs as
   `--build-arg`, tagging `design-pattern-tacit:$(DOCKER_TAG)` and `:latest`.
5. On the train path only: `docker create` + `docker cp` extracts `/models/tacit` into
   `models/tacit-4b/`, then writes `.extract-complete` **last**, containing the image tag,
   a UTC timestamp and the sha256 of every train input. A partial copy never leaves a
   reusable directory.

### 4.4 Verify the artifact

```bash
ls -lh models/tacit-4b/                      # config.json, self-distill.json, *.safetensors
python3 -m json.tool models/tacit-4b/self-distill.json | head -40
head -3 models/tacit-4b/.extract-complete    # tag + timestamp, then sha256 lines
```

`self-distill.json` is the provenance record: `repo`, `problems`, `examples`, `kept` per
kind, `dropped` per kind and reason, `epochs`/`lr`/`batch`/`accum`, the floors, the split
seed, `states`/`gate_states`/`style_states`/`style_problems`/`rerank_problems`,
`teacher_shards`, both gate thresholds, the per-seam `seams` block and the `eval` metrics.

The build refuses to ship an under-trained seam, but certifies only what it has evidence for:
the eval split is **stratified by seam**, and a seam needs ≥10 eval rows before its agreement
(or the weights MAE) is enforced — below that the number is binomial noise, so the seam is
skipped with a `too thin to certify` warning instead of failing the build. A full build holds
~10% of 800+ rows per seam and certifies all six; the smoke shape cannot certify the
low-frequency seams (`category`, `gate`, the two style stages) and logs them as skipped.
`TACIT_MIN_KIND_ROWS` is independent and always enforced.

### 4.5 Rebuild without a GPU (optional)

The extracted checkpoint is the portable artifact, but its marker is validated against
**that host's** `TACIT_TRAIN_INPUTS` — the trainer script, `src/decisions/seam_prompts.py`,
`qa/qa.jsonl`, `qa/typed/rerank_pairs.jsonl` and `docker/Dockerfile.tacit` must be
byte-identical there. Start from an empty directory so stale shards cannot survive the copy:

```bash
ssh <build-host> 'rm -rf ~/design-pattern-mcp/models/tacit-4b'
rsync -a models/tacit-4b/ <build-host>:~/design-pattern-mcp/models/tacit-4b/
ssh <build-host> 'bash -lc "cd ~/design-pattern-mcp && make docker-build-tacit"'
```

When the inputs differ the rule refuses with `is stale` (it does not silently retrain). Then
either restore the exact bytes, or build the cached stage directly and own the provenance
claim — `docker build --target runtime-distilled-cached -f docker/Dockerfile.tacit .` — which
needs no GPU and skips the marker check. `.dockerignore` keeps `models/tacit-4b/` in the
build context; the `runtime-distilled-cached` stage `COPY`s it and fails fast when
`config.json` or `self-distill.json` is empty. The weight shards are deliberately untracked
in git — never commit them.

### 4.6 Smoke the wire contract, then calibrate

```bash
make docker-up-decisions      # sidecar + stack, one GPU
make decisions-smoke          # POST /v1/decide + GET /v1/stats on 127.0.0.1:18100
make docker-down-decisions
```

Every seam ships uncalibrated. The Ch2 style seam stays off until the floor is measured:

```bash
make docker-up-decisions
make tacit-calibrate          # two-stage replay over qa/typed/style.jsonl
```

`models/tacit-calibration.json` either recommends a floor — set
`decisions.style.confidence_floor` to it and `DECISIONS_STYLE_ENABLED=true` — or recommends
`null` (exit 2), in which case the seam stays off. Do not hand-pick a floor to make it pass.

## 5. Cache and retrain semantics

| Situation | Behaviour |
|---|---|
| All three sentinels present, marker shas match | `runtime-distilled-cached`, no GPU, no train |
| A train input changed | Rule aborts before `docker build`: `ERROR: models/tacit-4b/ is stale` |
| Marker in a legacy format (no `sha  path` lines) | `sha256sum -c` fails closed (`no properly formatted checksum lines found`) — retrain |
| Knobs changed but inputs untouched | **No invalidation.** `rm -rf models/tacit-4b` first |
| Retrain from scratch | `rm -rf models/tacit-4b && make docker-build-tacit` |
| Released weights, no training, no GPU | `make docker-build-tacit-base` (`:base` tag) |

## 6. Out-of-band training (no GPU entitlement on the builder)

Build the trainer stage, run it against a GPU, then bake the output:

```bash
docker build --target self-distill -t tacit-trainer -f docker/Dockerfile.tacit .
docker run --gpus all -v "$PWD/out:/out" tacit-trainer python3 /build/tacit_self_distill.py \
  --init-dir /models/tacit-4b --out /out/tacit-4b \
  --seam-prompts /build/seam_prompts.py --seed-texts /build/qa/qa.jsonl \
  --rerank-pairs /build/qa/rerank_pairs.jsonl
```

Then `make docker-build-tacit-base` for the released variant, or bring the checkpoint back
into the make path. The trainer writes the checkpoint and `self-distill.json`, **not** the
cache marker — only the make train path writes `.extract-complete`. Without it
`TACIT_BUILD_TARGET` stays on `runtime-distilled` and the next build retrains (or fails on a
CPU-only host), so write the marker in the same shape:

```bash
mkdir -p models/tacit-4b && cp -a out/tacit-4b/. models/tacit-4b/
{ echo "# out-of-band $(date -u +%FT%TZ)"; \
  sha256sum scripts/decision_models/tacit_self_distill.py src/decisions/seam_prompts.py \
            qa/qa.jsonl qa/typed/rerank_pairs.jsonl docker/Dockerfile.tacit; } \
  > models/tacit-4b/.extract-complete
make docker-build-tacit
```

The guard verifies the marker against **that host's** `TACIT_TRAIN_INPUTS`, so the copied
weights are only reusable where the trainer script, prompt vocabulary, `qa/qa.jsonl`, the
rerank pairs and the Dockerfile are byte-identical. Where they cannot be, bypass the guard
with `docker build --target runtime-distilled-cached -f docker/Dockerfile.tacit .` — that
builds the baked checkpoint with no GPU and no provenance check, so the assertion is yours.
`--skip` copies the released checkpoint without training — for out-of-band inspection only,
never for a ship.

## 7. Publish

```bash
make docker-publish-tacit
```

Builds first (so the pushed bytes are the ones pinned by `.extract-complete`), prints the
manifest provenance, then pushes `design-pattern-tacit:$(DOCKER_TAG)` and `:latest` to
`olkowa/design-pattern-tacit` (Docker Hub) and `ghcr.io/olk/design-pattern-tacit` (GHCR).
It refuses a manifest that does not certify every seam (≥10 eval rows each), so a smoke
build cannot ship by accident; `TACIT_ALLOW_UNCERTIFIED=1` overrides with a warning.
It is deliberately **not** part of `docker-publish-all`: the image is ~29 GB. Bump
`pyproject.toml` version first if the release is versioned — the tag comes from there.

Consume it with the compose overlay:

```bash
TACIT_IMAGE=olkowa/design-pattern-tacit:1.0.1 make docker-up-decisions
```

## 8. Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `ERROR: models/tacit-4b/ is stale` | A train input changed since extraction | Restore the exact input bytes, retrain on a GPU host, or build `--target runtime-distilled-cached` directly (no provenance check) |
| `sha256sum: ... no properly formatted checksum lines found` | Legacy marker format | Same as stale: retrain, or bypass with `--target runtime-distilled-cached` |
| A cached build shipped smoke weights | Knobs are not hashed — only `TACIT_TRAIN_INPUTS` | `rm -rf models/tacit-4b` after every smoke build (§4.2) |
| `torch.cuda.is_available() is false: no GPU visible to the trainer` | CDI device not mounted into the RUN | Confirm `--allow device=nvidia.com/gpu=all` and the `ldconfig` prefix in `docker/Dockerfile.tacit:144` |
| `teacher landed on cpu, not CUDA` | accelerate fell back to CPU | Check the CDI mount; the run refuses an hours-long CPU train |
| `too few usable teacher rows for [...]` | Floors too strict or budget too small | Lower `--answer-mass-floor`/floor, raise `TACIT_PROBLEMS` |
| `too thin to certify [...]` warning | A seam has fewer than 10 eval rows (smoke-sized build) | Not a failure — certification needs the full build |
| Seam gate failure (`under-trained seam must not ship`) | A *certified* seam's agreement is below `TACIT_MIN_SEAM_AGREEMENT` or its MAE above 0.2 | Raise volume, not the threshold |
| `runc create ... open sysctl net.ipv4.ip_unprivileged_port_start ... permission denied` | Unprivileged Proxmox LXC: `docker build` works, `docker run`/compose cannot start | Train on that host via this make rule; serve elsewhere, or use `--network host` with `TACIT_BIND_HOST=127.0.0.1` |
| First container start takes minutes | vLLM loads ~8 GB of weights; healthcheck `start_period` is 300 s | Wait for `healthy` |

## 9. File map

| Path | Role |
|---|---|
| `Makefile:386` (`docker-build-tacit`) | The rule: payload check, cache switch, build, extract, marker |
| `Makefile:65-88` | Knob defaults |
| `Makefile:117-119` | `TACIT_TRAIN_INPUTS` — the only bytes the marker hashes |
| `docker/Dockerfile.tacit:94-165` | `self-distill` stage and the GPU `RUN` |
| `docker/Dockerfile.tacit:196-213` | `runtime-released` / `runtime-distilled-cached` / `runtime-distilled` |
| `scripts/decision_models/tacit_self_distill.py` | The trainer |
| `src/decisions/seam_prompts.py` | Served prompt vocabulary, shared byte-for-byte |
| `qa/qa.jsonl` | Topic hints for state/description writing |
| `qa/typed/rerank_pairs.jsonl` | J1 stage-2 item bytes (`make tacit-payloads`) |
| `models/tacit-4b/` | Extracted checkpoint + `self-distill.json` + `.extract-complete` |
| `scripts/decision_models/tacit_calibrate.py` | Floor calibration for the style seam |
| `scripts/decision_models/decision_smoke.py` | Wire-contract smoke test |
