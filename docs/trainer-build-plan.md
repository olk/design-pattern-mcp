# Decision-Model Trainer Build Plan — AnyJev Tacit-4B in Proxmox LXC

Build-time training, serve-only image. All training runs inside the LXC container during `docker build`; the final image executes only `vllm serve` + the anyjev gateway. A human must review GPU passthrough and training commands before execution on the trainer.

## Prerequisites

1. Proxmox host: Proxmox VE 8.x, kernel with cgroupv2, 4x RTX 3090 visible. Ampere CC 8.6 satisfies the compute gate in the `docker/Dockerfile.tacit` header (`>= 7.5 + bfloat16` for QLoRA).

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv
ls -l /dev/nvidia* /dev/nvidia-caps/*   # derive cgroup majors; do not assume
pveversion   # record; plan assumes PVE 8.x
```

2. Host NVIDIA driver MUST be >= 575.x. The pinned base `vllm/vllm-openai:v0.17.1` (`docker/Dockerfile.tacit`) carries CUDA 12.9.1, whose Linux driver floor is 575.57.08 — a 550 host fails at the first `vllm serve` with "CUDA driver version is insufficient".

```bash
cat /proc/driver/nvidia/version   # e.g. NVRM 580.178.04; record exact version
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
```

3. LXC resources: 16 vCPU, 64 GB RAM, 8 GB swap, 400 GB storage (one `vllm-openai` layer set, ~8 GB Tacit-4B, build cache, one ~25-40 GB final image), `nesting=1`, network egress to `github.com`, `huggingface.co`, `docker.io`.
4. Model sources (all pinned in the Dockerfile; overrides only via build args):
   - Tacit: `morriszjm/Tacit-4B` (`TACIT_MODEL_REPO`), `anyjev[client]==0.3.0`.
   - Bases: `python:3.11-slim` (downloader stages) + `vllm/vllm-openai:${VLLM_IMAGE_TAG}`, default `v0.17.1`.
5. Secrets: `HF_TOKEN` only if any HF repo is gated. The Dockerfile reads it via `RUN --mount=type=secret,id=hf_token` and proceeds with empty token otherwise. Runtime needs no Hub access (`HF_HUB_OFFLINE=1` in the `:runtime-base` stage).
6. In-repo training inputs (must exist in the clone): `scripts/decision_models/tacit_self_distill.py`, `src/decisions/seam_prompts.py`, `qa/qa.jsonl`, `qa/typed/rerank_pairs.jsonl`, `docker/tacit-entrypoint.sh` — kept in context by `.dockerignore` negations while the rest of `scripts/` and `docker/` stays excluded.

## Training Design at Build Time

Each bullet traces base image → COPY/RUN → cached artifact → what the final image carries. Nothing below runs at `docker run`.

1. Tacit `docker/Dockerfile.tacit` `:weights` (`FROM python:3.11-slim`): `snapshot_download(morriszjm/Tacit-4B)` → cached artifact `/models/tacit-4b`.
2. Tacit `docker/Dockerfile.tacit` `:self-distill` (`FROM vllm/vllm-openai:${VLLM_IMAGE_TAG}`): `pip install anyjev[client]==0.3.0 peft bitsandbytes datasets`, `COPY --from=weights`, `COPY tacit_self_distill.py`, `COPY src/decisions/seam_prompts.py`, `COPY qa/qa.jsonl`, `COPY qa/typed/rerank_pairs.jsonl` to `/build/qa/rerank_pairs.jsonl`, then the single GPU `RUN --device=nvidia.com/gpu=all python3 /build/tacit_self_distill.py --init-dir /models/tacit-4b --out /out/tacit-4b ...` with knobs `TACIT_PROBLEMS=6000 / TACIT_STATES=0 / TACIT_STYLE_STATES=0 / TACIT_RERANK_PROBLEMS=0 / TACIT_TEACHER_SHARDS=0 / TACIT_MIN_SEAM_AGREEMENT=0.6 / TACIT_CONFIDENCE_FLOOR=0.5 / TACIT_GATE_CONFIDENCE_FLOOR= / TACIT_MIN_KIND_ROWS=1 / TACIT_EPOCHS=1 / TACIT_LR=1e-4 / TACIT_BATCH=4 / TACIT_ACCUM=8 / TACIT_COT_MAX_TOKENS=2048 / TACIT_MAX_NEW_TOKENS=256` and flags `--style-states ... --rerank-pairs /build/qa/rerank_pairs.jsonl --rerank-problems ... --teacher-shards ... --min-seam-agreement ...`. The script runs teacher state-writing + a CoT `generate` answer pass sharded across the visible GPUs (`--teacher-shards 0` = one shard per visible CUDA device), trains all six seams from the one checkpoint — category, weights, gate, style-stage1, style-stage2 over description-shaped states, rerank over all ~1400 train pairs — builds gate states from the dataset's own descriptions in the exact `tacit.gate_state` shape, keeps `score` rows unfiltered and `choice` rows only above `TACIT_CONFIDENCE_FLOOR` (dropping any teacher result that omits an option key), and fails the build when a serving kind keeps no rows, when a per-seam student-vs-teacher agreement gate drops below 0.6, or when the weights expected-level MAE exceeds 0.2; then QLoRA-trains a 4-bit nf4/bf16 student, `merge_and_unload()` + `save_pretrained(..., safe_serialization=True)` plus tokenizer files and `self-distill.json` to `/out/tacit-4b`. Default cost: ~8-9k teacher questions, roughly 20-30 h on one 3090 / ~5-8 h with default auto-sharding on a 4-GPU host.
3. Tacit `docker/Dockerfile.tacit` `:runtime-base`: fresh `vllm-openai` shell, `pip install anyjev[client]==0.3.0`, `COPY docker/tacit-entrypoint.sh`, `ENV HF_HUB_OFFLINE=1`, `TACIT_ADAPTIVE=false`, `EXPOSE 8100`, `HEALTHCHECK` on `127.0.0.1:8100/v1/stats`, `ENTRYPOINT tacit-entrypoint.sh`. Never carries `/build`.
4. Tacit `docker/Dockerfile.tacit` `:runtime-distilled` (DEFAULT as last stage): `COPY --from=self-distill /out/tacit-4b /models/tacit` — the only trained artifact in the shipped image. (`:runtime-released` COPYs untrained weights instead; fallback only.)
5. Tacit serve path (`ENTRYPOINT` + `docker/tacit-entrypoint.sh`): asserts `/models/tacit/config.json`, backgrounds `vllm serve ${MODEL_DIR} --served-model-name tacit-4b --host 127.0.0.1 --port 8000 --max-model-len 8192`, polls `/health` (900 s timeout), then `exec python3 -m anyjev.serve --model ... --upstream http://127.0.0.1:8000/v1 --host ${TACIT_BIND_HOST:-0.0.0.0} --port 8100` (the compose publish is what confines the gateway to loopback, so a `--network host` run must pass `TACIT_BIND_HOST=127.0.0.1`, since anyjev.serve has no auth). Final image runs only `vllm serve` + `anyjev.serve` (`POST /v1/decide`, `GET /v1/stats`).
6. Build mechanics the Dockerfile relies on: `# syntax=docker/dockerfile:1.14-labs` (L1) is mandatory — `RUN --device` does not parse on 1.7-1.13; `RUN --gpus` does not exist. The GPU is granted by `--device=nvidia.com/gpu=all` on exactly one RUN plus the `docker build --allow device=nvidia.com/gpu=all` entitlement. Daemon side: Docker ≥ 25, `nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`, `docker info` shows `cdi: nvidia.com/gpu=all`.

## Build Procedure in LXC

The single GPU RUN requests `nvidia.com/gpu=all` and the teacher pass shards across the visible GPUs, so no other GPU work may run during the build. Explicitly NOT used: the `docker run --gpus all tacit-trainer ...` out-of-band recipe in the `docker/Dockerfile.tacit` header — no training at `docker run`, no training outside the LXC build.

1. Proxmox host: confirm drivers + 4x GPU visibility (Prerequisite 1-2 commands). Record the exact driver version.

2. Create the trainer LXC (host).

```bash
CTID=220
pveam update
pveam available --section system | grep debian-12   # pick the actual versioned filename; there is no "-latest" template
TMPL=$(pveam available --section system | grep -oP 'debian-12-standard_\S+_amd64\.tar\.zst' | sort -V | tail -1)
pveam download local "$TMPL"
pct create $CTID "local:vztmpl/$TMPL" \
  --hostname gpu-trainer --cores 16 --memory 65536 --swap 8192 \
  --rootfs local-lvm:400 --net0 name=eth0,bridge=vmbr0,ip=dhcp \
  --features nesting=1,keyctl=1 --unprivileged 0 --start 0
# privileged (unprivileged 0) is the simple trainer shape; see R1 for the unprivileged alternative
```

3. Pass through all 4x 3090 devices (host, container stopped). Compute cgroup majors from `stat -c '%t'` (hex → decimal); never hardcode — a trainer with different UVM/caps majors would be denied at container start.

```bash
CTID=220
ls -l /dev/nvidia* /dev/nvidia-caps/*
MAJ_GPU=$((16#$(stat -c '%t' /dev/nvidia0)))
MAJ_UVM=$((16#$(stat -c '%t' /dev/nvidia-uvm)))
MAJ_CAP=$((16#$(stat -c '%t' /dev/nvidia-caps/nvidia-cap1)))
echo "majors: gpu=$MAJ_GPU uvm=$MAJ_UVM caps=$MAJ_CAP"   # typically 195/509/234
printf 'lxc.cgroup2.devices.allow: c %s:* rwm\n' "$MAJ_GPU" "$MAJ_UVM" "$MAJ_CAP" >> /etc/pve/lxc/$CTID.conf
cat >> /etc/pve/lxc/$CTID.conf <<'EOF'
lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
lxc.mount.entry: /dev/nvidia1 dev/nvidia1 none bind,optional,create=file
lxc.mount.entry: /dev/nvidia2 dev/nvidia2 none bind,optional,create=file
lxc.mount.entry: /dev/nvidia3 dev/nvidia3 none bind,optional,create=file
lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-caps dev/nvidia-caps none bind,optional,create=dir
EOF
pct start $CTID
pct exec $CTID -- ls -l /dev/nvidia*
```

4. Inside LXC: exact-version user-space + Docker ≥ 25 + buildx + CDI + runtime. The nested dockerd resolves `RUN --device=nvidia.com/gpu=all` against the LXC filesystem, so `nvidia-ctk cdi generate` must run INSIDE the LXC; a host-generated spec's lib paths won't resolve.

```bash
DRV=$(grep -oP 'Module\s+\K[0-9.]+' /proc/driver/nvidia/version | head -1); echo "host driver: $DRV"
CTID=220
pct exec $CTID -- stat -c '%t %n' /dev/nvidia-uvm   # hex major; cross-checks the computed Step 3 majors
pct stop $CTID
# Resolve exact-version host libs by filename: ldconfig -p lists only sonames (libcuda.so.1),
# never libcuda.so.$DRV, so a grep-ldconfig lookup always reports MISSING. find() cannot miss.
for lib in libcuda.so.$DRV libnvidia-ml.so.$DRV libnvidia-cfg.so.$DRV libnvidia-ptxjitcompiler.so.$DRV libnvidia-nvvm.so.$DRV; do
  hostpath=$(find /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu -name "$lib" -print -quit 2>/dev/null)
  [ -n "$hostpath" ] || { echo "MISSING on host: $lib" >&2; exit 1; }
  echo "lxc.mount.entry: $hostpath usr/lib/x86_64-linux-gnu/$lib none bind,optional,create=file" >> /etc/pve/lxc/$CTID.conf
done
# The nvidia-smi binary itself: it dlopens libnvidia-ml at runtime (ldd shows no nvidia deps),
# so the explicit lib list above — not ldd-chasing — is what makes it work inside the LXC.
SMI=$(command -v nvidia-smi)
echo "lxc.mount.entry: $SMI usr/bin/nvidia-smi none bind,optional,create=file" >> /etc/pve/lxc/$CTID.conf
pct start $CTID
pct exec $CTID -- bash -lc 'ldconfig && ldconfig -p | grep -E "libcuda|libnvidia-ml" && nvidia-smi -L'
```

```bash
pct exec 220 -- bash -lc '
set -euxo pipefail
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#" > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin nvidia-container-toolkit
command -v nvidia-ctk && docker buildx version
nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
docker info | grep -i -E "cdi|runtime"
'
```

5. CDI smoke + CUDA probe in the pinned image (no training spent; proves the 575.x floor before the multi-hour build). The probe overrides the image's `["vllm","serve"]` ENTRYPOINT with `--entrypoint python3` — without it the probe runs `vllm serve python3 -c ...` and fails.

```bash
pct exec 220 -- bash -lc '
set -euxo pipefail
printf "FROM busybox\nRUN --device=nvidia.com/gpu=all ls -l /dev/nvidia0\n" > /tmp/Dockerfile.gpu-smoke
docker build --allow device=nvidia.com/gpu=all -f /tmp/Dockerfile.gpu-smoke /tmp
# --entrypoint python3 overrides the stock ["vllm","serve"] ENTRYPOINT:
docker run --rm --entrypoint python3 --device nvidia.com/gpu=0 vllm/vllm-openai:v0.17.1 -c \
  "import torch; assert torch.cuda.is_available(), \"no cuda\"; print(torch.cuda.get_device_name(0), torch.version.cuda)"
'
```

6. Clone the repo inside the LXC (the required trainer workflow).

```bash
pct exec 220 -- bash -lc '
set -euxo pipefail
git clone <repo-url> /root/design-pattern-mcp
cd /root/design-pattern-mcp
git rev-parse HEAD
ls docker/Dockerfile.tacit docker/tacit-entrypoint.sh scripts/decision_models/tacit_self_distill.py src/decisions/seam_prompts.py qa/qa.jsonl qa/typed/rerank_pairs.jsonl
'
```

7. Lint the Dockerfile without spending GPU.

```bash
pct exec 220 -- bash -lc '
set -euxo pipefail; cd /root/design-pattern-mcp
docker build --check --allow device=nvidia.com/gpu=all -f docker/Dockerfile.tacit .
'
```

8. Build Tacit (GPU training at build time; ~5-8 h with default auto-sharding on the 4-GPU host, ~20-30 h on one 3090). `make docker-build-tacit` runs `tacit-payloads-check` first and refuses a stale cached host dir (`sha256sum -c models/tacit-4b/.extract-complete`). Capture the log with `BUILDKIT_PROGRESS=plain` so trainer stdout lands in the file — it is the proof-of-training gate.

```bash
pct exec 220 -- bash -lc '
set -euxo pipefail; cd /root/design-pattern-mcp
BUILDKIT_PROGRESS=plain make docker-build-tacit 2>&1 | tee build-tacit.log
'
```

```bash
# Raw form when a proxy is needed (the Makefile forwards the HF secret but no proxy args):
# BUILDKIT_PROGRESS=plain docker build --allow device=nvidia.com/gpu=all \
#   -f docker/Dockerfile.tacit --target runtime-distilled \
#   --secret id=hf_token,src=/root/.cache/hf_token \
#   --build-arg VLLM_IMAGE_TAG=v0.17.1 --build-arg TACIT_MODEL_REPO=morriszjm/Tacit-4B \
#   -t design-pattern-tacit:latest .
# Quick smoke shape (do NOT ship): make docker-build-tacit TACIT_PROBLEMS=200 TACIT_COT_MAX_TOKENS=512
# Fresh-run log without wiping the world: --no-cache-filter self-distill
```

9. Record what was built.

```bash
pct exec 220 -- bash -lc '
docker images --format "{{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedSince}}" | grep -E "design-pattern-tacit"
docker inspect design-pattern-tacit:latest --format "{{.Config.Entrypoint}} {{.Config.Healthcheck}}"
'
```

## Verification and Serve

1. Prove training ran: assert the trainer's own log lines (a `CACHED` rebuild also passes presence checks, so presence alone proves nothing), then diff baked weights against the `:base` image.

```bash
pct exec 220 -- bash -lc '
set -euxo pipefail; cd /root/design-pattern-mcp
grep -E "teacher kept [0-9]+ of [0-9]+ problems" build-tacit.log || { echo "GATE FAIL: no teacher-kept line in build-tacit.log" >&2; exit 1; }
grep -E "wrote distilled checkpoint to /out/tacit-4b" build-tacit.log || { echo "GATE FAIL: no wrote-checkpoint line in build-tacit.log" >&2; exit 1; }
# Every seam carried problems, and the per-seam fidelity gates ran (no SystemExit in the log).
grep -E "problems: category/score/gate [0-9]+, style [0-9]+ \(states [0-9]+\), rerank [0-9]+" build-tacit.log || { echo "GATE FAIL: no problem-composition line" >&2; exit 1; }
grep -E "an under-trained seam must not ship|has no eval rows" build-tacit.log && { echo "GATE FAIL: a seam gate failed" >&2; exit 1; } || true
make docker-build-tacit-base
docker run --rm --entrypoint cat design-pattern-tacit:latest /models/tacit/self-distill.json
'
```

```bash
# Expected: the Tacit log shows teacher-kept + wrote-checkpoint lines; the per-seam
# agreement gates (>= 0.6) and the weights expected-level MAE gate (<= 0.2) fail the
# build below their floors, so a passing build already cleared them.
# Tacit :latest carries /models/tacit/self-distill.json (absent in :base).
# Logs need BUILDKIT_PROGRESS=plain (Build Step 8) or trainer stdout never reaches the file.
# "epoch %d step %d loss" only logs at step % 50 == 0 —
# do not gate tiny smoke builds on it; gate on "wrote distilled checkpoint" + self-distill.json.
```

2. Serve Tacit on one GPU; smoke `vllm serve` + `anyjev.serve` gateway (CDI-native run; wires mirror `make decisions-smoke`).

```bash
pct exec 220 -- bash -lc '
set -euxo pipefail
docker run -d --rm --name tacit-smoke --device nvidia.com/gpu=0 -p 127.0.0.1:18100:8100 design-pattern-tacit:latest
for i in $(seq 1 60); do curl -fsS http://127.0.0.1:18100/v1/stats >/dev/null && break; sleep 10; done
curl -fsS http://127.0.0.1:18100/v1/stats | python3 -m json.tool
curl -fsS http://127.0.0.1:18100/v1/decide -H "Content-Type: application/json" \
  -d "{\"state\":\"System goal: serve order views\\nDecision part: invalidate cached views\",\"question\":\"Which design-pattern category best describes the concern of this decision part?\",\"options\":[\"persistence-state\",\"behavior-selection\"],\"kind\":\"choice\"}" \
  | python3 -m json.tool
docker stop tacit-smoke
'
```

3. Serve-offline proof (the image must not phone home) + compose path check.

```bash
pct exec 220 -- bash -lc '
set -euxo pipefail
docker inspect design-pattern-tacit:latest --format "{{.Config.Env}}" | grep -o HF_HUB_OFFLINE=1
docker info | grep -i -E "cdi|runtime"   # cdi serves RUN --device; nvidia runtime serves compose driver:nvidia
'
```

4. Pass criteria: the `HEALTHCHECK` green within `start-period` (300 s + up to 900 s entrypoint timeout), the gateway wire returns 200 with a choice payload, loopback-only publish (`127.0.0.1:18100` — the gateway is unauthenticated). After the `nvidia-ctk runtime configure` line in Build Step 4, `make docker-up-decisions` works unchanged against the compose `driver: nvidia` reservations.

## Cached-Checkpoint Provenance

`make docker-build-tacit` extracts the distilled checkpoint to `models/tacit-4b/` once; later
runs re-bake that host dir with no GPU. The `.extract-complete` marker records the image
tag + date and the sha256 of every train input (`scripts/decision_models/tacit_self_distill.py`,
`src/decisions/seam_prompts.py`, `qa/qa.jsonl`, `qa/typed/rerank_pairs.jsonl`,
`docker/Dockerfile.tacit`). A cached rebuild verifies the marker with `sha256sum -c` and refuses a
stale host dir: if you change the corpus or the trainer, `rm -rf models/tacit-4b` and
rebuild to retrain. `models/tacit-calibration.json` (from `make tacit-calibrate`) is the
second piece of evidence and lives outside the checkpoint dir — it belongs to a serving run,
not to the checkpoint.

Two guards keep the rerank-pair artifact honest, because the catalog itself is not
hashed (7290 files): `qa/typed/rerank_pairs.jsonl` is hashed into the extract marker, so a
regenerated pair file invalidates a cached checkpoint build; and `docker-build-tacit` depends on
`make tacit-payloads-check`, which re-derives the pairs to a temp path and byte-compares,
so a catalog edit that was never re-derived fails the build instead of replaying
`document_text` the server no longer sends.

## Calibrate Before Enabling the Style Seam

The distilled checkpoint ships uncalibrated: `decisions.style.confidence_floor` defaults to
0.5 with no measurement behind it, and the compose overlay therefore keeps
`DECISIONS_STYLE_ENABLED=false`. Enable the seam only after a calibration run names a
floor.

```bash
make docker-up-decisions          # or start only design-pattern-tacit + design-pattern-mcp
make tacit-calibrate              # replays the 30 held-out style rows against :18100
```

`tacit-calibrate` writes `models/tacit-calibration.json` and prints a `recommended.floor`:
the lowest floor whose acted answers clear `TACIT_CAL_MIN_ACTED_ACCURACY` at an acted
rate of at least `TACIT_CAL_MIN_ACTED_RATE`, with ECE/Brier and the confidence
distribution alongside. If nothing qualifies it exits 2 — a checkpoint that cannot act
accurately must not be enabled, and the release step says so instead of guessing. Set
the floor in the same command as the enable:

```bash
DECISIONS_STYLE_ENABLED=true DECISIONS_STYLE_CONFIDENCE_FLOOR=<recommended.floor> make docker-up-decisions
```

Re-run after every retrain — the floor belongs to the weights, not the repo. Note the
seam's own A/B against the LLM style is `candidate-loss`
(`docs/JEV_BENCHMARK_REPORT.md`), so an accurate checkpoint is necessary but not sufficient:
the gate measures the model, not whether acting on it helps selection.

## Measure the Seam Before Trusting the Checkpoint

The build gates bound the checkpoint; they do not measure the seam. That measurement is
`make benchmark-style-ab`, which runs the pre-registered A/B twice through the full
stack with the decision overlay — `DECISIONS_STYLE_ENABLED=false` then `=true` with the
calibrated floor (read from `models/tacit-calibration.json`; the target refuses to run
without it) — over `tests/benchmark/scenarios/full.json`, `SPLIT=holdout`, `REPEAT=3`,
and hands both run dirs to `benchmark-compare` (+15% holdout p95 rule). Both arms use the
same corpus, split, repeat count and images, which is what makes the comparison
pre-registered rather than exploratory. Record the verdict next to the calibration
artifact; `candidate-loss` means keep the seam off regardless of checkpoint accuracy.

## Risks, Gaps, and Follow-ups

- R1 — Privileged LXC vs unprivileged + idmap. Plan uses `--unprivileged 0` for trainer simplicity; a privileged container with 4 GPUs and Docker is a wide blast radius. Alternative: unprivileged + `lxc.idmap` for `nvidia-caps` gids. Human must choose before Build Step 2. Owner: infra/human reviewer.
- R2 — No concurrent GPU work during the build. The single GPU RUN requests `nvidia.com/gpu=all` and the teacher pass shards across every visible GPU by default, so a second build or serve on the same host collides and OOMs. Plan runs the one build alone (Build Step 8). Owner: operator (do not background anything else on the GPUs).
- R3 — Driver floor 575.x + exact-version user-space. Host driver must be ≥ 575.x (CUDA 12.9.1 floor), enforced by the in-image `torch.cuda` probe with `--entrypoint python3` (Build Step 5) before any multi-hour build. LXC user-space libs must match the host driver exactly (resolved by `find -name`, NOT `ldconfig -p`, and bind-mounted with the `nvidia-smi` binary in Build Step 4; the list includes `libnvidia-cfg` for NVML). Cgroup majors are computed from `stat -c '%t'`, never hardcoded. Owner: infra.
- R4 — VRAM headroom per 3090 (24 GB). Serve budget: Tacit ~8 GB — one sidecar per card for serve; training peaks higher. If Tacit OOMs, lower `TACIT_BATCH=4`/`TACIT_COT_MAX_TOKENS`. Owner: ML engineer.
- R5 — Makefile forwards the HF secret but no proxy args. `make docker-build-tacit` passes `--secret id=hf_token,src=$(HF_TOKEN_FILE)` whenever `~/.cache/hf_token` is non-empty (and fails loudly for an explicitly-set missing path), but it forwards no `http_proxy/*` build args. Public repos + direct egress: fine. Gated repos: drop the token file in place. Proxied trainer: use the raw `docker build` form in Build Step 8. Owner: repo maintainer (Makefile) / operator (workaround).
- R6 — Base-image tags only half-pinned. `VLLM_IMAGE_TAG=v0.17.1` is pinned; `python:3.11-slim` floats on patch, `peft/bitsandbytes/datasets` are not pinned. Rebuilds months apart may drift. Owner: repo maintainer.
- R7 — Upstream fragility: `morriszjm/Tacit-4B`. No fallback if the repo is renamed or gated. Mitigation: the `:runtime-released` target builds with no GPU as a smoke/standby image. Owner: ML engineer.
- R8 — Dockerfile-vs-Context conflict (resolved by Constraints). The `docker/Dockerfile.tacit` header documents training via `docker run --gpus all tacit-trainer ...` outside the build; this plan forbids it — all training is the in-`docker-build` `RUN --device` stages. Owner: acknowledged; no action.
- R9 — Build-cache invalidation cost + log-grep gate. `COPY qa/...` + trainer scripts sit below the heavy `pip install` layers; any edit re-runs the full multi-hour GPU RUN (no checkpoint resume in the trainer). Keep datasets frozen during a build; use the small-knob smoke shape for iteration. A `CACHED` rebuild must never masquerade as a fresh train — the Verification Step 1 log-grep gate (`teacher kept` / `wrote distilled checkpoint`) plus the `:base` weight diff enforces this. Logs are captured with `BUILDKIT_PROGRESS=plain` so trainer stdout lands in the file. Owner: operator.
- R10 — Missing operator inputs. Clone URL for Build Step 6, Proxmox storage pool name (`local-lvm` assumed), bridge name (`vmbr0` assumed), host driver version. Owner: human reviewer must fill before execution.
- R11 — Image distribution. Plan yields images inside the trainer LXC's Docker daemon only. Pushing to a registry or `docker save` transfer to serving hosts is out of scope. Owner: infra/repo maintainer.
