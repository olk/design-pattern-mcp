# Decision-Model Trainer Build Plan — CLM-8B + Tacit-4B in Proxmox LXC

Build-time training, serve-only images. All training runs inside the LXC container during `docker build`; final images execute only `vllm serve` + gateway. A human must review GPU passthrough and training commands before execution on the trainer.

## Prerequisites

1. Proxmox host: Proxmox VE 8.x, kernel with cgroupv2, 4x RTX 3090 visible. Ampere CC 8.6 satisfies the compute gates in `docker/Dockerfile.clm` header L37-38 (`>= 7.0`) and `docker/Dockerfile.tacit` header L37-39 (`>= 7.5 + bfloat16` for QLoRA).

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv
ls -l /dev/nvidia* /dev/nvidia-caps/*   # derive cgroup majors; do not assume
pveversion   # record; plan assumes PVE 8.x
```

2. Host NVIDIA driver MUST be >= 575.x. The pinned base `vllm/vllm-openai:v0.17.1` (`docker/Dockerfile.clm` L53, `docker/Dockerfile.tacit` L64) carries CUDA 12.9.1, whose Linux driver floor is 575.57.08 — a 550 host fails at the first `vllm serve` with "CUDA driver version is insufficient".

```bash
cat /proc/driver/nvidia/version   # e.g. NVRM 580.178.04; record exact version
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
```

3. LXC resources: 16 vCPU, 64 GB RAM, 8 GB swap, 400 GB storage (two `vllm-openai` layer sets shared, ~16 GB Qwen3-8B encoder, ~8 GB Tacit-4B, build cache, two ~25-40 GB final images), `nesting=1`, network egress to `github.com`, `huggingface.co`, `docker.io`.
4. Model sources (all pinned in the Dockerfiles; overrides only via build args):
   - CLM: `Qwen/Qwen3-8B` (`docker/Dockerfile.clm:encoder` L62 `CLM_ENCODER_REPO`) + `Contrastive-LM/CLM-v0.1-8B` file `CLM_v0.1-8B.pt` (`docker/Dockerfile.clm:heads` L82-93) + CLM git pin `d5f9ef0` (`docker/Dockerfile.clm:finetune` L98, `:runtime-base` L144).
   - Tacit: `morriszjm/Tacit-4B` (`docker/Dockerfile.tacit:weights` L73 `TACIT_MODEL_REPO`), `anyjev[client]==0.3.0` (`docker/Dockerfile.tacit:self-distill` L105, `:runtime-base` L132).
   - Bases: `python:3.11-slim` (downloader stages) + `vllm/vllm-openai:${VLLM_IMAGE_TAG}`, default `v0.17.1`.
5. Secrets: `HF_TOKEN` only if any HF repo is gated. Both Dockerfiles read it via `RUN --mount=type=secret,id=hf_token` (`docker/Dockerfile.clm` L70/L90, `docker/Dockerfile.tacit` L81) and proceed with empty token otherwise. Runtime needs no Hub access (`HF_HUB_OFFLINE=1` in `docker/Dockerfile.clm:runtime-base` L156 and `docker/Dockerfile.tacit:runtime-base` L138).
6. In-repo training inputs (must exist in the clone): `scripts/decision_models/tacit_self_distill.py`, `scripts/decision_models/export_clm_dataset.py`, `src/decisions/seam_prompts.py`, `qa/qa.jsonl`, `qa/typed/choice.jsonl`, `docker/*-entrypoint.sh` — kept in context by `.dockerignore` L19-22 negations while the rest of `scripts/` and `docker/` stays excluded.

## Training Design at Build Time

Each bullet traces base image → COPY/RUN → cached artifact → what the final image carries. Nothing below runs at `docker run`.

1. CLM `docker/Dockerfile.clm` `:encoder` (`FROM python:3.11-slim` L56): `pip install huggingface_hub`, then `snapshot_download(Qwen/Qwen3-8B)` → cached artifact `/models/qwen3-8b`.
2. CLM `docker/Dockerfile.clm` `:heads` (`FROM python:3.11-slim` L76): `hf_hub_download(Contrastive-LM/CLM-v0.1-8B, CLM_v0.1-8B.pt)` → cached artifact `/clm/heads-released/CLM_v0.1-8B.pt`.
3. CLM `docker/Dockerfile.clm` `:finetune` (`FROM vllm/vllm-openai:${VLLM_IMAGE_TAG}` L96): `COPY --from=encoder` + `COPY --from=heads` (L101-102), `git clone CLM && git checkout d5f9ef0 && pip install -e . pyarrow` (L104-109), `COPY` exporter + `qa/typed/choice.jsonl` to `/build` (L111-112), then the single GPU `RUN --device=nvidia.com/gpu=all` (L117-139): starts `vllm serve /models/qwen3-8b --runner pooling --max-model-len 2048` on `127.0.0.1:8090`, polls `/health`, runs `export_clm_dataset.py` (train split only; holdout excluded by construction) to `/clm/data/typed-choice`, runs `/opt/CLM/train/finetune.py --task choice --data ... --init-ckpt /clm/heads-released/CLM_v0.1-8B.pt --out-dir /clm/heads-trained --embed-url .../v1/embeddings --batch ${CLM_FINETUNE_BATCH}`, then kills vLLM via trap. Cached artifact: `/clm/heads-trained/`. The `--task clm` rank fine-tune is explicitly NOT in this build (header L47-51); `/v1/rank` serves from the released-head lineage.
4. CLM `docker/Dockerfile.clm` `:runtime-base` (L142): fresh `vllm-openai` shell, `pip install git+CLM@${CLM_REPO_PIN}` (L148), `COPY --from=encoder /models/qwen3-8b`, `COPY docker/clm-entrypoint.sh` (L150-152), `ENV HF_HUB_OFFLINE=1`, `EXPOSE 8700`, `HEALTHCHECK` on `127.0.0.1:8700/health` (L165-166), `ENTRYPOINT clm-entrypoint.sh` (L168).
5. CLM `docker/Dockerfile.clm` `:runtime-finetuned` (L176-178, DEFAULT as last stage): `COPY --from=finetune /clm/heads-trained /clm/heads` — the only trained artifact in the shipped image. (`:runtime-released` L171-173 COPYs the untrained `.pt` instead; fallback only.)
6. CLM serve path (`ENTRYPOINT` L168 + `docker/clm-entrypoint.sh`): asserts `/clm/heads/best_head.pt` and `/models/qwen3-8b/config.json`, backgrounds `vllm serve ${MODEL_DIR} --runner pooling --max-model-len 2048 --host 127.0.0.1 --port 8090`, polls `/health` (900 s timeout), then `exec clm-serve --host 0.0.0.0 --port 8700 --emb-url .../v1/embeddings --ckpt ... --no-download --no-ui`. Final image runs only `vllm serve` + `clm-serve` (`POST /v1/systemone`, `POST /v1/rank`, `GET /health`).
7. Tacit `docker/Dockerfile.tacit` `:weights` (`FROM python:3.11-slim` L67): `snapshot_download(morriszjm/Tacit-4B)` → cached artifact `/models/tacit-4b`.
8. Tacit `docker/Dockerfile.tacit` `:self-distill` (`FROM vllm/vllm-openai:${VLLM_IMAGE_TAG}` L87): `pip install anyjev[client]==0.3.0 peft bitsandbytes datasets` (L105), `COPY --from=weights`, `COPY tacit_self_distill.py`, `COPY src/decisions/seam_prompts.py`, `COPY qa/qa.jsonl` (L107-111), then the single GPU `RUN --device=nvidia.com/gpu=all python3 /build/tacit_self_distill.py --init-dir /models/tacit-4b --out /out/tacit-4b ...` (L113-125) with knobs `TACIT_PROBLEMS=2000 / EPOCHS=1 / LR=1e-4 / BATCH=4 / ACCUM=8 / COT_MAX=2048 / MAX_NEW=256` (L89-102). The script runs teacher state-writing + sequential per-problem CoT `generate`, drops unsharp teacher distributions, QLoRA-trains a 4-bit nf4/bf16 student with `device_map={"": 0}` (single GPU, `scripts/decision_models/tacit_self_distill.py` L304-311), then `merge_and_unload()` + `save_pretrained(..., safe_serialization=True)` plus tokenizer files and `self-distill.json` to `/out/tacit-4b` (L366-374, L424-442). Build time scales with `PROBLEMS × COT_MAX_TOKENS` (single-digit hours on one 3090 at defaults; header L46-50).
9. Tacit `docker/Dockerfile.tacit` `:runtime-base` (L128): fresh `vllm-openai` shell, `pip install anyjev[client]==0.3.0` (L132), `COPY docker/tacit-entrypoint.sh` (L134-135), `ENV HF_HUB_OFFLINE=1`, `TACIT_ADAPTIVE=false` (L138-143), `EXPOSE 8100`, `HEALTHCHECK` on `127.0.0.1:8100/v1/stats` (L149-150), `ENTRYPOINT tacit-entrypoint.sh` (L152). Never carries `/build` (header L53).
10. Tacit `docker/Dockerfile.tacit` `:runtime-distilled` (L160-162, DEFAULT): `COPY --from=self-distill /out/tacit-4b /models/tacit` — the only trained artifact in the shipped image. (`:runtime-released` L155-157 COPYs untrained weights instead; fallback only.)
11. Tacit serve path (`ENTRYPOINT` L152 + `docker/tacit-entrypoint.sh`): asserts `/models/tacit/config.json`, backgrounds `vllm serve ${MODEL_DIR} --served-model-name tacit-4b --host 127.0.0.1 --port 8000 --max-model-len 8192`, polls `/health` (900 s timeout), then `exec python3 -m anyjev.serve --model ... --upstream http://127.0.0.1:8000/v1 --host 0.0.0.0 --port 8100`. Final image runs only `vllm serve` + `anyjev.serve` (`POST /v1/decide`, `GET /v1/stats`).
12. Build mechanics both Dockerfiles share: `# syntax=docker/dockerfile:1.14-labs` (L1) is mandatory — `RUN --device` does not parse on 1.7-1.13; `RUN --gpus` does not exist. The GPU is granted by `--device=nvidia.com/gpu=all` on exactly one RUN per Dockerfile (CLM L117, Tacit L113) plus the `docker build --allow device=nvidia.com/gpu=all` entitlement (headers CLM L30, Tacit L30). Daemon side: Docker ≥ 25, `nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`, `docker info` shows `cdi: nvidia.com/gpu=all` (headers CLM L32-35, Tacit L32-35).

## Build Procedure in LXC

Run sequentially. Both GPU RUNs request `nvidia.com/gpu=all` but train single-GPU (Tacit `device_map={"": 0}`, CLM one pooling server), so parallel builds collide on GPU0. Explicitly NOT used: the `docker run --gpus all tacit-trainer ...` out-of-band recipe in `docker/Dockerfile.tacit` header L52-59 — no training at `docker run`, no training outside the LXC build.

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

5. CDI smoke + CUDA probe in the pinned image (no training spent; proves the 575.x floor before the multi-hour builds). The probe overrides the image's `["vllm","serve"]` ENTRYPOINT with `--entrypoint python3` — without it the probe runs `vllm serve python3 -c ...` and fails.

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
ls docker/Dockerfile.clm docker/Dockerfile.tacit scripts/decision_models/tacit_self_distill.py scripts/decision_models/export_clm_dataset.py qa/qa.jsonl qa/typed/choice.jsonl src/decisions/seam_prompts.py
'
```

7. Lint both Dockerfiles without spending GPU.

```bash
pct exec 220 -- bash -lc '
set -euxo pipefail; cd /root/design-pattern-mcp
docker build --check --allow device=nvidia.com/gpu=all -f docker/Dockerfile.tacit .
docker build --check --allow device=nvidia.com/gpu=all -f docker/Dockerfile.clm .
'
```

8. Build Tacit FIRST (GPU training at build time; single-digit hours at defaults). Capture the log with `BUILDKIT_PROGRESS=plain` so trainer stdout lands in the file — it is the proof-of-training gate.

```bash
pct exec 220 -- bash -lc '
set -euxo pipefail; cd /root/design-pattern-mcp
BUILDKIT_PROGRESS=plain make docker-build-tacit 2>&1 | tee build-tacit.log
'
```

```bash
# Raw form when HF auth or proxy is needed (the Makefile forwards neither secret nor proxy args):
# BUILDKIT_PROGRESS=plain docker build --allow device=nvidia.com/gpu=all \
#   -f docker/Dockerfile.tacit --target runtime-distilled \
#   --secret id=hf_token,src=/root/.cache/hf_token \
#   --build-arg VLLM_IMAGE_TAG=v0.17.1 --build-arg TACIT_MODEL_REPO=morriszjm/Tacit-4B \
#   -t design-pattern-tacit:latest .
# Quick smoke shape (do NOT ship): make docker-build-tacit TACIT_PROBLEMS=200 TACIT_COT_MAX_TOKENS=512
# Fresh-run log without wiping the world: --no-cache-filter self-distill
```

9. Build CLM SECOND, only after Tacit finishes.

```bash
pct exec 220 -- bash -lc '
set -euxo pipefail; cd /root/design-pattern-mcp
BUILDKIT_PROGRESS=plain make docker-build-clm 2>&1 | tee build-clm.log
'
```

```bash
# Raw form with the secret the Makefile omits:
# BUILDKIT_PROGRESS=plain docker build --allow device=nvidia.com/gpu=all \
#   -f docker/Dockerfile.clm --target runtime-finetuned \
#   --secret id=hf_token,src=/root/.cache/hf_token \
#   --build-arg VLLM_IMAGE_TAG=v0.17.1 \
#   --build-arg CLM_ENCODER_REPO=Qwen/Qwen3-8B \
#   --build-arg CLM_HEADS_REPO=Contrastive-LM/CLM-v0.1-8B \
#   --build-arg CLM_REPO_PIN=d5f9ef0 \
#   -t design-pattern-clm:latest .
# Fresh-run log without wiping the world: --no-cache-filter finetune
```

10. Record what was built.

```bash
pct exec 220 -- bash -lc '
docker images --format "{{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedSince}}" | grep -E "design-pattern-(tacit|clm)"
docker inspect design-pattern-tacit:latest --format "{{.Config.Entrypoint}} {{.Config.Healthcheck}}"
docker inspect design-pattern-clm:latest --format "{{.Config.Entrypoint}} {{.Config.Healthcheck}}"
'
```

## Verification and Serve

1. Prove training ran: assert the trainers' own log lines (a `CACHED` rebuild or `ls`/`torch.load` of a renamed released head also passes presence checks, so presence alone proves nothing), then diff baked weights against the `:base` images.

```bash
pct exec 220 -- bash -lc '
set -euxo pipefail; cd /root/design-pattern-mcp
grep -E "teacher kept [0-9]+ of [0-9]+ problems" build-tacit.log || { echo "GATE FAIL: no teacher-kept line in build-tacit.log" >&2; exit 1; }
grep -E "wrote distilled checkpoint to /out/tacit-4b" build-tacit.log || { echo "GATE FAIL: no wrote-checkpoint line in build-tacit.log" >&2; exit 1; }
grep -E "exported [0-9]+ train / [0-9]+ test rows" build-clm.log || { echo "GATE FAIL: no exported-rows line in build-clm.log" >&2; exit 1; }
grep -c -E "finetune.py|best_head|heads-trained" build-clm.log || true   # informational only, NOT a gate
make docker-build-tacit-base docker-build-clm-base
docker run --rm --entrypoint cat design-pattern-tacit:latest /models/tacit/self-distill.json
docker run --rm --entrypoint sha256sum design-pattern-clm:latest /clm/heads/best_head.pt
docker run --rm --entrypoint sha256sum design-pattern-clm:base /clm/heads/best_head.pt
'
```

```bash
# Expected: Tacit log shows teacher-kept + wrote-checkpoint lines (tacit_self_distill.py:290/:443);
# CLM log shows exported-rows line (export_clm_dataset.py:164); Tacit :latest carries
# /models/tacit/self-distill.json (absent in :base); CLM :latest vs :base best_head.pt hashes differ.
# Logs need BUILDKIT_PROGRESS=plain (Build Steps 8-9) or trainer stdout never reaches the file.
# "epoch %d step %d loss" (tacit_self_distill.py:341) only logs at step % 50 == 0 —
# do not gate tiny smoke builds on it; gate on "wrote distilled checkpoint" + self-distill.json.
# The finetune.py|best_head count above is informational (|| true): a no-match grep under
# set -e would otherwise abort the check on a successful build.
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

3. Serve CLM on one GPU; smoke `vllm serve --runner pooling` + `clm-serve` gateway.

```bash
pct exec 220 -- bash -lc '
set -euxo pipefail
docker run -d --rm --name clm-smoke --device nvidia.com/gpu=1 -p 127.0.0.1:18700:8700 design-pattern-clm:latest
for i in $(seq 1 90); do curl -fsS http://127.0.0.1:18700/health >/dev/null && break; sleep 10; done
curl -fsS http://127.0.0.1:18700/health
curl -fsS http://127.0.0.1:18700/v1/systemone -H "Content-Type: application/json" \
  -d "{\"model\":\"clm-latest\",\"state\":\"A checkout service with flash sales.\",\"questions\":{\"style\":{\"type\":\"choice\",\"instructions\":\"Which style fits?\",\"criteria\":{\"event-driven\":\"Event flow and reactions\",\"layered\":\"Tiered separation\"}}}}" \
  | python3 -m json.tool
curl -fsS http://127.0.0.1:18700/v1/rank -H "Content-Type: application/json" \
  -d "{\"context\":\"\",\"question\":\"Wrap external payment providers.\",\"answers\":[\"Adapter\",\"Observer\",\"Memento\"]}" \
  | python3 -m json.tool
docker stop clm-smoke
'
```

4. Serve-offline proof (both images must not phone home) + compose path check.

```bash
pct exec 220 -- bash -lc '
set -euxo pipefail
docker inspect design-pattern-tacit:latest --format "{{.Config.Env}}" | grep -o HF_HUB_OFFLINE=1
docker inspect design-pattern-clm:latest --format "{{.Config.Env}}" | grep -o HF_HUB_OFFLINE=1
docker info | grep -i -E "cdi|runtime"   # cdi serves RUN --device; nvidia runtime serves compose driver:nvidia
'
```

5. Pass criteria: both `HEALTHCHECK`s green within `start-period` (300 s + up to 900 s entrypoint timeout), all three gateway wires return 200 with ranked/choice payloads, loopback-only publishes (`127.0.0.1:18x00` — both gateways are unauthenticated per `docker/docker-compose.decision-models.yml` L29-33/L60-63). After the `nvidia-ctk runtime configure` line in Build Step 4, `make docker-up-decisions` works unchanged against the compose `driver: nvidia` reservations.

## Risks, Gaps, and Follow-ups

- R1 — Privileged LXC vs unprivileged + idmap. Plan uses `--unprivileged 0` for trainer simplicity; a privileged container with 4 GPUs and Docker is a wide blast radius. Alternative: unprivileged + `lxc.idmap` for `nvidia-caps` gids. Human must choose before Build Step 2. Owner: infra/human reviewer.
- R2 — No parallel builds. Both GPU RUNs request `nvidia.com/gpu=all` but train single-GPU (Tacit `device_map={"": 0}` in `tacit_self_distill.py` L311; CLM one pooling server in `docker/Dockerfile.clm` L119-120). Parallel builds collide on GPU0 / OOM. Plan runs them sequentially (Build Steps 8→9). Owner: operator (do not background both).
- R3 — Driver floor 575.x + exact-version user-space. Host driver must be ≥ 575.x (CUDA 12.9.1 floor), enforced by the in-image `torch.cuda` probe with `--entrypoint python3` (Build Step 5) before any multi-hour build. LXC user-space libs must match the host driver exactly (resolved by `find -name`, NOT `ldconfig -p`, and bind-mounted with the `nvidia-smi` binary in Build Step 4; the list includes `libnvidia-cfg` for NVML). Cgroup majors are computed from `stat -c '%t'`, never hardcoded. Owner: infra.
- R4 — VRAM headroom per 3090 (24 GB). Serve budgets from `docker-compose.decision-models.yml` L7-9: CLM encoder ~16 GB, Tacit ~8 GB — one sidecar per card for serve; training peaks higher. If the CLM fat RUN OOMs, lower `CLM_FINETUNE_BATCH` (default 32, `docker/Dockerfile.clm` L99); if Tacit OOMs, lower `TACIT_BATCH=4`/`TACIT_COT_MAX_TOKENS` (`docker/Dockerfile.tacit` L93/L101). Owner: ML engineer.
- R5 — Makefile forwards no `--secret` and no proxy args. Both Dockerfiles accept `hf_token` secret + `http_proxy/*` build args, but `make docker-build-tacit/clm` (Makefile L318-353) pass neither. Public repos + direct egress: fine. Gated repos or proxied trainer: use the raw `docker build` forms in Build Steps 8-9. Owner: repo maintainer (Makefile) / operator (workaround).
- R6 — Base-image tags only half-pinned. `VLLM_IMAGE_TAG=v0.17.1` is pinned; `python:3.11-slim` floats on patch, CLM `pip install -e .` deps are not locked, `peft/bitsandbytes/datasets` are not pinned (`docker/Dockerfile.tacit` L105). Rebuilds months apart may drift. Owner: repo maintainer.
- R7 — Upstream fragility: `morriszjm/Tacit-4B` + `Contrastive-LM/CLM` + `finetune.py` CLI. No fallback if the Tacit repo is renamed/gated or the CLM CLI drifts from pin `d5f9ef0`. Mitigation: the `:runtime-released` targets (`docker/Dockerfile.clm` L171-173, `docker/Dockerfile.tacit` L155-157) build with no GPU as smoke/standby images. Owner: ML engineer.
- R8 — Dockerfile-vs-Context conflict (resolved by Constraints). `docker/Dockerfile.tacit` header L52-59 documents training via `docker run --gpus all tacit-trainer ...` outside the build; this plan forbids it — all training is the in-`docker-build` `RUN --device` stages. Owner: acknowledged; no action.
- R9 — Build-cache invalidation cost + log-grep gate. `COPY qa/...` + trainer scripts sit below the heavy `git clone`/`pip install` layers; any edit re-runs the full multi-hour GPU RUN (no checkpoint resume in either trainer). Keep datasets frozen during a build; use the small-knob smoke shape for iteration. A `CACHED` rebuild must never masquerade as a fresh train — the Verification Step 1 log-grep gate (`teacher kept` / `wrote distilled checkpoint` / `exported rows`) plus the `:base` weight diff enforces this. Logs are captured with `BUILDKIT_PROGRESS=plain` so trainer stdout lands in the file; the speculative `finetune.py|best_head` grep is informational (`|| true`), never a gate. Owner: operator.
- R10 — Missing operator inputs. Clone URL for Build Step 6, Proxmox storage pool name (`local-lvm` assumed), bridge name (`vmbr0` assumed), host driver version. Owner: human reviewer must fill before execution.
- R11 — Image distribution. Plan yields images inside the trainer LXC's Docker daemon only. Pushing to a registry or `docker save` transfer to serving hosts is out of scope. Owner: infra/repo maintainer.
