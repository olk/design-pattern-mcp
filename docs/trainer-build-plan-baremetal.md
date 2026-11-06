# Decision-Model Trainer Build Plan — AnyJev Tacit-4B on Bare Metal

Build-time training, serve-only image. All training runs on the bare-metal trainer during `docker build`; the final image executes only `vllm serve` + the anyjev gateway. No Proxmox, no LXC, no `pct` — Docker runs directly on the host OS with the NVIDIA driver installed natively. A human must review training commands before execution on the trainer. Commands assume a shell with Docker access (root, or a user in the `docker` group after re-login); `apt`/`systemctl`/`nvidia-ctk` lines use `sudo`.

## Prerequisites

1. Bare-metal trainer: Ubuntu 24.04 LTS or Debian 12, 4x RTX 3090 visible. Ampere CC 8.6 satisfies the compute gate in the `docker/Dockerfile.tacit` header (`>= 7.5 + bfloat16` for QLoRA).

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv
ls -l /dev/nvidia* /dev/nvidia-caps/*
uname -r; . /etc/os-release && echo "$ID $VERSION_CODENAME"   # record; plan covers ubuntu noble / debian bookworm
```

2. Host NVIDIA driver MUST be >= 575.x. The pinned base `vllm/vllm-openai:v0.17.1` (`docker/Dockerfile.tacit`) carries CUDA 12.9.1, whose Linux driver floor is 575.57.08 — a 550 host fails at the first `vllm serve` with "CUDA driver version is insufficient".

```bash
cat /proc/driver/nvidia/version   # e.g. NVRM 580.178.04; record exact version
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
```

3. Host resources: 16 CPU, 64 GB RAM, 8 GB swap, 400 GB free on the Docker data root (`/var/lib/docker` by default; one `vllm-openai` layer set, ~8 GB Tacit-4B, build cache, one ~25-40 GB final image), network egress to `github.com`, `huggingface.co`, `docker.io`. No other GPU workloads during the build.
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

## Build Procedure on Bare Metal

The single GPU RUN requests `nvidia.com/gpu=all` and the teacher pass shards across the visible GPUs, so no other GPU work may run during the build. Explicitly NOT used: the `docker run --gpus all tacit-trainer ...` out-of-band recipe in the `docker/Dockerfile.tacit` header — no training at `docker run`, no training outside `docker build`.

1. Confirm drivers + 4x GPU visibility (Prerequisite 1-2 commands). Record the exact driver version. If `nvidia-smi` already reports >= 575.x, skip to Step 3.

2. Install the NVIDIA driver >= 575.x (only if missing or below floor). Driver and user-space come from the same package set — no bind-mounts, no version matching across a container boundary.

```bash
set -euxo pipefail
. /etc/os-release
case "$ID" in
  ubuntu)
    ubuntu-drivers devices   # pick the recommended branch at >= 575.x (expect nvidia-driver-580+)
    sudo apt update && sudo apt install -y "nvidia-driver-580"
    ;;
  debian)
    # bookworm-backports alone may not reach 575.57.08 — check before trusting it:
    grep -r bookworm-backports /etc/apt/sources.list* || \
      echo "deb http://deb.debian.org/debian bookworm-backports main contrib non-free non-free-firmware" \
        | sudo tee /etc/apt/sources.list.d/backports.list
    sudo apt update
    apt-cache policy nvidia-driver | head -20
    # If no candidate >= 575.x: use NVIDIA's CUDA repo (https://developer.nvidia.com/cuda-downloads,
    # "deb (network)" for Debian 12) or the .run installer — do NOT proceed on a < 575.x driver:
    #   wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
    #   sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
    sudo apt install -y nvidia-driver firmware-misc-nonfree
    ;;
  *) echo "unsupported ID=$ID; need Ubuntu >= 24.04 or Debian 12 path" >&2; exit 1 ;;
esac
sudo reboot
```

```bash
set -euxo pipefail
nvidia-smi -L   # must list 4x 3090 + driver >= 575.x before proceeding
```

3. Install Docker ≥ 25 + buildx + NVIDIA Container Toolkit, then generate the CDI spec (`nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`). CDI serves **both** sides: `RUN --device=nvidia.com/gpu=all` at build time and the overlay's `devices: ["nvidia.com/gpu=N"]` request at run time — the legacy `nvidia` runtime is no longer needed by this stack.

```bash
set -euxo pipefail
. /etc/os-release   # $ID/$VERSION_CODENAME: ubuntu/noble or debian/bookworm
sudo apt update && sudo apt install -y curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL "https://download.docker.com/linux/$ID/gpg" | sudo tee /etc/apt/keyrings/docker.asc >/dev/null
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/$ID $VERSION_CODENAME stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#" \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit
sudo apt install -y git make curl ca-certificates   # Step 5 git-clone + Step 7 make; curl/ca-certificates for key fetches
sudo usermod -aG docker "$USER" || true
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# Re-login (or: newgrp docker) BEFORE the lines below — the group change is not active in this shell:
newgrp docker <<'EOF'
command -v nvidia-ctk && docker buildx version   # nvidia-ctk ships INSIDE nvidia-container-toolkit
docker compose version   # from docker-compose-plugin; unknown without it
docker info | grep -i -E "cdi|runtime"   # expect cdi: nvidia.com/gpu=all AND runtimes include nvidia
docker version --format "{{.Server.Version}}"   # expect >= 25
EOF
```

4. CDI smoke + CUDA probe in the pinned image (no training spent; proves the 575.x floor before the multi-hour builds). The probe overrides the image's `["vllm","serve"]` ENTRYPOINT with `--entrypoint python3` — without it the probe runs `vllm serve python3 -c ...` and fails.

```bash
set -euxo pipefail
printf "FROM busybox\nRUN --device=nvidia.com/gpu=all ls -l /dev/nvidia0\n" > /tmp/Dockerfile.gpu-smoke
docker build --allow device=nvidia.com/gpu=all -f /tmp/Dockerfile.gpu-smoke /tmp
# --entrypoint python3 overrides the stock ["vllm","serve"] ENTRYPOINT:
docker run --rm --entrypoint python3 --device nvidia.com/gpu=0 vllm/vllm-openai:v0.17.1 -c \
  "import torch; assert torch.cuda.is_available(), \"no cuda\"; print(torch.cuda.get_device_name(0), torch.version.cuda)"
```

5. Clone the repo on the trainer (the required workflow).

```bash
set -euxo pipefail
git clone <repo-url> ~/design-pattern-mcp
cd ~/design-pattern-mcp
git rev-parse HEAD
ls docker/Dockerfile.tacit scripts/decision_models/tacit_self_distill.py src/decisions/seam_prompts.py qa/qa.jsonl qa/typed/rerank_pairs.jsonl docker/tacit-entrypoint.sh
```

6. Lint the Dockerfile without spending GPU.

```bash
set -euxo pipefail; cd ~/design-pattern-mcp
docker build --check --allow device=nvidia.com/gpu=all -f docker/Dockerfile.tacit .
```

7. Build Tacit (GPU training at build time; ~5-8 h with default auto-sharding on the 4-GPU host, ~20-30 h on one 3090). `make docker-build-tacit` runs `tacit-payloads-check` and `check-dockerfile-shell` first and refuses a stale cached host dir (`sha256sum -c models/tacit-4b/.extract-complete`). Capture the log with `BUILDKIT_PROGRESS=plain` so trainer stdout lands in the file — it is the proof-of-training gate.

```bash
set -euxo pipefail; cd ~/design-pattern-mcp
BUILDKIT_PROGRESS=plain make docker-build-tacit 2>&1 | tee build-tacit.log
```

```bash
# Raw form when a proxy is needed (the Makefile forwards the HF secret but no proxy args):
# BUILDKIT_PROGRESS=plain docker build --allow device=nvidia.com/gpu=all \
#   -f docker/Dockerfile.tacit --target runtime-distilled \
#   --secret id=hf_token,src=~/.cache/hf_token \
#   --build-arg VLLM_IMAGE_TAG=v0.17.1 --build-arg TACIT_MODEL_REPO=morriszjm/Tacit-4B \
#   -t design-pattern-tacit:latest .
# Quick smoke shape (do NOT ship): make docker-build-tacit TACIT_PROBLEMS=200 TACIT_COT_MAX_TOKENS=512
# Fresh-run log without wiping the world: --no-cache-filter self-distill
```

8. Record what was built.

```bash
set -euxo pipefail
docker images --format "{{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedSince}}" | grep -E "design-pattern-tacit"
docker inspect design-pattern-tacit:latest --format "{{.Config.Entrypoint}} {{.Config.Healthcheck}}"
```

## Verification and Serve

1. Prove training ran: assert the trainer's own log lines (a `CACHED` rebuild also passes presence checks, so presence alone proves nothing), then diff baked weights against the `:base` image.

```bash
set -euxo pipefail; cd ~/design-pattern-mcp
grep -E "teacher kept [0-9]+ of [0-9]+ problems" build-tacit.log || { echo "GATE FAIL: no teacher-kept line in build-tacit.log" >&2; exit 1; }
grep -E "wrote distilled checkpoint to /out/tacit-4b" build-tacit.log || { echo "GATE FAIL: no wrote-checkpoint line in build-tacit.log" >&2; exit 1; }
# Every seam carried problems, and the per-seam fidelity gates ran (no SystemExit in the log).
grep -E "problems: category/score/gate [0-9]+, style [0-9]+ \(states [0-9]+\), rerank [0-9]+" build-tacit.log || { echo "GATE FAIL: no problem-composition line" >&2; exit 1; }
grep -E "an under-trained seam must not ship|has no eval rows" build-tacit.log && { echo "GATE FAIL: a seam gate failed" >&2; exit 1; } || true
make docker-build-tacit-base
docker run --rm --entrypoint cat design-pattern-tacit:latest /models/tacit/self-distill.json
```

```bash
# Expected: the Tacit log shows teacher-kept + wrote-checkpoint lines; the per-seam
# agreement gates (>= 0.6) and the weights expected-level MAE gate (<= 0.2) fail the
# build below their floors, so a passing build already cleared them.
# Tacit :latest carries /models/tacit/self-distill.json (absent in :base).
# Logs need BUILDKIT_PROGRESS=plain (Build Step 7) or trainer stdout never reaches the file.
# "epoch %d step %d loss" only logs at step % 50 == 0 —
# do not gate tiny smoke builds on it; gate on "wrote distilled checkpoint" + self-distill.json.
```

2. Serve Tacit on one GPU; smoke `vllm serve` + `anyjev.serve` gateway (CDI-native run; wires mirror `make decisions-smoke`).

```bash
set -euxo pipefail
docker run -d --rm --name tacit-smoke --device nvidia.com/gpu=0 -p 127.0.0.1:18100:8100 design-pattern-tacit:latest
for i in $(seq 1 60); do curl -fsS http://127.0.0.1:18100/v1/stats >/dev/null && break; sleep 10; done
curl -fsS http://127.0.0.1:18100/v1/stats | python3 -m json.tool
curl -fsS http://127.0.0.1:18100/v1/decide -H "Content-Type: application/json" \
  -d "{\"state\":\"System goal: serve order views\\nDecision part: invalidate cached views\",\"question\":\"Which design-pattern category best describes the concern of this decision part?\",\"options\":[\"persistence-state\",\"behavior-selection\"],\"kind\":\"choice\"}" \
  | python3 -m json.tool
docker stop tacit-smoke
```

3. Serve-offline proof (the image must not phone home) + compose path check.

```bash
set -euxo pipefail
docker inspect design-pattern-tacit:latest --format "{{.Config.Env}}" | grep -o HF_HUB_OFFLINE=1
docker info | grep -i -E "cdi|runtime"   # cdi serves RUN --device; nvidia runtime serves compose driver:nvidia
```

4. Pass criteria: the `HEALTHCHECK` green within `start-period` (300 s + up to 900 s entrypoint timeout), the gateway wire returns 200 with a choice payload, loopback-only publish (`127.0.0.1:18100` — the gateway is unauthenticated). `make docker-up-decisions` is a dev-stack command, not part of this plan's proof: it also starts the MCP server + TEI services, whose images this plan never builds (`make docker-build-all`), so only the sidecar smoke in Step 2 gates the build — the compose GPU path is proven separately by the CDI spec from Build Step 3 plus `docker info` showing `cdi: nvidia.com/gpu=…`.

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

## Host Runtime Preconditions

`docker compose ... up` needs a net namespace it can configure. When the host cannot
write the sysctl Docker asks for, **every** sandboxed container fails at create time
(reported on two different 4×3090 hosts):

```
Error response from daemon: failed to create task for container: …
OCI runtime create failed: runc create failed: … open sysctl
net.ipv4.ip_unprivileged_port_start file: reopen fd 8: permission denied
```

The Docker Engine we could inspect (29.9.2 on the first host) asks for
`net.ipv4.ip_unprivileged_port_start=0` in each sandbox's netns so non-root images can
bind low ports — the version that introduced that was not bisected, so treat "recent
Docker" as the trigger rather than a specific release. The sysctl is not in
`HostConfig` (`docker inspect` shows `Sysctls: null`) — it is applied by the network
driver, so `--sysctl`, `--user 0:0`, `--privileged` and
`--security-opt systempaths=unconfined` all fail identically. `docker build` is
unaffected (build containers get no sandbox netns), which is why the image build above
succeeds on a host where `docker run` does not.

### Classify the host before changing anything

Five different causes produce that identical message, so run the write test first:

```bash
# 1. can this host's root write the sysctl at all?
echo 0 > /proc/sys/net/ipv4/ip_unprivileged_port_start   # → permission denied = cause A
# 2. which daemon is even running?
docker info --format '{{json .SecurityOptions}} rootless={{.Rootless}}'
grep -o '"userns-remap"[^,}]*' /etc/docker/daemon.json 2>/dev/null
uname -r                                                 # a '-pve'/'-lxc' suffix = guest
# 3. is the guest itself confined by a generated LXC profile?
cat /proc/self/attr/current                              # an 'lxc-…' label, not 'unconfined' = cause D
docker version --format '{{.Server.Version}}'; runc --version | head -1
```

- **A. The host itself is namespaced** (`/proc/sys/net/**` read-only for root: an
  unprivileged Proxmox LXC, or a guest whose `/proc/sys` is mounted read-only). Leaf
  net-namespace sysctls are writable only by a root whose user namespace owns that netns,
  so nothing inside such a guest can grant it: fix it on the Proxmox host
  (`unprivileged: 0`), or run Docker on the physical host, or use the workaround below.
  Verified on the first host (step 1 → `Keine Berechtigung`). Note the converse does not
  hold: `unprivileged: 0` alone does not make `docker run` work on a current PVE host —
  that is cause D, which is denied by path, not by capability.
- **B. Rootless `dockerd`** (`rootless=true`, `DOCKER_HOST=unix:///run/user/<uid>/docker.sock`).
  Not verified here; the check is one line, and the fix is to run the system daemon
  (`systemctl enable --now docker`) instead of `dockerd-rootless`.
- **C. `userns-remap` in `daemon.json`** (the grep prints a mapping). Not verified here
  either — treat it as a suspect the one-line check clears cheaply, not as a diagnosis.
  The fix is to drop the remap for this host, or run these services with
  `--userns=host` / `userns_mode: host`, or use the workaround below.
- **D. The sysctl write is path-denied by the guest's LXC AppArmor profile** — Docker
  ≥ 20.10 with runc ≥ 1.3, on a Proxmox CT, **privileged (`unprivileged: 0`) included**.
  The engine injects `net.ipv4.ip_unprivileged_port_start=0` into every container it gives a
  **new** net namespace (`daemon/oci_linux.go`: `if c.HostConfig.NetworkMode.IsPrivate()`,
  "only if we create a new net namespace" — moby PR #41030; no daemon option turns it off).
  runc applies it by *reopening the procfs node through a detached procfs mount*
  (`internal/sys/sysctl_linux.go`: `procfs.OpenUnsafeProcRoot` → `pathrs.Reopen(…, O_WRONLY|O_TRUNC|O_CLOEXEC)`,
  which is where the "fd 8" in the message comes from). AppArmor
  resolves paths inside a detached mount from `d_name`, so
  `/proc/sys/net/ipv4/ip_unprivileged_port_start` is mediated as
  `/sys/net/ipv4/ip_unprivileged_port_start`. Proxmox confines every CT with a *generated*
  profile — the rules live in the `lxc-pve` binary, not in `/etc/apparmor.d/`, and the `/sys`
  write denials sit in an audit-quiet block, so **no `dmesg` DENIED line appears** and the
  write returns EACCES. Tracking: opencontainers/runc#4968 ("CVE-2025-52881: fd reopening
  causes issues with AppArmor profiles"), lxc/lxc#4606 (Proxmox: the restrictions are dropped
  in the *nesting* case).
  Ordering is what makes it unfixable from the container side: runc registers the container's
  profile as a **change on exec** (`libcontainer/apparmor/apparmor_linux.go`:
  `ApplyProfile` → `changeOnExec`, writing `/proc/self/attr/apparmor/exec`) and only then
  calls `sys.WriteSysctls` (`libcontainer/standard_init_linux.go`), so at the moment of the
  write the label is still the **inherited** one — the CT's generated profile. That is why
  `--security-opt apparmor=unconfined`, a compose `security_opt`, `--privileged` and
  `--sysctl` all change nothing, and why the same daemon on bare metal (inherited label
  `unconfined`) works.
  This is the one cause where the write test in step 1 is **not** a discriminator: a
  privileged CT owns its netns, so `echo 0 > …` succeeds while `docker run` still fails
  identically. Distinguish D from B/C by step 3 — an `lxc-…` label from
  `/proc/self/attr/current` plus a passing write test means profile-level.
- **E. The runtime's own reopen of the sysctl handle is denied** — runc 1.2.x and
  1.4.0+ (1.5.2 is the current upstream release) fail, **runc 1.3.0 passes**, and the
  failure is independent of AppArmor and of capabilities. Reproduced on the 3090 CT with a
  20-line OCI bundle (a new netns plus `linux.sysctl: {net.ipv4.ip_unprivileged_port_start:
  "0"}`, rootfs = one static binary): the same bundle *without* the sysctl starts; with it,
  it fails with the message above. `strace -y` shows the failing pair — the sysctl write is
  never attempted:

  ```
  openat2(11</>, "./sys/net/ipv4/ip_unprivileged_port_start",
          {O_PATH, RESOLVE_NO_XDEV|RESOLVE_NO_MAGICLINKS|RESOLVE_BENEATH}) = 8
  openat(12</1/task/1/fd>, "8", O_WRONLY|O_TRUNC|O_CLOEXEC) = -1 EACCES
  ```

  Version matrix on that bundle: 1.3.0 passes; 1.2.8, 1.4.0, 1.5.0, 1.5.1, 1.5.2 fail; the
  key does not matter (`net.ipv4.ping_group_range` fails identically), nor does the pid
  namespace (dropping `pid` from `linux.namespaces` still fails) or `noNewPrivileges`. On
  the same CT `/proc/self/attr/current` is `unconfined` for PID 1 and every shell, and root
  **can** write the sysctl directly (`echo 0 > /proc/sys/net/ipv4/ip_unprivileged_port_start`
  → rc 0), so causes A and D do not apply there: do not "fix" this with a profile, a
  capability or a privileged conversion.
  Fix: run the containers under an OCI runtime that writes the sysctl without the detached
  reopen — **crun** (verified 1.21: the failing bundle exits 0 and `docker run` on a bridge
  network then reads `net.ipv4.ip_unprivileged_port_start = 0` inside the container). With
  docker, register it and make it the default runtime, then recreate the containers:

  ```bash
  # /etc/docker/daemon.json (merge, keep any existing keys)
  #   "runtimes": {"crun": {"path": "/usr/bin/crun"}}, "default-runtime": "crun"
  systemctl restart docker
  docker compose … down && docker compose … up -d   # stored runtime is part of the container
  ```

  Containers created before the switch keep `HostConfig.Runtime: runc` and fail exactly as
  before — a daemon change alone looks like it did nothing until they are recreated.

Whichever of A–E applies, the discriminator is the same: step 1 failing means the cause
is host-level and no daemon or compose setting can fix it; step 1 succeeding while
`docker run` still fails means the cause is daemon-level (B or C), profile-level (D) or
runtime-level (E) — read step 3 and the runc version before touching `privileged:`.

Three ways out, in order of preference:

1. **Run Docker where the netns the engine configures is not denied** — bare metal, a
   root daemon, or a daemon without `userns-remap`; on a Proxmox CT, cause D has to be
   cleared host-side first, and `unprivileged: 0` is *not* the lever that clears it:
   - Preferred: update the host's `lxc-pve` (fixed in `6.0.5-2`, shipped with PVE 9.1; the
     8.4 line takes its newest package) and keep `features: nesting=1,keyctl=1` — the fixed
     generated profile drops the `/sys` denials **only in the nesting case**.
   - If the host cannot be updated: add the documented per-CT AppArmor off switch to
     `/etc/pve/lxc/<CTID>.conf`, then do a full stop/start (the profile is generated at CT
     start; `pct reboot` does not re-read it):

     ```bash
     pct set <CTID> -features nesting=1,keyctl=1
     grep -q '^lxc\.apparmor\.profile' /etc/pve/lxc/<CTID>.conf || \
       echo 'lxc.apparmor.profile: unconfined' >> /etc/pve/lxc/<CTID>.conf
     pct stop <CTID> && pct start <CTID>
     # verified by: pct exec <CTID> -- docker run --rm alpine sysctl net.ipv4.ip_unprivileged_port_start
     #   → prints "… = 0" on a fixed host, the OCI/runc error above on an unfixed one
     ```

     `pct` prints `explicitly configured lxc.apparmor.profile overrides the following
     settings: features:nesting` — expected, `unconfined` is a superset. If dockerd then
     reports `Could not check if docker-default AppArmor profile was loaded: … open
     /sys/kernel/security/apparmor/profiles: permission denied`, also add
     `lxc.mount.entry: /dev/null sys/module/apparmor/parameters/enabled none bind 0 0` and
     stop/start again. Both lines are a workaround: drop them and re-verify once the host
     runs a fixed `lxc-pve`. Do **not** edit `/etc/apparmor.d/abstractions/lxc/*` on a PVE
     host — the generated profile ignores it. Converting an unprivileged CT in place
     (`pct set <CTID> --unprivileged 0`) still bricks its uid-shifted rootfs (cause A's
     `vzdump` → `pct restore` sequence is the only safe conversion) and is not needed for D.

   Nothing in this repo changes either way, and the compose overlay works as written.
2. **Run the sidecar with `--network host` and bind it to loopback.** Host networking
   has no sandbox netns, so no sysctl write, but it also removes the port publish that
   the overlay's loopback-only rule depends on — the gateway listens on `0.0.0.0` and
   has no auth (`anyjev.serve` has none), so the bind knob is mandatory, never optional:

```bash
# Tacit sidecar, host networking, loopback-only (needed by `make tacit-calibrate`).
# DOCKER_TAG is pyproject.toml's version — 1.0.1 at this writing; `make` would expand
# $(DOCKER_TAG), a shell will not, so pass the literal tag here.
docker run -d --name design-pattern-tacit-host --network host \
    --device nvidia.com/gpu=all -e TACIT_BIND_HOST=127.0.0.1 \
    design-pattern-tacit:${DOCKER_TAG:-1.0.1}
# the gateway binds 18100 only if you also pass -e TACIT_PORT=18100;
# the image default is 8100
make tacit-calibrate TACIT_BASE_URL=http://127.0.0.1:8100 \
    TACIT_SERVED_CONTAINER=design-pattern-tacit-host
```

`TACIT_SERVED_CONTAINER` is what `docker inspect` names for the report's `served` block; it
defaults to the compose service's `design-pattern-tacit-dev`, so this lane must pass the
`docker run` name above — otherwise the report records null provenance for a perfectly good
run.

`docker-up-decisions` itself cannot be rescued by the knobs: compose publishes
`127.0.0.1:18100` from a bridge network, so the stack needs cause A/B/C/D fixed, or the
runtime lever below. The host-network recipe is the sidecar-only path, and it is what
calibration needs. With host networking the MCP service would reach the sidecar at
`127.0.0.1:8100` instead of the compose service name, so a full
host-network stack would need `DECISIONS_STYLE_BASE_URL`,
`DECISIONS_WEIGHTS_BASE_URL`, `DECISIONS_CATEGORY_BASE_URL` and
`DECISIONS_REASONING_GATE_BASE_URL` overridden — a four-URL overlay rewrite, which
is why we did not ship one.

3. **Change the OCI runtime (cause E only, verified on the 3090 CT).** `crun` writes the
   sysctl without runc's detached reopen, so a bridge stack starts where runc cannot:

   ```bash
   # /etc/docker/daemon.json — merge into the existing object, keep "runtimes.nvidia"
   #   "runtimes": { "crun": { "path": "/usr/bin/crun" } }, "default-runtime": "crun"
   systemctl restart docker
   docker compose -f docker/docker-compose.yml -f docker/docker-compose.decision-models.yml down
   make docker-up-decisions          # stored runtime is per container: `down` is required
   ```

   The catch is GPU delivery. The legacy reservation (`deploy.resources.reservations.devices`
   with `driver: nvidia`) only injects `/dev/nvidia*` and the driver libraries when the
   container's runtime is `nvidia-container-runtime` (a runc shim); under crun the sidecar
   starts with no CUDA and vLLM dies with `CUDA unknown error`. The overlay therefore asks
   for the GPU by **CDI device name** (`devices: ["nvidia.com/gpu=0"]`), which the daemon
   resolves into the OCI spec and which works under runc *and* crun (both verified). A GPU
   that is bound into the CT but missing from its device allowlist fails the same way —
   check the node inside the CT before blaming the runtime:

   ```bash
   pct exec <CTID> -- sh -c 'head -c0 /dev/nvidia-uvm; echo uvm_rc=$?'   # EPERM = allowlist gap
   # the CT config needs BOTH majors, not just the 195 (GPU) one:
   #   lxc.cgroup2.devices.allow: c 195:* rwm     # nvidiactl / nvidiaN
   #   lxc.cgroup2.devices.allow: c 511:* rwm     # nvidia-uvm / nvidia-uvm-tools
   #   lxc.cgroup2.devices.allow: c 237:* rwm     # nvidia-caps
   # a legacy `lxc.cgroup.devices.allow: a` (cgroup v1 syntax) is a no-op under cgroup v2
   ```

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

- R1 — No isolation on bare metal. Builds run on the host Docker daemon with `gpu=all`; a runaway training RUN can pin the GPUs and host RAM for hours, and a driver upgrade needs a host reboot. Run nothing else on the GPUs during Build Step 7. Owner: operator.
- R2 — No concurrent GPU work during the build. The single GPU RUN requests `nvidia.com/gpu=all` and the teacher pass shards across every visible GPU by default, so a second build or serve on the same host collides and OOMs. Plan runs the one build alone (Build Step 7). Owner: operator (do not background anything else on the GPUs).
- R3 — Driver floor 575.x on the host. Host driver must be ≥ 575.x (CUDA 12.9.1 floor), enforced by the in-image `torch.cuda` probe with `--entrypoint python3` (Build Step 4) before any multi-hour build. No cross-boundary user-space matching: the driver package installs kernel module and libs together (Build Step 2). Owner: infra.
- R4 — VRAM headroom per 3090 (24 GB). Serve budget: Tacit ~8 GB — one sidecar per card for serve; training peaks higher. If Tacit OOMs, lower `TACIT_BATCH=4`/`TACIT_COT_MAX_TOKENS`. Owner: ML engineer.
- R5 — Makefile forwards the HF secret but no proxy args. `make docker-build-tacit` passes `--secret id=hf_token,src=$(HF_TOKEN_FILE)` whenever `~/.cache/hf_token` is non-empty (and fails loudly for an explicitly-set missing path), but it forwards no `http_proxy/*` build args. Public repos + direct egress: fine. Gated repos: drop the token file in place. Proxied trainer: use the raw `docker build` form in Build Step 7. Owner: repo maintainer (Makefile) / operator (workaround).
- R6 — Base-image tags only half-pinned. `VLLM_IMAGE_TAG=v0.17.1` is pinned; `python:3.11-slim` floats on patch, `peft/bitsandbytes/datasets` are not pinned. Rebuilds months apart may drift. Owner: repo maintainer.
- R7 — Upstream fragility: `morriszjm/Tacit-4B`. No fallback if the repo is renamed or gated. Mitigation: the `:runtime-released` target builds with no GPU as a smoke/standby image. Owner: ML engineer.
- R8 — Dockerfile-vs-Constraints conflict (resolved). The `docker/Dockerfile.tacit` header documents training via `docker run --gpus all tacit-trainer ...` outside the build; this plan forbids it — all training is the in-`docker-build` `RUN --device` stages. Owner: acknowledged; no action.
- R9 — Build-cache invalidation cost + log-grep gate. `COPY qa/...` + trainer scripts sit below the heavy `pip install` layers; any edit re-runs the full multi-hour GPU RUN (no checkpoint resume in the trainer). Keep datasets frozen during a build; use the small-knob smoke shape for iteration. A `CACHED` rebuild must never masquerade as a fresh train — the Verification Step 1 log-grep gate (`teacher kept` / `wrote distilled checkpoint`) plus the `:base` weight diff enforces this. Logs are captured with `BUILDKIT_PROGRESS=plain` so trainer stdout lands in the file. Owner: operator.
- R10 — Missing operator inputs. Clone URL for Build Step 5, host OS variant if not Ubuntu 24.04 / Debian 12, host driver version. Owner: human reviewer must fill before execution.
- R11 — Image distribution. Plan yields images inside the trainer host's Docker daemon only. Pushing to a registry or `docker save` transfer to serving hosts is out of scope. Owner: infra/repo maintainer.
