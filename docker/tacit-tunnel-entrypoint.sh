#!/bin/sh
# tacit-tunnel entrypoint: one ssh -L forward, fail-loud on any misconfiguration.
#
# Required env (all from the compose service; none has a default worth guessing):
#   TACIT_TUNNEL_SSH_HOST   GPU host, as the ssh client resolves it (alias or address)
#   TACIT_TUNNEL_SSH_USER   login user on the GPU host (key auth, no passwords)
# Key material arrives as read-only mounts at /ssh (see compose): id_ed25519 plus
# known_hosts. Refusing to run without host-key verification is deliberate — an ssh
# tunnel that trusts on first use is a MITM slot in front of an unauthenticated API.
set -eu

: "${TACIT_TUNNEL_SSH_HOST:?set TACIT_TUNNEL_SSH_HOST (GPU host alias or address)}"
: "${TACIT_TUNNEL_SSH_USER:?set TACIT_TUNNEL_SSH_USER (login user on the GPU host)}"

if [ ! -f /ssh/id_ed25519 ]; then
    echo "tacit-tunnel: /ssh/id_ed25519 missing — mount the deployment key read-only" >&2
    exit 1
fi
if [ ! -f /ssh/known_hosts ]; then
    echo "tacit-tunnel: /ssh/known_hosts missing — pin the GPU host key, no TOFU" >&2
    exit 1
fi

exec ssh -N \
    -i /ssh/id_ed25519 \
    -o UserKnownHostsFile=/ssh/known_hosts \
    -o StrictHostKeyChecking=yes \
    -o IdentitiesOnly=yes \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -L "0.0.0.0:8100:127.0.0.1:8100" \
    "${TACIT_TUNNEL_SSH_USER}@${TACIT_TUNNEL_SSH_HOST}"
