#!/usr/bin/env bash
# Start JupyterLab with the environment Isaac Sim needs. Kernels inherit it.
#   usage: scripts/jupyter.sh [port]   (default 8001, the forwarded port; extra jupyter args after the port)
# From the Mac:  ssh -N -L <port>:localhost:<port> <dev-node>, then open the printed URL.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${1:-8001}"
shift || true

# libGLU.so.1 (no sudo on the dev node); without it Kit hangs at startup
export LD_LIBRARY_PATH="$PWD/deps/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMNI_KIT_ACCEPT_EULA=YES

echo "Tunnel from your laptop:  ssh -N -L $PORT:localhost:$PORT $(hostname)"
exec uv run jupyter lab --no-browser --ip 127.0.0.1 --port "$PORT" --notebook-dir "$PWD" "$@"
