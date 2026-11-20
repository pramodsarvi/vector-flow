#!/usr/bin/env bash
# RunPod start script. Works as the container start command of the StructuralSync image:
#     bash /app/runpod_entrypoint.sh
# or on a stock RunPod PyTorch pod after extracting the tarball into /workspace/structuralsync:
#     bash runpod_entrypoint.sh
# Everything mutable lives on the persistent volume (/workspace) so a pod restart keeps projects,
# catalogue edits and the ~13 GB of model weights.
set -euo pipefail
APP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOL="${RUNPOD_VOLUME:-/workspace}"
[[ -d "$VOL" && -w "$VOL" ]] || { echo "warning: $VOL missing/not writable; using /tmp (nothing persists)" >&2; VOL=/tmp/structsync; mkdir -p "$VOL"; }

export STRUCTSYNC_DATA="${STRUCTSYNC_DATA:-$VOL/data}"
export STRUCTSYNC_CATALOG="${STRUCTSYNC_CATALOG:-$VOL/catalog}"
export HF_HOME="${HF_HOME:-$VOL/hf}"
export STRUCTSYNC_PRELOAD="${STRUCTSYNC_PRELOAD:-1}"
export PYTHONUNBUFFERED=1
PORT="${PORT:-8000}"
mkdir -p "$STRUCTSYNC_DATA" "$HF_HOME"

# public proxy URL => refuse to run open
if [[ -z "${STRUCTSYNC_TOKEN:-}" && "${ALLOW_NO_TOKEN:-0}" != 1 ]]; then
  echo "error: set STRUCTSYNC_TOKEN (pod env var). RunPod proxy URLs are public. (ALLOW_NO_TOKEN=1 overrides)" >&2
  exit 1
fi

# seed the catalogue once; later edits/imports persist on the volume
if [[ ! -f "$STRUCTSYNC_CATALOG/catalog.json" ]]; then
  echo "seeding catalogue -> $STRUCTSYNC_CATALOG"
  mkdir -p "$STRUCTSYNC_CATALOG" && cp -a "$APP/catalog/." "$STRUCTSYNC_CATALOG/"
fi

# Python: use the image's interpreter, else build a venv on the volume (stock pod fallback)
PY=python
if ! python -c "import fastapi, torch, transformers" 2>/dev/null; then
  PY="$VOL/venv/bin/python"
  if [[ ! -x "$PY" ]]; then
    echo "installing pinned dependencies into $VOL/venv (first start only, several minutes)"
    pip install -q uv
    uv venv --python 3.12 "$VOL/venv"
    uv pip install -p "$PY" -r "$APP/requirements.lock"
  fi
fi

# model weights: download once into the volume
if [[ ! -f "$HF_HOME/.prefetched" ]]; then
  echo "downloading model weights -> $HF_HOME"
  (cd "$APP" && "$PY" prefetch.py) && touch "$HF_HOME/.prefetched"
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "warning: no GPU visible" >&2
cd "$APP"
exec "$PY" -m uvicorn web.server:app --host 0.0.0.0 --port "$PORT"
