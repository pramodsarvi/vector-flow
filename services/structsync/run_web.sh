#!/usr/bin/env bash
# Start the web app (backend API + frontend UI, one process).
# Usage: ./run_web.sh [--public] [--preload]
#   --public   bind 0.0.0.0 so other machines can reach it (set STRUCTSYNC_TOKEN first!)
#   --preload  load all models at start-up (slower start, faster first request)
# Env: PORT (default 8000), plus any STRUCTSYNC_* variable (see .env.example). A .env file is read if present.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  echo "error: .venv not found. Run first:  make setup   (or: bash setup_gpu.sh --download-models)" >&2
  exit 1
fi
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

HOST=127.0.0.1
for arg in "$@"; do
  case "$arg" in
    --public)  HOST=0.0.0.0
               [[ -z "${STRUCTSYNC_TOKEN:-}" ]] && echo "warning: exposing without STRUCTSYNC_TOKEN" >&2 ;;
    --preload) export STRUCTSYNC_PRELOAD=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done
PORT="${PORT:-8000}"

if ! .venv/bin/python -c "import socket,sys; socket.socket().bind((sys.argv[1], int(sys.argv[2])))" "$HOST" "$PORT" 2>/dev/null; then
  echo "error: port $PORT is already in use. Pick another:  PORT=8001 ./run_web.sh" >&2
  exit 1
fi

echo "StructuralSync → http://$([[ $HOST == 0.0.0.0 ]] && echo localhost || echo "$HOST"):$PORT   (API docs: /docs)"
exec .venv/bin/python -m uvicorn web.server:app --host "$HOST" --port "$PORT"
