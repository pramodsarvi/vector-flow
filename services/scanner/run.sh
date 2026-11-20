#!/usr/bin/env bash
# Start the Spatial Scanner service (API + web viewer, one process).
#
#   ./run.sh                 http://127.0.0.1:8765
#   PORT=9000 ./run.sh       another port
#   ./run.sh --public        listen on all interfaces (set SCANNER_TOKEN first)
#   ./run.sh --reload        auto-reload on source changes (development)
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "No virtualenv found at .venv" >&2
  echo "Create one with:  make setup      (or: uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt)" >&2
  exit 1
fi

HOST="127.0.0.1"
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --public) HOST="0.0.0.0" ;;
    --reload) EXTRA+=(--reload) ;;
    *) EXTRA+=("$arg") ;;
  esac
done

PORT="${PORT:-8765}"

if [[ "$HOST" == "0.0.0.0" && -z "${SCANNER_TOKEN:-}" ]]; then
  echo "Refusing to listen on all interfaces without SCANNER_TOKEN set." >&2
  echo "Set it in .env (see .env.example) so the API is not open to your whole network." >&2
  exit 1
fi

echo "Spatial Scanner  ->  http://${HOST}:${PORT}/viewer/     (API docs: /docs)"
exec "$PYTHON" -m uvicorn engine.service.app:app --host "$HOST" --port "$PORT" "${EXTRA[@]}"
