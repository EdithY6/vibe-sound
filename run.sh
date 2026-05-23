#!/bin/bash
# Start Streamlit (run from repo root: bash run.sh)
set -e
set -u
cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

if [ ! -d .venv ]; then
  echo "No .venv — run: bash setup.sh"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

PORT="${STREAMLIT_SERVER_PORT:-8502}"
ADDR="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
MUSIC_BACKEND="${VIBESOUND_MUSIC_BACKEND:-auto}"

echo "Starting Streamlit on ${ADDR}:${PORT} ..."
echo "Music backend: ${MUSIC_BACKEND} (set VIBESOUND_MUSIC_BACKEND=local in .env for server)"
echo "Tunnel: /tmp/cloudflared tunnel --url http://127.0.0.1:${PORT}"

exec streamlit run app.py \
  --server.port="${PORT}" \
  --server.address="${ADDR}" \
  --server.headless=true
