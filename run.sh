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

PORT="${STREAMLIT_SERVER_PORT:-8501}"
ADDR="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"

echo "Starting Streamlit on ${ADDR}:${PORT} ..."
echo "Open: http://imz250.ust.hk:8129/proxy/${PORT}/"

exec streamlit run app.py \
  --server.port="${PORT}" \
  --server.address="${ADDR}" \
  --server.headless=true
