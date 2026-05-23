#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ ! -d .venv ]]; then
  echo "Run: bash scripts/setup.sh first"
  exit 1
fi

source .venv/bin/activate

export STREAMLIT_SERVER_PORT="${STREAMLIT_SERVER_PORT:-8501}"
export STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"

echo "Starting Streamlit on port ${STREAMLIT_SERVER_PORT}..."
echo "Try: http://imz250.ust.hk:8129/proxy/${STREAMLIT_SERVER_PORT}/"

exec streamlit run app.py \
  --server.port="${STREAMLIT_SERVER_PORT}" \
  --server.address="${STREAMLIT_SERVER_ADDRESS}" \
  --server.headless=true
