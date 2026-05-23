#!/bin/bash
# bash fix_blank_page.sh YOUR-SUBDOMAIN.trycloudflare.com
set -e
cd "$(dirname "$0")"

HOST="${1:-}"
if [ -z "$HOST" ]; then
  echo "Usage: bash fix_blank_page.sh YOUR-SUBDOMAIN.trycloudflare.com"
  exit 1
fi
HOST="${HOST#https://}"
HOST="${HOST%%/*}"

# shellcheck disable=SC1091
source .venv/bin/activate
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
export VIBESOUND_MUSIC_BACKEND="${VIBESOUND_MUSIC_BACKEND:-local}"
export STREAMLIT_BROWSER_SERVER_ADDRESS="$HOST"
export STREAMLIT_BROWSER_SERVER_PORT=443

pkill -9 -f streamlit 2>/dev/null || true
sleep 2

echo "Restarting app for link: https://$HOST"
streamlit run app.py \
  --server.port=8502 \
  --server.address=127.0.0.1 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false &

sleep 8
echo "Done. Refresh your browser tab (Cmd+Shift+R)."
