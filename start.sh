#!/bin/bash
# One command: bash start.sh
# Keeps this window open and prints your public link.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Run setup first: bash setup.sh"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
export VIBESOUND_MUSIC_BACKEND="${VIBESOUND_MUSIC_BACKEND:-local}"

echo "==> Stopping old copies..."
pkill -9 -f streamlit 2>/dev/null || true
pkill -9 -f cloudflared 2>/dev/null || true
sleep 2

echo "==> Starting VibeSound app (port 8502)..."
streamlit run app.py \
  --server.port=8502 \
  --server.address=127.0.0.1 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false &

echo "    Waiting for app..."
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8502/ | grep -q 200; then
    echo "    App is ready."
    break
  fi
  sleep 2
done

if ! curl -s -o /dev/null http://127.0.0.1:8502/; then
  echo "ERROR: App did not start. Check .env and run: bash setup.sh"
  exit 1
fi

if [ ! -x /tmp/cloudflared ]; then
  echo "==> Downloading cloudflared (first time only)..."
  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /tmp/cloudflared
  chmod +x /tmp/cloudflared
fi

echo ""
echo "============================================================"
echo "  KEEP THIS WINDOW OPEN"
echo "  Copy the https://....trycloudflare.com link below"
echo "  Open it in Chrome/Safari (use a new incognito tab)"
echo "============================================================"
echo ""

exec /tmp/cloudflared tunnel --url http://127.0.0.1:8502 --protocol http2
