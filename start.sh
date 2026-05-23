#!/bin/bash
# bash start.sh  — one terminal, keeps running, prints public link
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

if [ ! -x /tmp/cloudflared ]; then
  echo "==> Downloading cloudflared..."
  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /tmp/cloudflared
  chmod +x /tmp/cloudflared
fi

CF_LOG=/tmp/vibesound-cf.log
rm -f "$CF_LOG"

echo "==> Starting public tunnel..."
/tmp/cloudflared tunnel --url http://127.0.0.1:8502 --protocol http2 2>&1 | tee "$CF_LOG" &
CF_PID=$!

TUNNEL_HOST=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20 25 30; do
  TUNNEL_HOST=$(grep -oE '[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" | head -1 || true)
  if [ -n "$TUNNEL_HOST" ]; then
    break
  fi
  sleep 1
done

if [ -z "$TUNNEL_HOST" ]; then
  echo "ERROR: Could not get trycloudflare URL from tunnel log."
  kill "$CF_PID" 2>/dev/null || true
  exit 1
fi

export STREAMLIT_BROWSER_SERVER_ADDRESS="$TUNNEL_HOST"
export STREAMLIT_BROWSER_SERVER_PORT=443

echo "==> Starting VibeSound for https://$TUNNEL_HOST ..."
streamlit run app.py \
  --server.port=8502 \
  --server.address=127.0.0.1 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false &

ST_PID=$!
sleep 8
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8502/ | grep -q 200; then
    break
  fi
  sleep 2
done

echo ""
echo "============================================================"
echo "  OPEN THIS LINK (incognito / private window):"
echo "  https://$TUNNEL_HOST"
echo "  KEEP THIS WINDOW OPEN"
echo "============================================================"
echo ""

wait "$CF_PID" || true
kill "$ST_PID" 2>/dev/null || true
