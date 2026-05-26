#!/bin/bash
# VibeSound — fresh container → public https link (trycloudflare + Streamlit)
# Edit HF_TOKEN below, then:  bash deploy.sh
set -euo pipefail

# ================== EDIT ==================
HF_TOKEN="${HF_TOKEN:-hf_PASTE_YOUR_TOKEN_HERE}"
APP_FILE="${APP_FILE:-app_clap.py}"
PORT="${PORT:-8503}"
# ==========================================

REPO_DIR="${REPO_DIR:-$HOME/vibe-sound}"
REPO_URL="${REPO_URL:-https://github.com/EdithY6/vibe-sound.git}"

echo "========== VibeSound deploy =========="

# --- repo ---
if [ ! -d "$REPO_DIR/.git" ]; then
  echo ">> git clone"
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin
git reset --hard origin/main

shopt -s nullglob
for f in *.sh app*.py music_gen.py ui_theme.py .env.example .env; do
  [ -f "$f" ] && sed -i 's/\r$//' "$f" 2>/dev/null || true
done
shopt -u nullglob

# --- .env ---
[ -f .env ] || { [ -f .env.example ] && cp .env.example .env; } || touch .env
if [ -n "$HF_TOKEN" ] && [[ "$HF_TOKEN" == hf_* ]]; then
  if grep -q '^HF_TOKEN=' .env 2>/dev/null; then
    sed -i "s|^HF_TOKEN=.*|HF_TOKEN=$HF_TOKEN|" .env
  else
    echo "HF_TOKEN=$HF_TOKEN" >> .env
  fi
fi
grep -q '^VIBESOUND_MUSIC_BACKEND=' .env 2>/dev/null || echo 'VIBESOUND_MUSIC_BACKEND=local' >> .env
grep -q '^VIBESOUND_AUDIO_FORMAT=' .env 2>/dev/null || echo 'VIBESOUND_AUDIO_FORMAT=mp4' >> .env
sed -i 's/\r$//' .env 2>/dev/null || true

if ! grep -q '^HF_TOKEN=hf_' .env 2>/dev/null; then
  echo "ERROR: Set HF_TOKEN at top of deploy.sh (or in .env)"
  exit 1
fi

if [ ! -f "$APP_FILE" ]; then
  echo "ERROR: $APP_FILE not in repo — upload to GitHub first"
  exit 1
fi
if [ "$APP_FILE" = "app_clap.py" ] && [ ! -f ui_theme.py ]; then
  echo "ERROR: ui_theme.py missing (required by app_clap.py)"
  exit 1
fi

# --- python ---
if [ ! -d .venv ]; then
  echo ">> setup.sh (first time, ~10–20 min)"
  bash setup.sh
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> Pin Streamlit <1.40 (fixes blank trycloudflare page)"
pip -q install 'streamlit>=1.32.0,<1.40.0' -r requirements.txt
python -c "import streamlit as st; print('Streamlit', st.__version__)"

if command -v nvidia-smi >/dev/null 2>&1; then
  python -c "import torch; print('CUDA:', torch.cuda.is_available())" || true
  if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo ">> CUDA PyTorch (cu124)"
    pip -q install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  fi
fi

if ! command -v ffmpeg >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
  conda install -y -c conda-forge ffmpeg 2>/dev/null || true
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

python -c "import ui_theme; print('ui_theme ok')"

pkill -f "streamlit run" 2>/dev/null || true
pkill -f cloudflared 2>/dev/null || true
sleep 2

if [ ! -x /tmp/cloudflared ]; then
  curl -fsSL -o /tmp/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /tmp/cloudflared
fi

# --- 1) tunnel first (get hostname) ---
CF_LOG=/tmp/vibesound-cf.log
rm -f "$CF_LOG"
echo ">> cloudflared → 127.0.0.1:${PORT}"
/tmp/cloudflared tunnel --url "http://127.0.0.1:${PORT}" 2>&1 | tee "$CF_LOG" &
CF_PID=$!

TUNNEL_HOST=""
for _ in $(seq 1 45); do
  TUNNEL_HOST=$(grep -oE '[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" | head -1 || true)
  [ -n "$TUNNEL_HOST" ] && break
  sleep 1
done
if [ -z "$TUNNEL_HOST" ]; then
  echo "ERROR: no trycloudflare host in log:"
  tail -30 "$CF_LOG"
  kill "$CF_PID" 2>/dev/null || true
  exit 1
fi

export STREAMLIT_BROWSER_SERVER_ADDRESS="$TUNNEL_HOST"
export STREAMLIT_BROWSER_SERVER_PORT=443

# --- 2) streamlit in background (must start AFTER tunnel host is known) ---
echo ">> streamlit run $APP_FILE"
streamlit run "$APP_FILE" \
  --server.port="$PORT" \
  --server.address=127.0.0.1 \
  --server.headless=true \
  --server.enableCORS=true \
  --server.enableXsrfProtection=false \
  --browser.serverAddress="$TUNNEL_HOST" \
  --browser.serverPort=443 \
  > /tmp/vibesound-st.log 2>&1 &
ST_PID=$!

READY=0
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PORT}/_stcore/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if curl -sf -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/" 2>/dev/null | grep -qE '200|30[0-9]'; then
    READY=1
    break
  fi
  if ! kill -0 "$ST_PID" 2>/dev/null; then
    echo "ERROR: Streamlit exited. Log:"
    tail -40 /tmp/vibesound-st.log
    kill "$CF_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 1
done

if [ "$READY" -ne 1 ]; then
  echo "ERROR: Streamlit not responding on :$PORT"
  tail -40 /tmp/vibesound-st.log
  kill "$ST_PID" "$CF_PID" 2>/dev/null || true
  exit 1
fi

echo ""
echo "============================================================"
echo "  LOCAL OK:  http://127.0.0.1:${PORT}"
echo "  OPEN THIS: https://${TUNNEL_HOST}"
echo "  (incognito / hard refresh — NOT http://...:443)"
echo "  Logs: tail -f /tmp/vibesound-st.log"
echo "  KEEP THIS WINDOW OPEN"
echo "============================================================"
echo ""

# block on tunnel; cleanup on exit
trap 'kill "$ST_PID" 2>/dev/null || true' EXIT
wait "$CF_PID"
