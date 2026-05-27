# VibeSound — School cloud server deploy (copy/paste)

Streamlit app that turns a **Reel photo (+ optional caption)** into **background music**.

**Entrypoint:** `app_clap.py`

## What it does (current `main`)

- **Image caption**: `nlpconnect/vit-gpt2-image-captioning` (ViT-GPT2)
- **Mood classifier (gated)**: `MelodyWEN7/vibesound-music-mood-classifier`
- **Prompt builder**: CLAP tag ranking (`laion/clap-htsat-fused`) over a curated music tag vocabulary
- **Music generation**: MusicGen
  - local `facebook/musicgen-small` when CUDA is available
  - fallback: public `facebook/MusicGen` HF Space queue (`VIBESOUND_MUSIC_BACKEND=space`)

## Requirements

- `python3`, `git`, `curl`
- Optional:
  - NVIDIA GPU + CUDA (faster local MusicGen)
  - `ffmpeg` (MP4/AAC download; otherwise WAV)

## Hugging Face token (required)

The mood classifier is gated:
1. Create token: `https://huggingface.co/settings/tokens`
2. Accept access: `https://huggingface.co/MelodyWEN7/vibesound-music-mood-classifier`
3. You will paste it into `HF_TOKEN` in the script below.

## Cloud server deploy (ONE paste)

Paste this into your school cloud server terminal. **Set `HF_TOKEN`** first.

```bash
bash <<'ENDSCRIPT'
set -euo pipefail

# ================= EDIT BEFORE PASTING =================
HF_TOKEN=""                 # <-- set hf_...
APP_FILE="app_clap.py"
STREAMLIT_PORT=8503
# =======================================================

REPO="$HOME/vibe-sound"
REPO_URL="https://github.com/EdithY6/vibe-sound.git"

echo "========== VibeSound =========="

# --- repo sync ---
if [ ! -d "$REPO/.git" ]; then
  echo ">> Cloning repo..."
  git clone "$REPO_URL" "$REPO"
fi
cd "$REPO"
echo ">> git fetch + reset --hard origin/main"
git fetch origin
git reset --hard origin/main

# normalize CRLF if needed
for f in *.sh app*.py music_gen.py ui_theme.py .env.example .env; do
  [ -f "$f" ] && sed -i 's/\r$//' "$f" 2>/dev/null || true
done

# --- .env ---
[ -f .env ] || { [ -f .env.example ] && cp .env.example .env; } || touch .env
if [ -n "${HF_TOKEN:-}" ]; then
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
  echo "ERROR: HF_TOKEN missing. Put your hf_... token in HF_TOKEN at top of this script."
  exit 1
fi

set -a
. ./.env
set +a

# --- python deps ---
if [ ! -d .venv ]; then
  echo ">> First-time setup (10–20 min)..."
  bash setup.sh
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ensure CUDA torch if GPU exists
if command -v nvidia-smi >/dev/null 2>&1; then
  python -c "import torch; print('CUDA:', torch.cuda.is_available())" || true
  if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo ">> Installing CUDA PyTorch (cu124)..."
    pip install -q -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  fi
fi

# --- validate app file ---
if [ ! -f "$APP_FILE" ]; then
  echo "ERROR: $APP_FILE not found in repo."
  exit 1
fi

# --- cloudflared ---
if [ ! -x /tmp/cloudflared ]; then
  curl -fsSL -o /tmp/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /tmp/cloudflared
fi

# --- kill old processes ---
pkill -f "streamlit run" 2>/dev/null || true
pkill -f cloudflared 2>/dev/null || true
sleep 2

# --- 1) start tunnel (needs to run first to get hostname) ---
CF_LOG=/tmp/vibesound-cf.log
rm -f "$CF_LOG"
echo ">> cloudflared → 127.0.0.1:${STREAMLIT_PORT}"
/tmp/cloudflared tunnel --url "http://127.0.0.1:${STREAMLIT_PORT}" 2>&1 | tee "$CF_LOG" &
CF_PID=$!

TUNNEL_HOST=""
for _ in $(seq 1 45); do
  TUNNEL_HOST="$(grep -oE '[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" | head -1 || true)"
  [ -n "$TUNNEL_HOST" ] && break
  sleep 1
done
if [ -z "$TUNNEL_HOST" ]; then
  echo "ERROR: no tunnel URL (see $CF_LOG)"
  kill "$CF_PID" 2>/dev/null || true
  exit 1
fi

export STREAMLIT_BROWSER_SERVER_ADDRESS="$TUNNEL_HOST"
export STREAMLIT_BROWSER_SERVER_PORT=443

# --- 2) start streamlit ---
echo ">> streamlit run $APP_FILE (port $STREAMLIT_PORT)"
streamlit run "$APP_FILE" \
  --server.port="$STREAMLIT_PORT" \
  --server.address=127.0.0.1 \
  --server.headless=true \
  --server.enableCORS=true \
  --server.enableXsrfProtection=false \
  --browser.serverAddress="$TUNNEL_HOST" \
  --browser.serverPort=443 \
  > /tmp/vibesound-st.log 2>&1 &

# wait until ready
READY=0
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${STREAMLIT_PORT}/_stcore/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done
if [ "$READY" -ne 1 ]; then
  echo "ERROR: Streamlit not responding. Last logs:"
  tail -40 /tmp/vibesound-st.log || true
  kill "$CF_PID" 2>/dev/null || true
  exit 1
fi

echo ""
echo "============================================================"
echo "OPEN: https://${TUNNEL_HOST}"
echo "Logs: tail -f /tmp/vibesound-st.log"
echo "KEEP THIS WINDOW OPEN"
echo "============================================================"
echo ""

wait "$CF_PID"
ENDSCRIPT


## Config knobs (`.env`)

- `VIBESOUND_MUSIC_BACKEND=auto|local|space`
- `VIBESOUND_AUDIO_FORMAT=mp4|wav`
- `FFMPEG_PATH=/path/to/ffmpeg` *(optional)*
- `MUSICGEN_MODEL=facebook/musicgen-small` *(optional)*
- `VIBESOUND_CLAP_MODEL=laion/clap-htsat-fused` *(optional)*

## Troubleshooting

- **401 / mood model won’t load**
  - `HF_TOKEN` missing/invalid **or** you didn’t accept the gated model:
  - `https://huggingface.co/MelodyWEN7/vibesound-music-mood-classifier`

- **Music too slow**
  - CPU MusicGen is slow → use GPU **or** set `VIBESOUND_MUSIC_BACKEND=space`.

- **No tunnel URL**
  - `tail -n 200 /tmp/vibesound-cf.log`

- **Streamlit died**
  - `tail -n 200 /tmp/vibesound-st.log`

- **Wanted MP4, got WAV**
  - ffmpeg missing/failed → install ffmpeg or set `FFMPEG_PATH`, or set `VIBESOUND_AUDIO_FORMAT=wav`.
