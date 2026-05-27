# VibeSound (school cloud deploy)

Streamlit app that turns a **Reel photo (+ optional caption)** into **background music**.

Current app entrypoint: `app_clap.py`

Pipeline (what’s in `main`):
- **Image caption**: `nlpconnect/vit-gpt2-image-captioning` (ViT-GPT2)
- **Mood classifier (gated)**: `MelodyWEN7/vibesound-music-mood-classifier`
- **Prompt builder**: CLAP tag ranking (`laion/clap-htsat-fused`) over a curated tag vocab
- **Music generation**: MusicGen
  - local `facebook/musicgen-small` when CUDA is available
  - fallback: public `facebook/MusicGen` HF Space queue (`VIBESOUND_MUSIC_BACKEND=space`)

---

## Repo layout (current)

- `app_clap.py` — Streamlit UI + end-to-end pipeline
- `music_gen.py` — MusicGen local/Space backend + WAV→MP4 packaging (ffmpeg)
- `ui_theme.py` — UI theme/helpers + HF token loader
- `setup.sh` — creates `.venv`, installs PyTorch + deps, tries to install ffmpeg
- `deploy.sh` — deploy script with trycloudflare + Streamlit
- `.env.example` — env template (copy to `.env`, never commit)
- `.streamlit/config.toml` — Streamlit server/theme config
- `requirements.txt` — Python deps (Streamlit pinned `<1.40`)
- `LICENSE` — GPL-3.0

---

## Requirements

- `python3`
- Optional but recommended:
  - NVIDIA GPU + CUDA (local MusicGen is much faster)
  - `ffmpeg` (enables MP4/AAC download; otherwise WAV)

---

## Hugging Face token (required)

The mood classifier is gated:
1. Create token: `https://huggingface.co/settings/tokens`
2. Accept access: `https://huggingface.co/MelodyWEN7/vibesound-music-mood-classifier`
3. Provide token via `.env` or environment variable:

```env
HF_TOKEN=hf_...

---

## Run on school cloud server (copy/paste deploy)
Paste this into the server terminal, set HF_TOKEN, and it will:

- clone/update ~/vibe-sound
- create .env from .env.example + inject HF_TOKEN
- create .venv via setup.sh if missing
- start a trycloudflare.com tunnel to Streamlit
- print a public https://...trycloudflare.com URL

bash << 'ENDSCRIPT'
set -e

# ========== EDIT BEFORE PASTING ==========
HF_TOKEN=""                  # <-- set hf_...
APP_FILE="app_clap.py"
STREAMLIT_PORT=8503
# =======================================

REPO="$HOME/vibe-sound"
REPO_URL="https://github.com/EdithY6/vibe-sound.git"

echo "========== VibeSound =========="

if [ ! -d "$REPO/.git" ]; then
  echo ">> Cloning repo..."
  git clone "$REPO_URL" "$REPO" || { echo "Clone failed"; exit 1; }
fi
cd "$REPO"

echo ">> git fetch + reset to origin/main"
git fetch origin
git reset --hard origin/main

for f in *.sh app*.py music_gen.py .env.example; do
  [ -f "$f" ] && sed -i 's/\r$//' "$f" 2>/dev/null || true
done

if [ ! -f .env ]; then
  cp -f .env.example .env 2>/dev/null || true
fi

if [ -n "$HF_TOKEN" ]; then
  if grep -q '^HF_TOKEN=' .env 2>/dev/null; then
    sed -i "s|^HF_TOKEN=.*|HF_TOKEN=$HF_TOKEN|" .env
  else
    echo "HF_TOKEN=$HF_TOKEN" >> .env
  fi
fi

grep -q '^VIBESOUND_MUSIC_BACKEND=' .env 2>/dev/null || \
  echo 'VIBESOUND_MUSIC_BACKEND=local' >> .env
sed -i 's/\r$//' .env 2>/dev/null || true

if ! grep -q '^HF_TOKEN=hf_' .env 2>/dev/null; then
  echo "ERROR: Set HF_TOKEN= at the top of this script."
  exit 1
fi

set -a
. ./.env
set +a
export VIBESOUND_MUSIC_BACKEND="${VIBESOUND_MUSIC_BACKEND:-local}"

if [ ! -d .venv ]; then
  echo ">> First-time setup (10–20 min)..."
  bash setup.sh
fi
source .venv/bin/activate

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo ">> Installing CUDA PyTorch..."
    pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  fi
  python -c "import torch; print('CUDA:', torch.cuda.is_available())"
fi

if [ ! -f "$APP_FILE" ]; then
  echo "ERROR: $APP_FILE not found on server."
  exit 1
fi
echo ">> Running: streamlit run $APP_FILE (port $STREAMLIT_PORT)"

if [ ! -x /tmp/cloudflared ]; then
  curl -L -o /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /tmp/cloudflared
fi

pkill -9 -f streamlit 2>/dev/null || true
pkill -9 -f cloudflared 2>/dev/null || true
sleep 2

CF_LOG=/tmp/vibesound-cf.log
rm -f "$CF_LOG"
/tmp/cloudflared tunnel --url "http://127.0.0.1:${STREAMLIT_PORT}" --protocol http2 2>&1 | tee "$CF_LOG" &
CF_PID=$!

TUNNEL_HOST=""
for _ in $(seq 1 30); do
  TUNNEL_HOST=$(grep -oE '[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" | head -1 || true)
  [ -n "$TUNNEL_HOST" ] && break
  sleep 1
done

if [ -z "$TUNNEL_HOST" ]; then
  echo "ERROR: no tunnel URL"
  kill "$CF_PID" 2>/dev/null || true
  exit 1
fi

export STREAMLIT_BROWSER_SERVER_ADDRESS="$TUNNEL_HOST"
export STREAMLIT_BROWSER_SERVER_PORT=443

streamlit run "$APP_FILE" \
  --server.port="$STREAMLIT_PORT" \
  --server.address=127.0.0.1 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false &

for _ in $(seq 1 20); do
  curl -s -o /dev/null "http://127.0.0.1:${STREAMLIT_PORT}/" && break
  sleep 2
done

echo ""
echo "App:     $APP_FILE"
echo "OPEN:    https://$TUNNEL_HOST"
echo "KEEP THIS WINDOW OPEN"
echo ""

wait "$CF_PID"
ENDSCRIPT

Configuration (env vars)
Set these in .env (see .env.example):

HF_TOKEN=hf_...
VIBESOUND_MUSIC_BACKEND=auto|local|space
auto (default in code): local if CUDA available; else Space
local: force local MusicGen
space: force HF Space queue
VIBESOUND_AUDIO_FORMAT=mp4|wav
mp4 needs ffmpeg; otherwise falls back to wav
FFMPEG_PATH=/path/to/ffmpeg (optional)
MUSICGEN_MODEL=facebook/musicgen-small (optional override)
VIBESOUND_CLAP_MODEL=laion/clap-htsat-fused (optional override)
Troubleshooting
Mood model 401 / load failure
HF_TOKEN missing/invalid or gated access not accepted.
Music is slow
CPU MusicGen is slow; use GPU or set VIBESOUND_MUSIC_BACKEND=space.
MP4 download becomes WAV
ffmpeg missing/encoding failed; install ffmpeg or set FFMPEG_PATH.
Tunnel URL not printed
check /tmp/vibesound-cf.log.


License
GPL-3.0 — see LICENSE.
