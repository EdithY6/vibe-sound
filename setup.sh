#!/bin/bash
# VibeSound one-time install (run from repo root: bash setup.sh)
set -e
set -u
cd "$(dirname "$0")"

echo "==> VibeSound setup ($(pwd))"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found"
  exit 1
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -U pip wheel

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> GPU detected — installing requirements.txt"
  pip install -r requirements.txt
else
  echo "==> No GPU — CPU PyTorch + requirements"
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
fi

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
elif [ ! -f .env ] && [ -f .env.example ]; then
  echo ""
  echo "Tip: cp .env.example .env and set HF_TOKEN=hf_..."
fi

if [ -n "${HF_TOKEN:-}" ]; then
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli login --token "$HF_TOKEN" || true
  else
    python -m huggingface_hub.commands.huggingface_cli login --token "$HF_TOKEN" || true
  fi
fi

echo ""
echo "Setup done."
echo "  1. cp .env.example .env   (if you have not yet)"
echo "  2. nano .env              (set HF_TOKEN=hf_...)"
echo "  3. bash run.sh"
echo "  4. Open http://imz250.ust.hk:8129/proxy/8501/"
