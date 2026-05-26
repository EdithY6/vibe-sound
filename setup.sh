#!/bin/bash
# VibeSound one-time install (from repo root: bash setup.sh)
# Fresh containers: use bash deploy.sh instead (runs setup + tunnel + app).
if grep -q $'\r' "$0" 2>/dev/null; then
  sed -i 's/\r$//' "$0"
  exec bash "$0" "$@"
fi
set -eu
cd "$(dirname "$0")"

echo "==> VibeSound setup ($(pwd))"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found"
  exit 1
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
else
  echo "==> .venv already exists — reusing"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -U pip wheel

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> GPU detected — CUDA PyTorch (cu124) + requirements"
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
  echo "==> No GPU — CPU PyTorch + requirements"
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> Python deps (Streamlit pinned <1.40 for trycloudflare)"
pip install 'streamlit>=1.32.0,<1.40.0' -r requirements.txt

python -c "import streamlit as st; print('Streamlit', st.__version__)"
if command -v nvidia-smi >/dev/null 2>&1; then
  python -c "import torch; print('CUDA:', torch.cuda.is_available())" || true
fi

# .env
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo "==> Created .env from .env.example — set HF_TOKEN=hf_..."
fi
if [ -f .env ]; then
  sed -i 's/\r$//' .env 2>/dev/null || true
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

if [ -n "${HF_TOKEN:-}" ] && [[ "${HF_TOKEN}" == hf_* ]]; then
  if command -v hf >/dev/null 2>&1; then
    hf auth login --token "$HF_TOKEN" 2>/dev/null || true
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
  fi
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo ""
  echo "==> ffmpeg not found (MP4 export). Trying conda..."
  if command -v conda >/dev/null 2>&1; then
    conda install -y -c conda-forge ffmpeg 2>/dev/null || true
  fi
fi
if command -v ffmpeg >/dev/null 2>&1; then
  echo "==> ffmpeg OK: $(command -v ffmpeg)"
else
  echo "==> WARN: no ffmpeg — downloads will be WAV until you install ffmpeg"
fi

echo ""
echo "Setup done."
echo "  1. Edit .env → HF_TOKEN=hf_... (accept gated mood model on Hugging Face)"
echo "  2. bash deploy.sh"
echo "  3. Open the https://....trycloudflare.com link printed by deploy.sh"
