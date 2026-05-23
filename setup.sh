#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> VibeSound setup"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

if command -v nvidia-smi &>/dev/null; then
  echo "==> GPU detected — installing default requirements (CUDA torch from PyPI)"
  pip install -r requirements.txt
else
  echo "==> No GPU — CPU torch + requirements"
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN" || true
fi

echo ""
echo "Setup done. Next: bash scripts/run.sh"
echo "Open: http://imz250.ust.hk:8129/proxy/8501/  (Jupyter proxy URL may vary)"
