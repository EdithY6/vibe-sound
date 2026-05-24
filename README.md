# VibeSound — UST server deploy (Option A)

Streamlit app for ISOM5240: ViT-GPT2 caption → mood classifier → context-aware prompt → MusicGen.

Designed for **JupyterLab on `imz250.ust.hk`** (more RAM than Streamlit Cloud 1GB).

## Quick start (on server)

```bash
git clone https://github.com/YOUR_USER/vibesound.git
cd vibesound
cp .env.example .env
# edit .env → set HF_TOKEN=hf_...

bash setup.sh
bash run.sh
```

Open in browser (Jupyter proxy — adjust port if needed):

`http://imz250.ust.hk:8129/proxy/8501/`

## Hugging Face token

1. Create token: https://huggingface.co/settings/tokens  
2. Accept gated model: https://huggingface.co/MelodyWEN7/vibesound-music-mood-classifier  
3. Put token in `.env` as `HF_TOKEN=hf_...` (never commit `.env`)

## Music backend

| `VIBESOUND_MUSIC_BACKEND` | Behavior |
|---------------------------|----------|
| `auto` (default) | Local MusicGen on GPU; else public HF Space |
| `local` | Force `facebook/musicgen-small` on this machine |
| `space` | Force facebook/MusicGen HF Space queue |

Check GPU:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI + pipelines 1–3 |
| `music_gen.py` | Local MusicGen or Gradio 3 Space |
| `requirements.txt` | Python deps |
| `setup.sh` | venv + pip install |
| `run.sh` | start Streamlit |

## Assignment note

If you must submit a **Streamlit Cloud** URL as well, keep this server as the “real” backend or demo the working version from `imz250` if your course allows it.

## Troubleshooting

- **Mood model 401** → `HF_TOKEN` missing or model terms not accepted on Hub  
- **Music slow on CPU** → normal; use GPU or `VIBESOUND_MUSIC_BACKEND=space`  
- **Proxy 404** → ask TA for correct Jupyter proxy path for port 8501  
- **OOM** → pipelines already load one model at a time; restart kernel between heavy tests
