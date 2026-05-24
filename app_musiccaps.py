"""
ISOM5240 — VibeSound (UST server / JupyterLab deploy)

P1: ViT-GPT2 image caption (local)
P2: Fine-tuned mood classifier (local)
P3: MusicCaps retrieval (music-domain data; no LLM)
P4: MusicGen (local on GPU if available)
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    pipeline,
)

from music_gen import MUSICGEN_MODEL, generate_music, resolve_backend

# ── CONFIG ───────────────────────────────────────────────────────────────────
USE_FINETUNED_MODEL = True
QUANTIZE_MOOD_MODEL = True

PLACEHOLDER_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"
FINETUNED_MODEL = "MelodyWEN7/vibesound-music-mood-classifier"

IMAGE_CAPTION_MODEL = "nlpconnect/vit-gpt2-image-captioning"
CAPTION_MODEL_LABEL = "ViT-GPT2 image captioning"

PROMPT_BUILDER_LABEL = "MusicCaps retrieval"
MUSICCAPS_MAX_ROWS = int(os.environ.get("VIBESOUND_MUSICCAPS_MAX_ROWS", "8000"))
EMBED_MODEL = os.environ.get(
    "VIBESOUND_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

PLACEHOLDER_REMAP = {
    "joy": "happy",
    "sadness": "sad",
    "love": "romantic",
    "anger": "intense",
    "fear": "intense",
    "surprise": "surprised",
}

FINETUNED_LABELS = ["happy", "sad", "romantic", "intense", "surprised", "neutral"]

MOOD_EMOJI = {
    "happy": "😊",
    "sad": "😢",
    "romantic": "❤️",
    "intense": "😠",
    "surprised": "😲",
    "neutral": "😐",
}

MOOD_COLOR = {
    "happy": "#FFD700",
    "sad": "#4A90D9",
    "romantic": "#E91E8C",
    "intense": "#E74C3C",
    "surprised": "#FF6B35",
    "neutral": "#888888",
}


def get_hf_token() -> str:
    try:
        token = st.secrets.get("HF_TOKEN", "")
        if token:
            return token
    except (FileNotFoundError, KeyError):
        pass
    return os.environ.get("HF_TOKEN", "")


def _free(*objs) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def normalise_mood(label: str) -> str:
    label = label.lower()
    if USE_FINETUNED_MODEL:
        return label if label in FINETUNED_LABELS else "neutral"
    return PLACEHOLDER_REMAP.get(label, "neutral")


@torch.inference_mode()
def run_image_caption(image: Image.Image) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = ViTImageProcessor.from_pretrained(IMAGE_CAPTION_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(IMAGE_CAPTION_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(
        IMAGE_CAPTION_MODEL,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(
        device=device, dtype=dtype
    )
    out = model.generate(pixel_values, max_length=50, num_beams=4)
    caption = tokenizer.decode(out[0], skip_special_tokens=True)
    _free(model, processor, tokenizer, pixel_values, out)
    return caption


@torch.inference_mode()
def run_mood(text: str) -> tuple[str, float, list[tuple[str, float]]]:
    if not text.strip():
        return "neutral", 1.0, [("neutral", 1.0)]

    model_name = FINETUNED_MODEL if USE_FINETUNED_MODEL else PLACEHOLDER_MODEL
    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=get_hf_token() or None)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        token=get_hf_token() or None,
        torch_dtype=dtype,
    )
    model.eval()

    # int8 dynamic quant is CPU-only; on GPU use fp16 weights instead
    if cuda:
        model = model.to("cuda")
    elif QUANTIZE_MOOD_MODEL and USE_FINETUNED_MODEL:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    device = 0 if cuda else -1
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        device=device,
    )
    raw_scores = sorted(clf(text.strip())[0], key=lambda x: x["score"], reverse=True)
    _free(clf, model, tokenizer)

    top_mood = normalise_mood(raw_scores[0]["label"])
    top_score = raw_scores[0]["score"]
    merged: dict[str, float] = {}
    for s in raw_scores:
        k = normalise_mood(s["label"])
        merged[k] = merged.get(k, 0.0) + s["score"]
    display_scores = sorted(merged.items(), key=lambda x: -x[1])
    return top_mood, top_score, display_scores


def _index_path() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "vibesound" / f"musiccaps_{MUSICCAPS_MAX_ROWS}.npz"


@st.cache_resource(show_spinner=False)
def _load_musiccaps_index():
    """
    Loads or builds a small MusicCaps embedding index.
    This is retrieval, not a template: we reuse human-written music descriptions.
    """
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    idx_path = _index_path()
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    if idx_path.is_file():
        data = np.load(idx_path, allow_pickle=False)
        captions = data["captions"].astype(str).tolist()
        emb = data["emb"].astype(np.float32)
        return captions, emb, SentenceTransformer(EMBED_MODEL)

    ds = load_dataset("laion/musiccaps", split="train")
    rows = min(MUSICCAPS_MAX_ROWS, len(ds))
    captions = [str(ds[i]["caption"]) for i in range(rows)]

    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(
        captions,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    np.savez_compressed(idx_path, captions=np.array(captions), emb=emb)
    return captions, emb, model


def retrieve_musiccaps(query: str, k: int = 3) -> list[tuple[str, float]]:
    captions, emb, model = _load_musiccaps_index()
    q = model.encode([query], normalize_embeddings=True).astype(np.float32)[0]
    sims = emb @ q  # cosine since normalized
    idx = np.argsort(-sims)[:k]
    return [(captions[int(i)], float(sims[int(i)])) for i in idx]


def build_music_prompt(
    caption: str,
    display_scores: list[tuple[str, float]],
    user_text: str,
) -> tuple[str, list[tuple[str, float]]]:
    top_mood = display_scores[0][0] if display_scores else "neutral"
    query = (
        f"Scene: {caption}. "
        f"Reel text: {user_text or 'none'}. "
        f"Target mood: {top_mood}. "
        f"Write a short description of suitable background music."
    )
    retrieved = retrieve_musiccaps(query, k=3)
    best_caption = retrieved[0][0] if retrieved else ""

    # Use the retrieved human caption as the core, then lightly constrain for MusicGen.
    prompt = (
        f"{best_caption} "
        f"Target mood: {top_mood}. "
        f"Scene: {caption}. "
        f"Instagram reel background instrumental, cohesive production, no vocals."
    )
    words = prompt.split()
    if len(words) > 70:
        prompt = " ".join(words[:70])
    return prompt, retrieved


# ── PAGE ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibeSound — Reel Music Generator (MusicCaps)",
    page_icon="🎵",
    layout="centered",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0D0D0D 0%, #1A0A2E 100%); }
    .title-text { font-size: 2.6rem; font-weight: 700;
                  background: linear-gradient(90deg, #E91E8C, #FF6B35);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subtitle { color: #aaa; font-size: 1rem; margin-top: -8px; }
    .step-label { color: #888; font-size: 0.8rem; font-weight: 600;
                  text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .caption-box { color: #eee; font-style: italic; font-size: 1rem; padding: 12px;
                   background: rgba(255,255,255,0.04);
                   border-left: 3px solid #E91E8C; border-radius: 4px; }
    .prompt-box { color: #eee; font-size: 0.95rem; padding: 12px;
                  background: rgba(255,255,255,0.04);
                  border-left: 3px solid #FF6B35; border-radius: 4px; }
    .mood-badge { display: inline-block; padding: 6px 18px; border-radius: 30px;
                  font-weight: 600; font-size: 1rem; color: #fff; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="title-text">🎵 VibeSound</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">MusicCaps retrieval prompt builder · UST server</p>',
    unsafe_allow_html=True,
)

music_seconds = st.sidebar.slider(
    "Music length (seconds)",
    min_value=4,
    max_value=20,
    value=8,
    step=1,
)
music_max_new_tokens = int(music_seconds * 50)

with st.form("vibesound_input", border=True):
    st.markdown("**Upload once, then generate**")
    col_photo, col_text = st.columns(2)
    with col_photo:
        uploaded = st.file_uploader("Reel photo", type=["jpg", "jpeg", "png"])
    with col_text:
        user_text = st.text_area(
            "How are you feeling? (optional)",
            placeholder="e.g. best day ever with my girls",
            height=120,
        )
    submitted = st.form_submit_button(
        "🎵 Generate Background Music",
        type="primary",
        use_container_width=True,
    )

if uploaded is not None:
    preview = Image.open(uploaded).convert("RGB")
    st.image(preview, caption="Preview", width=280)

if submitted:
    if uploaded is None:
        st.error("Please upload a reel photo first.")
        st.stop()

    image = Image.open(uploaded).convert("RGB")
    hf_token = get_hf_token()

    st.markdown("---")
    st.markdown(
        f'<p class="step-label">★ Pipeline 1 — {CAPTION_MODEL_LABEL}</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("Reading your photo..."):
        caption = run_image_caption(image)
    st.markdown(f'<div class="caption-box">📝 Scene: {caption}</div>', unsafe_allow_html=True)

    mood_label = FINETUNED_MODEL if USE_FINETUNED_MODEL else PLACEHOLDER_MODEL
    st.markdown(
        f'<p class="step-label">★ Pipeline 2 — {mood_label}</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("Detecting mood..."):
        top_mood, top_score, display_scores = run_mood(user_text)

    emoji = MOOD_EMOJI.get(top_mood, "🎶")
    color = MOOD_COLOR.get(top_mood, "#888")
    col_mood, col_chart = st.columns([1, 2])
    with col_mood:
        st.markdown(
            f'<div class="mood-badge" style="background:{color};">'
            f"{emoji} {top_mood.capitalize()}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Confidence: {top_score * 100:.1f}%")
    with col_chart:
        if user_text.strip():
            top3 = {k.capitalize(): round(v, 3) for k, v in display_scores[:3]}
            st.bar_chart(top3, height=100)

    st.markdown(
        f'<p class="step-label">★ Pipeline 3 — {PROMPT_BUILDER_LABEL}</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("Retrieving MusicCaps prompt... (first run builds index)"):
        music_prompt, retrieved = build_music_prompt(caption, display_scores, user_text)

    st.markdown(
        f'<div class="prompt-box">🎼 Music prompt: {music_prompt}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Debug: retrieved MusicCaps captions"):
        st.write(retrieved)
        st.caption(f"Embed model: {EMBED_MODEL} · rows: {MUSICCAPS_MAX_ROWS}")

    backend_hint = resolve_backend()
    st.markdown(
        f'<p class="step-label">Music — MusicGen (`{MUSICGEN_MODEL}`)</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Backend: **{backend_hint}** · length: **{music_seconds}s** "
        f"(set `VIBESOUND_MUSIC_BACKEND=local|space|auto`)"
    )
    with st.spinner("Composing music..."):
        audio_bytes, backend_used = generate_music(
            music_prompt, hf_token, max_new_tokens=music_max_new_tokens
        )

    st.success(f"✅ Ready — music via {backend_used}")
    st.audio(audio_bytes, format="audio/wav")
    st.download_button(
        label="⬇️ Download Music (.wav)",
        data=audio_bytes,
        file_name=f"vibesound_{top_mood}.wav",
        mime="audio/wav",
    )

with st.sidebar:
    st.markdown("### 🔬 Architecture")
    st.success(f"**P1** {CAPTION_MODEL_LABEL}")
    st.success(f"**P2** `{FINETUNED_MODEL if USE_FINETUNED_MODEL else PLACEHOLDER_MODEL}`")
    st.success(f"**P3** {PROMPT_BUILDER_LABEL}")
    st.caption(f"Embed: `{EMBED_MODEL}` · MusicCaps rows: `{MUSICCAPS_MAX_ROWS}`")
    cuda = torch.cuda.is_available()
    mb = resolve_backend()
    st.success(f"**Music** backend=`{mb}` · CUDA=`{cuda}`")
    st.caption(f"CUDA: `{cuda}` · device count: `{torch.cuda.device_count() if cuda else 0}`")

