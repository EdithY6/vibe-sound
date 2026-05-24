"""
ISOM5240 — VibeSound (UST server / JupyterLab deploy)

P1: ViT-GPT2 image caption (local)
P2: Fine-tuned mood classifier (local)
P3: flan-t5-small text prompt (seq2seq LLM baseline)
P4: MusicGen (local on GPU if available)
"""
from __future__ import annotations

import gc
import os
import re

import streamlit as st
import torch
from PIL import Image
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
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

FLAN_MODEL = os.environ.get("VIBESOUND_FLAN_MODEL", "google/flan-t5-small")
PROMPT_BUILDER_LABEL = f"flan-t5-small ({FLAN_MODEL})"

PLACEHOLDER_REMAP = {
    "joy": "happy",
    "sadness": "sad",
    "love": "romantic",
    "anger": "intense",
    "fear": "intense",
    "surprise": "surprised",
}

FINETUNED_LABELS = ["happy", "sad", "romantic", "intense", "surprised", "neutral"]

MOOD_FALLBACK: dict[str, str] = {
    "happy": "upbeat indie pop, acoustic guitar, bright melody, fast tempo, instrumental",
    "sad": "slow piano ballad, soft strings, minor key, emotional instrumental",
    "romantic": "warm nylon guitar, tender melody, slow intimate tempo, instrumental",
    "intense": "cinematic drums, brass swells, driving rhythm, epic instrumental",
    "surprised": "playful ukulele, bouncy rhythm, quirky bright accents, instrumental",
    "neutral": "calm lo-fi, mellow electric piano, soft drums, ambient instrumental",
}

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

_BAD_FLAN = frozenset({"0", "1", "none", "n/a", "null", "scream", "error"})


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
    token = get_hf_token() or None
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        token=token,
        torch_dtype=dtype,
    )
    model.eval()

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


@st.cache_resource(show_spinner=False)
def _load_flan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        FLAN_MODEL,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def _flan_input(caption: str, mood: str, user_text: str) -> str:
    reel = user_text.strip() or "none"
    return (
        "Write a short MusicGen prompt listing genre, instruments, tempo, and mood. "
        "One sentence. Instrumental only. No vocals. No explanation.\n"
        f"Scene: {caption}\n"
        f"Mood: {mood}\n"
        f"Reel text: {reel}\n"
        "Music prompt:"
    )


def _sanitize_flan(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""

    # Drop echoed instruction lines
    for prefix in ("music prompt:", "description:", "answer:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].strip()

    text = re.sub(r"\s+", " ", text).strip(" \"'")
    low = text.lower()
    if low in _BAD_FLAN or low.isdigit():
        return ""
    if len(text) < 12 or len(text.split()) < 3:
        return ""
    if re.fullmatch(r"[\W\d]+", text):
        return ""
    return text


def _fallback_prompt(
    caption: str,
    display_scores: list[tuple[str, float]],
    user_text: str,
) -> str:
    mood = display_scores[0][0] if display_scores else "neutral"
    base = MOOD_FALLBACK.get(mood, MOOD_FALLBACK["neutral"])
    parts = [base]
    if caption.strip():
        parts.append(f"Scene: {caption.strip()}.")
    if user_text.strip():
        parts.append(f"Reel message: {user_text.strip()[:80]}.")
    parts.append("Instagram reel background instrumental, no vocals.")
    return " ".join(parts)


@torch.inference_mode()
def build_music_prompt(
    caption: str,
    display_scores: list[tuple[str, float]],
    user_text: str,
) -> tuple[str, str]:
    """Seq2seq prompt via flan-t5-small; falls back to mood template if junk."""
    top_mood = display_scores[0][0] if display_scores else "neutral"
    tokenizer, model, device = _load_flan()
    prompt_in = _flan_input(caption, top_mood, user_text)
    inputs = tokenizer(
        prompt_in,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=int(os.environ.get("VIBESOUND_FLAN_MAX_NEW_TOKENS", "72")),
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    raw = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    clean = _sanitize_flan(raw)

    if not clean:
        final = _fallback_prompt(caption, display_scores, user_text)
        return final, raw

    if "instrumental" not in clean.lower() and "no vocal" not in clean.lower():
        clean = f"{clean} Instrumental, no vocals."
    if caption.strip() and caption.lower() not in clean.lower():
        clean = f"{clean} Scene: {caption.strip()}."

    words = clean.split()
    if len(words) > 70:
        clean = " ".join(words[:70])
    return clean, raw


# ── PAGE ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibeSound — Reel Music Generator (flan-t5)",
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
    '<p class="subtitle">flan-t5-small prompt builder · UST server</p>',
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
    color = MOOD_COLOR.get(top_mood, "#888888")
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
    with st.spinner("Generating music prompt with flan-t5..."):
        music_prompt, raw_flan = build_music_prompt(caption, display_scores, user_text)

    st.markdown(
        f'<div class="prompt-box">🎼 Music prompt: {music_prompt}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Debug: raw flan-t5 output"):
        st.write(raw_flan if raw_flan.strip() else "(empty)")

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
    st.caption(f"Model: `{FLAN_MODEL}`")
    cuda = torch.cuda.is_available()
    mb = resolve_backend()
    st.success(f"**Music** backend=`{mb}` · CUDA=`{cuda}`")
    st.caption(f"CUDA: `{cuda}` · device count: `{torch.cuda.device_count() if cuda else 0}`")
