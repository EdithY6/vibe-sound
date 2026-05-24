"""
ISOM5240 — VibeSound (UST server / JupyterLab deploy)

Pipelines 1–3: local transformers (sequential load/unload).
Music: local MusicGen on GPU if available, else facebook/MusicGen HF Space.
"""
from __future__ import annotations

import gc
import os

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
FLANT5_MODEL = "google/flan-t5-small"

PLACEHOLDER_REMAP = {
    "joy": "happy",
    "sadness": "sad",
    "love": "romantic",
    "anger": "intense",
    "fear": "intense",
    "surprise": "surprised",
}

FINETUNED_LABELS = ["happy", "sad", "romantic", "intense", "surprised", "neutral"]

MOOD_FALLBACK = {
    "happy": "upbeat acoustic guitar, bright cheerful, fast tempo, major key",
    "sad": "slow melancholic piano, minor key, cinematic, emotional",
    "romantic": "soft romantic acoustic guitar, warm gentle, slow tempo",
    "intense": "dramatic orchestra, heavy drums, fast, dark, powerful",
    "surprised": "playful quirky ukulele, bouncy, dynamic, bright",
    "neutral": "calm ambient background, soft instrumental, peaceful",
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
    out = model.generate(
        pixel_values,
        max_length=50,
        num_beams=4,
    )
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


@torch.inference_mode()
def run_prompt_builder(caption: str, top_mood: str) -> str:
    device = 0 if torch.cuda.is_available() else -1
    gen = pipeline(
        "text2text-generation",
        model=FLANT5_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        max_new_tokens=40,
        device=device,
    )
    instruction = (
        f"Generate background music keywords for an Instagram reel. "
        f"Scene: {caption}. "
        f"Mood: {top_mood}. "
        f"Output comma-separated music style keywords only, maximum 15 words."
    )
    raw_output = gen(instruction)[0]["generated_text"].strip()
    _free(gen)
    if (
        raw_output
        and len(raw_output.split()) <= 20
        and not raw_output.isdigit()
        and len(raw_output) > 2
    ):
        return raw_output
    return MOOD_FALLBACK.get(top_mood, "calm ambient music")


# ── PAGE ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibeSound — Reel Music Generator",
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
    .card { background: rgba(255,255,255,0.05); border-radius: 16px;
            padding: 20px; margin: 10px 0; border: 1px solid rgba(255,255,255,0.1); }
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
    '<p class="subtitle">Background music for your Instagram Reel · UST server</p>',
    unsafe_allow_html=True,
)

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
        f'<p class="step-label">★ Pipeline 1 — {IMAGE_CAPTION_MODEL}</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("Reading your photo..."):
        try:
            caption = run_image_caption(image)
        except Exception as e:
            st.error(f"Image captioning failed: {e}")
            caption = "a scenic photo"
            st.warning("Using fallback caption.")

    st.markdown(f'<div class="caption-box">📝 Scene: {caption}</div>', unsafe_allow_html=True)

    mood_label = (
        FINETUNED_MODEL if USE_FINETUNED_MODEL else PLACEHOLDER_MODEL
    )
    st.markdown(
        f'<p class="step-label">★ Pipeline 2 — {mood_label}</p>',
        unsafe_allow_html=True,
    )
    with st.spinner("Detecting mood..."):
        try:
            top_mood, top_score, display_scores = run_mood(user_text)
        except Exception as e:
            st.error(f"Mood detection failed: {e}")
            if "gated" in str(e).lower() or "401" in str(e):
                st.info("Set `HF_TOKEN` and accept the model terms on Hugging Face Hub.")
            st.stop()

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
        if not user_text.strip():
            st.caption("*(no text → neutral)*")
    with col_chart:
        if user_text.strip():
            top3 = {k.capitalize(): round(v, 3) for k, v in display_scores[:3]}
            st.bar_chart(top3, height=100)

    st.markdown('<p class="step-label">★ Pipeline 3 — flan-t5-small</p>', unsafe_allow_html=True)
    with st.spinner("Building music prompt..."):
        try:
            music_prompt = run_prompt_builder(caption, top_mood)
        except Exception:
            music_prompt = MOOD_FALLBACK.get(top_mood, "calm ambient music")

    st.markdown(
        f'<div class="prompt-box">🎼 Music prompt: {music_prompt}</div>',
        unsafe_allow_html=True,
    )

    backend_hint = resolve_backend()
    st.markdown(
        f'<p class="step-label">Music — MusicGen (`{MUSICGEN_MODEL}`)</p>',
        unsafe_allow_html=True,
    )
    st.caption(f"Backend: **{backend_hint}** (set `VIBESOUND_MUSIC_BACKEND=local|space|auto`)")
    with st.spinner("Composing music (1–3 min on first run)..."):
        try:
            audio_bytes, backend_used = generate_music(music_prompt, hf_token)
        except Exception as e:
            st.error(f"Music generation failed: {e}")
            st.info(
                "Export `HF_TOKEN` before starting. On CPU-only hosts, music uses the public "
                "MusicGen Space (slow). With GPU, music runs locally."
            )
            st.stop()

    st.success(f"✅ Ready — music via {backend_used}")
    st.audio(audio_bytes, format="audio/wav")
    st.download_button(
        label="⬇️ Download Music (.wav)",
        data=audio_bytes,
        file_name=f"vibesound_{top_mood}.wav",
        mime="audio/wav",
    )

    st.markdown("---")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Summary")
    for k, v in {
        "Scene": caption,
        "Text": user_text.strip() or "*(none)*",
        "Mood": f"{emoji} {top_mood.capitalize()} ({top_score * 100:.1f}%)",
        "Prompt": music_prompt,
        "Music backend": backend_used,
    }.items():
        st.markdown(f"**{k}:** {v}")
    st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔬 Architecture")
    st.success(f"**P1** `{IMAGE_CAPTION_MODEL}`")
    st.success(f"**P2** `{FINETUNED_MODEL if USE_FINETUNED_MODEL else PLACEHOLDER_MODEL}`")
    st.success("**P3** flan-t5 (local)")
    cuda = torch.cuda.is_available()
    mb = resolve_backend()
    st.success(f"**Music** backend=`{mb}` · CUDA=`{cuda}`")
    st.caption(f"CUDA: `{cuda}` · device count: `{torch.cuda.device_count() if cuda else 0}`")
    if not get_hf_token():
        st.warning("Set `HF_TOKEN` in environment for gated mood model.")
    st.markdown("---")
    for mood, em in MOOD_EMOJI.items():
        st.markdown(f"{em} {mood.capitalize()}")
    st.caption("ISOM5240 · VibeSound")
