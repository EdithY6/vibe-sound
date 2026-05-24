"""
ISOM5240 — VibeSound (UST server / JupyterLab deploy)

P1: ViT-GPT2 image caption (local)
P2: Fine-tuned mood classifier (local)
P3: CLAP music tag ranking (music-domain model; no LLM)
P4: MusicGen (local on GPU if available)
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
    ClapModel,
    ClapProcessor,
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

CLAP_MODEL = os.environ.get("VIBESOUND_CLAP_MODEL", "laion/clap-htsat-fused")
PROMPT_BUILDER_LABEL = f"CLAP tags ({CLAP_MODEL})"

PLACEHOLDER_REMAP = {
    "joy": "happy",
    "sadness": "sad",
    "love": "romantic",
    "anger": "intense",
    "fear": "intense",
    "surprise": "surprised",
}

FINETUNED_LABELS = ["happy", "sad", "romantic", "intense", "surprised", "neutral"]

# Small but useful music vocabulary; CLAP ranks these against the reel text/caption.
TAG_VOCAB: list[str] = [
    # genres / styles
    "indie pop",
    "lo-fi hip hop",
    "surf rock",
    "dance pop",
    "acoustic pop",
    "cinematic",
    "orchestral",
    "ambient",
    "synthwave",
    "EDM",
    "house",
    "trap",
    "jazz",
    "bossa nova",
    # instruments
    "acoustic guitar",
    "electric guitar",
    "nylon guitar",
    "piano",
    "electric piano",
    "strings",
    "brass",
    "synth pad",
    "synth lead",
    "bass",
    "kick drum",
    "snare",
    "hand claps",
    "ukulele",
    # tempo / feel
    "fast tempo",
    "mid tempo",
    "slow tempo",
    "groovy",
    "driving rhythm",
    "bouncy",
    "soft",
    "punchy drums",
    "warm",
    "bright",
    "dark",
    # mood color
    "joyful",
    "happy",
    "romantic",
    "intense",
    "surprised",
    "calm",
    "melancholic",
    "nostalgic",
    "uplifting",
    "emotional",
    # production constraints
    "instrumental",
    "no vocals",
    "clean mix",
    "background music",
]

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


@st.cache_resource(show_spinner=False)
def _load_clap_and_tags():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClapModel.from_pretrained(CLAP_MODEL).to(device)
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)

    # Pre-embed tag vocabulary once
    with torch.inference_mode():
        tag_inputs = processor(text=TAG_VOCAB, return_tensors="pt", padding=True).to(
            device
        )
        tag_features = model.get_text_features(**tag_inputs)
        tag_features = torch.nn.functional.normalize(tag_features, dim=-1)
    return model, processor, tag_features, device


@torch.inference_mode()
def build_music_prompt(
    caption: str,
    display_scores: list[tuple[str, float]],
    user_text: str,
) -> tuple[str, list[tuple[str, float]]]:
    """
    Build a MusicGen prompt by ranking a music-vocabulary with CLAP.
    Returns (prompt, top_tags).
    """
    top_mood = display_scores[0][0] if display_scores else "neutral"
    # Use mood as a hint, not a template
    query = (
        f"photo scene: {caption}. "
        f"reel text: {user_text or 'none'}. "
        f"target mood: {top_mood}."
    ).strip()

    model, processor, tag_features, device = _load_clap_and_tags()
    q_inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    q_feat = model.get_text_features(**q_inputs)
    q_feat = torch.nn.functional.normalize(q_feat, dim=-1)

    scores = (q_feat @ tag_features.T).squeeze(0)  # [num_tags]
    topk = min(12, scores.shape[0])
    vals, idx = torch.topk(scores, k=topk)
    top_tags = [(TAG_VOCAB[i], float(v)) for i, v in zip(idx.tolist(), vals.tolist())]

    tag_line = ", ".join([t for t, _ in top_tags[:10]])
    prompt = (
        f"{tag_line}. "
        f"Scene: {caption}. "
        f"Message: {user_text or 'none'}. "
        f"Instagram reel background instrumental, cohesive production, no vocals."
    )
    words = prompt.split()
    if len(words) > 70:
        prompt = " ".join(words[:70])
    return prompt, top_tags


# ── PAGE ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibeSound — Reel Music Generator (CLAP)",
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
    '<p class="subtitle">CLAP-tag prompt builder · UST server</p>',
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
    with st.spinner("Ranking music tags..."):
        music_prompt, top_tags = build_music_prompt(caption, display_scores, user_text)

    st.markdown(
        f'<div class="prompt-box">🎼 Music prompt: {music_prompt}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Debug: top CLAP tags"):
        st.write(top_tags)

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
    cuda = torch.cuda.is_available()
    mb = resolve_backend()
    st.success(f"**Music** backend=`{mb}` · CUDA=`{cuda}`")
    st.caption(f"CUDA: `{cuda}` · device count: `{torch.cuda.device_count() if cuda else 0}`")

