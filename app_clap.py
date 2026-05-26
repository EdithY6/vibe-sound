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
from ui_theme import (
    PAGE_CONFIG,
    get_hf_token,
    info_card,
    inject_day_theme,
    mood_badge,
    render_hero,
    render_sidebar_footer,
    wide_button_kwargs,
    wide_image_kwargs,
)

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
st.set_page_config(**PAGE_CONFIG)
inject_day_theme()
render_hero()

music_seconds = st.sidebar.slider(
    "Track length (seconds)",
    min_value=4,
    max_value=20,
    value=8,
    step=1,
    help="Shorter = faster to generate.",
)
music_max_new_tokens = int(music_seconds * 50)

with st.form("vibesound_input"):
    col_photo, col_text = st.columns(2)
    with col_photo:
        st.markdown("##### 📷 Your Reel photo")
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="A clear photo of your scene works best.",
        )
        if uploaded is not None:
            preview = Image.open(uploaded).convert("RGB")
            st.image(preview, **wide_image_kwargs())
    with col_text:
        st.markdown("##### ✏️ Caption (optional)")
        user_text = st.text_area(
            "Caption",
            placeholder="e.g. Best sunset with friends — chill summer vibes",
            height=160,
            label_visibility="collapsed",
            help="Helps set mood. Leave empty for a calm neutral feel.",
        )
        st.caption("No caption? We still read your photo; mood defaults to **neutral**.")
    submitted = st.form_submit_button(
        "Generate my background music",
        **wide_button_kwargs(),
    )

if submitted:
    if uploaded is None:
        st.warning("Please upload a Reel photo first, then tap **Generate**.")
        st.stop()

    image = Image.open(uploaded).convert("RGB")
    hf_token = get_hf_token()

    st.markdown("---")
    st.subheader("Your results")

    with st.status("Creating your track…", expanded=True) as status:
        st.write("Reading your photo…")
        caption = run_image_caption(image)
        info_card("What we see in your photo", caption, variant="scene")

        st.write("Understanding your vibe…")
        top_mood, top_score, display_scores = run_mood(user_text)
        emoji = MOOD_EMOJI.get(top_mood, "🎶")
        color = MOOD_COLOR.get(top_mood, "#888888")

        mood_col, detail_col = st.columns([1, 2])
        with mood_col:
            mood_badge(emoji, top_mood, color)
            if user_text.strip():
                st.caption(f"Mood confidence: {top_score * 100:.0f}%")
            else:
                st.caption("No caption — using **neutral** mood. Add text next time to steer feel.")
        with detail_col:
            if user_text.strip():
                top3 = {k.capitalize(): round(v, 3) for k, v in display_scores[:3]}
                st.bar_chart(top3, height=120)
            else:
                st.info("Tip: add a short caption to personalize mood and music style.")

        st.write("Picking music style (CLAP)…")
        music_prompt, top_tags = build_music_prompt(caption, display_scores, user_text)
        info_card("Music style prompt", music_prompt, variant="prompt")

        st.write(f"Composing ~{music_seconds}s of music…")
        audio_bytes, backend_used, audio_ext = generate_music(
            music_prompt, hf_token, max_new_tokens=music_max_new_tokens
        )
        status.update(label="Done — your track is ready", state="complete")

    with st.expander("Style tags we matched"):
        tag_labels = [f"{t} ({s:.2f})" for t, s in top_tags[:8]]
        st.write(", ".join(tag_labels) if tag_labels else "—")

    audio_mime = "audio/mp4" if audio_ext == "mp4" else "audio/wav"
    st.success("Your background music is ready.")
    if audio_ext != "mp4":
        st.caption("MP4 export needs `ffmpeg` on the server — downloaded as WAV this time.")
    st.audio(audio_bytes, format=audio_mime)
    st.download_button(
        label="Download MP4" if audio_ext == "mp4" else "Download WAV",
        data=audio_bytes,
        file_name=f"vibesound_{top_mood}.{audio_ext}",
        mime=audio_mime,
        **wide_button_kwargs(),
    )

cuda = torch.cuda.is_available()
mb = resolve_backend()
mood_model = FINETUNED_MODEL if USE_FINETUNED_MODEL else PLACEHOLDER_MODEL
render_sidebar_footer(
    p1_label=CAPTION_MODEL_LABEL,
    p2_model=mood_model,
    p3_label="CLAP music tags",
    music_backend=mb,
    cuda=cuda,
)

