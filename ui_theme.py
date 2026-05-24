"""Shared Streamlit UI — light day mode, user-facing copy."""
from __future__ import annotations

import html

import streamlit as st

PAGE_CONFIG = {
    "page_title": "VibeSound — Reel Music",
    "page_icon": "🎵",
    "layout": "centered",
    "initial_sidebar_state": "expanded",
}


def inject_day_theme() -> None:
    st.markdown(
        """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }
  .stApp {
    background: linear-gradient(165deg, #FAFBFC 0%, #F0F4FF 45%, #FFF7ED 100%);
  }
  [data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
  }
  .vs-hero { margin-bottom: 0.25rem; }
  .vs-title {
    font-size: 2.35rem; font-weight: 700; letter-spacing: -0.02em;
    color: #0F172A; margin: 0;
  }
  .vs-title span {
    background: linear-gradient(90deg, #DB2777, #EA580C);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .vs-sub { color: #64748B; font-size: 1.05rem; margin: 0.35rem 0 1rem 0; }
  .vs-steps {
    display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.75rem 0 1.25rem 0;
  }
  .vs-step-pill {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 999px;
    padding: 0.35rem 0.85rem; font-size: 0.82rem; color: #475569; font-weight: 600;
  }
  .vs-step-pill strong { color: #DB2777; }
  .vs-card {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
    padding: 1rem 1.1rem; margin: 0.5rem 0 0.75rem 0;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
  }
  .vs-card-label {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: #94A3B8; margin-bottom: 0.35rem;
  }
  .vs-card-body { color: #1E293B; font-size: 1rem; line-height: 1.5; }
  .vs-card-scene { border-left: 4px solid #DB2777; }
  .vs-card-prompt { border-left: 4px solid #EA580C; }
  .vs-mood-badge {
    display: inline-block; padding: 0.45rem 1rem; border-radius: 999px;
    font-weight: 700; font-size: 1rem; color: #fff;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12);
  }
  div[data-testid="stForm"] {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px;
    padding: 0.25rem 0.5rem 0.5rem 0.5rem;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
  }
</style>
""",
        unsafe_allow_html=True,
    )


def esc(text: str) -> str:
    return html.escape(str(text))


def render_hero() -> None:
    st.markdown(
        """
<div class="vs-hero">
  <p class="vs-title">🎵 <span>VibeSound</span></p>
  <p class="vs-sub">Turn your Reel photo into custom background music — in a few taps.</p>
  <div class="vs-steps">
    <span class="vs-step-pill"><strong>1</strong> Upload photo</span>
    <span class="vs-step-pill"><strong>2</strong> Add caption (optional)</span>
    <span class="vs-step-pill"><strong>3</strong> Generate &amp; download</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def info_card(label: str, body: str, variant: str = "scene") -> None:
    klass = "vs-card-scene" if variant == "scene" else "vs-card-prompt"
    st.markdown(
        f'<div class="vs-card {klass}">'
        f'<div class="vs-card-label">{esc(label)}</div>'
        f'<div class="vs-card-body">{esc(body)}</div></div>',
        unsafe_allow_html=True,
    )


def mood_badge(emoji: str, mood: str, color: str) -> None:
    st.markdown(
        f'<span class="vs-mood-badge" style="background:{color};">'
        f"{esc(emoji)} {esc(mood.capitalize())}</span>",
        unsafe_allow_html=True,
    )


def render_sidebar_footer(
    *,
    show_tech: bool = True,
    p1_label: str,
    p2_model: str,
    p3_label: str,
    music_backend: str,
    cuda: bool,
) -> None:
    st.sidebar.markdown("### ⚙️ Settings")
    st.sidebar.caption("Adjust before you generate.")

    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.markdown(
        "- **Photo** drives the scene (beach, party, city…)\n"
        "- **Caption** steers mood — leave blank for a calm neutral vibe\n"
        "- First run downloads models (1–3 min)"
    )

    if show_tech:
        with st.sidebar.expander("How it works (technical)"):
            st.markdown(
                f"**1. Scene** — {p1_label}\n\n"
                f"**2. Mood** — `{p2_model}`\n\n"
                f"**3. Style tags** — {p3_label}\n\n"
                f"**4. Audio** — MusicGen · `{music_backend}` · CUDA `{cuda}`"
            )
