#!/usr/bin/env python3
"""
Batch-evaluate VibeSound (CLAP pipeline) on many image + text pairs.

Fast mode (default): P1 caption + P2 mood + P3 prompt only (no MusicGen).
Loads each model once, then loops over samples.

Manifest CSV columns (header required):
  sample_id,image_path,user_text
  - image_path: path to jpg/png (relative to --data-dir or absolute)
  - user_text: optional caption / feeling (empty → neutral mood)

Example:
  python batch_eval.py --manifest samples.csv --data-dir ./my_50_samples --out results/
  python batch_eval.py --manifest samples.csv --data-dir ./my_50_samples --out results/ --music
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

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

# Reuse config from app_clap without importing streamlit
FINETUNED_MODEL = "MelodyWEN7/vibesound-music-mood-classifier"
IMAGE_CAPTION_MODEL = "nlpconnect/vit-gpt2-image-captioning"
CLAP_MODEL = os.environ.get("VIBESOUND_CLAP_MODEL", "laion/clap-htsat-fused")
FINETUNED_LABELS = ["happy", "sad", "romantic", "intense", "surprised", "neutral"]

TAG_VOCAB: list[str] = [
    "indie pop", "lo-fi hip hop", "surf rock", "dance pop", "acoustic pop",
    "cinematic", "orchestral", "ambient", "synthwave", "EDM", "house", "trap",
    "jazz", "bossa nova", "acoustic guitar", "electric guitar", "nylon guitar",
    "piano", "electric piano", "strings", "brass", "synth pad", "synth lead",
    "bass", "kick drum", "snare", "hand claps", "ukulele", "fast tempo",
    "mid tempo", "slow tempo", "groovy", "driving rhythm", "bouncy", "soft",
    "punchy drums", "warm", "bright", "dark", "joyful", "happy", "romantic",
    "intense", "surprised", "calm", "melancholic", "nostalgic", "uplifting",
    "emotional", "instrumental", "no vocals", "clean mix", "background music",
]


def get_hf_token() -> str:
    return os.environ.get("HF_TOKEN", "").strip()


def normalise_mood(label: str) -> str:
    label = label.lower()
    return label if label in FINETUNED_LABELS else "neutral"


class VibeSoundBatch:
    """Load models once; run many samples."""

    def __init__(self, hf_token: str | None = None) -> None:
        self.token = hf_token or get_hf_token() or None
        if not self.token:
            print("WARN: HF_TOKEN not set — gated mood model may 401", file=sys.stderr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        print("Loading ViT-GPT2 caption model...")
        self._cap_processor = ViTImageProcessor.from_pretrained(IMAGE_CAPTION_MODEL)
        self._cap_tokenizer = AutoTokenizer.from_pretrained(IMAGE_CAPTION_MODEL)
        self._cap_model = VisionEncoderDecoderModel.from_pretrained(
            IMAGE_CAPTION_MODEL, torch_dtype=self.dtype, low_cpu_mem_usage=True
        ).to(self.device)
        self._cap_model.eval()

        print(f"Loading mood model {FINETUNED_MODEL}...")
        self._mood_tokenizer = AutoTokenizer.from_pretrained(
            FINETUNED_MODEL, token=self.token
        )
        mood_model = AutoModelForSequenceClassification.from_pretrained(
            FINETUNED_MODEL,
            token=self.token,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        mood_model.eval()
        if self.device.type == "cuda":
            mood_model = mood_model.to(self.device)
        self._mood_clf = pipeline(
            "text-classification",
            model=mood_model,
            tokenizer=self._mood_tokenizer,
            top_k=None,
            device=0 if self.device.type == "cuda" else -1,
        )

        print(f"Loading CLAP {CLAP_MODEL}...")
        self._clap_model = ClapModel.from_pretrained(CLAP_MODEL).to(self.device)
        self._clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL)
        with torch.inference_mode():
            tag_inputs = self._clap_processor(
                text=TAG_VOCAB, return_tensors="pt", padding=True
            ).to(self.device)
            self._tag_features = self._clap_model.get_text_features(**tag_inputs)
            self._tag_features = torch.nn.functional.normalize(self._tag_features, dim=-1)
        print("Models ready.\n")

    @torch.inference_mode()
    def caption(self, image: Image.Image) -> str:
        pixel_values = self._cap_processor(images=image, return_tensors="pt").pixel_values.to(
            device=self.device, dtype=self.dtype
        )
        out = self._cap_model.generate(pixel_values, max_length=50, num_beams=4)
        return self._cap_tokenizer.decode(out[0], skip_special_tokens=True)

    @torch.inference_mode()
    def mood(self, text: str) -> tuple[str, float]:
        if not text.strip():
            return "neutral", 1.0
        raw = sorted(
            self._mood_clf(text.strip())[0], key=lambda x: x["score"], reverse=True
        )
        merged: dict[str, float] = {}
        for s in raw:
            k = normalise_mood(s["label"])
            merged[k] = merged.get(k, 0.0) + s["score"]
        top = max(merged.items(), key=lambda x: x[1])
        return top[0], float(top[1])

    @torch.inference_mode()
    def music_prompt(
        self, caption: str, user_text: str, top_mood: str
    ) -> tuple[str, list[tuple[str, float]]]:
        query = (
            f"photo scene: {caption}. "
            f"reel text: {user_text or 'none'}. "
            f"target mood: {top_mood}."
        ).strip()
        q_inputs = self._clap_processor(text=[query], return_tensors="pt", padding=True).to(
            self.device
        )
        q_feat = self._clap_model.get_text_features(**q_inputs)
        q_feat = torch.nn.functional.normalize(q_feat, dim=-1)
        scores = (q_feat @ self._tag_features.T).squeeze(0)
        topk = min(12, scores.shape[0])
        vals, idx = torch.topk(scores, k=topk)
        top_tags = [(TAG_VOCAB[i], float(v)) for i, v in zip(idx.tolist(), vals.tolist())]
        tag_line = ", ".join([t for t, _ in top_tags[:10]])
        prompt = (
            f"{tag_line}. Scene: {caption}. Message: {user_text or 'none'}. "
            "Instagram reel background instrumental, cohesive production, no vocals."
        )
        words = prompt.split()
        if len(words) > 70:
            prompt = " ".join(words[:70])
        return prompt, top_tags

    def process(
        self,
        image_path: Path,
        user_text: str,
        *,
        gen_music: bool = False,
        music_tokens: int = 400,
        audio_dir: Path | None = None,
    ) -> dict:
        t0 = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        caption = self.caption(image)
        top_mood, mood_score = self.mood(user_text)
        prompt, top_tags = self.music_prompt(caption, user_text, top_mood)

        row: dict = {
            "image_path": str(image_path),
            "user_text": user_text,
            "scene_caption": caption,
            "mood": top_mood,
            "mood_score": round(mood_score, 4),
            "music_prompt": prompt,
            "top_tags": json.dumps(top_tags[:8]),
            "audio_path": "",
            "audio_ext": "",
            "seconds": 0.0,
        }

        if gen_music:
            from music_gen import generate_music

            t1 = time.perf_counter()
            audio_bytes, _backend, ext = generate_music(
                prompt, self.token or "", max_new_tokens=music_tokens
            )
            if audio_dir:
                audio_dir.mkdir(parents=True, exist_ok=True)
                out_audio = audio_dir / f"{image_path.stem}.{ext}"
                out_audio.write_bytes(audio_bytes)
                row["audio_path"] = str(out_audio)
                row["audio_ext"] = ext
            row["seconds"] = round(time.perf_counter() - t1, 2)

        row["total_seconds"] = round(time.perf_counter() - t0, 2)
        return row


def load_manifest(path: Path, data_dir: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Manifest CSV needs a header row")
        for i, r in enumerate(reader):
            sid = (r.get("sample_id") or r.get("id") or str(i)).strip()
            img = (r.get("image_path") or r.get("image") or r.get("file") or "").strip()
            text = (r.get("user_text") or r.get("text") or r.get("caption") or "").strip()
            if not img:
                continue
            p = Path(img)
            if not p.is_absolute():
                p = data_dir / p
            rows.append(
                {
                    "sample_id": sid,
                    "image_path": p,
                    "user_text": text,
                    "picture_no": sid.lstrip("0") or "0",
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch VibeSound (CLAP) evaluation")
    ap.add_argument("--manifest", required=True, help="CSV: sample_id,image_path,user_text")
    ap.add_argument("--data-dir", default=".", help="Base dir for relative image paths")
    ap.add_argument("--out", default="batch_results", help="Output directory")
    ap.add_argument(
        "--music",
        action="store_true",
        help="Also run MusicGen (slow: ~30–90s per sample on GPU)",
    )
    ap.add_argument(
        "--music-seconds",
        type=int,
        default=8,
        help="Target music length if --music (tokens = seconds * 50)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max samples (0 = all)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest).resolve()

    samples = load_manifest(manifest, data_dir)
    if args.limit > 0:
        samples = samples[: args.limit]

    if not samples:
        print("No samples in manifest.", file=sys.stderr)
        sys.exit(1)

    print(f"Samples: {len(samples)} | music: {args.music} | out: {out_dir}")

    pipe = VibeSoundBatch()
    music_tokens = int(args.music_seconds * 50)
    audio_dir = out_dir / "audio" if args.music else None

    results_path = out_dir / "results.csv"
    fieldnames = [
        "sample_id",
        "picture_no",
        "image_path",
        "user_text",
        "scene_caption",
        "mood",
        "mood_score",
        "music_prompt",
        "top_tags",
        "audio_path",
        "audio_ext",
        "seconds",
        "total_seconds",
    ]

    t_start = time.perf_counter()
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for n, s in enumerate(samples, 1):
            sid = s["sample_id"]
            img_path: Path = s["image_path"]
            if not img_path.is_file():
                print(f"[{n}/{len(samples)}] SKIP {sid}: missing {img_path}")
                writer.writerow(
                    {
                        "sample_id": sid,
                        "image_path": str(img_path),
                        "user_text": s["user_text"],
                        "scene_caption": "ERROR: file not found",
                        "mood": "",
                        "mood_score": "",
                        "music_prompt": "",
                        "top_tags": "[]",
                        "audio_path": "",
                        "audio_ext": "",
                        "seconds": "",
                        "total_seconds": "",
                    }
                )
                continue
            print(f"[{n}/{len(samples)}] {sid} ...", flush=True)
            row = pipe.process(
                img_path,
                s["user_text"],
                gen_music=args.music,
                music_tokens=music_tokens,
                audio_dir=audio_dir,
            )
            writer.writerow(
                {
                    "sample_id": sid,
                    "picture_no": sid.lstrip("0") or "0",
                    **row,
                }
            )
            print(
                f"    mood={row['mood']} | {row['total_seconds']}s | "
                f"{row['music_prompt'][:80]}..."
            )

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed:.1f}s ({elapsed / len(samples):.1f}s/sample avg)")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
