#!/usr/bin/env python3
"""
One-shot: 50-sample batch — create grading Excel, run pipeline + music (8s), merge.

Layout:
  my_batch/
    images/1.jpg … 50.jpg
    user_texts.csv          (optional: picture_no,user_text)
    batch_manifest.csv      (generated)
    vibesound_batch_grading.xlsx
    batch_results/
      results.csv
      audio/

Usage:
  python run_batch_50.py --data-dir ./my_batch
  python run_batch_50.py --data-dir ./my_batch --prompts-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        required=True,
        help="Folder containing images/ (numbered files)",
    )
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--texts", default="", help="user_texts.csv inside data-dir if set")
    ap.add_argument(
        "--prompts-only",
        action="store_true",
        help="Skip MusicGen (fast: captions + prompts only)",
    )
    ap.add_argument("--music-seconds", type=int, default=8)
    ap.add_argument("--rater", default="R1")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data = Path(args.data_dir).resolve()
    images = data / "images"
    if not images.is_dir():
        raise SystemExit(f"Expected {images}/ with numbered jpg/png files")

    texts = args.texts or str(data / "user_texts.csv")
    texts_arg = ["--texts", texts] if Path(texts).is_file() else []

    py = sys.executable
    grading = root / "grading_workbook.py"
    batch = root / "batch_eval.py"

    # 1) Excel + manifest
    subprocess.check_call(
        [
            py,
            str(grading),
            "create",
            "--images-dir",
            str(images),
            "--out-dir",
            str(data),
            "--count",
            str(args.count),
            "--rater",
            args.rater,
            "--listen-seconds",
            str(args.music_seconds),
            *texts_arg,
        ]
    )

    manifest = data / "batch_manifest.csv"
    out_results = data / "batch_results"
    workbook = data / "vibesound_batch_grading.xlsx"

    # 2) batch_eval (data-dir = parent of images/ so paths match manifest)
    cmd = [
        py,
        str(batch),
        "--manifest",
        str(manifest),
        "--data-dir",
        str(images),
        "--out",
        str(out_results),
    ]
    if not args.prompts_only:
        cmd.extend(["--music", "--music-seconds", str(args.music_seconds)])
    subprocess.check_call(cmd)

    # 3) Merge pipeline outputs into Excel
    results = out_results / "results.csv"
    if results.is_file():
        subprocess.check_call(
            [
                py,
                str(grading),
                "merge",
                "--workbook",
                str(workbook),
                "--results",
                str(results),
            ]
        )

    print(f"\nDone. Grade in: {workbook}")
    if not args.prompts_only:
        print(f"Audio: {out_results / 'audio'}/")


if __name__ == "__main__":
    main()
