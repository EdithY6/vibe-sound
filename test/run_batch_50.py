#!/usr/bin/env python3
"""One-click batch test. Uses existing batch_manifest.csv + images/ only."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "batch_manifest.csv"
IMAGES = ROOT / "images"
OUT = ROOT / "batch_results"
MUSIC_SECONDS = int(os.environ.get("MUSIC_SECONDS", "8"))


def main() -> None:
    if not MANIFEST.is_file():
        sys.exit(f"Missing {MANIFEST}")
    if not IMAGES.is_dir():
        sys.exit(f"Missing {IMAGES}/")

    env = os.environ.copy()
    dotenv = ROOT / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip())

    cmd = [
        sys.executable,
        str(ROOT / "batch_eval.py"),
        "--manifest",
        str(MANIFEST),
        "--data-dir",
        str(IMAGES),
        "--out",
        str(OUT),
        "--music",
        "--music-seconds",
        str(MUSIC_SECONDS),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT, env=env)
    print(f"\nDone.\n  {OUT / 'results.csv'}\n  {OUT / 'audio'}/")


if __name__ == "__main__":
    main()
