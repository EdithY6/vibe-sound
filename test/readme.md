# `test/` folder

Batch evaluation + grading utilities for VibeSound.

## Files (current)

- `batch_eval.py`
  - Runs VibeSound’s **caption → mood → CLAP prompt** pipeline over a manifest CSV.
  - Loads models once, then loops samples.
  - Optional `--music` to also run MusicGen and write audio outputs.

- `batch_manifest.csv`
  - Example manifest for `batch_eval.py`.
  - Columns: `sample_id,image_path,user_text`
  - `image_path` is interpreted relative to `--data-dir` (or absolute).

- `grading_workbook.py`
  - `create`: builds
    - `batch_manifest.csv` (for batch_eval)
    - `vibesound_batch_grading.xlsx` (Excel scoring sheet with weighted 0–100 formula)
    from numbered images (e.g. `1.jpg`, `02.png`, `003.jpeg`, …).
  - `merge`: fills workbook columns (`scene_caption`, `target_mood`, `mood_score`, `music_prompt`, `output_audio`) from a `results.csv`.

- `run_batch_50.py`
  - Convenience runner that executes `batch_eval.py` using:
    - `test/batch_manifest.csv`
    - `test/images/`
    - output: `test/batch_results/`
  - Reads optional `test/.env` if present (for `HF_TOKEN`, etc.).

- `requirements.txt`
  - Test deps (same stack + `openpyxl` for Excel).

- `music_gen.py`
  - Music generation helper used when `batch_eval.py --music` is enabled.

- `images/`
  - Image folder referenced by the manifest / scripts.

## Output locations (what the scripts write)

- `batch_eval.py --out <dir>` writes `<dir>/results.csv`
- With `--music`, also writes `<dir>/audio/*.(mp4|wav)`
- `grading_workbook.py create` writes `batch_manifest.csv` + `vibesound_batch_grading.xlsx` (to `--out-dir`, default `.`)
- `grading_workbook.py merge` updates the given workbook in-place

## Minimal usage examples

```bash
# Fast: no MusicGen
python batch_eval.py --manifest batch_manifest.csv --data-dir images --out batch_results

# Slow: includes MusicGen + audio files
python batch_eval.py --manifest batch_manifest.csv --data-dir images --out batch_results --music --music-seconds 8

# Create workbook+manifest from numbered images
python grading_workbook.py create --images-dir images --count 50

# Merge batch results back into workbook
python grading_workbook.py merge --workbook vibesound_batch_grading.xlsx --results batch_results/results.csv
