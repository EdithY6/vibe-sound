"""Music generation: local MusicGen (GPU) or facebook/MusicGen Space (Gradio 3 queue)."""
from __future__ import annotations

import io
import json
import os
import secrets
import time
from pathlib import Path

import requests
import torch

MUSICGEN_MODEL = os.environ.get("MUSICGEN_MODEL", "facebook/musicgen-small")
MUSICGEN_SPACE_URL = "https://facebook-musicgen.hf.space"
MUSIC_BACKEND = os.environ.get("VIBESOUND_MUSIC_BACKEND", "auto")  # auto | local | space


def _read_audio_result(result) -> bytes:
    if isinstance(result, (list, tuple)):
        for item in result:
            try:
                return _read_audio_result(item)
            except (TypeError, ValueError):
                continue
        raise ValueError("No audio in response tuple")

    if isinstance(result, dict):
        for key in ("path", "url", "name"):
            if key in result and result[key]:
                return _read_audio_result(result[key])
        raise ValueError(f"Unknown audio dict keys: {result.keys()}")

    if isinstance(result, bytes):
        return result

    if isinstance(result, str):
        if result.startswith("http"):
            r = requests.get(result, timeout=120)
            r.raise_for_status()
            return r.content
        path = Path(result)
        if path.is_file():
            return path.read_bytes()

    raise ValueError(f"Unsupported audio result type: {type(result)}")


def _pick_audio_output(result) -> bytes:
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        for item in (result[1], result[0]):
            try:
                return _read_audio_result(item)
            except (TypeError, ValueError):
                continue
    return _read_audio_result(result)


@torch.inference_mode()
def generate_music_local(prompt: str, max_new_tokens: int = 256) -> bytes:
    from scipy.io.wavfile import write as wav_write
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(MUSICGEN_MODEL)
    model = MusicgenForConditionalGeneration.from_pretrained(
        MUSICGEN_MODEL,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    audio = model.generate(**inputs, max_new_tokens=max_new_tokens)
    sr = int(getattr(model.config, "sampling_rate", 32000))
    samples = audio[0, 0].float().cpu().numpy()
    peak = max(abs(samples.max()), abs(samples.min()), 1e-8)
    samples = (samples / peak * 0.95 * 32767).astype("int16")

    buf = io.BytesIO()
    wav_write(buf, rate=sr, data=samples)
    del model, processor, inputs, audio
    if device == "cuda":
        torch.cuda.empty_cache()
    return buf.getvalue()


def _space_headers(token: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _parse_queue_line(line: str) -> dict | None:
    line = line.strip()
    if not line or line == "data: [DONE]":
        return None
    if line.startswith("data:"):
        line = line[5:].strip()
    if not line:
        return None
    return json.loads(line)


def _poll_gradio3_queue(
    base: str, session_hash: str, headers: dict[str, str], timeout: float
) -> list:
    deadline = time.monotonic() + timeout
    last_error = ""

    while time.monotonic() < deadline:
        for method, kwargs in (
            ("GET", {"url": f"{base}/queue/data", "params": {"session_hash": session_hash}}),
            ("POST", {"url": f"{base}/queue/data", "json": {"session_hash": session_hash}}),
        ):
            try:
                with requests.request(
                    method, headers=headers, stream=True, timeout=120, **kwargs
                ) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines(decode_unicode=True):
                        payload = _parse_queue_line(raw_line)
                        if not payload:
                            continue
                        msg = payload.get("msg")
                        if msg == "process_completed":
                            output = payload.get("output") or {}
                            data = output.get("data", output)
                            if data is None:
                                raise RuntimeError(f"Space returned no data: {output}")
                            return data
                        if msg in ("process_failed", "process_error"):
                            raise RuntimeError(
                                payload.get("error")
                                or payload.get("title")
                                or str(payload)
                            )
                        if msg == "queue_full":
                            raise RuntimeError("MusicGen Space queue full — retry later.")
                        last_error = str(payload)
            except requests.HTTPError:
                continue
        time.sleep(2)

    raise TimeoutError(
        f"MusicGen Space timed out after {int(timeout)}s"
        + (f" (last: {last_error})" if last_error else "")
    )


def generate_music_space(prompt: str, token: str | None) -> bytes:
    """facebook/MusicGen Space — Gradio 3.34 queue API."""
    headers = _space_headers(token)
    base = MUSICGEN_SPACE_URL
    session_hash = secrets.token_hex(16)
    payload = {
        "data": [prompt, None],
        "fn_index": 0,
        "session_hash": session_hash,
        "event_data": None,
    }

    join = requests.post(f"{base}/queue/join", headers=headers, json=payload, timeout=120)
    if not join.ok:
        raise RuntimeError(f"queue/join failed ({join.status_code}): {join.text[:300]}")
    body = join.json() if join.text else {}
    if body.get("error"):
        raise RuntimeError(f"MusicGen queue error: {body['error']}")

    data = _poll_gradio3_queue(base, session_hash, headers, timeout=600)
    return _pick_audio_output(data)


def resolve_backend() -> str:
    if MUSIC_BACKEND in ("local", "space"):
        return MUSIC_BACKEND
    # auto: prefer local on server (more RAM); Space queue API breaks often
    if torch.cuda.is_available():
        return "local"
    # CPU-only: still local if explicitly not space-only; caller sets VIBESOUND_MUSIC_BACKEND=local
    return "space"


def generate_music(
    prompt: str, hf_token: str = "", max_new_tokens: int = 256
) -> tuple[bytes, str]:
    """Returns (wav_bytes, backend_label)."""
    backend = resolve_backend()
    if backend == "local":
        try:
            return (
                generate_music_local(prompt, max_new_tokens=max_new_tokens),
                f"local `{MUSICGEN_MODEL}`",
            )
        except Exception as e:
            if MUSIC_BACKEND == "local":
                raise
            return generate_music_space(prompt, hf_token or None), f"space (local failed: {e})"
    return generate_music_space(prompt, hf_token or None), "HF Space (Gradio 3)"
