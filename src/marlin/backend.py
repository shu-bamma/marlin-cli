"""Marlin inference client — one OpenAI-compatible interface, local or hosted.

Video is sent as a base64 data URL in a `video_url` content part. This works
against vLLM (local and hosted) without `--allowed-local-media-path`, and
chunks are 30s/480p so payloads stay small (~1-3 MB).

Chunking to <=30s is a correctness requirement, not just a cost choice:
vLLM's Qwen3-VL path compresses timestamps on long videos (vllm#30847);
short clips ground correctly and match Marlin's training distribution.
"""

from __future__ import annotations

import base64
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx
from openai import OpenAI

from .config import Config
from .contract import (
    CAPTION_DETAIL_PROMPT,
    CAPTION_PROMPT,
    GROUND_PROMPT,
    Event,
    parse_caption,
    parse_span,
    strip_thinking,
)

_MIME = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".m4v": "video/x-m4v",
}


def probe(base_url: str, api_key: str = "", timeout: float = 3.0) -> bool:
    """True if an OpenAI-compatible server answers at base_url."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=timeout)
        return r.status_code in (200, 401, 403) if api_key == "" else r.status_code == 200
    except httpx.HTTPError:
        return False


def _video_part(path: Path) -> dict:
    mime = _MIME.get(path.suffix.lower(), "video/mp4")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{b64}"}}


# Marlin's per-frame budget (modeling_marlin.py: VIDEO_MAX_PIXELS=200704, 2 fps,
# patch×merge = 16×2 = 32). The server already smart_resizes to this, but only
# after decoding full-res frames — so we downscale on the client to the same
# target first: same frames the model would see, far cheaper to decode + much
# less memory (a 4K decode is the thing that OOMs weak machines).
VIDEO_MAX_PIXELS = 200704
VIDEO_FPS = 2.0
_FACTOR = 32


def _have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _probe_dims(path: Path) -> "tuple[int, int] | None":
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(path)],
            capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL).stdout.strip()
        w, h = (int(x) for x in out.split("x")[:2])
        return (w, h) if w > 0 and h > 0 else None
    except Exception:
        return None


def _target(w: int, h: int, max_pixels: int) -> "tuple[int, int] | None":
    """smart_resize (downscale path), rounded to the patch factor — mirrors the
    model. None if already within budget (don't upscale)."""
    if w * h <= max_pixels:
        return None
    beta = math.sqrt(w * h / max_pixels)
    w2 = max(_FACTOR, math.floor(w / beta / _FACTOR) * _FACTOR)
    h2 = max(_FACTOR, math.floor(h / beta / _FACTOR) * _FACTOR)
    return w2, h2


def downscale_proxy(path: Path, max_pixels: int = VIDEO_MAX_PIXELS,
                    fps: float = VIDEO_FPS) -> "tuple[Path, str | None]":
    """Return (path_to_send, note). Re-encodes a small proxy at the model's pixel
    budget + fps; falls back to the original (note=None) if ffmpeg is missing,
    dims are unknown, the clip is already within budget, or the encode fails.
    When note is set the returned path is a temp file — caller deletes its parent."""
    if not _have_ffmpeg():
        return path, None
    dims = _probe_dims(path)
    if not dims:
        return path, None
    w, h = dims
    tgt = _target(w, h, max_pixels)
    if not tgt:
        return path, None
    w2, h2 = tgt
    tmp = Path(tempfile.mkdtemp(prefix="marlin_proxy_")) / f"{path.stem}.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
             "-vf", f"scale={w2}:{h2},fps={fps}", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "28", "-pix_fmt", "yuv420p", "-an", str(tmp)],
            check=True, stdin=subprocess.DEVNULL)
    except Exception:
        shutil.rmtree(tmp.parent, ignore_errors=True)
        return path, None
    return tmp, f"{w}×{h} → {w2}×{h2}"


class Marlin:
    def __init__(self, cfg: Config, max_pixels: int = VIDEO_MAX_PIXELS,
                 fps: float = VIDEO_FPS, full_res: bool = False):
        self.cfg = cfg
        self.max_pixels = max_pixels
        self.fps = fps
        self.full_res = full_res
        self.last_note: "str | None" = None  # set per _ask (e.g. "1920×832 → 672×288")
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.resolved_api_key,
            timeout=httpx.Timeout(300.0, connect=10.0),
            max_retries=2,
        )

    def _ask(self, video: Path, prompt: str, max_tokens: int = 1024) -> str:
        send, note = (video, None) if self.full_res else downscale_proxy(video, self.max_pixels, self.fps)
        self.last_note = note
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{
                    "role": "user",
                    "content": [_video_part(send), {"type": "text", "text": prompt}],
                }],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return strip_thinking(resp.choices[0].message.content or "")
        finally:
            if note and send != video:  # delete the temp proxy
                shutil.rmtree(send.parent, ignore_errors=True)

    def caption_events(self, video: Path) -> tuple[str, list[Event], str]:
        """Dense caption → (scene paragraph, per-event rows, raw text)."""
        raw = self._ask(video, CAPTION_PROMPT)
        scene, events = parse_caption(raw)
        return scene, events, raw

    def caption(self, video: Path) -> str:
        return self._ask(video, CAPTION_DETAIL_PROMPT)

    def ground(self, video: Path, query: str) -> tuple[tuple[float, float], str]:
        """Locate `query` in a clip → ((start_s, end_s) relative to clip, tier)."""
        raw = self._ask(video, GROUND_PROMPT.format(query=query), max_tokens=64)
        span, tier = parse_span(raw)
        return (span, tier)
