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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from .contract import (
    CAPTION_DETAIL_PROMPT,
    CAPTION_PROMPT,
    GROUND_PROMPT,
    Event,
    parse_caption,
    parse_span,
    strip_thinking,
)
from .logging import get_logger
from .models import Config
from .video_processor import CHUNK_SECONDS, OVERLAP_SECONDS

logger = get_logger("backend")

_MIME = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".m4v": "video/x-m4v",
}


def probe(base_url: str, api_key: str = "", timeout: float = 3.0) -> bool:
    """Return whether an OpenAI-compatible server answers.

    Parameters
    ----------
    base_url
        OpenAI-compatible API base URL.
    api_key
        Optional bearer token.
    timeout
        Request timeout in seconds.

    Returns
    -------
    bool
        ``True`` when the server responds with an acceptable status code.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=timeout)
        return r.status_code in (200, 401, 403) if api_key == "" else r.status_code == 200
    except httpx.HTTPError as exc:
        logger.debug("server probe failed for {}: {}", base_url, exc)
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


def _probe_dims(path: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            stdin=subprocess.DEVNULL,
        ).stdout.strip()
        w, h = (int(x) for x in out.split("x")[:2])
        return (w, h) if w > 0 and h > 0 else None
    except Exception as exc:
        logger.debug("could not probe video dimensions for {}: {}", path, exc)
        return None


def _target(w: int, h: int, max_pixels: int) -> tuple[int, int] | None:
    """Return the model-compatible downscaled resolution.

    Parameters
    ----------
    w
        Source width.
    h
        Source height.
    max_pixels
        Per-frame pixel budget.

    Returns
    -------
    tuple[int, int] | None
        Target width and height, or ``None`` when no downscale is needed.
    """
    if w * h <= max_pixels:
        return None
    beta = math.sqrt(w * h / max_pixels)
    w2 = max(_FACTOR, math.floor(w / beta / _FACTOR) * _FACTOR)
    h2 = max(_FACTOR, math.floor(h / beta / _FACTOR) * _FACTOR)
    return w2, h2


def downscale_proxy(
    path: Path, max_pixels: int = VIDEO_MAX_PIXELS, fps: float = VIDEO_FPS
) -> tuple[Path, str | None]:
    """Create a small proxy video when the source exceeds the model budget.

    Parameters
    ----------
    path
        Source video path.
    max_pixels
        Per-frame pixel budget.
    fps
        Sampling frame rate for the proxy.

    Returns
    -------
    tuple[Path, str | None]
        Path to send to inference and an optional human-readable resize note.
    """
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
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vf",
                f"scale={w2}:{h2},fps={fps}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(tmp),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        logger.warning("failed to create downscale proxy for {}: {}", path, exc)
        shutil.rmtree(tmp.parent, ignore_errors=True)
        return path, None
    return tmp, f"{w}×{h} → {w2}×{h2}"


class Marlin:
    """OpenAI-compatible Marlin inference client.

    Parameters
    ----------
    cfg
        Runtime configuration.
    max_pixels
        Per-frame pixel budget used for optional client-side downscaling.
    fps
        Video sampling frame rate.
    full_res
        Whether to bypass client-side downscaling.
    """

    def __init__(
        self,
        cfg: Config,
        max_pixels: int = VIDEO_MAX_PIXELS,
        fps: float = VIDEO_FPS,
        full_res: bool = False,
    ):
        self.cfg = cfg
        self.max_pixels = max_pixels
        self.fps = fps
        self.full_res = full_res
        self.last_note: str | None = None  # set per _ask (e.g. "1920×832 → 672×288")
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.resolved_api_key,
            timeout=httpx.Timeout(300.0, connect=10.0),
            max_retries=2,
        )

    def _ask(self, video: Path, prompt: str, max_tokens: int = 1024) -> str:
        send, note = (
            (video, None) if self.full_res else downscale_proxy(video, self.max_pixels, self.fps)
        )
        self.last_note = note
        if note:
            logger.info("using downscaled video proxy: {}", note)
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {
                        "role": "user",
                        "content": [_video_part(send), {"type": "text", "text": prompt}],
                    }
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return strip_thinking(resp.choices[0].message.content or "")
        except Exception:
            logger.exception("model request failed")
            raise
        finally:
            if note and send != video:  # delete the temp proxy
                shutil.rmtree(send.parent, ignore_errors=True)

    def caption_events(self, video: Path) -> tuple[str, list[Event], str]:
        """Return a scene description, parsed events, and raw model text.

        Parameters
        ----------
        video
            Clip to caption.

        Returns
        -------
        tuple[str, list[Event], str]
            Scene paragraph, parsed event rows, and raw response text.
        """
        raw = self._ask(video, CAPTION_PROMPT)
        scene, events = parse_caption(raw)
        return scene, events, raw

    def caption(self, video: Path) -> str:
        """Return one free-form dense caption paragraph.

        Parameters
        ----------
        video
            Clip to caption.
        """
        return self._ask(video, CAPTION_DETAIL_PROMPT)

    def ground(self, video: Path, query: str) -> tuple[tuple[float, float], str]:
        """Locate a query inside a single clip.

        Parameters
        ----------
        video
            Clip to search.
        query
            Natural-language event description.

        Returns
        -------
        tuple[tuple[float, float], str]
            Relative ``(start, end)`` span and parser tier.
        """
        raw = self._ask(video, GROUND_PROMPT.format(query=query), max_tokens=64)
        span, tier = parse_span(raw)
        return (span, tier)

    def ground_video(
        self,
        video: Path,
        query: str,
        on_chunk_start: Callable[[int, int, float, float], Any] | None = None,
        chunk_seconds: float = CHUNK_SECONDS,
        overlap_seconds: float = OVERLAP_SECONDS,
    ) -> GroundResult:
        """Locate a query in a video of any length.

        Probes the video duration; if longer than chunk_seconds the video is
        automatically chunked and each chunk is grounded via ``self.ground``.
        Results are mapped to global timestamps, deduplicated, and returned
        as a single ``GroundResult``.

        Parameters
        ----------
        video
            Video file to search.
        query
            Natural-language event description.
        on_chunk_start
            Optional progress callback ``(idx, total, start_sec, end_sec)``.
        chunk_seconds
            Duration per chunk window in seconds.
        overlap_seconds
            Overlap between consecutive chunks in seconds.

        Returns
        -------
        GroundResult
        """
        from .video_processor import (
            find_in_long_video,
            hits_to_visualizer_events,
            probe_duration_seconds,
        )

        try:
            duration = probe_duration_seconds(video)
        except Exception:
            duration = None

        # Chunk whenever the video exceeds the *requested* window, so a custom
        # --chunk-seconds actually takes effect (the module default is only the
        # fallback). Using the constant here would silently ignore the flag for
        # any video between chunk_seconds and CHUNK_SECONDS.
        chunked = duration is not None and duration > chunk_seconds

        if chunked:
            long_result = find_in_long_video(
                video_path=video,
                query=query,
                ground_fn=self.ground,
                on_chunk_start=on_chunk_start,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
            )
            events = hits_to_visualizer_events(long_result)
            return GroundResult(
                events=events,
                found=len(events) > 0,
                duration=long_result.duration_seconds,
                chunked=True,
                # single-clip fields left at defaults
            )

        # Short video — direct single-clip grounding.
        (start, end), tier = self.ground(video, query)
        found = tier != "no_match"
        events = []
        if found:
            events.append(
                {
                    "global_start": round(start, 2),
                    "global_end": round(end, 2),
                    "description": query,
                    "chunk_id": 0,
                }
            )
        return GroundResult(
            events=events,
            found=found,
            duration=duration,
            chunked=False,
            start=start,
            end=end,
            tier=tier,
        )


@dataclass
class GroundResult:
    """Unified result from ``Marlin.ground_video``."""

    events: list[dict] = field(default_factory=list)
    found: bool = False
    duration: float | None = None
    chunked: bool = False
    # populated only when chunked=False (single-clip path)
    start: float = 0.0
    end: float = 0.0
    tier: str = ""
