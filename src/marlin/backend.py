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


class Marlin:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.resolved_api_key,
            timeout=httpx.Timeout(300.0, connect=10.0),
            max_retries=2,
        )

    def _ask(self, video: Path, prompt: str, max_tokens: int = 1024) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[{
                "role": "user",
                "content": [_video_part(video), {"type": "text", "text": prompt}],
            }],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return strip_thinking(resp.choices[0].message.content or "")

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
