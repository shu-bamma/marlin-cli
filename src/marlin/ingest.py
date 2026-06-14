"""Input resolution: local files, directories, and YouTube/HTTP URLs.

URLs are downloaded once via yt-dlp into ~/.marlin/downloads/ and then
treated exactly like local files (Marlin has no native URL ingestion).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .chunker import VIDEO_EXTS
from .config import DOWNLOADS_DIR
from .output import status

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def is_url(s: str) -> bool:
    return bool(_URL_RE.match(s))


def download_url(url: str) -> Path:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    out_tpl = str(DOWNLOADS_DIR / "%(id)s.%(ext)s")
    status(f"downloading {url} …")
    r = subprocess.run(
        ["yt-dlp", "-f", "mp4/bestvideo*+bestaudio/best", "--merge-output-format", "mp4",
         "-o", out_tpl, "--no-playlist", "--print", "after_move:filepath", url],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {r.stderr.strip().splitlines()[-1] if r.stderr else 'unknown error'}")
    path = Path(r.stdout.strip().splitlines()[-1])
    if not path.exists():
        raise RuntimeError("yt-dlp reported success but no file found")
    return path


def resolve_inputs(inputs: list[str]) -> list[Path]:
    """Expand URLs (download), directories (recurse), and files into video paths."""
    videos: list[Path] = []
    for item in inputs:
        if is_url(item):
            videos.append(download_url(item))
            continue
        p = Path(item).expanduser()
        if p.is_dir():
            videos.extend(sorted(
                f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTS and f.is_file()
            ))
        elif p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
    # de-dupe, keep order
    seen: set[Path] = set()
    out = []
    for v in videos:
        rv = v.resolve()
        if rv not in seen:
            seen.add(rv)
            out.append(rv)
    return out
