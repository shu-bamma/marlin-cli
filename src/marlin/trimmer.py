"""Clip extraction for search results — ±2s padding, copy-first, re-encode fallback."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .chunker import FFMPEG, probe_duration

PAD_S = 2.0


def trim(source: Path, start: float, end: float, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(source)
    s = max(0.0, start - PAD_S)
    e = min(duration or end + PAD_S, end + PAD_S)
    dur = max(e - s, 0.5)
    name = f"{source.stem}_{int(s) // 60:02d}m{int(s) % 60:02d}s-{int(e) // 60:02d}m{int(e) % 60:02d}s.mp4"
    out = out_dir / name

    r = subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-ss", f"{s:.2f}", "-t", f"{dur:.2f}",
         "-i", str(source), "-c", "copy", "-avoid_negative_ts", "make_zero", str(out)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        r = subprocess.run(
            [FFMPEG, "-y", "-v", "error", "-ss", f"{s:.2f}", "-t", f"{dur:.2f}",
             "-i", str(source), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-c:a", "aac", str(out)],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not out.exists():
            return None
    return out


def open_in_player(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", str(path)])
