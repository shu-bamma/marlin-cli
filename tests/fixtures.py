"""Test fixtures — synthesize sample videos on the fly via ffmpeg.

We deliberately do NOT commit a sample video (this is a published PyPI
package); long-video tests generate one in a temp dir and skip when ffmpeg
is unavailable. Each second is labelled with a burned-in timer so the clip
is visually verifiable by a human running the live demo.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def make_sample_video(out_path: Path, duration: float = 12.0, fps: int = 5) -> Path:
    """Generate a small test video of *duration* seconds.

    Uses the built-in ``testsrc2`` source (a moving pattern that already renders
    a per-frame timer), so it needs no fonts/filters beyond core ffmpeg. Tiny (a
    few hundred KB) and fast to encode.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=320x240:rate={fps}:duration={duration}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-t",
        str(duration),
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    return out_path
