"""Long-video chunking and grounding for Marlin-2B.

Pure chunking library — no model loading, no file I/O beyond temp chunks,
no CLI. Splits a video into overlapping chunks, calls a user-supplied
ground_fn on each, normalises local timestamps to global, deduplicates
overlap events, and returns the combined result in memory.

Usage from cli.py::

    from .video_processor import probe_duration_seconds, find_in_long_video

    duration = probe_duration_seconds(path)
    if duration > 30.0:
        result = find_in_long_video(path, query, m.ground)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .logging import get_logger

logger = get_logger("chunk")

# ── constants ─────────────────────────────────────────────────────────────────

# 30s/5s matches Config.chunk_seconds/chunk_overlap and the existing chunker.py.
# It is a correctness requirement, not just a cost choice: vLLM's Qwen3-VL path
# compresses timestamps on long clips (vllm#30847), so chunks must stay <=30s to
# ground inside the model's training distribution. See backend.py module docstring.
CHUNK_SECONDS = 30.0
OVERLAP_SECONDS = 5.0
DEDUP_TOLERANCE_SECONDS = 5.0
DEDUP_IOU_THRESHOLD = 0.5
# Fold a trailing chunk shorter than this into its predecessor.
MIN_CHUNK_SECONDS = 2.0


# ── exceptions ────────────────────────────────────────────────────────────────


class VideoChunkingError(Exception):
    """Base exception for chunking errors."""


class FFmpegError(VideoChunkingError):
    """FFmpeg or ffprobe subprocess failed."""


# ── data models ───────────────────────────────────────────────────────────────
@dataclass
class VideoChunk:
    """One time-window inside the source video."""

    chunk_id: int
    start: float
    end: float
    duration: float
    path: Path | None = None


@dataclass
class GroundingHit:
    """A single grounding match mapped to global time."""

    chunk_id: int
    local_start: float
    local_end: float
    global_start: float
    global_end: float
    description: str
    tier: str


@dataclass
class LongVideoFindResult:
    """Aggregated result of grounding across all chunks."""

    video_path: Path
    duration_seconds: float
    chunk_seconds: float
    overlap_seconds: float
    query: str
    hits: list[GroundingHit]


# ── helpers ───────────────────────────────────────────────────────────────────


def _fmt_time(seconds: float) -> str:
    """Format seconds into MM:SS.ff for log alignment."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


# ── phase 1: probe & plan ────────────────────────────────────────────────────


def probe_duration_seconds(video_path: str | Path) -> float:
    """Return video duration in seconds via ffprobe."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoChunkingError(f"Video file not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {proc.stderr.strip()}")

    try:
        data = json.loads(proc.stdout)
        return float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        raise FFmpegError(f"Could not parse ffprobe output: {exc}") from exc


def generate_chunks(
    duration_seconds: float,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
) -> list[VideoChunk]:
    """Build a list of overlapping VideoChunk windows across the video."""
    if duration_seconds <= 0:
        raise VideoChunkingError("duration_seconds must be > 0")
    if chunk_seconds <= 0:
        raise VideoChunkingError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise VideoChunkingError("overlap_seconds must be >= 0")
    if overlap_seconds >= chunk_seconds:
        raise VideoChunkingError("overlap_seconds must be < chunk_seconds")

    step = chunk_seconds - overlap_seconds
    chunks: list[VideoChunk] = []
    start = 0.0
    chunk_id = 0

    while start < duration_seconds:
        end = min(start + chunk_seconds, duration_seconds)
        if chunks and abs(end - chunks[-1].end) < 1e-6:
            break
        chunks.append(
            VideoChunk(
                chunk_id=chunk_id,
                start=round(start, 6),
                end=round(end, 6),
                duration=round(end - start, 6),
            )
        )
        chunk_id += 1
        if end >= duration_seconds:
            break
        start += step

    # Avoid a uselessly short trailing chunk: fold it into its predecessor so we
    # never spend a model pass on a sub-second tail.
    if len(chunks) > 1 and chunks[-1].duration < MIN_CHUNK_SECONDS:
        tail = chunks.pop()
        prev = chunks[-1]
        prev.end = tail.end
        prev.duration = round(prev.end - prev.start, 6)

    return chunks


# ── phase 2: extract ─────────────────────────────────────────────────────────


def extract_chunk(
    input_video: Path,
    chunk: VideoChunk,
    output_dir: Path,
) -> VideoChunk:
    """Extract a single chunk from the source video via FFmpeg.

    Re-encodes (does NOT stream-copy). With ``-ss`` before ``-i`` a re-encode is
    both fast (keyframe pre-seek) and frame-accurate: the first output frame is
    exactly at ``chunk.start``. Stream-copy (``-c copy``) snaps to the previous
    keyframe and would shift every global timestamp late by up to one GOP, which
    silently corrupts grounding results — the whole point of ``find``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    start_ms = int(chunk.start * 1000)
    end_ms = int(chunk.end * 1000)
    filename = f"chunk_{chunk.chunk_id:04d}_{start_ms}_{end_ms}.mp4"
    output_path = output_dir / filename

    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(chunk.start),
        "-i",
        str(input_video),
        "-t",
        str(chunk.duration),
        "-map",
        "0:v:0",
        "-an",  # grounding is video-only; dropping audio speeds the re-encode
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-reset_timestamps",
        "1",
        str(output_path),
    ]

    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    elapsed = time.monotonic() - t0
    if proc.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        raise FFmpegError(f"FFmpeg failed for chunk {chunk.chunk_id}: {proc.stderr.strip()}")

    chunk.path = output_path
    logger.debug("extracted chunk {} in {:.1f}s", chunk.chunk_id, elapsed)
    return chunk


# ── phase 3: dedup ────────────────────────────────────────────────────────────


def dedup_hits(
    hits: list[GroundingHit],
    tolerance: float = DEDUP_TOLERANCE_SECONDS,
    iou_threshold: float = DEDUP_IOU_THRESHOLD,
) -> list[GroundingHit]:
    """Merge near-duplicate hits from overlapping chunks.

    Two hits are treated as the same physical event (and the longer span kept)
    when their spans overlap substantially — intersection-over-union >=
    *iou_threshold*, or they overlap at all and start within *tolerance*
    seconds. Distinct events that merely sit close together (no overlap) are
    preserved, which a start-proximity-only rule would wrongly collapse.
    """
    if not hits:
        return []

    sorted_hits = sorted(hits, key=lambda h: h.global_start)
    kept: list[GroundingHit] = [sorted_hits[0]]

    for candidate in sorted_hits[1:]:
        prev = kept[-1]
        inter = max(
            0.0,
            min(candidate.global_end, prev.global_end)
            - max(candidate.global_start, prev.global_start),
        )
        span = max(candidate.global_end, prev.global_end) - min(
            candidate.global_start, prev.global_start
        )
        iou = inter / span if span > 0 else 0.0
        close_start = abs(candidate.global_start - prev.global_start) <= tolerance

        if iou >= iou_threshold or (inter > 0 and close_start):
            cand_dur = candidate.global_end - candidate.global_start
            prev_dur = prev.global_end - prev.global_start
            if cand_dur > prev_dur:
                kept[-1] = candidate
        else:
            kept.append(candidate)

    return kept


# ── phase 4: main pipeline ───────────────────────────────────────────────────

# Type alias for the callable the CLI passes in (backend.Marlin.ground).
# Signature: ground_fn(video: Path, query: str) -> ((start, end), tier)
GroundFn = Callable[[Path, str], tuple[tuple[float, float], str]]


def find_in_long_video(
    video_path: Path,
    query: str,
    ground_fn: GroundFn,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
    dedup_tolerance: float = DEDUP_TOLERANCE_SECONDS,
    on_chunk_start: Callable[[int, int, float, float], Any] | None = None,
) -> LongVideoFindResult:
    """Chunk a long video, run *ground_fn* on each chunk, and return merged hits.

    Parameters
    ----------
    video_path
        Path to the source video file.
    query
        Natural-language description of the event to locate.
    ground_fn
        Callable with signature ``(video: Path, query: str) -> ((start, end), tier)``.
        Typically ``backend.Marlin.ground``.
    chunk_seconds
        Duration per chunk window.
    overlap_seconds
        Overlap between consecutive chunks.
    dedup_tolerance
        Start-time tolerance for deduplication.
    on_chunk_start
        Optional callback ``(chunk_idx, total_chunks, start_sec, end_sec)``
        invoked before each chunk is processed (for progress UI).

    Returns
    -------
    LongVideoFindResult
        Aggregated, deduplicated grounding hits in global time.

    Raises
    ------
    VideoChunkingError
        If the video is missing, or every chunk fails to extract/ground (so an
        all-failure run is never silently reported as "not found").
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoChunkingError(f"Video file not found: {video_path}")

    duration = probe_duration_seconds(video_path)
    chunks = generate_chunks(duration, chunk_seconds, overlap_seconds)
    total = len(chunks)

    logger.info(
        "chunking {} ({:.1f}s) into {} chunks ({}s window, {}s overlap)",
        video_path.name,
        duration,
        total,
        chunk_seconds,
        overlap_seconds,
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="marlin_chunks_"))
    raw_hits: list[GroundingHit] = []
    errored = 0

    try:
        for chunk in chunks:
            if on_chunk_start:
                on_chunk_start(chunk.chunk_id, total, chunk.start, chunk.end)

            try:
                extract_chunk(video_path, chunk, temp_dir)
                (local_start, local_end), tier = ground_fn(chunk.path, query)
            except Exception as exc:
                errored += 1
                logger.warning("chunk {} failed: {}", chunk.chunk_id, exc)
                continue
            finally:
                # Don't let chunk files accumulate across a multi-hour video.
                if chunk.path is not None:
                    Path(chunk.path).unlink(missing_ok=True)

            if tier == "no_match":
                continue

            # Clamp to chunk boundaries first, then drop empty/inverted spans.
            local_start = max(0.0, min(local_start, chunk.duration))
            local_end = max(0.0, min(local_end, chunk.duration))
            if local_end <= local_start:
                continue

            raw_hits.append(
                GroundingHit(
                    chunk_id=chunk.chunk_id,
                    local_start=round(local_start, 2),
                    local_end=round(local_end, 2),
                    global_start=round(chunk.start + local_start, 2),
                    global_end=round(chunk.start + local_end, 2),
                    description=query,
                    tier=tier,
                )
            )

        if total > 0 and errored == total:
            raise VideoChunkingError(
                f"all {total} chunks failed to extract/ground — see warnings above"
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    deduped = dedup_hits(raw_hits, tolerance=dedup_tolerance)

    logger.info(
        "chunking complete: {} raw hits → {} after dedup",
        len(raw_hits),
        len(deduped),
    )

    return LongVideoFindResult(
        video_path=video_path,
        duration_seconds=duration,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        query=query,
        hits=deduped,
    )


# ── serialisation (for visualizer) ────────────────────────────────────────────


def hits_to_visualizer_events(result: LongVideoFindResult) -> list[dict]:
    """Convert hits to the dict format expected by the HTML visualizer."""
    return [
        {
            "global_start": h.global_start,
            "global_end": h.global_end,
            "description": h.description,
            "chunk_id": h.chunk_id,
        }
        for h in result.hits
    ]
