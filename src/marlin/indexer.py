"""Index pipeline: chunk → Marlin caption (timestamped events) → embed → LanceDB.

One row per parsed event (its own span) plus one row for the full chunk
caption, plus optional speech rows from faster-whisper. Resume-safe via a
chunks_done table keyed on deterministic (video, chunk_start).
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import lancedb
import pyarrow as pa

from .backend import Marlin
from .chunker import Chunk, chunk_spans, extract_chunk, is_still_chunk, probe_duration
from .config import Config
from .output import status

EVENTS_TABLE = "events"
DONE_TABLE = "chunks_done"

_EMBEDDER = None
_WHISPER = None


def get_embedder(model_name: str):
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        status(f"loading embedder {model_name} …")
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER


def _row_id(video: str, start: float, kind: str, text: str) -> str:
    return hashlib.sha256(f"{video}:{start:.2f}:{kind}:{text[:80]}".encode()).hexdigest()[:16]


def _events_schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("video", pa.string()),
        pa.field("start", pa.float32()),
        pa.field("end", pa.float32()),
        pa.field("chunk_start", pa.float32()),
        pa.field("chunk_end", pa.float32()),
        pa.field("kind", pa.string()),
        pa.field("text", pa.string()),
        pa.field("indexed_at", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])


def open_db(cfg: Config):
    return lancedb.connect(cfg.db_path)


def _open_tables(db, dim: int):
    names = set(db.table_names())
    events = (
        db.open_table(EVENTS_TABLE)
        if EVENTS_TABLE in names
        else db.create_table(EVENTS_TABLE, schema=_events_schema(dim))
    )
    done = (
        db.open_table(DONE_TABLE)
        if DONE_TABLE in names
        else db.create_table(
            DONE_TABLE,
            schema=pa.schema([pa.field("video", pa.string()), pa.field("chunk_start", pa.float32())]),
        )
    )
    return events, done


def _done_starts(done_tbl, video: str) -> set[float]:
    try:
        rows = done_tbl.search().where(f"video = '{video}'").limit(100_000).to_list()
    except Exception:
        rows = [r for r in done_tbl.to_pandas().to_dict("records") if r["video"] == video]
    return {round(float(r["chunk_start"]), 2) for r in rows}


def _transcribe(chunk: Chunk) -> list[tuple[float, float, str]]:
    """Optional STT on the raw chunk. Requires `pip install 'marlin-cli[stt]'`."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "speech indexing needs faster-whisper: pip install 'marlin-cli[stt]'"
        ) from None
    global _WHISPER
    if _WHISPER is None:
        _WHISPER = WhisperModel("distil-large-v3", compute_type="auto")
    segments, _info = _WHISPER.transcribe(str(chunk.raw), vad_filter=True)
    return [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]


@dataclass
class IndexStats:
    videos: int = 0
    chunks: int = 0
    skipped_still: int = 0
    skipped_done: int = 0
    events: int = 0
    speech_segments: int = 0
    errors: list[str] = field(default_factory=list)


def index_videos(
    cfg: Config,
    videos: list[Path],
    *,
    stt: bool = False,
    on_progress: Callable[[IndexStats, str], None] | None = None,
) -> IndexStats:
    marlin = Marlin(cfg)
    embedder = get_embedder(cfg.embed_model)
    dim = embedder.get_embedding_dimension()
    db = open_db(cfg)
    events_tbl, done_tbl = _open_tables(db, dim)

    stats = IndexStats()
    now = datetime.now(timezone.utc).isoformat()

    for video in videos:
        stats.videos += 1
        vkey = str(video)
        duration = probe_duration(video)
        if duration <= 0:
            stats.errors.append(f"unreadable: {video}")
            continue
        spans = chunk_spans(duration, cfg.chunk_seconds, cfg.chunk_overlap)
        done = _done_starts(done_tbl, vkey)

        with tempfile.TemporaryDirectory(prefix="marlin_") as td:
            workdir = Path(td)
            for start, end in spans:
                if round(start, 2) in done:
                    stats.skipped_done += 1
                    continue
                chunk = extract_chunk(video, start, end, workdir)
                if chunk is None:
                    stats.errors.append(f"ffmpeg failed: {video} @{start:.0f}s")
                    continue

                rows: list[dict] = []
                if is_still_chunk(chunk):
                    stats.skipped_still += 1
                else:
                    try:
                        scene, events, _raw = marlin.caption_events(chunk.proxy)
                    except Exception as e:
                        stats.errors.append(f"model error: {video} @{start:.0f}s: {e}")
                        chunk.raw.unlink(missing_ok=True)
                        chunk.proxy.unlink(missing_ok=True)
                        continue
                    if scene:
                        rows.append({
                            "video": vkey, "start": start, "end": end,
                            "chunk_start": start, "chunk_end": end,
                            "kind": "scene", "text": scene,
                        })
                    for ev in events:
                        abs_start = min(start + ev.start, end)
                        abs_end = min(start + ev.end, end)
                        rows.append({
                            "video": vkey, "start": abs_start, "end": abs_end,
                            "chunk_start": start, "chunk_end": end,
                            "kind": "event", "text": ev.text,
                        })
                        stats.events += 1
                    if stt:
                        for s_start, s_end, s_text in _transcribe(chunk):
                            rows.append({
                                "video": vkey,
                                "start": min(start + s_start, end),
                                "end": min(start + s_end, end),
                                "chunk_start": start, "chunk_end": end,
                                "kind": "speech", "text": s_text,
                            })
                            stats.speech_segments += 1

                if rows:
                    vectors = embedder.encode(
                        [r["text"] for r in rows], normalize_embeddings=True
                    )
                    for r, v in zip(rows, vectors):
                        r["id"] = _row_id(r["video"], r["start"], r["kind"], r["text"])
                        r["indexed_at"] = now
                        r["vector"] = v.tolist()
                    events_tbl.add(rows)

                done_tbl.add([{"video": vkey, "chunk_start": start}])
                stats.chunks += 1
                chunk.raw.unlink(missing_ok=True)
                chunk.proxy.unlink(missing_ok=True)
                if on_progress:
                    on_progress(stats, f"{video.name} @{start:.0f}s")

    try:
        events_tbl.create_fts_index("text", replace=True)
    except Exception as e:
        stats.errors.append(f"fts index: {e}")
    return stats
