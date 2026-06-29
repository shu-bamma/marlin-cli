"""Tests for the long-video chunking grounding pipeline (video_processor.py).

Hermetic: unit tests mock ffprobe/ffmpeg and use a real temp file so the
`exists()` guard passes anywhere (not just the author's machine). The
integration tests synthesize a tiny video via ffmpeg and skip when ffmpeg
is unavailable.

Runnable via:
    uv run --with pytest pytest tests/test_video_processor.py -q
or:
    PYTHONPATH=src python3 tests/test_video_processor.py
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marlin.video_processor import (  # noqa: E402
    GroundingHit,
    VideoChunkingError,
    dedup_hits,
    extract_chunk,
    find_in_long_video,
    generate_chunks,
    probe_duration_seconds,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import have_ffmpeg, make_sample_video  # noqa: E402


def _real_temp_video() -> Path:
    """Return a path to a real (non-empty) file so exists() guards pass."""
    fd, name = tempfile.mkstemp(suffix=".mp4")
    Path(name).write_bytes(b"\x00")
    return Path(name)


def _hit(gs, ge):
    return GroundingHit(0, 0.0, 0.0, gs, ge, "q", "from_pair")


# ── generate_chunks ───────────────────────────────────────────────────────────


def test_generate_chunks_boundaries():
    chunks = generate_chunks(210.0, chunk_seconds=30.0, overlap_seconds=5.0)
    assert chunks[0].start == 0.0
    assert chunks[-1].end == 210.0  # last chunk reaches the end
    step = 30.0 - 5.0
    for i in range(1, len(chunks) - 1):
        assert abs(chunks[i].start - i * step) < 1e-6  # uniform stepping
        assert chunks[i].duration == 30.0
    # every window stays inside the video
    assert all(0.0 <= c.start < c.end <= 210.0 for c in chunks)


def test_generate_chunks_folds_tiny_tail():
    # chunk=10, overlap=0 -> step=10; a 21s video would leave a 1s tail chunk.
    chunks = generate_chunks(21.0, chunk_seconds=10.0, overlap_seconds=0.0)
    assert all(c.duration >= 2.0 for c in chunks)  # no sub-2s tail
    assert chunks[-1].end == 21.0


def test_generate_chunks_rejects_bad_args():
    for bad in (
        lambda: generate_chunks(-1.0),
        lambda: generate_chunks(10.0, chunk_seconds=0.0),
        lambda: generate_chunks(10.0, chunk_seconds=5.0, overlap_seconds=5.0),
    ):
        try:
            bad()
            raise AssertionError("expected VideoChunkingError")
        except VideoChunkingError:
            pass


# ── dedup_hits ────────────────────────────────────────────────────────────────


def test_dedup_merges_true_overlap_keeping_longer():
    # Same event seen in two overlapping chunks: heavy overlap -> one hit, longer kept.
    kept = dedup_hits([_hit(10.0, 20.0), _hit(11.0, 22.0)])
    assert len(kept) == 1
    assert (kept[0].global_start, kept[0].global_end) == (11.0, 22.0)


def test_dedup_keeps_distinct_far_events():
    kept = dedup_hits([_hit(10.0, 12.0), _hit(30.0, 32.0)])
    assert len(kept) == 2


def test_dedup_keeps_close_but_nonoverlapping_events():
    # starts 4s apart (within tolerance) but spans don't overlap -> NOT merged.
    # A start-proximity-only rule (the previous behaviour) would wrongly collapse these.
    kept = dedup_hits([_hit(10.0, 12.0), _hit(14.0, 16.0)])
    assert len(kept) == 2


# ── find_in_long_video (mocked extract/probe) ─────────────────────────────────


def _fake_extract(input_video, chunk, output_dir):
    chunk.path = Path("fake_chunk.mp4")
    return chunk


def test_find_maps_local_to_global_timestamps():
    video = _real_temp_video()
    ground = MagicMock(side_effect=[((10.0, 20.0), "from_pair"), ((15.0, 25.0), "from_pair")])
    with (
        patch("marlin.video_processor.probe_duration_seconds", return_value=210.0),
        patch("marlin.video_processor.extract_chunk", side_effect=_fake_extract) as mx,
    ):
        res = find_in_long_video(
            video, "soccer goal", ground, chunk_seconds=120.0, overlap_seconds=30.0
        )
    assert mx.call_count == 2 and ground.call_count == 2
    assert len(res.hits) == 2
    assert (res.hits[0].global_start, res.hits[0].global_end) == (10.0, 20.0)  # chunk 0 @ 0s
    assert (res.hits[1].global_start, res.hits[1].global_end) == (105.0, 115.0)  # chunk 1 @ 90s


def test_find_skips_no_match_and_clamps_empty():
    video = _real_temp_video()
    # chunk0: span runs off the end -> clamps to chunk.duration; chunk1: no_match.
    ground = MagicMock(side_effect=[((5.0, 999.0), "from_pair"), ((0.0, 0.0), "no_match")])
    with (
        patch("marlin.video_processor.probe_duration_seconds", return_value=210.0),
        patch("marlin.video_processor.extract_chunk", side_effect=_fake_extract),
    ):
        res = find_in_long_video(video, "q", ground, chunk_seconds=120.0, overlap_seconds=30.0)
    assert len(res.hits) == 1
    assert res.hits[0].global_end <= 120.0  # clamped to chunk duration, not 999


def test_find_raises_when_all_chunks_fail():
    video = _real_temp_video()
    ground = MagicMock(side_effect=RuntimeError("model down"))
    with (
        patch("marlin.video_processor.probe_duration_seconds", return_value=210.0),
        patch("marlin.video_processor.extract_chunk", side_effect=_fake_extract),
    ):
        try:
            find_in_long_video(video, "q", ground, chunk_seconds=120.0, overlap_seconds=30.0)
            raise AssertionError("expected VideoChunkingError when every chunk fails")
        except VideoChunkingError:
            pass


def test_find_rejects_missing_video():
    try:
        find_in_long_video(Path("/no/such/file.mp4"), "q", MagicMock())
        raise AssertionError("expected VideoChunkingError")
    except VideoChunkingError:
        pass


# ── integration: real ffmpeg extraction (skips without ffmpeg) ────────────────


def test_extract_chunk_real_ffmpeg_is_duration_accurate():
    if not have_ffmpeg():
        print("skip test_extract_chunk_real_ffmpeg (no ffmpeg)")
        return
    with tempfile.TemporaryDirectory() as td:
        src = make_sample_video(Path(td) / "sample.mp4", duration=12.0)
        chunks = generate_chunks(12.0, chunk_seconds=5.0, overlap_seconds=1.0)
        c = chunks[1]
        extract_chunk(src, c, Path(td) / "chunks")
        assert c.path is not None and c.path.exists() and c.path.stat().st_size > 0
        got = probe_duration_seconds(c.path)
        # Frame-accurate re-encode: extracted duration tracks the planned window.
        assert abs(got - c.duration) < 0.5, f"chunk dur {got} vs planned {c.duration}"


def test_find_in_long_video_real_ffmpeg_pipeline():
    if not have_ffmpeg():
        print("skip test_find_in_long_video_real_ffmpeg (no ffmpeg)")
        return
    with tempfile.TemporaryDirectory() as td:
        src = make_sample_video(Path(td) / "sample.mp4", duration=12.0)

        def stub_ground(_path, _query):
            return (1.0, 2.0), "from_pair"  # a hit 1-2s into every chunk

        res = find_in_long_video(
            src, "anything", stub_ground, chunk_seconds=5.0, overlap_seconds=1.0
        )
        assert res.hits, "expected at least one hit from the real pipeline"
        assert all(
            0.0 <= h.global_start < h.global_end <= res.duration_seconds + 0.5 for h in res.hits
        )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all video processor tests passed")
