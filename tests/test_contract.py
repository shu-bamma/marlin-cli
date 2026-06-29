"""Contract parser tests — runnable without a test runner:
PYTHONPATH=src python3 tests/test_contract.py
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marlin import __version__  # noqa: E402
from marlin.contract import parse_caption, parse_span, strip_thinking  # noqa: E402
from marlin.models import Config, Event  # noqa: E402

# Verbatim Marlin-2B output captured from the live hosted endpoint
# (caption_video.mp4, 2026-06-12) — the format the parser must handle.
REAL_CAPTION = (
    "Scene: A modern bathroom setting where a person is washing a golden "
    "retriever. The dog is positioned in a stainless steel sink, with its front "
    "paws resting on the edge. A person wearing a green shirt is visible on the "
    "right side of the frame, holding a pink and white plush toy. The lighting "
    "is bright and even, typical of a domestic bathroom environment.\n\n"
    "Events:\n"
    "<0.0 - 1.0> The person holds a plush toy near the dog.\n"
    "<1.0 - 3.0> The person moves the plush toy under the water.\n"
    "<3.0 - 4.0> The dog lowers its head toward the sink.\n"
    "<4.0 - 8.5> The dog licks its nose and mouth repeatedly."
)


def test_real_caption_regression():
    scene, events = parse_caption(REAL_CAPTION)
    assert scene.startswith("A modern bathroom setting")
    assert "Events:" not in scene and "<0.0" not in scene  # clean split
    assert len(events) == 4
    assert (events[0].start, events[0].end) == (0.0, 1.0)
    assert events[0].text == "The person holds a plush toy near the dog."
    assert (events[3].start, events[3].end) == (4.0, 8.5)
    assert events[3].text == "The dog licks its nose and mouth repeatedly."


def test_caption_units_and_brackets():
    scene, events = parse_caption(
        "Scene: a lab.\nEvents:\n<1.8 seconds - 3.4 sec> a beaker bubbles\n5-9: smoke rises"
    )
    assert scene == "a lab."
    assert (events[0].start, events[0].end, events[0].text) == (1.8, 3.4, "a beaker bubbles")
    assert (events[1].start, events[1].end, events[1].text) == (5.0, 9.0, "smoke rises")


def test_caption_no_headers_fallback():
    # No Scene:/Events: headers — everything before the first event is the scene.
    scene, events = parse_caption("a quiet suburban street\n2-5: a cat crosses")
    assert scene == "a quiet suburban street"
    assert len(events) == 1 and events[0].text == "a cat crosses"


def test_caption_strips_thinking():
    scene, events = parse_caption("<think>\nScene: x\nEvents:\n0-1: y")
    assert scene == "x" and len(events) == 1


def test_parse_span_release_from():
    assert parse_span("From 3.5 to 4.5.") == ((3.5, 4.5), "from_pair")
    assert parse_span("From 12.5s to 20.1 sec") == ((12.5, 20.1), "from_pair")
    # think-stripped before matching
    assert parse_span("<think>hmm</think>From 4 to 8.") == ((4.0, 8.0), "from_pair")


def test_parse_span_fallback_cascade():
    # No "From..to" → cascade tiers
    assert parse_span("The event happens in 3.5 - 8.2 seconds.") == ((3.5, 8.2), "dash_pair")
    assert parse_span("between 01:15 and 01:21") == ((75.0, 81.0), "mmss_pair")
    assert parse_span("the deer is not visible") == ((0.0, 0.0), "no_match")


def test_strip_thinking():
    assert strip_thinking("<think>hmm 1 2</think>From 3.5 to 8.2.") == "From 3.5 to 8.2."
    assert strip_thinking("<think>\nFrom 3.5 to 8.2.") == "From 3.5 to 8.2."
    assert strip_thinking("From 3.5 to 8.2.</think>") == "From 3.5 to 8.2."
    assert strip_thinking("From 3.5 to 8.2.") == "From 3.5 to 8.2."


def _capture_cli_json(fn, *args, **kwargs):
    from marlin.output import set_json

    set_json(True)
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            fn(*args, **kwargs)
    finally:
        set_json(False)
    return json.loads(stdout.getvalue())


class _FakeMarlin:
    def __init__(self, *args, **kwargs):
        self.last_note = None

    def caption_events(self, video):
        return (
            "a quiet platform at night",
            [Event(start=0.0, end=2.5, text="a train arrives")],
            "raw",
        )

    def ground(self, video, query):
        return (4.0, 6.25), "from_pair"

    def ground_video(
        self,
        video,
        query,
        on_chunk_start=None,
        chunk_seconds=30.0,
        overlap_seconds=5.0,
    ):
        from marlin.backend import GroundResult

        return GroundResult(
            events=[
                {
                    "global_start": 4.0,
                    "global_end": 6.25,
                    "description": query,
                    "chunk_id": 0,
                }
            ],
            found=True,
            duration=None,
            chunked=False,
            start=4.0,
            end=6.25,
            tier="from_pair",
        )


def test_caption_command_json_shape():
    from marlin import backend, cli

    original_ready_clip = cli._ready_clip
    original_marlin = backend.Marlin
    try:
        cli._ready_clip = lambda video: (Config(), Path(video))
        backend.Marlin = _FakeMarlin
        payload = _capture_cli_json(
            cli.caption,
            "clip.mp4",
            detail=False,
            max_pixels=200704,
            fps=2.0,
            full_res=False,
        )
    finally:
        cli._ready_clip = original_ready_clip
        backend.Marlin = original_marlin

    assert payload == {
        "video": "clip.mp4",
        "scene": "a quiet platform at night",
        "events": [{"start": 0.0, "end": 2.5, "text": "a train arrives"}],
    }


def test_find_command_json_shape():
    from marlin import backend, cli

    original_ready_clip = cli._ready_clip
    original_marlin = backend.Marlin
    try:
        cli._ready_clip = lambda video: (Config(), Path(video))
        backend.Marlin = _FakeMarlin
        payload = _capture_cli_json(
            cli.find,
            "clip.mp4",
            "train arrives",
            max_pixels=200704,
            fps=2.0,
            full_res=False,
        )
    finally:
        cli._ready_clip = original_ready_clip
        backend.Marlin = original_marlin

    assert payload == {
        "video": "clip.mp4",
        "query": "train arrives",
        "start": 4.0,
        "end": 6.25,
        "found": True,
        "tier": "from_pair",
        "events": [
            {
                "global_start": 4.0,
                "global_end": 6.25,
                "description": "train arrives",
                "chunk_id": 0,
            }
        ],
    }


def test_json_stdout_stays_parseable_when_logging_to_stderr():
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    with tempfile.TemporaryDirectory() as home:
        env["PYTHONPATH"] = str(repo / "src")
        env["MARLIN_HOME"] = home
        env["MARLIN_LOG_STDERR"] = "1"
        env["MARLIN_LOG_LEVEL"] = "DEBUG"
        env.pop("MARLIN_LOG_FILE_ENABLED", None)
        env.pop("MARLIN_LOG_FILE", None)
        env.pop("MARLIN_LOG_DIR", None)

        proc = subprocess.run(
            [sys.executable, "-m", "marlin.cli", "--json", "version"],
            capture_output=True,
            text=True,
            env=env,
            timeout=20,
            check=False,
        )

        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout) == {"version": __version__}
        assert "cli entrypoint initialized" in proc.stderr
        assert "cli entrypoint initialized" not in proc.stdout
        assert not (Path(home) / "logs" / "marlin.log").exists()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all contract tests passed")
