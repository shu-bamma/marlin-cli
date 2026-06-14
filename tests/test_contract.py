"""Contract parser tests — runnable with plain python3 (no deps):
    PYTHONPATH=src python3 tests/test_contract.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marlin.contract import parse_caption, parse_span, strip_thinking  # noqa: E402

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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all contract tests passed")
