"""Marlin-2B model I/O contract: canonical prompts + parsers.

Prompts and parsers mirror the released model's own remote code
(NemoStation/Marlin-2B `modeling_marlin.py`), which warns the prompt strings
"must match exactly what the model was fine-tuned on." `parse_caption`,
`parse_span`, and `strip_thinking` are aligned to that file. `parse_span`
additionally falls back to the TencentARC TimeLens 4-tier regex cascade for
stray output formats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# Release-canonical grounding prompt → 'From <start> to <end>.' in seconds.
GROUND_PROMPT = (
    'Identify the timestamps during which "{query}" takes place. '
    'Output the time range as "From <start> to <end>." (numbers in seconds).'
)

# Release-canonical caption prompt → 'Scene: <paragraph>  Events: <a - b> desc'.
CAPTION_PROMPT = (
    "Provide a spatial description of this clip followed by time-ranged events.\n"
    "For each event, give the time range as <start - end> and a short description."
)

# Tarsier-paper baseline (DREAM-1K default) — free-form dense caption.
CAPTION_DETAIL_PROMPT = "Describe the video in detail."


# --- thinking-tag stripping (verbatim from modeling_marlin.strip_thinking) ---
# Marlin's ms-swift template prefixes responses with a bare `<think>\n` (no
# close tag) and occasionally emits full `<think>...</think>` blocks.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_PREFIX = re.compile(r"^\s*<think>\s*\n*", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>\s*", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    out = _THINK_BLOCK.sub("", text)
    out = _THINK_PREFIX.sub("", out)
    out = _THINK_CLOSE.sub("", out)
    return out.strip()


# --- Mode 1: dense caption parser (mirrors modeling_marlin.parse_caption) ---


@dataclass
class Event:
    start: float
    end: float
    text: str


# Tolerates `<1.2 - 3.4>` / `1.2 - 3.4` / `1.2-3.4` with optional units, units
# ordered longest-first so "1.8 seconds" consumes the whole word.
_EVENT_LINE = re.compile(
    r"^\s*<?\s*(\d+\.?\d*)\s*(?:seconds?|secs?|s)?\s*-\s*"
    r"(\d+\.?\d*)\s*(?:seconds?|secs?|s)?\s*>?\s*[:\-]?\s*(.+?)\s*$"
)


def _parse_event_lines(block: str) -> list[Event]:
    out: list[Event] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _EVENT_LINE.match(line)
        if not m:
            continue
        start, end = float(m.group(1)), float(m.group(2))
        desc = m.group(3).strip().lstrip("-").strip()
        if end <= start or not desc:
            continue
        out.append(Event(start=start, end=end, text=desc))
    return out


def parse_caption(text: str) -> tuple[str, list[Event]]:
    """Parse a Mode-1 caption into ``(scene, events)``.

    Trained format::

        Scene: <one-paragraph spatial description>

        Events:
        <start - end> <description>
        <start - end> <description>

    Tolerant: missing ``Scene:``/``Events:`` headers fall back to "everything
    before the first event line is the scene."
    """
    cleaned = strip_thinking(text)

    scene_match = re.search(
        r"(?:^|\n)\s*Scene\s*:\s*(.*?)(?=\n\s*Events\s*:|\Z)",
        cleaned, re.IGNORECASE | re.DOTALL,
    )
    events_match = re.search(
        r"(?:^|\n)\s*Events\s*:\s*(.*)\Z", cleaned, re.IGNORECASE | re.DOTALL,
    )

    if scene_match:
        scene = scene_match.group(1).strip()
    else:
        scene_lines: list[str] = []
        for line in cleaned.splitlines():
            if _EVENT_LINE.match(line.strip()):
                break
            scene_lines.append(line)
        scene = "\n".join(scene_lines).strip()

    block = events_match.group(1) if events_match else cleaned
    return scene, _parse_event_lines(block)


# --- Mode 2: temporal grounding parser ---

ParsedTier = Literal[
    "from_pair", "mmss_pair", "dash_pair", "to_pair", "any_pair", "no_match"
]

# Release-canonical span: 'From 1.2 to 3.4.', 'From 1.2s to 3.4 sec'.
_SPAN_FROM = re.compile(
    r"From\s+(\d+\.?\d*)\s*(?:s|sec)?\s+to\s+(\d+\.?\d*)\s*(?:s|sec)?\.?",
    re.IGNORECASE,
)
# Fallback cascade (TencentARC TimeLens) for stray formats.
_TIME_REGEX = re.compile(r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b")
_DASH_PAIR = re.compile(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)")
_TO_PAIR = re.compile(r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)")
_BARE_NUM = re.compile(r"\b(\d+\.\d+|\d+)\b")


def _mmss_to_seconds(token: str) -> float:
    parts = token.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(token)


def parse_span(text: str) -> tuple[tuple[float, float], ParsedTier]:
    """Return ((start_s, end_s), tier). Release 'From X to Y' first, then cascade."""
    cleaned = strip_thinking(text)
    if not cleaned:
        return (0.0, 0.0), "no_match"

    m = _SPAN_FROM.search(cleaned)
    if m:
        s, e = float(m.group(1)), float(m.group(2))
        if e > s:
            return (s, e), "from_pair"

    mmss = _TIME_REGEX.findall(cleaned)
    if len(mmss) >= 2:
        return (_mmss_to_seconds(mmss[0]), _mmss_to_seconds(mmss[1])), "mmss_pair"

    m = _DASH_PAIR.search(cleaned)
    if m:
        return (float(m.group(1)), float(m.group(2))), "dash_pair"

    m = _TO_PAIR.search(cleaned)
    if m:
        return (float(m.group(1)), float(m.group(2))), "to_pair"

    nums = _BARE_NUM.findall(cleaned)
    if len(nums) >= 2:
        return (float(nums[0]), float(nums[1])), "any_pair"

    return (0.0, 0.0), "no_match"
