"""HTML visualizer helper module for Marlin grounding results.

Generates a self-contained interactive web player from events data and
opens it in the default browser.
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

from .models.media import Event

TEMPLATE_PATH = Path(__file__).parent / "templates/view"


def generate_and_open(
    video_path: Path,
    events: list[dict] | list[Event],
    query: str,
    *,
    duration: float | None = None,
    output_html_path: Path | None = None,
) -> None:
    """Generate the HTML visualizer dashboard and open it in a browser."""
    if duration is None:
        try:
            from .video_processor import probe_duration_seconds

            duration = probe_duration_seconds(video_path)
        except Exception:
            max_end = 0.0
            for e in events:
                if hasattr(e, "end"):
                    max_end = max(max_end, e.end)
                elif isinstance(e, dict):
                    max_end = max(max_end, e.get("end", e.get("global_end", 0.0)))
            duration = max_end + 1.0

    if output_html_path is None:
        from . import config as cfg_mod

        prefix = "marlin_caption" if query == "Dense Captioning" else "marlin_find"
        output_html_path = cfg_mod.CONFIG_DIR / "views" / f"{prefix}_{video_path.stem}.html"

    normalized_events = []
    for e in events:
        if hasattr(e, "start") and hasattr(e, "end") and hasattr(e, "text"):
            normalized_events.append(
                {
                    "global_start": e.start,
                    "global_end": e.end,
                    "description": e.text,
                    "chunk_id": getattr(e, "chunk_id", 0),
                }
            )
        elif isinstance(e, dict):
            if "start" in e and "end" in e and "text" in e:
                normalized_events.append(
                    {
                        "global_start": e["start"],
                        "global_end": e["end"],
                        "description": e["text"],
                        "chunk_id": e.get("chunk_id", 0),
                    }
                )
            else:
                normalized_events.append(
                    {
                        "global_start": e.get("global_start", 0.0),
                        "global_end": e.get("global_end", 0.0),
                        "description": e.get("description", ""),
                        "chunk_id": e.get("chunk_id", 0),
                    }
                )
        else:
            normalized_events.append(e)

    data = {
        "duration_seconds": duration,
        "events": normalized_events,
        "query": query,
    }

    # Obtain absolute file URI of the video for secure local playback
    video_uri = video_path.resolve().as_uri()

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Visualizer template not found at {TEMPLATE_PATH}")
    html_template = TEMPLATE_PATH.read_text(encoding="utf-8")

    # Escape the JSON for safe <script> embedding: json.dumps does not escape
    # "</", so a description containing "</script>" would break out of the tag.
    safe = json.dumps(data, indent=2).replace("</", "<\\/").replace("<!--", "<\\!--")
    patched_html = html_template.replace("__DATA_JSON_PLACEHOLDER__", safe).replace(
        "__VIDEO_PATH_PLACEHOLDER__", video_uri
    )

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.write_text(patched_html, encoding="utf-8")

    from .output import console, is_json

    if not is_json():
        console.print(f"  [ok]✓[/ok] visualizer → [bold]{output_html_path.name}[/bold]")

    webbrowser.open(output_html_path.as_uri())
