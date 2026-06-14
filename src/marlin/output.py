"""Output discipline: --json gates everything; auto-JSON when stdout is piped.

Agents parse stdout. Humans get rich tables. One switch, checked everywhere.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from rich.console import Console
from rich.theme import Theme

# NemoStation brand palette (company/brand.md). marlin = coral; hero numbers =
# splash orange; CTA/links = accent red; warm = amber; secondary = inkLight.
# No green/blue/yellow — they're off-palette.
BRAND = Theme({
    "model": "#E76F57",            # marlinCoral — model/CLI name + accents
    "accent": "#E76F57",
    "ok": "bold #FF644E",          # splashOrange — success / done
    "num": "#FF644E",              # hero numbers
    "link": "#BF3131 underline",   # accentRed — gated link / CTA
    "warn": "#D97706",             # chartAmber
    "err": "bold #BF3131",
    "muted": "#5C4A46",            # inkLight — secondary text
})

console = Console(theme=BRAND)
err_console = Console(stderr=True, theme=BRAND)

_FORCE_JSON = False


def set_json(force: bool) -> None:
    global _FORCE_JSON
    _FORCE_JSON = force


def is_json() -> bool:
    return _FORCE_JSON or not sys.stdout.isatty()


def emit(data: Any, human=None) -> None:
    """JSON to stdout in agent mode; `human()` callback (or repr) otherwise."""
    if is_json():
        sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
        sys.stdout.flush()
    elif human is not None:
        human()
    else:
        console.print(data)


def status(msg: str) -> None:
    """Progress lines go to stderr so they never corrupt JSON stdout."""
    err_console.print(f"[dim]{msg}[/dim]")


def banner() -> None:
    """The marlin banner — human mode only (callers guard with emit/is_json)."""
    console.print()
    console.print("      [model]___[/model]")
    console.print("  [model]⟩⟩⟩─< °  )≡≡≡≡▷[/model]   [model]marlin[/model]")
    console.print("      [model]‾‾‾[/model]           [muted]video understanding, on your Mac[/muted]")
    console.print("                      [muted]NemoStation ·[/muted] [model]Marlin-2B[/model]")
    console.print()
