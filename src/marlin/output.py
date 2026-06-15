"""Output discipline: --json gates everything; auto-JSON when stdout is piped.

Agents parse stdout. Humans get rich tables. One switch, checked everywhere.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
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
    "status.spinner": "#E76F57",   # override Rich's green default spinner
})

console = Console(theme=BRAND)
err_console = Console(stderr=True, theme=BRAND)

# Brand spinner — a marlin swimming (design ②: mascot-as-motion). Registered
# into Rich's spinner table so console.status(spinner="marlin", …) can use it.
try:
    from rich._spinners import SPINNERS

    SPINNERS.setdefault("marlin", {
        "interval": 110,
        "frames": [
            "><>      ", " ><>     ", "  ><>    ", "   ><>   ", "    ><>  ",
            "     ><> ", "      ><>", "     <>< ", "    <><  ", "   <><   ",
            "  <><    ", " <><     ", "<><      ",
        ],
    })
except Exception:  # pragma: no cover — spinner is cosmetic; never block on it
    pass

# First-run / --version hero: the gradient block wordmark. Vertical fade from
# splash-orange to accent-red; Rich auto-degrades on non-truecolor terminals
# and honors NO_COLOR. Shown only by setup + version, never on hot-path.
_HERO_LINES = (
    "███╗   ███╗ █████╗ ██████╗ ██╗     ██╗███╗   ██╗",
    "████╗ ████║██╔══██╗██╔══██╗██║     ██║████╗  ██║",
    "██╔████╔██║███████║██████╔╝██║     ██║██╔██╗ ██║",
    "██║╚██╔╝██║██╔══██║██╔══██╗██║     ██║██║╚██╗██║",
    "██║ ╚═╝ ██║██║  ██║██║  ██║███████╗██║██║ ╚████║",
)
_HERO_GRADIENT = ("#FF644E", "#EF5747", "#DF4B40", "#CF3E38", "#BF3131")

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
    """First-run / version hero — the gradient block wordmark (design ②).

    Human mode only (callers guard with emit/is_json). Reserved for setup and
    `version`; hot-path commands print no banner so piped/scripted output stays
    clean.
    """
    console.print()
    for line, color in zip(_HERO_LINES, _HERO_GRADIENT):
        console.print(f"  [{color}]{line}[/]")
    console.print("  [muted]video understanding, on your Mac ·[/muted] [model]Marlin-2B[/model]")
    console.print()


@contextmanager
def spinner(title: str, *, fish: bool = False):
    """Hide a slow, noisy step behind one clean live line.

    Human mode: a spinner on stderr whose label is swapped via the yielded
    ``log(msg)`` — a swimming marlin (splash-orange) when ``fish`` else calm
    coral dots. Agent/JSON mode: plain dim stderr lines (no spinner, no control
    codes to corrupt a piped log). Either way callers get a ``log``; success /
    failure lines are the caller's job, printed after the block.
    """
    if is_json():
        err_console.print(f"[muted]{title}…[/muted]")
        yield lambda m: err_console.print(f"[muted]  {m}[/muted]")
    else:
        name, style = ("marlin", "#FF644E") if fish else ("dots", "model")
        with err_console.status(f"[model]{title}…[/model]", spinner=name, spinner_style=style) as st:
            yield lambda m: st.update(f"[model]{title} — {m}…[/model]")


@contextmanager
def build_spinner(title: str):
    """Spinner + live elapsed clock for an opaque multi-minute build.

    No %-bar on purpose: the heavy phases (Metal-kernel compile, PyTorch/MLX
    download) expose no parseable progress and vary by machine, so a fill-bar
    would stall near 100%. Callers pass "[k/N] phase" labels via the yielded
    ``log``; the elapsed clock proves it's still moving. Agent/JSON mode: plain
    dim stderr lines.
    """
    if is_json():
        err_console.print(f"[muted]{title}…[/muted]")
        yield lambda m: err_console.print(f"[muted]  {m}[/muted]")
    else:
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        with Progress(
            SpinnerColumn(spinner_name="dots", style="#E76F57"),
            TextColumn("[model]{task.description}[/model]"),
            TextColumn("[muted]·[/muted]"),
            TimeElapsedColumn(),
            console=err_console,
            transient=True,
        ) as prog:
            task = prog.add_task(title, total=None)
            yield lambda m: prog.update(task, description=f"{title} — {m}")
