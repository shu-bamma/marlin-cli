"""Agent-first Marlin CLI.

Every verb honors ``--json``; stdout is parseable when piped, and progress goes
to stderr.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.table import Table

from . import __version__
from . import config as cfg_mod
from .config import DEFAULT_LOCAL_URL
from .logging import configure_logging, get_logger
from .models import Config
from .output import banner, build_spinner, console, emit, err_console, is_json, set_json, spinner
from .output import status as echo

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
skills_app = typer.Typer(no_args_is_help=True)
app.add_typer(skills_app, name="skills", help="Install agent skills that ride this CLI.")
logger = get_logger("cli")


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Force JSON output (auto when piped)."),
):
    set_json(json_out)
    if ctx.invoked_subcommand is not None:
        return
    # Bare `marlin`: onboard on first run (no separate setup step), else show help.
    if not cfg_mod.configured():
        _do_setup()
    else:
        typer.echo(ctx.get_help())


def _require_config() -> Config:
    if cfg_mod.configured():
        return cfg_mod.load()
    # Unconfigured + interactive terminal: onboard inline so `marlin find …`
    # works on the very first command. Non-interactive/piped → clear error.
    if not is_json() and sys.stdin.isatty():
        _do_setup()
        if cfg_mod.configured():
            return cfg_mod.load()
    err_console.print(
        "[err]not configured[/err] — run [bold]marlin setup[/bold] "
        "(or set MARLIN_BASE_URL for non-interactive use)"
    )
    logger.warning("command blocked because Marlin is not configured")
    raise typer.Exit(2)


def _require_signin() -> None:
    """Require Google sign-in for interactive human sessions."""
    from . import auth

    if auth.email() or is_json() or not sys.stdin.isatty():
        return
    if auth.google_enabled() is False:
        return
    console.print(
        "  [bold]Sign in to use Marlin[/bold] [muted](opens your browser — "
        "2 quick questions, then Google)[/muted]"
    )
    try:
        auth.login(log=echo)
    except RuntimeError as e:
        logger.exception("interactive sign-in failed")
        err_console.print(f"  [err]sign-in required[/err] — {e}")
        raise typer.Exit(1) from None


def _require_index_extra() -> None:
    """Require optional dependencies for hidden index/search commands."""
    try:
        import lancedb  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        logger.warning("index/search command requested without optional index dependencies")
        err_console.print(
            r"  [err]index/search needs the optional extra[/err] — install it:"
            "\n"
            r"      [bold]uv tool install 'nemostation\[index]'[/bold]  "
            r"[muted](or: pip install 'nemostation\[index]')[/muted]"
        )
        raise typer.Exit(2) from None


def _platform_human(p: str) -> str:
    return {
        "apple_silicon": "Apple Silicon (Metal)",
        "nvidia": "NVIDIA CUDA GPU",
        "other": "no local GPU detected (need Apple Silicon or NVIDIA)",
    }.get(p, p)


def _short_platform(p: str) -> str:
    return {"apple_silicon": "Apple Silicon", "nvidia": "NVIDIA", "other": "this machine"}.get(p, p)


def _next_steps() -> None:
    console.print()
    console.print("  Try it on a clip:")
    console.print(
        "      [bold]marlin caption clip.mp4[/bold]                 "
        "[muted]# describe a video[/muted]"
    )
    console.print(
        '      [bold]marlin find clip.mp4 "a deer crossing"[/bold]  '
        "[muted]# locate a moment[/muted]"
    )
    console.print()


def _do_setup(
    *,
    engine: str = "",
    build: bool = True,
    non_interactive: bool = False,
    local: bool = False,
    hosted: bool = False,
    base_url: str = "",
    api_key: str = "",
) -> None:
    """Run shared first-run and explicit setup logic.

    Parameters
    ----------
    engine
        Explicit engine override.
    build
        Whether to build a missing local engine inline.
    non_interactive
        Disable prompts.
    local
        Hidden local-mode compatibility flag.
    hosted
        Hidden hosted-mode flag.
    base_url
        Hosted or local OpenAI-compatible base URL.
    api_key
        Hosted API key.
    """
    from . import engines
    from .backend import probe
    from .output import is_json

    # No ffmpeg gate: caption/find send the clip to the engine as-is. ffmpeg is
    # only needed to window videos >2 min (optional) — never block setup on it.

    cfg = cfg_mod.load()
    detected = engines.detect_platform()
    rec = engines.default_engine()  # mlx | vllm | hosted
    logger.info("setup started: detected_platform={} recommended_engine={}", detected, rec)

    human_mode = not is_json()

    # Engine = the machine, not a question. The CLI ships the Apple-Silicon (MLX)
    # build only for now — it's the validated, public, 8-bit path. NVIDIA/other
    # auto-detect exits gracefully; the vLLM path stays reachable via the explicit
    # (hidden) --engine vllm flag for internal use. --hosted is also honored.
    eng = engine or ("hosted" if hosted else "")
    if not eng:
        if detected == "apple_silicon":
            eng = rec  # mlx
        else:
            err_console.print(
                "   [warn]Apple Silicon only for now[/warn] — Marlin's CLI ships the "
                "Metal (MLX) build, which needs an Apple-Silicon Mac"
            )
            if detected == "nvidia":
                err_console.print(
                    "  [muted]NVIDIA detected — an optimized NVIDIA build is coming as a "
                    "separate release.[/muted]"
                )
            logger.warning("setup stopped on unsupported platform: {}", detected)
            raise typer.Exit(2)
    interactive = human_mode and not non_interactive

    if human_mode:
        banner()
        if eng != "hosted":
            console.print(
                f"  [muted]{_platform_human(detected)} detected — "
                "Marlin runs on this machine, free.[/muted]"
            )

    if eng == "hosted":
        cfg.mode, cfg.engine = "hosted", "hosted"
        cfg.base_url = (
            base_url
            or (
                typer.prompt("  hosted base URL", default=cfg.base_url)
                if interactive
                else cfg.base_url
            )
        ).rstrip("/")
        cfg.api_key = (
            api_key
            or os.environ.get("MARLIN_API_KEY", "")
            or (typer.prompt("  API key", hide_input=True) if interactive else "")
        )
        if not cfg.base_url:
            logger.warning("hosted setup failed because base URL was empty")
            err_console.print("[err]hosted mode needs --base-url[/err]")
            raise typer.Exit(2)
    else:
        cfg.mode, cfg.engine = "local", eng
        cfg.base_url = (base_url or DEFAULT_LOCAL_URL).rstrip("/")
        cfg.api_key = api_key

    path = cfg_mod.save(cfg)

    # Required Google sign-in (one-time, lead capture). Interactive only.
    _require_signin()

    # Local: build the engine inline so onboarding is one command to ready
    # (Ollama-style). Skips if already built or --no-build; agents build via
    # `marlin engine install` or auto-build on the first find.
    build_error = None
    if interactive and build and eng in ("mlx", "vllm") and not engines.engine_ready(eng):
        if eng == "vllm":
            console.print(
                "\n  [warn]vLLM not found[/warn] — install it: [bold]uv tool install vllm[/bold]"
            )
        else:
            console.print()
            try:
                with build_spinner("building the local engine (one time)") as log:
                    engines.install_mlx(log=log)
            except RuntimeError as e:
                logger.exception("inline engine build failed during setup")
                build_error = str(e)

    # Pre-fetch the weights now (progress bar) so setup leaves the machine fully
    # ready and the first caption is instant — not racing the serve timeout.
    if interactive and eng == "mlx" and not build_error and engines.engine_ready(eng):
        engines.ensure_weights(cfg, echo)

    reachable = probe(cfg.base_url, cfg.api_key)
    ready = engines.engine_ready(eng)

    result = {
        "configured": True,
        "mode": cfg.mode,
        "engine": eng,
        "base_url": cfg.base_url,
        "model": cfg.model,
        "engine_installed": ready,
        "server_reachable": reachable,
        "config_path": str(path),
    }
    if eng == "mlx":
        result["weights"] = cfg.mlx_weights
    if build_error:
        result["build_error"] = build_error

    def human():
        console.print()
        if eng == "hosted":
            console.print("  [ok]✓[/ok] configured — NemoStation hosted")
            console.print(
                "  [ok]✓[/ok] endpoint reachable"
                if reachable
                else "  [warn]⚠ endpoint not reachable[/warn] — check the URL / key"
            )
            _next_steps()
            return

        console.print(f"  [ok]✓[/ok] configured — local on {_short_platform(detected)}")
        if build_error:
            console.print(
                "  [err]✗ engine build failed[/err] — retry: [bold]marlin engine install[/bold]"
            )
            tail = build_error.strip().splitlines()[-1][:100] if build_error.strip() else ""
            if tail:
                console.print(f"    [muted]{tail}[/muted]")
        elif ready:
            console.print("  [ok]✓[/ok] engine ready")
        else:
            console.print(
                "  [muted]engine builds on your first search "
                "(or run: marlin engine install)[/muted]"
            )

        if eng == "mlx":
            console.print("  [ok]✓[/ok] weights ready")
        _next_steps()

    emit(result, human)
    logger.info(
        "setup completed: mode={} engine={} engine_installed={} server_reachable={}",
        cfg.mode,
        eng,
        ready,
        reachable,
    )


@app.command()
def setup(
    engine: str = typer.Option(
        "", "--engine", help="Force the local engine: mlx | vllm (default: auto-detect)."
    ),
    build: bool = typer.Option(
        True, "--build/--no-build", help="Build the local engine inline during setup."
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", help="No prompts; flags/env only."
    ),
    local: bool = typer.Option(False, "--local", hidden=True),
    hosted: bool = typer.Option(False, "--hosted", hidden=True),
    base_url: str = typer.Option("", "--base-url", hidden=True),
    api_key: str = typer.Option("", "--api-key", hidden=True),
):
    """Set up Marlin to run locally on Apple Silicon."""
    _do_setup(
        engine=engine,
        build=build,
        non_interactive=non_interactive,
        local=local,
        hosted=hosted,
        base_url=base_url,
        api_key=api_key,
    )


def _ready_clip(video: str):
    """Validate a clip path and ensure the local engine answers."""
    from . import daemon

    cfg = _require_config()
    _require_signin()
    path = Path(video)
    if not path.is_file():
        logger.warning("video path rejected because it is not a file: {}", video)
        emit(
            {"error": f"not a file: {video}"},
            lambda: err_console.print(f"[err]not a file:[/err] {video}"),
        )
        raise typer.Exit(1)
    try:
        daemon.ensure_running(cfg, log=echo)
    except RuntimeError as e:
        logger.exception("engine readiness check failed")
        message = str(e)
        emit({"error": message}, lambda: err_console.print(f"[err]{message}[/err]"))
        raise typer.Exit(2) from None
    return cfg, path


@app.command()
def caption(
    video: str = typer.Argument(..., help="A single video file (a bounded clip, ~≤2 min)."),
    detail: bool = typer.Option(
        False, "--detail", help="One free-form paragraph instead of the scene + event timeline."
    ),
    max_pixels: int = typer.Option(
        200704,
        "--max-pixels",
        help="Per-frame pixel budget for auto-downscaling; lower for weak machines.",
    ),
    fps: float = typer.Option(2.0, "--fps", help="Frames/sec sampled (the model uses 2.0)."),
    full_res: bool = typer.Option(
        False, "--full-res", help="Send the clip at full resolution (skip the auto-downscale)."
    ),
    view: bool = typer.Option(
        False, "--view", help="Generate an interactive HTML visualizer and open it."
    ),
):
    """Describe what is in one video clip."""
    from .backend import Marlin

    cfg, path = _ready_clip(video)
    m = Marlin(cfg, max_pixels=max_pixels, fps=fps, full_res=full_res)
    try:
        with spinner("captioning", fish=True):
            if detail:
                result = {"video": str(path), "caption": m.caption(path)}
            else:
                scene, events, _ = m.caption_events(path)
                result = {
                    "video": str(path),
                    "scene": scene,
                    "events": [{"start": e.start, "end": e.end, "text": e.text} for e in events],
                }
    except Exception as e:
        logger.exception("caption command failed")
        message = str(e)
        emit({"error": message}, lambda: err_console.print(f"[err]{message}[/err]"))
        raise typer.Exit(1) from None
    if m.last_note and not is_json():
        err_console.print(
            f"  [muted]↓ auto-downscaled {m.last_note} for speed/memory — "
            "--full-res to keep it[/muted]"
        )

    def human():
        console.print()
        if detail:
            console.print(f"  {result['caption']}\n")
            return
        if result["scene"]:
            console.print(f"  {result['scene']}\n")
        for ev in result["events"]:
            console.print(f"  [num]{ev['start']:6.1f}s–{ev['end']:6.1f}s[/num]  {ev['text']}")
        if not result["events"] and not result["scene"]:
            console.print("  [warn]no caption returned[/warn]")
        console.print()

    emit(result, human)

    # ── visualizer ──────────────────────────────────────────────────────────
    if view and not detail and result.get("events"):
        from .visualizer import generate_and_open

        generate_and_open(
            video_path=path,
            events=result["events"],
            query="Dense Captioning",
        )


@app.command()
def find(
    video: str = typer.Argument(..., help="A video file; long videos are auto-chunked."),
    query: str = typer.Argument(..., help="What to locate, in plain language."),
    max_pixels: int = typer.Option(
        200704,
        "--max-pixels",
        help="Per-frame pixel budget for auto-downscaling; lower for weak machines.",
    ),
    fps: float = typer.Option(2.0, "--fps", help="Frames/sec sampled (the model uses 2.0)."),
    full_res: bool = typer.Option(
        False, "--full-res", help="Send the clip at full resolution (skip the auto-downscale)."
    ),
    view: bool = typer.Option(
        False, "--view", help="Generate an interactive HTML visualizer and open it."
    ),
    chunk_seconds: float = typer.Option(
        30.0,
        "--chunk-seconds",
        help="Chunk window (s) for long videos; keep <=30 for accurate timestamps.",
    ),
    overlap: float = typer.Option(
        5.0,
        "--overlap",
        help="Overlap (s) between consecutive chunks.",
    ),
):
    """Find when an event happens in a video (auto-chunks videos longer than the chunk window)."""
    from .backend import Marlin

    cfg, path = _ready_clip(video)
    m = Marlin(cfg, max_pixels=max_pixels, fps=fps, full_res=full_res)

    def _on_chunk(idx, total, c_start, c_end):
        if not is_json():
            err_console.print(
                f"  [muted]chunk {idx + 1}/{total}  {c_start:.0f}s\u2013{c_end:.0f}s[/muted]"
            )

    try:
        with spinner("finding the moment", fish=True):
            gr = m.ground_video(
                path,
                query,
                on_chunk_start=_on_chunk,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap,
            )
    except Exception as e:
        logger.exception("find command failed")
        message = str(e)
        emit({"error": message}, lambda: err_console.print(f"[err]{message}[/err]"))
        raise typer.Exit(1) from None

    if m.last_note and not is_json():
        err_console.print(
            f"  [muted]\u2193 auto-downscaled {m.last_note} for speed/memory \u2014 "
            "--full-res to keep it[/muted]"
        )

    result = {
        "video": str(path),
        "query": query,
        "events": gr.events,
        "found": gr.found,
    }
    if not gr.chunked:
        result.update({"start": gr.start, "end": gr.end, "tier": gr.tier})
    if gr.duration:
        result["duration"] = gr.duration

    def human():
        console.print()
        if gr.found:
            for i, ev in enumerate(gr.events):
                console.print(
                    f"  [bold]#{i + 1}  {ev['global_start']:.1f}s \u2192 "
                    f"{ev['global_end']:.1f}s[/bold]"
                )
            console.print(f"\n  [muted]{path.name} \u00b7 \u201c{query}\u201d[/muted]\n")
        else:
            console.print(
                f"  [warn]not found[/warn] \u2014 \u201c{query}\u201d didn\u2019t match "
                f"anything in {path.name}\n"
            )

    emit(result, human)

    if view and gr.found:
        from .visualizer import generate_and_open

        generate_and_open(
            video_path=path,
            events=gr.events,
            query=query,
            duration=gr.duration,
        )


@app.command()
def serve(
    port: int = typer.Option(8000, "--port"),
    engine: str = typer.Option("", "--engine", help="mlx | vllm (default: auto / config)."),
    detach: bool = typer.Option(
        False, "--detach", help="Run in the background (logs → ~/.marlin/engine.log)."
    ),
):
    """Launch the local Marlin-2B server."""
    from . import daemon, engines

    cfg = cfg_mod.load()
    eng = engine or engines.resolve_engine(cfg)
    if eng == "hosted":
        logger.warning("serve requested in hosted mode")
        err_console.print(
            "[err]hosted mode has no local server[/err] — you point at a remote endpoint."
        )
        raise typer.Exit(2)
    if not engines.engine_ready(eng):
        logger.warning("serve requested before engine install: {}", eng)
        err_console.print(
            f"[err]{eng} engine not installed[/err] — run [bold]marlin engine install[/bold]"
        )
        raise typer.Exit(2)

    if detach:
        info = daemon.start(cfg, log=echo, port=port)
        logger.info("detached engine started: {}", info)
        emit(
            info,
            lambda: console.print(
                f"[ok]serving[/ok] {engines.label(eng)} (pid {info.get('pid')}) — {cfg.base_url}"
            ),
        )
        return

    argv, env = engines.serve_command(cfg, eng, port)
    echo(" ".join(argv))
    os.execvpe(argv[0], argv, env)


@app.command()
def stop():
    """Stop the background local engine."""
    from . import daemon

    info = daemon.stop(log=echo)
    logger.info("stop completed: {}", info)
    emit(
        info,
        lambda: console.print("[ok]stopped[/ok]" if info.get("stopped") else "no running engine"),
    )


@app.command()
def login(
    force: bool = typer.Option(False, "--force", help="Re-run sign-in even if already signed in."),
):
    """Sign in with the browser OAuth flow."""
    from . import auth

    existing = auth.email()
    if existing and not force:
        emit(
            {"email": existing, "already": True},
            lambda: console.print(
                f"  [ok]✓[/ok] signed in as [bold]{existing}[/bold] "
                "[muted](--force to switch)[/muted]"
            ),
        )
        return
    try:
        info = auth.login(log=echo)
    except RuntimeError as e:
        logger.exception("login command failed")
        message = str(e)
        emit(
            {"error": message},
            lambda: err_console.print(f"  [err]sign-in failed[/err] — {message}"),
        )
        raise typer.Exit(1) from None
    emit(
        info,
        lambda: console.print(f"\n  [ok]✓[/ok] signed in as [bold]{info.get('email')}[/bold]\n"),
    )


@app.command()
def logout():
    """Sign out and clear the local session."""
    from . import auth

    out = auth.logout()
    logger.info("logout completed: signed_out={}", out)
    emit(
        {"signed_out": out},
        lambda: console.print("  [ok]✓[/ok] signed out" if out else "  not signed in"),
    )


engine_app = typer.Typer(no_args_is_help=True)
app.add_typer(engine_app, name="engine", help="Install / manage the local inference engine.")


@engine_app.command("install")
def engine_install():
    """Install the local engine for this machine."""
    from . import engines
    from .output import spinner

    eng = engines.default_engine()
    if eng == "hosted":
        logger.warning("engine install requested without supported local GPU")
        err_console.print(
            "[warn]no local GPU detected[/warn] (need Apple Silicon or NVIDIA) — "
            "use [bold]marlin setup --hosted[/bold]"
        )
        raise typer.Exit(2)
    if eng == "vllm":
        if engines.vllm_ready():
            emit(
                {"engine": "vllm", "installed": True},
                lambda: console.print("  [ok]✓[/ok] vLLM already installed"),
            )
            return
        err_console.print("install vLLM: [bold]uv tool install vllm[/bold]  (or: pip install vllm)")
        raise typer.Exit(2)

    already = engines.mlx_ready()
    if not already:
        try:
            with spinner("building the local engine — SGLang-MLX (Metal), one time") as log:
                engines.install_mlx(log=log)
        except RuntimeError as e:
            logger.exception("engine install failed")
            message = str(e)
            emit(
                {"error": message},
                lambda: err_console.print(
                    f"  [err]✗ build failed[/err] — {message.strip().splitlines()[-1][:120]}"
                ),
            )
            raise typer.Exit(1) from None

    # Actually fetch the weights so "weights ready" is true (progress bar, resumable).
    engines.ensure_weights(cfg_mod.load(), echo)

    def human():
        console.print(
            "  [ok]✓[/ok] engine ready" + (" [muted](already built)[/muted]" if already else "")
        )
        console.print("  [ok]✓[/ok] weights ready")
        _next_steps()

    emit(
        {
            "engine": "mlx",
            "installed": True,
            "already_built": already,
            "weights": cfg_mod.load().mlx_weights,
        },
        human,
    )


@app.command(hidden=True)
def index(
    inputs: list[str] = typer.Argument(..., help="Video files, folders, or YouTube/HTTP URLs."),
    stt: bool = typer.Option(False, "--stt", help="Also index speech (faster-whisper)."),
    background: bool = typer.Option(
        False, "--async", help="Detach; returns a job id for `marlin status`."
    ),
    job: str = typer.Option("", "--job", hidden=True),
):
    """Build the experimental local video index."""
    _require_index_extra()
    from . import jobs as jobs_mod
    from .indexer import IndexStats, index_videos
    from .ingest import resolve_inputs

    cfg = _require_config()

    if background:
        job_id = jobs_mod.spawn_index(inputs, stt=stt)
        logger.info("background index job started: job_id={}", job_id)
        emit(
            {"job_id": job_id, "state": "started", "hint": f"marlin status {job_id}"},
            lambda: console.print(
                f"indexing in background — job [bold]{job_id}[/bold] "
                f"(check: marlin status {job_id})"
            ),
        )
        return

    from . import daemon

    try:
        daemon.ensure_running(cfg, log=echo)
    except RuntimeError as e:
        logger.exception("engine readiness check failed before index")
        message = str(e)
        emit({"error": message}, lambda: err_console.print(f"[err]{message}[/err]"))
        raise typer.Exit(2) from None

    videos = resolve_inputs(inputs)
    if not videos:
        logger.warning("index command resolved no videos")
        emit(
            {"error": "no videos found", "inputs": inputs},
            lambda: err_console.print("[err]no videos found[/err]"),
        )
        raise typer.Exit(1)

    echo(f"indexing {len(videos)} video(s) → {cfg.db_path}")

    def on_progress(s: IndexStats, label: str):
        if job:
            jobs_mod.write_job(
                job,
                {
                    "job_id": job,
                    "state": "running",
                    "current": label,
                    "chunks": s.chunks,
                    "events": s.events,
                    "errors": len(s.errors),
                },
            )
        else:
            echo(f"  {label}  (chunks={s.chunks} events={s.events})")

    stats = index_videos(cfg, videos, stt=stt, on_progress=on_progress)
    logger.info(
        "index completed: videos={} chunks={} events={} errors={}",
        stats.videos,
        stats.chunks,
        stats.events,
        len(stats.errors),
    )
    result = {
        "videos": stats.videos,
        "chunks": stats.chunks,
        "events": stats.events,
        "speech_segments": stats.speech_segments,
        "skipped_still": stats.skipped_still,
        "skipped_done": stats.skipped_done,
        "errors": stats.errors,
        "db": cfg.db_path,
    }
    if job:
        jobs_mod.write_job(job, {"job_id": job, "state": "done", **result})

    def human():
        t = Table(title=f"indexed {stats.videos} video(s)")
        for col in ("chunks", "events", "speech", "still-skipped", "resumed", "errors"):
            t.add_column(col, justify="right")
        t.add_row(
            str(stats.chunks),
            str(stats.events),
            str(stats.speech_segments),
            str(stats.skipped_still),
            str(stats.skipped_done),
            str(len(stats.errors)),
        )
        console.print(t)
        for e in stats.errors[:5]:
            err_console.print(f"[warn]{e}[/warn]")

    emit(result, human)


@app.command(hidden=True)
def search(
    query: str = typer.Argument(..., help="What to find, in plain language."),
    in_path: str = typer.Option(
        "", "--in", help="Scope to a folder/file (indexes it first if needed)."
    ),
    k: int = typer.Option(5, "-k", help="Number of results."),
    ground: bool = typer.Option(True, "--ground/--no-ground", help="Stage-2 precise grounding."),
    clip: bool = typer.Option(False, "--clip", help="Trim result clips to ./marlin_clips/."),
    open_player: bool = typer.Option(False, "--open", help="Open the top clip in a player."),
):
    """Search the experimental local video index."""
    _require_index_extra()
    from .indexer import index_videos
    from .ingest import resolve_inputs
    from .search import search as run_search
    from .trimmer import open_in_player, trim

    cfg = _require_config()
    from . import daemon

    try:
        daemon.ensure_running(cfg, log=echo)
    except RuntimeError as e:
        logger.exception("engine readiness check failed before search")
        message = str(e)
        emit({"error": message}, lambda: err_console.print(f"[err]{message}[/err]"))
        raise typer.Exit(2) from None
    scope = None

    if in_path:
        videos = resolve_inputs([in_path])
        if not videos:
            logger.warning("search scope resolved no videos: {}", in_path)
            emit(
                {"error": f"no videos under {in_path}"},
                lambda: err_console.print(f"[err]no videos under {in_path}[/err]"),
            )
            raise typer.Exit(1)
        echo(f"ensuring {len(videos)} video(s) are indexed …")
        index_videos(cfg, videos)  # resume-safe: already-done chunks are skipped
        common = (
            Path(os.path.commonpath([str(v) for v in videos])) if len(videos) > 1 else videos[0]
        )
        scope = str(common)

    try:
        with spinner("finding the moment", fish=True):
            hits = run_search(cfg, query, k=k, scope=scope, ground=ground)
    except RuntimeError as e:
        logger.exception("search command failed")
        message = str(e)
        emit({"error": message}, lambda: err_console.print(f"[err]{message}[/err]"))
        raise typer.Exit(1) from None

    results = [h.to_dict() for h in hits]
    clip_dir = Path.cwd() / "marlin_clips"
    if clip or open_player:
        for i, h in enumerate(hits):
            p = trim(Path(h.video), h.start, h.end, clip_dir)
            results[i]["clip"] = str(p) if p else None
        if open_player and results and results[0].get("clip"):
            open_in_player(Path(results[0]["clip"]))

    def human():
        if not hits:
            console.print("[warn]no matches[/warn] — is the footage indexed? (marlin index <path>)")
            return
        for i, h in enumerate(hits, 1):
            mark = "[ok]⏱ grounded[/ok]" if h.grounded else "[dim]index span[/dim]"
            console.print(
                f"\n[bold]#{i}[/bold] [{h.score:.3f}] {Path(h.video).name}  "
                f"[bold]{h.start:.1f}s → {h.end:.1f}s[/bold]  {mark}"
            )
            console.print(f"   {h.text[:160]}")
            if results[i - 1].get("clip"):
                console.print(f"   [link]{results[i - 1]['clip']}[/link]")

    emit({"query": query, "results": results}, human)


@app.command(hidden=True)
def status(job_id: str = typer.Argument("", help="Job id (omit to list all jobs)")):
    """Check background index jobs."""
    from . import jobs as jobs_mod

    if job_id:
        data = jobs_mod.read_job(job_id)
        if data is None:
            emit(
                {"error": f"unknown job {job_id}"},
                lambda: err_console.print(f"[err]unknown job {job_id}[/err]"),
            )
            raise typer.Exit(1)
        emit(data, lambda: console.print(data))
    else:
        all_jobs = jobs_mod.list_jobs()
        emit(all_jobs, lambda: console.print(all_jobs or "no jobs"))


def _skill_source() -> Path:
    packaged = Path(__file__).parent / "skills" / "video-understanding" / "SKILL.md"
    if packaged.is_file():
        return packaged
    repo = Path(__file__).resolve().parents[2] / "skills" / "video-understanding" / "SKILL.md"
    return repo


@skills_app.command("install")
def skills_install(
    target: str = typer.Option("auto", "--target", help="auto | claude | agents"),
    global_install: bool = typer.Option(
        False, "--global", help="Install to ~ instead of the project."
    ),
):
    """Install the bundled video-understanding skill."""
    src = _skill_source()
    if not src.is_file():
        logger.error("bundled skill source missing: {}", src)
        emit(
            {"error": "bundled SKILL.md not found"},
            lambda: err_console.print("[err]bundled SKILL.md not found[/err]"),
        )
        raise typer.Exit(1)

    base = Path.home() if global_install else Path.cwd()
    dests: list[Path] = []
    if target in ("auto", "claude") and (
        target == "claude" or (base / ".claude").is_dir() or global_install
    ):
        dests.append(base / ".claude" / "skills" / "video-understanding" / "SKILL.md")
    if target in ("auto", "agents"):
        dests.append(base / ".agents" / "skills" / "video-understanding" / "SKILL.md")
    if not dests:
        dests.append(base / ".claude" / "skills" / "video-understanding" / "SKILL.md")

    installed = []
    for d in dests:
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(src.read_text())
        installed.append(str(d))

    emit(
        {"installed": installed},
        lambda: [console.print(f"[ok]installed[/ok] {p}") for p in installed],
    )


@app.command()
def version():
    """Print the Marlin CLI version."""

    def human():
        banner()
        console.print(f"  marlin [model]{__version__}[/model]\n")

    emit({"version": __version__}, human)


def main():
    """Run the Typer application."""
    configure_logging()
    logger.debug("cli entrypoint initialized")
    app()


if __name__ == "__main__":
    main()
