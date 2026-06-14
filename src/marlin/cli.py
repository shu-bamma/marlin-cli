"""marlin — agent-first CLI. Every verb honors --json; stdout is parseable
when piped; progress goes to stderr."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import typer
from rich.table import Table

from . import __version__, config as cfg_mod
from .config import Config, DEFAULT_LOCAL_URL, DEFAULT_MODEL
from .output import banner, console, emit, err_console, set_json
from .output import status as echo

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")
skills_app = typer.Typer(no_args_is_help=True)
app.add_typer(skills_app, name="skills", help="Install agent skills that ride this CLI.")


@app.callback()
def _root(json_out: bool = typer.Option(False, "--json", help="Force JSON output (auto when piped).")):
    set_json(json_out)


def _require_config() -> Config:
    if not cfg_mod.configured():
        err_console.print(
            "[err]not configured[/err] — run [bold]marlin setup[/bold] "
            "(or set MARLIN_BASE_URL / MARLIN_API_KEY for non-interactive use)"
        )
        raise typer.Exit(2)
    return cfg_mod.load()


def _platform_human(p: str) -> str:
    return {
        "apple_silicon": "Apple Silicon (Metal)",
        "nvidia": "NVIDIA CUDA GPU",
        "other": "no local GPU detected (need Apple Silicon or NVIDIA)",
    }.get(p, p)


@app.command()
def setup(
    local: bool = typer.Option(False, "--local", help="Run locally (auto-detects MLX on Apple Silicon, vLLM on NVIDIA)."),
    hosted: bool = typer.Option(False, "--hosted", help="Use the hosted NemoStation endpoint."),
    engine: str = typer.Option("", "--engine", help="Force engine: mlx | vllm | hosted (default: auto-detect)."),
    base_url: str = typer.Option("", "--base-url", help="Override endpoint URL."),
    api_key: str = typer.Option("", "--api-key", help="API key for hosted mode."),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="No prompts; flags/env only."),
):
    """Configure marlin. Auto-detects your platform: Apple Silicon (MLX),
    NVIDIA (vLLM), or hosted (API key)."""
    from . import engines
    from .backend import probe
    from .chunker import check_ffmpeg

    if not check_ffmpeg():
        err_console.print("[err]ffmpeg/ffprobe not found[/err] — install first: brew install ffmpeg")
        raise typer.Exit(2)

    cfg = cfg_mod.load()
    detected = engines.detect_platform()
    rec = engines.default_engine()  # mlx | vllm | hosted

    eng = engine or ("hosted" if hosted else (rec if local else ""))
    if not eng and not non_interactive:
        console.print(f"\n[bold]Detected:[/bold] {_platform_human(detected)}")
        console.print("[bold]How should Marlin-2B run?[/bold]")
        if detected in ("apple_silicon", "nvidia"):
            console.print(f"  1. local — {engines.label(rec)}  [ok](recommended)[/ok]")
        else:
            console.print("  1. local — needs Apple Silicon or NVIDIA (none detected here)")
        console.print(f"  2. hosted — {engines.label('hosted')}")
        default_choice = "1" if detected in ("apple_silicon", "nvidia") else "2"
        choice = typer.prompt("choice", default=default_choice).strip()
        eng = rec if choice == "1" else "hosted"
    eng = eng or rec

    if eng == "hosted":
        cfg.mode, cfg.engine = "hosted", "hosted"
        cfg.base_url = (base_url or (
            typer.prompt("hosted base URL", default=cfg.base_url) if not non_interactive else cfg.base_url
        )).rstrip("/")
        cfg.api_key = api_key or os.environ.get("MARLIN_API_KEY", "") or (
            typer.prompt("API key", hide_input=True) if not non_interactive else ""
        )
        if not cfg.base_url:
            err_console.print("[err]hosted mode needs --base-url[/err]")
            raise typer.Exit(2)
    else:
        cfg.mode, cfg.engine = "local", eng
        cfg.base_url = (base_url or DEFAULT_LOCAL_URL).rstrip("/")
        cfg.api_key = api_key

    path = cfg_mod.save(cfg)
    reachable = probe(cfg.base_url, cfg.api_key)
    ready = engines.engine_ready(eng)
    result = {
        "configured": True, "mode": cfg.mode, "engine": eng, "base_url": cfg.base_url,
        "model": cfg.model, "engine_installed": ready, "server_reachable": reachable,
        "config_path": str(path),
    }
    if eng == "mlx":
        result["weights"], result["access_form"] = cfg.mlx_weights, engines.MLX_ACCESS_URL

    def human():
        banner()
        console.print(f"[ok]configured[/ok] → {path}")
        console.print(f"  engine: [bold]{engines.label(eng)}[/bold]")
        console.print(f"  endpoint: {cfg.base_url}")
        if eng == "hosted":
            console.print("  server: " + ("[ok]reachable[/ok]" if reachable else "[warn]unreachable — check URL/key[/warn]"))
        elif not ready:
            if eng == "mlx":
                console.print("  engine: [warn]not installed[/warn] → [bold]marlin engine install[/bold]")
                console.print(f"  weights: [warn]gated[/warn] — request access (1-click): [link]{engines.MLX_ACCESS_URL}[/link]")
            else:
                console.print("  engine: [warn]vLLM not found[/warn] → [bold]marlin engine install[/bold]")
        elif reachable:
            console.print("  server: [ok]running[/ok]")
        else:
            console.print("  engine: [ok]installed[/ok] — auto-starts on first [bold]find[/bold] (or run [bold]marlin serve[/bold])")
        console.print("\nnext: [bold]marlin index <folder>[/bold] then [bold]marlin find \"query\"[/bold]")

    emit(result, human)


@app.command()
def serve(
    port: int = typer.Option(8000, "--port"),
    engine: str = typer.Option("", "--engine", help="mlx | vllm (default: auto / config)."),
    detach: bool = typer.Option(False, "--detach", help="Run in the background (logs → ~/.marlin/engine.log)."),
):
    """Launch the local Marlin-2B server — MLX on Apple Silicon, vLLM on NVIDIA."""
    from . import daemon, engines

    cfg = cfg_mod.load()
    eng = engine or engines.resolve_engine(cfg)
    if eng == "hosted":
        err_console.print("[err]hosted mode has no local server[/err] — you point at a remote endpoint.")
        raise typer.Exit(2)
    if not engines.engine_ready(eng):
        err_console.print(f"[err]{eng} engine not installed[/err] — run [bold]marlin engine install[/bold]")
        if eng == "mlx":
            err_console.print(f"MLX weights are gated — request access: {engines.MLX_ACCESS_URL}")
        raise typer.Exit(2)

    if detach:
        info = daemon.start(cfg, log=echo, port=port)
        emit(info, lambda: console.print(
            f"[ok]serving[/ok] {engines.label(eng)} (pid {info.get('pid')}) — {cfg.base_url}"))
        return

    argv, env = engines.serve_command(cfg, eng, port)
    echo(" ".join(argv))
    os.execvpe(argv[0], argv, env)


@app.command()
def stop():
    """Stop the background local engine."""
    from . import daemon

    info = daemon.stop(log=echo)
    emit(info, lambda: console.print("[ok]stopped[/ok]" if info.get("stopped") else "no running engine"))


engine_app = typer.Typer(no_args_is_help=True)
app.add_typer(engine_app, name="engine", help="Install / manage the local inference engine.")


@engine_app.command("install")
def engine_install():
    """Install the local engine for this machine (SGLang-MLX on Apple Silicon, vLLM on NVIDIA)."""
    from . import engines

    eng = engines.default_engine()
    if eng == "hosted":
        err_console.print("[warn]no local GPU detected[/warn] (need Apple Silicon or NVIDIA) — use [bold]marlin setup --hosted[/bold]")
        raise typer.Exit(2)
    if eng == "vllm":
        if engines.vllm_ready():
            emit({"engine": "vllm", "installed": True}, lambda: console.print("[ok]vLLM already installed[/ok]"))
            return
        err_console.print("install vLLM: [bold]uv tool install vllm[/bold]  (or: pip install vllm)")
        raise typer.Exit(2)

    echo("installing the SGLang-MLX engine — a few minutes the first time …")
    try:
        engines.install_mlx(log=echo)
    except RuntimeError as e:
        emit({"error": str(e)}, lambda: err_console.print(f"[err]{e}[/err]"))
        raise typer.Exit(1)
    emit(
        {"engine": "mlx", "installed": True, "weights": cfg_mod.load().mlx_weights, "access_form": engines.MLX_ACCESS_URL},
        lambda: console.print(
            "[ok]MLX engine ready[/ok]\n"
            f"  weights are gated — request access (1-click): [link]{engines.MLX_ACCESS_URL}[/link]\n"
            "  then: [bold]marlin serve[/bold] (or it auto-starts on first find)"
        ),
    )


@app.command()
def index(
    inputs: list[str] = typer.Argument(..., help="Video files, folders, or YouTube/HTTP URLs."),
    stt: bool = typer.Option(False, "--stt", help="Also index speech (faster-whisper)."),
    background: bool = typer.Option(False, "--async", help="Detach; returns a job id for `marlin status`."),
    job: str = typer.Option("", "--job", hidden=True),
):
    """Caption + embed videos into the local index (resume-safe)."""
    from . import jobs as jobs_mod
    from .indexer import IndexStats, index_videos
    from .ingest import resolve_inputs

    cfg = _require_config()

    if background:
        job_id = jobs_mod.spawn_index(inputs, stt=stt)
        emit(
            {"job_id": job_id, "state": "started", "hint": f"marlin status {job_id}"},
            lambda: console.print(f"indexing in background — job [bold]{job_id}[/bold] (check: marlin status {job_id})"),
        )
        return

    from . import daemon
    try:
        daemon.ensure_running(cfg, log=echo)
    except RuntimeError as e:
        emit({"error": str(e)}, lambda: err_console.print(f"[err]{e}[/err]"))
        raise typer.Exit(2)

    videos = resolve_inputs(inputs)
    if not videos:
        emit({"error": "no videos found", "inputs": inputs},
             lambda: err_console.print("[err]no videos found[/err]"))
        raise typer.Exit(1)

    echo(f"indexing {len(videos)} video(s) → {cfg.db_path}")

    def on_progress(s: IndexStats, label: str):
        if job:
            jobs_mod.write_job(job, {
                "job_id": job, "state": "running", "current": label,
                "chunks": s.chunks, "events": s.events, "errors": len(s.errors),
            })
        else:
            echo(f"  {label}  (chunks={s.chunks} events={s.events})")

    stats = index_videos(cfg, videos, stt=stt, on_progress=on_progress)
    result = {
        "videos": stats.videos, "chunks": stats.chunks, "events": stats.events,
        "speech_segments": stats.speech_segments, "skipped_still": stats.skipped_still,
        "skipped_done": stats.skipped_done, "errors": stats.errors, "db": cfg.db_path,
    }
    if job:
        jobs_mod.write_job(job, {"job_id": job, "state": "done", **result})

    def human():
        t = Table(title=f"indexed {stats.videos} video(s)")
        for col in ("chunks", "events", "speech", "still-skipped", "resumed", "errors"):
            t.add_column(col, justify="right")
        t.add_row(str(stats.chunks), str(stats.events), str(stats.speech_segments),
                  str(stats.skipped_still), str(stats.skipped_done), str(len(stats.errors)))
        console.print(t)
        for e in stats.errors[:5]:
            err_console.print(f"[warn]{e}[/warn]")

    emit(result, human)


@app.command()
def find(
    query: str = typer.Argument(..., help="What to find, in plain language."),
    in_path: str = typer.Option("", "--in", help="Scope to a folder/file (indexes it first if needed)."),
    k: int = typer.Option(5, "-k", help="Number of results."),
    ground: bool = typer.Option(True, "--ground/--no-ground", help="Stage-2 precise grounding."),
    clip: bool = typer.Option(False, "--clip", help="Trim result clips to ./marlin_clips/."),
    open_player: bool = typer.Option(False, "--open", help="Open the top clip in a player."),
):
    """Find when something happens across your indexed footage."""
    from .indexer import index_videos
    from .ingest import resolve_inputs
    from .search import search as run_search
    from .trimmer import open_in_player, trim

    cfg = _require_config()
    from . import daemon
    try:
        daemon.ensure_running(cfg, log=echo)
    except RuntimeError as e:
        emit({"error": str(e)}, lambda: err_console.print(f"[err]{e}[/err]"))
        raise typer.Exit(2)
    scope = None

    if in_path:
        videos = resolve_inputs([in_path])
        if not videos:
            emit({"error": f"no videos under {in_path}"},
                 lambda: err_console.print(f"[err]no videos under {in_path}[/err]"))
            raise typer.Exit(1)
        echo(f"ensuring {len(videos)} video(s) are indexed …")
        index_videos(cfg, videos)  # resume-safe: already-done chunks are skipped
        common = Path(os.path.commonpath([str(v) for v in videos])) if len(videos) > 1 else videos[0]
        scope = str(common)

    try:
        hits = run_search(cfg, query, k=k, scope=scope, ground=ground)
    except RuntimeError as e:
        emit({"error": str(e)}, lambda: err_console.print(f"[err]{e}[/err]"))
        raise typer.Exit(1)

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


@app.command()
def status(job_id: str = typer.Argument("", help="Job id (omit to list all jobs)")):
    """Check background index jobs."""
    from . import jobs as jobs_mod

    if job_id:
        data = jobs_mod.read_job(job_id)
        if data is None:
            emit({"error": f"unknown job {job_id}"},
                 lambda: err_console.print(f"[err]unknown job {job_id}[/err]"))
            raise typer.Exit(1)
        emit(data, lambda: console.print(data))
    else:
        all_jobs = jobs_mod.list_jobs()
        emit(all_jobs, lambda: console.print(all_jobs or "no jobs"))


def _skill_source() -> Path:
    packaged = Path(__file__).parent / "skills" / "video-search" / "SKILL.md"
    if packaged.is_file():
        return packaged
    repo = Path(__file__).resolve().parents[2] / "skills" / "video-search" / "SKILL.md"
    return repo


@skills_app.command("install")
def skills_install(
    target: str = typer.Option("auto", "--target", help="auto | claude | agents"),
    global_install: bool = typer.Option(False, "--global", help="Install to ~ instead of the project."),
):
    """Install the video-search SKILL.md into your agent's skills directory."""
    src = _skill_source()
    if not src.is_file():
        emit({"error": "bundled SKILL.md not found"},
             lambda: err_console.print("[err]bundled SKILL.md not found[/err]"))
        raise typer.Exit(1)

    base = Path.home() if global_install else Path.cwd()
    dests: list[Path] = []
    if target in ("auto", "claude") and (target == "claude" or (base / ".claude").is_dir() or global_install):
        dests.append(base / ".claude" / "skills" / "video-search" / "SKILL.md")
    if target in ("auto", "agents"):
        dests.append(base / ".agents" / "skills" / "video-search" / "SKILL.md")
    if not dests:
        dests.append(base / ".claude" / "skills" / "video-search" / "SKILL.md")

    installed = []
    for d in dests:
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(src.read_text())
        installed.append(str(d))

    emit({"installed": installed},
         lambda: [console.print(f"[ok]installed[/ok] {p}") for p in installed])


@app.command()
def version():
    """Print version."""
    emit({"version": __version__}, lambda: console.print(f"marlin {__version__}"))


def main():
    app()


if __name__ == "__main__":
    main()
