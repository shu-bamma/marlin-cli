"""Background local engine — start / stop / status + auto-start on find/index.

Ollama-style: the local server (MLX or vLLM) runs detached so the ~40s warmup
is a one-time cost and later commands reuse it. State in ~/.marlin/daemon.json,
logs in ~/.marlin/engine.log. Hosted mode has no daemon (it's a remote URL).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from . import engines
from .backend import probe
from .config import CONFIG_DIR, Config

DAEMON_FILE = CONFIG_DIR / "daemon.json"
LOG_FILE = CONFIG_DIR / "engine.log"


def _read() -> dict:
    try:
        return json.loads(DAEMON_FILE.read_text())
    except Exception:
        return {}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def status(cfg: Config) -> dict:
    d = _read()
    pid = d.get("pid")
    running = bool(pid and _alive(pid))
    return {
        "engine": d.get("engine") or engines.resolve_engine(cfg),
        "pid": pid if running else None,
        "running": running,
        "reachable": probe(cfg.base_url, cfg.api_key),
        "base_url": cfg.base_url,
        "log": str(LOG_FILE),
    }


def start(cfg: Config, log, port: int = engines.LOCAL_PORT, wait_s: float = 600.0) -> dict:
    """Launch the local engine detached and wait until it answers. Idempotent."""
    engine = engines.resolve_engine(cfg)
    if engine == "hosted":
        raise RuntimeError("hosted mode has no local engine to start")
    if probe(cfg.base_url, cfg.api_key):
        log("engine already running.")
        return status(cfg)
    if not engines.engine_ready(engine):
        hint = f"\nMLX weights are gated — request access at {engines.MLX_ACCESS_URL}" if engine == "mlx" else ""
        raise RuntimeError(f"{engine} engine not installed — run `marlin engine install`{hint}")

    argv, env = engines.serve_command(cfg, engine, port)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    logf = open(LOG_FILE, "ab")
    log(f"starting {engines.label(engine)} … first run warms up (~40s).")
    proc = subprocess.Popen(argv, env=env, stdout=logf, stderr=logf, start_new_session=True)
    DAEMON_FILE.write_text(json.dumps({"pid": proc.pid, "engine": engine, "port": port}))

    deadline = time.time() + wait_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"engine exited (code {proc.returncode}) — see {LOG_FILE}")
        if probe(cfg.base_url, cfg.api_key):
            log("engine ready.")
            return status(cfg)
        time.sleep(2)
    raise RuntimeError(f"engine not ready in {wait_s:.0f}s — see {LOG_FILE}")


def stop(log) -> dict:
    d = _read()
    pid = d.get("pid")
    if not pid or not _alive(pid):
        DAEMON_FILE.unlink(missing_ok=True)
        log("no running engine.")
        return {"stopped": False}
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    DAEMON_FILE.unlink(missing_ok=True)
    log(f"stopped engine (pid {pid}).")
    return {"stopped": True, "pid": pid}


def ensure_running(cfg: Config, log) -> None:
    """Auto-start the local engine if it isn't answering. Hosted mode = no-op."""
    if engines.resolve_engine(cfg) == "hosted":
        return
    if not probe(cfg.base_url, cfg.api_key):
        start(cfg, log)
