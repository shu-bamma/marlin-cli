"""Engine selection — detect the platform and decide how Marlin-2B runs.

Three engines behind one OpenAI-compatible contract (see backend.Marlin):
  mlx    — Apple Silicon, SGLang-MLX fork (local, free)
  vllm   — NVIDIA CUDA, vLLM (local, free)
  hosted — NemoStation Modal endpoint (API key)

Only the *server* side differs; the client (backend.py) is identical for all
three — each is just a different base_url. The MLX engine runs in its own
venv (SGLang + MLX need Python 3.12 and a separate dependency set), installed
under ~/.marlin/engines/ by `marlin engine install`.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from .config import CONFIG_DIR, Config

# Public MLX weights (Apple Silicon) — same repo string as cfg.mlx_weights (HF
# canonical / fallback).
MLX_REPO = "NemoStation/Marlin-2B-MLX-8bit"

# Fast weights mirror — Azure Blob (anonymous, no HF rate limit). The CLI curls
# these into WEIGHTS_DIR and serves the engine from there; HF is the fallback.
# Override the mirror with MARLIN_MLX_WEIGHTS_URL.
MLX_WEIGHTS_URL = os.environ.get(
    "MARLIN_MLX_WEIGHTS_URL",
    "https://marlinweights-d7btf0h4c5dxbzdk.z01.azurefd.net/weights").rstrip("/")
WEIGHTS_DIR = CONFIG_DIR / "weights" / "marlin-2b-mlx-8bit"
_WEIGHTS_DONE = WEIGHTS_DIR / ".complete"
_WEIGHT_FILES = (
    "config.json", "generation_config.json", "chat_template.jinja",
    "model.safetensors", "model.safetensors.index.json", "modeling_marlin.py",
    "preprocessor_config.json", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json",
)

# Apple-Silicon multimodal SGLang support lives on this fork branch until upstream.
SGLANG_FORK = "https://github.com/Itssshikhar/sglang"
SGLANG_BRANCH = "codex/marlin-mlx-mm-support"

ENGINES_DIR = CONFIG_DIR / "engines"
MLX_ENGINE_DIR = ENGINES_DIR / "sglang-mlx"
LOCAL_PORT = 8000

# Validated SGLang-MLX video sampling (matches Marlin's training distribution)
# and the architecture override (config.json brands the arch as a Marlin
# subclass of Qwen3_5ForConditionalGeneration).
_MM_CONFIG = '{"video":{"fps":2.0,"min_frames":4,"max_frames":240,"max_pixels":200704}}'
_ARCH_OVERRIDE = '{"architectures":["Qwen3_5ForConditionalGeneration"]}'


# ── platform detection ──────────────────────────────────────────────────────

def detect_platform() -> str:
    """'apple_silicon' | 'nvidia' | 'other'."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "apple_silicon"
    if has_cuda():
        return "nvidia"
    return "other"


def has_cuda() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, timeout=5, check=True)
        return True
    except Exception:
        return False


def default_engine() -> str:
    """Engine that fits this machine: mlx (Mac), vllm (NVIDIA), else hosted."""
    return {"apple_silicon": "mlx", "nvidia": "vllm"}.get(detect_platform(), "hosted")


def resolve_engine(cfg: Config) -> str:
    """Concrete engine for an action. Hosted mode is hosted; else honor an
    explicit cfg.engine, falling back to auto-detection."""
    if cfg.mode == "hosted":
        return "hosted"
    if cfg.engine and cfg.engine != "auto":
        return cfg.engine
    return default_engine()


_LABELS = {
    "mlx": "Apple Silicon · SGLang-MLX (local, free)",
    "vllm": "NVIDIA CUDA · vLLM (local, free)",
    "hosted": "NemoStation hosted · API key",
}


def label(engine: str) -> str:
    return _LABELS.get(engine, engine)


# ── readiness ─────────────────────────────────────────────────────────────────

def mlx_python() -> Path:
    return MLX_ENGINE_DIR / ".venv" / "bin" / "python"


def mlx_ready() -> bool:
    return mlx_python().is_file()


def vllm_ready() -> bool:
    return shutil.which("vllm") is not None


def engine_ready(engine: str) -> bool:
    return {
        "mlx": mlx_ready,
        "vllm": vllm_ready,
        "hosted": lambda: True,
    }.get(engine, lambda: False)()


# ── serve commands ────────────────────────────────────────────────────────────

def serve_command(cfg: Config, engine: str, port: int = LOCAL_PORT) -> tuple[list[str], dict]:
    """(argv, env) to launch the local OpenAI-compatible server for `engine`."""
    if engine == "mlx":
        # Serve from the local Azure-mirrored dir when present (fast, no HF), else
        # the HF repo id. Both are self-contained (tokenizer + processor + config +
        # modeling_marlin.py), so model-path and tokenizer-path are the same.
        wpath = weights_path(cfg)
        argv = [
            str(mlx_python()), "-m", "sglang.launch_server",
            "--model-path", wpath,
            "--tokenizer-path", wpath,
            "--served-model-name", cfg.model,
            "--trust-remote-code", "--enable-multimodal",
            "--disable-cuda-graph", "--disable-radix-cache", "--disable-overlap-schedule",
            "--host", "127.0.0.1", "--port", str(port),
            "--mm-process-config", _MM_CONFIG,
            "--json-model-override-args", _ARCH_OVERRIDE,
        ]
        return argv, {**os.environ, "SGLANG_USE_MLX": "1"}
    if engine == "vllm":
        argv = [
            "vllm", "serve", cfg.model, "--host", "127.0.0.1", "--port", str(port),
            "--max-num-seqs", "8", "--max-model-len", "32768",
            "--hf-overrides", _ARCH_OVERRIDE,
        ]
        return argv, dict(os.environ)
    raise ValueError(f"engine {engine!r} has no local server (hosted runs remotely)")


def _remote_size(url: str) -> "int | None":
    try:
        r = subprocess.run(["curl", "-fsSLI", url], capture_output=True, text=True, timeout=30)
        for line in r.stdout.splitlines():
            if line.lower().startswith("content-length:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return None


def ensure_weights(cfg: Config, log) -> None:
    """Fetch the MLX weights (~2.5 GB) from the NemoStation Azure mirror into a
    local dir before serving. Anonymous Azure Blob is fast (HF anonymous pulls
    are rate-limited); the engine then serves from the local dir. Per-file resume
    (curl -C -) + skip-if-complete (size match); idempotent. Never fatal — if the
    mirror can't be reached, serve falls back to the HF repo id."""
    if _WEIGHTS_DONE.exists():
        return
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    announced = False
    for f in _WEIGHT_FILES:
        dest = WEIGHTS_DIR / f
        url = f"{MLX_WEIGHTS_URL}/{f}"
        rsize = _remote_size(url)
        if dest.exists() and rsize is not None and dest.stat().st_size == rsize:
            continue  # already complete
        if not announced:
            log("fetching Marlin-2B weights from the NemoStation mirror (one-time, ~2.5 GB — resumable)…")
            announced = True
        try:
            subprocess.run(["curl", "-fSL", "--retry", "3", "-C", "-", "-o", str(dest), url],
                           check=True, stdin=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            log(f"  weight download incomplete ({f}) — re-run to resume (serve falls back to Hugging Face)")
            return
    _WEIGHTS_DONE.write_text("ok\n")


def weights_path(cfg: Config) -> str:
    """Local Azure-mirrored weights dir if complete, else the HF repo id (fallback)."""
    return str(WEIGHTS_DIR) if _WEIGHTS_DONE.exists() else cfg.mlx_weights


# ── MLX engine install ─────────────────────────────────────────────────────────

def install_mlx(log) -> None:
    """Build the SGLang-MLX engine into ~/.marlin/engines/sglang-mlx.

    Ports the validated bring-up: fork branch, uv venv (py3.12), [all_mps],
    MLX + video deps, the mlx_lm tied-lm-head patch. Re-runnable: skips clone
    and venv if present; the patch is applied at most once. Raises RuntimeError
    naming the failing step.
    """
    if shutil.which("uv") is None:
        raise RuntimeError(
            "uv not found — install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
    ENGINES_DIR.mkdir(parents=True, exist_ok=True)
    venv = MLX_ENGINE_DIR / ".venv"

    # Number the perceptible phases up front so the spinner can show "[k/N]".
    # clone + venv are skipped on a re-run, hence the conditional total.
    total = 2 + (0 if (MLX_ENGINE_DIR / ".git").is_dir() else 1) + (0 if venv.is_dir() else 1)
    state = {"k": 0}

    def step(label):
        state["k"] += 1
        log(f"[{state['k']}/{total}] {label}")

    def run(cmd, cwd=None, env=None):
        # Always capture + close stdin: a build step must never block on a prompt
        # (e.g. `patch` asking "File to patch:"). Only a failure surfaces output.
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env,
                           stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "").strip()[-800:]
            raise RuntimeError(f"step failed ({' '.join(cmd[:3])}…):\n{tail}")
        return r

    if not (MLX_ENGINE_DIR / ".git").is_dir():
        step("cloning the fork")
        run(["git", "clone", "--depth", "1", "-b", SGLANG_BRANCH, SGLANG_FORK, str(MLX_ENGINE_DIR)])

    if not venv.is_dir():
        step("creating the Python 3.12 venv")
        run(["uv", "venv", "-p", "3.12", str(venv)], cwd=str(MLX_ENGINE_DIR))

    # The fork ships Apple/Metal build extras as pyproject_other.toml; select it.
    pp = MLX_ENGINE_DIR / "python" / "pyproject.toml"
    pp_other = MLX_ENGINE_DIR / "python" / "pyproject_other.toml"
    if pp_other.is_file():
        pp.unlink(missing_ok=True)
        pp_other.rename(pp)

    py = str(mlx_python())
    step("compiling the Metal kernel (slowest)")
    run(["uv", "pip", "install", "--python", py, "-e", "python[all_mps]"], cwd=str(MLX_ENGINE_DIR))
    step("downloading MLX + video deps")
    run(["uv", "pip", "install", "--python", py, "--upgrade",
         "mlx", "mlx-lm", "mlx-vlm", "transformers>=5.7.0", "torchcodec",
         "qwen-vl-utils>=0.0.14", "av", "huggingface-hub>=1.18.0"])

    # Marlin ships an explicit lm_head with tied embeddings; mlx_lm's qwen3_5
    # sanitizer needs a one-line patch (verbatim from the fork).
    patch = MLX_ENGINE_DIR / "benchmark" / "marlin_video" / "patches" / "mlx_lm_qwen3_5_tied_lm_head.patch"
    if patch.is_file():
        log("applying the tied-lm_head patch")
        site = run(
            [py, "-c", "import mlx_lm,os;print(os.path.dirname(os.path.dirname(mlx_lm.__file__)))"],
        ).stdout.strip()

        # Idempotent and NON-INTERACTIVE. `patch` prompts ("File to patch:",
        # "Unreversed patch detected! Ignore -R?") and hangs forever if stdin is
        # a tty — so every call is --batch with stdin closed. Decide by dry-run:
        #   forward dry-run clean → not yet applied → apply
        #   else reverse dry-run clean → already applied → skip
        #   else → real conflict (mlx_lm drift) → surface it
        def _dry(extra):
            return subprocess.run(
                ["patch", "-p0", "--batch", "--dry-run", *extra, "-i", str(patch)],
                cwd=site, capture_output=True, text=True, stdin=subprocess.DEVNULL,
            ).returncode == 0

        if _dry([]):
            run(["patch", "-p0", "--batch", "-i", str(patch)], cwd=site)
        elif not _dry(["-R"]):
            raise RuntimeError(
                "mlx_lm tied-lm_head patch did not apply cleanly "
                "(mlx_lm may have drifted from the fork's patch)"
            )
