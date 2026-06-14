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

# Gated MLX weights — same HF access form as the base model (high-signal leads).
MLX_REPO = "NemoStation/Marlin-2B-MLX-8bit"
MLX_ACCESS_URL = f"https://huggingface.co/{MLX_REPO}"

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
        argv = [
            str(mlx_python()), "-m", "sglang.launch_server",
            "--model-path", cfg.mlx_weights,
            "--tokenizer-path", cfg.model,
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

    def run(cmd, capture=False, cwd=None, env=None):
        log(f"  $ {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=capture, text=True, cwd=cwd, env=env)
        if r.returncode != 0:
            tail = (r.stderr or "")[-600:] if capture else "(see output above)"
            raise RuntimeError(f"step failed ({' '.join(cmd[:3])}…):\n{tail}")
        return r

    if not (MLX_ENGINE_DIR / ".git").is_dir():
        log("cloning the SGLang-MLX engine…")
        run(["git", "clone", "--depth", "1", "-b", SGLANG_BRANCH, SGLANG_FORK, str(MLX_ENGINE_DIR)])

    venv = MLX_ENGINE_DIR / ".venv"
    if not venv.is_dir():
        log("creating the Python 3.12 engine venv…")
        run(["uv", "venv", "-p", "3.12", str(venv)], cwd=str(MLX_ENGINE_DIR))

    # The fork ships Apple/Metal build extras as pyproject_other.toml; select it.
    pp = MLX_ENGINE_DIR / "python" / "pyproject.toml"
    pp_other = MLX_ENGINE_DIR / "python" / "pyproject_other.toml"
    if pp_other.is_file():
        log("selecting the Apple-Silicon build profile…")
        pp.unlink(missing_ok=True)
        pp_other.rename(pp)

    py = str(mlx_python())
    log("installing sglang[all_mps] (slow — builds the Metal extension)…")
    run(["uv", "pip", "install", "--python", py, "-e", "python[all_mps]"], cwd=str(MLX_ENGINE_DIR))
    log("installing MLX + video deps…")
    run(["uv", "pip", "install", "--python", py, "--upgrade",
         "mlx", "mlx-lm", "mlx-vlm", "transformers>=5.7.0", "torchcodec",
         "qwen-vl-utils>=0.0.14", "av", "huggingface-hub>=1.18.0"])

    # Marlin ships an explicit lm_head with tied embeddings; mlx_lm's qwen3_5
    # sanitizer needs a one-line patch (verbatim from the fork).
    patch = MLX_ENGINE_DIR / "benchmark" / "marlin_video" / "patches" / "mlx_lm_qwen3_5_tied_lm_head.patch"
    if patch.is_file():
        site = run(
            [py, "-c", "import mlx_lm,os;print(os.path.dirname(os.path.dirname(mlx_lm.__file__)))"],
            capture=True,
        ).stdout.strip()
        # reverse dry-run succeeds iff already applied → skip; else apply forward
        already = subprocess.run(
            ["patch", "-p0", "-R", "--dry-run", "-i", str(patch)],
            cwd=site, capture_output=True, text=True,
        ).returncode == 0
        if already:
            log("  (mlx_lm patch already applied)")
        else:
            log("patching mlx_lm (qwen3_5 tied lm_head)…")
            run(["patch", "-p0", "-i", str(patch)], capture=True, cwd=site)

    log("MLX engine ready.")
