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

import hashlib
import os
import platform
import shutil
import subprocess
from pathlib import Path

from .config import CONFIG_DIR
from .logging import get_logger
from .models import Config

logger = get_logger("engines")

# Public MLX weights (Apple Silicon) — same repo string as cfg.mlx_weights (HF
# canonical / fallback).
MLX_REPO = "NemoStation/Marlin-2B-MLX-8bit"

# Fast weights mirror — Azure Blob (anonymous, no HF rate limit). The CLI curls
# these into WEIGHTS_DIR and serves the engine from there; HF is the fallback.
# Override the mirror with MARLIN_MLX_WEIGHTS_URL.
MLX_WEIGHTS_URL = os.environ.get(
    "MARLIN_MLX_WEIGHTS_URL", "https://marlinweights-d7btf0h4c5dxbzdk.z01.azurefd.net/weights"
).rstrip("/")
WEIGHTS_DIR = CONFIG_DIR / "weights" / "marlin-2b-mlx-8bit"
_WEIGHTS_DONE = WEIGHTS_DIR / ".complete"  # Azure mirror fetched + verified here
_HF_DONE = WEIGHTS_DIR / ".hf_complete"  # HF (token) fetched into the HF cache
_WEIGHT_FILES = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "model.safetensors",
    "model.safetensors.index.json",
    "modeling_marlin.py",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
# Pinned SHA256 of each weight file (the known-good model). The engine runs the
# downloaded modeling_marlin.py via --trust-remote-code, so we verify every file
# after download: a tampered mirror, a hijacked MARLIN_MLX_WEIGHTS_URL, or a
# transient CDN corruption all fail closed → fall back to Hugging Face.
_WEIGHT_SHA256 = {
    "config.json": "d6ab48208818ce26017d65ab64b30f0c419227d4d219c69999305507e3584554",
    "generation_config.json": "0d54a28c36c5143413aa18a910f661d17483909e87321f34ad58859b66cf25b4",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "model.safetensors": "0575deb6f9e9f68405979f4e7d4cf42f0361774f722a480180dff95d414466bf",
    "model.safetensors.index.json": (
        "83ee945a045012893ec93b0074510f4e581ac4130b7284307137ffeb35ba0ef2"
    ),
    "modeling_marlin.py": "baec8f0a2cabf5d64a883d8b5ac1881a9119f91f2251e6964f45447d24b19733",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "processor_config.json": "566b6b6ff98913f6b18295b03c03244875bf6c15f96db586d8b024060e54f0c3",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b",
}

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
    """Return the supported local platform bucket."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "apple_silicon"
    if has_cuda():
        return "nvidia"
    return "other"


def has_cuda() -> bool:
    """Return whether an NVIDIA CUDA runtime appears available."""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, timeout=5, check=True)
        return True
    except Exception as exc:
        logger.debug("nvidia-smi check failed: {}", exc)
        return False


def default_engine() -> str:
    """Return the default engine for the current machine."""
    return {"apple_silicon": "mlx", "nvidia": "vllm"}.get(detect_platform(), "hosted")


def resolve_engine(cfg: Config) -> str:
    """Resolve the concrete engine for an action.

    Parameters
    ----------
    cfg
        Runtime configuration.
    """
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
    """Return a human-readable engine label."""
    return _LABELS.get(engine, engine)


# ── readiness ─────────────────────────────────────────────────────────────────


def mlx_python() -> Path:
    """Return the Python executable path for the MLX engine venv."""
    return MLX_ENGINE_DIR / ".venv" / "bin" / "python"


def mlx_ready() -> bool:
    """Return whether the MLX engine venv exists."""
    return mlx_python().is_file()


def vllm_ready() -> bool:
    """Return whether the ``vllm`` command is available."""
    return shutil.which("vllm") is not None


def engine_ready(engine: str) -> bool:
    """Return whether an engine is installed and ready to start."""
    return {
        "mlx": mlx_ready,
        "vllm": vllm_ready,
        "hosted": lambda: True,
    }.get(engine, lambda: False)()


# ── serve commands ────────────────────────────────────────────────────────────


def serve_command(cfg: Config, engine: str, port: int = LOCAL_PORT) -> tuple[list[str], dict]:
    """Build the command used to launch a local engine server.

    Parameters
    ----------
    cfg
        Runtime configuration.
    engine
        Engine name.
    port
        Local server port.

    Returns
    -------
    tuple[list[str], dict]
        Command arguments and environment variables.
    """
    if engine == "mlx":
        # Serve from the local Azure-mirrored dir when present (fast, no HF), else
        # the HF repo id. Both are self-contained (tokenizer + processor + config +
        # modeling_marlin.py), so model-path and tokenizer-path are the same.
        wpath = weights_path(cfg)
        argv = [
            str(mlx_python()),
            "-m",
            "sglang.launch_server",
            "--model-path",
            wpath,
            "--tokenizer-path",
            wpath,
            "--served-model-name",
            cfg.model,
            "--trust-remote-code",
            "--enable-multimodal",
            "--disable-cuda-graph",
            "--disable-radix-cache",
            "--disable-overlap-schedule",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--mm-process-config",
            _MM_CONFIG,
            "--json-model-override-args",
            _ARCH_OVERRIDE,
        ]
        return argv, {**os.environ, "SGLANG_USE_MLX": "1"}
    if engine == "vllm":
        argv = [
            "vllm",
            "serve",
            cfg.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-num-seqs",
            "8",
            "--max-model-len",
            "32768",
            "--hf-overrides",
            _ARCH_OVERRIDE,
        ]
        return argv, dict(os.environ)
    raise ValueError(f"engine {engine!r} has no local server (hosted runs remotely)")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_token() -> str | None:
    """Return the user's Hugging Face token if present (env or token file).

    Its presence means HF downloads are authenticated — fast + canonical — so we
    prefer HF; without it, anonymous HF is rate-limited and we use the mirror.
    """
    for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        t = os.environ.get(env)
        if t and t.strip():
            return t.strip()
    for p in (
        os.environ.get("HF_TOKEN_PATH"),
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ):
        try:
            if p and Path(p).is_file():
                t = Path(p).read_text().strip()
                if t:
                    return t
        except Exception:
            pass
    return None


def ensure_weights(cfg: Config, log) -> None:
    """Fetch the MLX weights (~2.5 GB) once before serving.

    Source preference: **Hugging Face when a token is configured**
    (authenticated → fast + canonical; via snapshot_download, which handles HF
    auth/redirects), **else the public Azure CDN mirror** (anonymous HF is
    rate-limited). The mirror path SHA256-verifies every file against a pinned
    hash — the engine runs the downloaded modeling_marlin.py via
    --trust-remote-code, so a tampered/transient mirror response fails closed
    (re-fetched once, else abort → serve falls back to the HF repo id).
    Idempotent + resumable; one clean progress bar for the big file.
    """
    if _WEIGHTS_DONE.exists() or _HF_DONE.exists():
        return

    # 1) HF first when authenticated — snapshot_download into the HF cache; serve
    #    then uses cfg.mlx_weights (weights_path returns the repo id).
    py = mlx_python()
    if _hf_token() and py.is_file():
        log("fetching Marlin-2B weights from your Hugging Face account (one-time, ~2.5 GB)…")
        code = (
            "import sys; from huggingface_hub import snapshot_download; "
            "snapshot_download(sys.argv[1])"
        )
        try:
            subprocess.run(
                [str(py), "-c", code, cfg.mlx_weights], check=True, stdin=subprocess.DEVNULL
            )
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            _HF_DONE.write_text("ok\n")
            return
        except subprocess.CalledProcessError:
            log("  Hugging Face fetch failed — falling back to the NemoStation mirror")

    # 2) No token (or HF failed) → Azure mirror, clean progress bar, SHA-verified.
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    announced = False
    for f in _WEIGHT_FILES:
        dest = WEIGHTS_DIR / f
        url = f"{MLX_WEIGHTS_URL}/{f}"
        want = _WEIGHT_SHA256[f]
        if dest.exists() and _sha256(dest) == want:
            continue  # present + verified
        if not announced:
            log("fetching Marlin-2B weights from the NemoStation mirror (one-time, ~2.5 GB)…")
            announced = True
        # one clean bar for the 2.5 GB file; the tiny configs download quietly.
        prog = ["--progress-bar"] if f == "model.safetensors" else ["-sS"]
        ok = False
        for attempt in (1, 2):  # attempt 2 = fresh re-fetch (transient/partial)
            if attempt == 2:
                dest.unlink(missing_ok=True)
            resume = ["-C", "-"] if attempt == 1 else []
            try:
                subprocess.run(
                    ["curl", "-fL", *prog, "--retry", "3", *resume, "-o", str(dest), url],
                    check=True,
                    stdin=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "weight download failed: file={} attempt={} error={}", f, attempt, exc
                )
                continue
            if dest.exists() and _sha256(dest) == want:
                ok = True
                break
            logger.warning("weight checksum mismatch: file={} attempt={}", f, attempt)
        if not ok:
            dest.unlink(missing_ok=True)
            logger.error("weight download failed after retries: {}", f)
            log(f"  ✗ {f}: download/checksum failed — serve falls back to Hugging Face")
            return
    _WEIGHTS_DONE.write_text("ok\n")
    logger.info("MLX weights verified at {}", WEIGHTS_DIR)


def weights_path(cfg: Config) -> str:
    """Return the verified local weights path or configured fallback repo."""
    return str(WEIGHTS_DIR) if _WEIGHTS_DONE.exists() else cfg.mlx_weights


# ── MLX engine install ─────────────────────────────────────────────────────────


def install_mlx(log) -> None:
    """Build the SGLang-MLX engine.

    Parameters
    ----------
    log
        Progress callback.

    Raises
    ------
    RuntimeError
        If ``uv`` is missing or an installation step fails.
    """
    if shutil.which("uv") is None:
        logger.error("uv is required to install the MLX engine but was not found")
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
        r = subprocess.run(
            cmd, capture_output=True, text=True, cwd=cwd, env=env, stdin=subprocess.DEVNULL
        )
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "").strip()[-800:]
            logger.error("engine install command failed: cmd={} tail={}", " ".join(cmd[:3]), tail)
            raise RuntimeError(f"step failed ({' '.join(cmd[:3])}…):\n{tail}")
        return r

    if not (MLX_ENGINE_DIR / ".git").is_dir():
        step("cloning the fork")
        logger.info("cloning SGLang fork into {}", MLX_ENGINE_DIR)
        run(["git", "clone", "--depth", "1", "-b", SGLANG_BRANCH, SGLANG_FORK, str(MLX_ENGINE_DIR)])

    if not venv.is_dir():
        step("creating the Python 3.12 venv")
        logger.info("creating MLX engine virtualenv at {}", venv)
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
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            py,
            "--upgrade",
            "mlx",
            "mlx-lm",
            "mlx-vlm",
            "transformers>=5.7.0",
            "torchcodec",
            "qwen-vl-utils>=0.0.14",
            "av",
            "huggingface-hub>=1.18.0",
        ]
    )

    # Marlin ships an explicit lm_head with tied embeddings; mlx_lm's qwen3_5
    # sanitizer needs a one-line patch (verbatim from the fork).
    patch = (
        MLX_ENGINE_DIR
        / "benchmark"
        / "marlin_video"
        / "patches"
        / "mlx_lm_qwen3_5_tied_lm_head.patch"
    )
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
            return (
                subprocess.run(
                    ["patch", "-p0", "--batch", "--dry-run", *extra, "-i", str(patch)],
                    cwd=site,
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                ).returncode
                == 0
            )

        if _dry([]):
            run(["patch", "-p0", "--batch", "-i", str(patch)], cwd=site)
        elif not _dry(["-R"]):
            logger.error("mlx_lm tied-lm_head patch failed to apply")
            raise RuntimeError(
                "mlx_lm tied-lm_head patch did not apply cleanly "
                "(mlx_lm may have drifted from the fork's patch)"
            )
