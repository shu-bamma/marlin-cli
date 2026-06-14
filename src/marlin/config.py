"""Config resolution: env vars > ~/.marlin/config.json > defaults.

Local mode needs no credentials at all. Hosted mode needs MARLIN_API_KEY
(or the key stored by `marlin setup`). The OpenAI SDK requires a non-empty
api_key string, so local uses the "no-key-required" placeholder.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("MARLIN_HOME", Path.home() / ".marlin"))
CONFIG_FILE = CONFIG_DIR / "config.json"
JOBS_DIR = CONFIG_DIR / "jobs"
DOWNLOADS_DIR = CONFIG_DIR / "downloads"
DB_DIR = CONFIG_DIR / "index.lancedb"

DEFAULT_MODEL = "NemoStation/Marlin-2B"
DEFAULT_LOCAL_URL = "http://localhost:8000/v1"
NO_KEY = "no-key-required"


@dataclass
class Config:
    mode: str = "local"  # "local" | "hosted"
    base_url: str = DEFAULT_LOCAL_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    engine: str = "auto"  # auto | mlx | vllm | hosted — how/where Marlin runs locally
    mlx_weights: str = "NemoStation/Marlin-2B-MLX-8bit"  # gated MLX repo (Apple Silicon)
    embed_model: str = "BAAI/bge-small-en-v1.5"
    chunk_seconds: float = 30.0
    chunk_overlap: float = 5.0
    db_path: str = str(DB_DIR)
    extra: dict = field(default_factory=dict)

    @property
    def resolved_api_key(self) -> str:
        return self.api_key or NO_KEY


def load() -> Config:
    cfg = Config()
    if CONFIG_FILE.is_file():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        except json.JSONDecodeError:
            pass
    # Env always wins — the non-interactive/agent path.
    if os.environ.get("MARLIN_BASE_URL"):
        cfg.base_url = os.environ["MARLIN_BASE_URL"].rstrip("/")
        cfg.mode = "local" if _looks_local(cfg.base_url) else "hosted"
    if os.environ.get("MARLIN_API_KEY"):
        cfg.api_key = os.environ["MARLIN_API_KEY"]
    if os.environ.get("MARLIN_MODEL"):
        cfg.model = os.environ["MARLIN_MODEL"]
    if os.environ.get("MARLIN_ENGINE"):
        cfg.engine = os.environ["MARLIN_ENGINE"]
    if os.environ.get("MARLIN_MLX_WEIGHTS"):
        cfg.mlx_weights = os.environ["MARLIN_MLX_WEIGHTS"]
    return cfg


def save(cfg: Config) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    CONFIG_FILE.chmod(0o600)
    return CONFIG_FILE


def configured() -> bool:
    return CONFIG_FILE.is_file() or bool(os.environ.get("MARLIN_BASE_URL"))


def _looks_local(url: str) -> bool:
    return any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"))
