"""Production logging configuration for Marlin.

The CLI keeps user-facing output in :mod:`marlin.output`; this module is for
operator logs only. Log records go to stderr and a rotating file sink, never to
stdout, so ``--json`` and piped command output remain parseable.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from pathlib import Path

from loguru import logger as _logger

from .config import CONFIG_DIR
from .models import LoggingConfig

_CONFIGURED = False
_FALSE_VALUES = {"0", "false", "no", "off", ""}
_LOG_FORMAT = (
    "{time:YYYY-MM-DDTHH:mm:ss.SSSZZ} | {level: <8} | pid={process} | "
    "{name}:{function}:{line} | {message} | {extra}"
)


class InterceptHandler(logging.Handler):
    """Forward standard-library logging records into Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a standard logging record through the configured Loguru sinks."""
        try:
            level: str | int = _logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = inspect.currentframe(), 0
        while frame:
            filename = frame.f_code.co_filename
            is_logging = filename == logging.__file__
            is_importlib = "importlib" in filename and "_bootstrap" in filename
            if depth > 0 and not (is_logging or is_importlib):
                break
            frame = frame.f_back
            depth += 1

        _logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _env_bool(name: str, default: bool) -> bool:
    """Return a boolean environment setting."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE_VALUES


def _resolve_log_file() -> Path:
    """Resolve the active log file path from environment and defaults."""
    explicit = os.environ.get("MARLIN_LOG_FILE")
    if explicit:
        return Path(explicit).expanduser()
    log_dir = Path(os.environ.get("MARLIN_LOG_DIR", CONFIG_DIR / "logs")).expanduser()
    return log_dir / "marlin.log"


def load_logging_config() -> LoggingConfig:
    """Load logging settings from ``MARLIN_LOG_*`` environment variables.

    Operator logging is opt-in: both sinks default off so a normal/agent
    invocation does no logging I/O. ``marlin`` is invoked constantly (every
    agent ``caption``/``find`` call), so it must not write a rotating file
    under ``$HOME`` by default — enable with ``MARLIN_LOG_FILE_ENABLED=1`` (or
    ``MARLIN_LOG_STDERR=1``) when debugging.
    """
    return LoggingConfig(
        stderr_enabled=_env_bool("MARLIN_LOG_STDERR", False),
        file_enabled=_env_bool("MARLIN_LOG_FILE_ENABLED", False),
        stderr_level=os.environ.get("MARLIN_LOG_LEVEL", "WARNING").upper(),
        file_level=os.environ.get("MARLIN_LOG_FILE_LEVEL", "INFO").upper(),
        log_file=_resolve_log_file(),
        rotation=os.environ.get("MARLIN_LOG_ROTATION", "10 MB"),
        retention=os.environ.get("MARLIN_LOG_RETENTION", "14 days"),
        compression=os.environ.get("MARLIN_LOG_COMPRESSION", "gz"),
        serialize=_env_bool("MARLIN_LOG_JSON", False),
        diagnose=_env_bool("MARLIN_LOG_DIAGNOSE", False),
        enqueue=_env_bool("MARLIN_LOG_ENQUEUE", True),
    )


def configure_logging(*, force: bool = False) -> LoggingConfig:
    """Configure Loguru sinks and stdlib logging interception.

    Parameters
    ----------
    force
        Rebuild sinks even if logging was already configured.

    Returns
    -------
    LoggingConfig
        The resolved logging settings.
    """
    global _CONFIGURED
    config = load_logging_config()
    if _CONFIGURED and not force:
        return config

    _logger.remove()
    if config.stderr_enabled:
        _logger.add(
            sys.stderr,
            level=config.stderr_level,
            format=_LOG_FORMAT,
            colorize=sys.stderr.isatty(),
            serialize=config.serialize,
            backtrace=True,
            diagnose=config.diagnose,
            enqueue=config.enqueue,
        )

    if config.file_enabled:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            config.log_file,
            level=config.file_level,
            format=_LOG_FORMAT,
            rotation=config.rotation,
            retention=config.retention,
            compression=config.compression,
            serialize=config.serialize,
            encoding="utf-8",
            backtrace=True,
            diagnose=config.diagnose,
            enqueue=config.enqueue,
        )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for noisy_logger in ("httpcore", "httpx", "openai"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _CONFIGURED = True
    _logger.debug("logging configured: {}", config)
    return config


def get_logger(component: str | None = None):
    """Return a Loguru logger bound to an optional component name.

    Parameters
    ----------
    component
        Component or module name to bind into the log record extras.

    Returns
    -------
    loguru.Logger
        Configured Loguru logger instance.
    """
    if component:
        return _logger.bind(component=component)
    return _logger
