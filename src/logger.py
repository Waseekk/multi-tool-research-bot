"""
src/logger.py
=============
Centralized logging configuration for the Research Bot.

All modules should import get_logger from here instead of calling
logging.basicConfig() directly — this ensures consistent formatting,
a single log file, and avoids duplicate handlers across module reloads.

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Tool initialized")
    logger.error("LLM call failed: %s", exc)
"""

import logging
import logging.handlers
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger writing to both console (INFO+) and a rotating
    log file at logs/app.log (DEBUG+).

    Safe to call multiple times with the same name — handlers are only
    attached once (guard on logger.handlers).

    Parameters
    ----------
    name : str
        Typically __name__ of the calling module. Shows up in log lines as
        the logger name, e.g. "src.nodes", "src.tools", "__main__".
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the same logger is retrieved again
    # (Streamlit reruns modules on every interaction)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console: INFO and above so startup noise stays readable
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Rotating file: DEBUG and above for full diagnostics
    # 5 MB per file, 3 backups → max ~20 MB on disk
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Don't propagate to the root logger — avoids double-printing when
    # third-party libraries also call logging.basicConfig()
    logger.propagate = False

    return logger
