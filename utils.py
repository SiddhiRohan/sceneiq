"""
SceneIQ — Shared utility functions.

Provides logging setup, reproducibility helpers, JSON I/O,
and a timing decorator used across all pipeline scripts.
"""

import json
import logging
import os
import random
import time
import functools
from pathlib import Path

import numpy as np
import torch

from config import RANDOM_SEED, LOGS_DIR


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with timestamps, writing to console and optionally a file.

    Args:
        log_file: If provided, logs are also written to this file inside LOGS_DIR.
        level: Logging level (default: INFO).

    Returns:
        Configured root logger.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler()]

    if log_file is not None:
        log_path = LOGS_DIR / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
    logger = logging.getLogger("sceneiq")
    logger.info("Logging initialised (level=%s, file=%s)", logging.getLevelName(level), log_file)
    return logger


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.getLogger("sceneiq").info("Random seed set to %d", seed)


# ── JSON I/O ─────────────────────────────────────────────────────────────────

def save_json(data, path: str | Path) -> None:
    """Save a Python object as pretty-printed JSON.

    Args:
        data: Serialisable Python object.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.getLogger("sceneiq").info("Saved JSON → %s (%d bytes)", path, path.stat().st_size)


def load_json(path: str | Path):
    """Load a JSON file with error handling.

    Args:
        path: Source file path.

    Returns:
        Parsed Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logging.getLogger("sceneiq").info("Loaded JSON ← %s", path)
    return data


# ── Timer decorator ──────────────────────────────────────────────────────────

def timer(func):
    """Decorator that logs the wall-clock execution time of a function.

    Usage:
        @timer
        def my_expensive_step(...):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("sceneiq")
        logger.info("START  %s", func.__qualname__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        minutes, seconds = divmod(elapsed, 60)
        logger.info("DONE   %s  [%dm %.1fs]", func.__qualname__, int(minutes), seconds)
        return result
    return wrapper
