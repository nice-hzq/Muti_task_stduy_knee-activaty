"""Logging utilities for the multi-task knee angle project."""

import logging
import os
import sys


def setup_logger(log_dir: str, name: str = "MultiTaskKnee") -> logging.Logger:
    """Create a logger that writes to both stdout and a log file.

    Args:
        log_dir: Directory to save the log file.
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger
