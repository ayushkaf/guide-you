"""Logger setup helper for the project."""
import logging
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and return a configured logger.

    Args:
        name: Logger name.
        level: Logging level (e.g., logging.INFO).

    Returns:
        Configured `logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.propagate = False
    return logger
