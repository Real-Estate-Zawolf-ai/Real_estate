"""Application logging configuration."""

from __future__ import annotations

import logging
import os


def configure_logging(level: str | int | None = None) -> None:
    """Configure a consistent process-wide log format."""
    configured_level = level or os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=configured_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
