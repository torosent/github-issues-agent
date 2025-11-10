"""Centralized logging setup for GitHub Issues AI Agent.

Provides idempotent initialization with a consistent format across modules.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

_HANDLER_MARKER = "issues_agent_root_handler"


def init_logging(level: str = "INFO") -> None:
    """Initialize root logging configuration.

    Idempotent: if a handler marked for this app already exists, does nothing.
    Level is case-insensitive; defaults to INFO. Falls back to INFO on invalid level.
    """
    root = logging.getLogger()
    for h in root.handlers:
        if getattr(h, "_marker", None) == _HANDLER_MARKER:
            return  # already initialized

    # Resolve level
    lvl = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(lvl)

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(fmt)
    handler.setLevel(lvl)
    handler._marker = _HANDLER_MARKER  # type: ignore[attr-defined]
    root.addHandler(handler)

    logging.getLogger(__name__).debug("Logging initialized level=%s", level.upper())


__all__ = ["init_logging"]
