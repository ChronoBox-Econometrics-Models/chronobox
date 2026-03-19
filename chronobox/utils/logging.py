"""Structured logging configuration for ChronoBox.

Uses structlog for structured, context-aware logging.
Falls back to standard logging if structlog is not installed.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_STRUCTLOG_AVAILABLE = False
try:
    import structlog

    _STRUCTLOG_AVAILABLE = True
except ImportError:
    pass


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
) -> None:
    """Configure ChronoBox logging.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    json_output : bool
        If True, output logs as JSON (useful for production).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if _STRUCTLOG_AVAILABLE:
        processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
        ]

        if json_output:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stderr)],
        )


def get_logger(name: str = "chronobox") -> Any:
    """Get a logger instance.

    Parameters
    ----------
    name : str
        Logger name (typically module name).

    Returns
    -------
    Logger
        structlog BoundLogger if available, else standard logger.
    """
    if _STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)
