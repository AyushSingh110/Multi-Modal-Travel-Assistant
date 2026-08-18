"""Structured logging and per-node timing helpers.

Two things live here:

1. :func:`configure_logging` - one consistent log format for the whole app.
2. :func:`timed` / :class:`Timer` - measure how long a node or tool took. Latency
   is a first-class output of this project (the parallel fan-out claim is only
   credible with numbers), so timing is instrumentation, not debug printing.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import sys
import time
from collections.abc import Callable
from types import TracebackType
from typing import Any, ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-38s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

_configured = False


def configure_logging(level: str = "INFO", *, force: bool = False) -> None:
    """Install the application-wide logging configuration.

    Idempotent: calling it repeatedly (Streamlit re-runs the whole script on every
    interaction) will not stack duplicate handlers.

    Args:
        level: Logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.
        force: Reconfigure even if logging was already set up.
    """
    global _configured
    if _configured and not force:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Third-party libraries are chatty; keep our own logs readable.
    for noisy in ("httpx", "httpcore", "urllib3", "faiss", "watchdog", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured logger instance.
    """
    return logging.getLogger(name)


class Timer:
    """Context manager that measures wall-clock duration in milliseconds.

    Attributes:
        elapsed_ms: Duration of the ``with`` block, populated on exit.
    """

    def __init__(self) -> None:
        """Initialise an unstarted timer."""
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        """Start timing.

        Returns:
            This timer instance.
        """
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop timing and record the elapsed duration."""
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def timed(label: str | None = None) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorate a sync or async callable so its duration is logged at DEBUG level.

    Args:
        label: Name used in the log line. Defaults to the wrapped function name.

    Returns:
        A decorator that preserves the wrapped callable's signature.
    """

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        name = label or func.__name__
        logger = get_logger(func.__module__)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Any:
                started = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    logger.debug("%s took %.1f ms", name, (time.perf_counter() - started) * 1000)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                logger.debug("%s took %.1f ms", name, (time.perf_counter() - started) * 1000)

        return sync_wrapper

    return decorator


__all__ = ["Timer", "configure_logging", "get_logger", "timed"]
