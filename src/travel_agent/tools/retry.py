"""Timeouts, bounded retries and rate-limit handling for external calls.

Every call that leaves the process goes through :func:`call_with_retry`. The
policy is deliberately conservative:

* a hard timeout, so one slow provider cannot stall the whole graph;
* a bounded number of attempts, so a broken provider fails fast instead of
  hammering a dead endpoint;
* exponential backoff with jitter, so parallel branches do not retry in lockstep;
* explicit ``Retry-After`` support, because Groq's free tier returns HTTP 429 with
  that header and guessing a delay when the server has told you the answer is
  simply wrong.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from travel_agent.exceptions import RateLimitError, RetryableError
from travel_agent.logging_setup import get_logger

_T = TypeVar("_T")

logger = get_logger(__name__)

DEFAULT_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 0.5
DEFAULT_MAX_DELAY = 8.0

# Failures that are worth trying again. Anything else (bad arguments, schema
# violations, programming errors) is raised immediately - retrying a deterministic
# failure just wastes the user's time.
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    RetryableError,
    RateLimitError,
    asyncio.TimeoutError,
    TimeoutError,
    ConnectionError,
)


def backoff_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    """Compute an exponential backoff delay with full jitter.

    Full jitter (a random value in ``[0, computed]``) is used rather than a fixed
    exponential because three fan-out branches failing at the same instant would
    otherwise retry at the same instant too.

    Args:
        attempt: Zero-based attempt number that just failed.
        base_delay: Delay after the first failure, in seconds.
        max_delay: Upper bound on the delay, in seconds.

    Returns:
        Seconds to sleep before the next attempt.
    """
    ceiling = min(max_delay, base_delay * (2**attempt))
    return random.uniform(0, ceiling)


async def call_with_retry(
    operation: Callable[[], Awaitable[_T]],
    *,
    label: str,
    attempts: int = DEFAULT_ATTEMPTS,
    timeout: float | None = 20.0,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    retry_on: tuple[type[BaseException], ...] = RETRYABLE_EXCEPTIONS,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> _T:
    """Run an async operation with a timeout and bounded retries.

    Args:
        operation: Zero-argument coroutine function to call. Wrap arguments in a
            lambda or ``functools.partial`` at the call site.
        label: Name used in log messages, e.g. ``"weather:mock"``.
        attempts: Maximum number of attempts, including the first.
        timeout: Per-attempt timeout in seconds. ``None`` disables it.
        base_delay: Backoff base delay in seconds.
        max_delay: Backoff ceiling in seconds.
        retry_on: Exception types that justify another attempt.
        on_retry: Optional callback invoked as
            ``(attempt_index, exception, delay_seconds)`` before each retry -
            used to push a "rate limited, retrying" event into the UI trace.

    Returns:
        Whatever ``operation`` returns.

    Raises:
        BaseException: The last failure, once the attempt budget is exhausted, or
            immediately for any exception not listed in ``retry_on``.
    """
    last_error: BaseException | None = None

    for attempt in range(attempts):
        try:
            if timeout is None:
                return await operation()
            return await asyncio.wait_for(operation(), timeout=timeout)
        except retry_on as exc:  # type: ignore[misc]
            last_error = exc
            is_final = attempt == attempts - 1
            if is_final:
                logger.warning("%s failed after %d attempt(s): %s", label, attempts, exc)
                break

            # A server that tells us how long to wait is more reliable than our
            # own guess, so Retry-After takes precedence over the backoff curve.
            if isinstance(exc, RateLimitError) and exc.retry_after is not None:
                delay = float(exc.retry_after)
                reason = f"rate limited, honouring Retry-After={delay:.1f}s"
            else:
                delay = backoff_delay(attempt, base_delay, max_delay)
                reason = f"{type(exc).__name__}: {exc}"

            logger.info(
                "%s attempt %d/%d failed (%s); retrying in %.2fs",
                label,
                attempt + 1,
                attempts,
                reason,
                delay,
            )
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            await asyncio.sleep(delay)

    assert last_error is not None  # noqa: S101 - loop cannot exit without an error
    raise last_error


def rate_limit_from_response(status_code: int, headers: dict[str, Any]) -> RateLimitError | None:
    """Convert an HTTP 429 response into a :class:`RateLimitError`.

    Args:
        status_code: HTTP status code from the provider.
        headers: Response headers, case-insensitive mapping or plain dict.

    Returns:
        A populated :class:`RateLimitError` when the response was a rate limit,
        otherwise ``None``.
    """
    if status_code != 429:
        return None

    raw = headers.get("retry-after") or headers.get("Retry-After")
    retry_after: float | None = None
    if raw is not None:
        try:
            retry_after = float(raw)
        except (TypeError, ValueError):
            retry_after = None

    return RateLimitError(
        f"provider returned HTTP 429 (retry_after={retry_after})", retry_after=retry_after
    )


__all__ = [
    "DEFAULT_ATTEMPTS",
    "RETRYABLE_EXCEPTIONS",
    "backoff_delay",
    "call_with_retry",
    "rate_limit_from_response",
]
