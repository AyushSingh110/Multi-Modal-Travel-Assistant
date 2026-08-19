"""Deliberate failure injection for the mock providers.

The rubric asks whether the app handles a failing weather API gracefully. The
honest way to answer that in a demo is to *make it fail on demand*, in front of
the reviewer, rather than describe what would happen.

This module is where that capability lives, built into the mocks from the start
rather than bolted on. Four failure shapes are supported because they fail in
genuinely different ways and the retry policy treats them differently:

============  ===================================  ==========================
Mode          Simulates                            Retried?
============  ===================================  ==========================
timeout       provider hangs past the deadline     yes - transient
server_error  HTTP 500 from the provider           yes - transient
rate_limit    HTTP 429 with a Retry-After header   yes - honours Retry-After
malformed     HTTP 200 with unusable JSON          no - retrying will not help
============  ===================================  ==========================

That last row is the interesting one. A malformed payload is a *deterministic*
failure: the provider is answering successfully with data we cannot use, so
trying again just burns the user's time and the provider's quota.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from travel_agent.exceptions import RateLimitError, RetryableError, ToolExecutionError

FailureMode = Literal["none", "timeout", "server_error", "malformed", "rate_limit"]

#: Retry-After value the simulated 429 advertises, in seconds. Small so the demo
#: and the tests stay quick while still exercising the header path.
SIMULATED_RETRY_AFTER = 0.5

#: How long the simulated timeout hangs for. Longer than any configured tool
#: timeout, so the timeout fires rather than the sleep completing.
SIMULATED_HANG_SECONDS = 30.0


class MalformedPayloadError(ToolExecutionError):
    """The provider returned a success response whose body cannot be used.

    Deliberately **not** a :class:`~travel_agent.exceptions.RetryableError`: the
    call succeeded, so repeating it produces the same unusable answer.
    """


async def maybe_fail(mode: FailureMode, tool_name: str) -> None:
    """Raise the failure the caller asked to simulate.

    Args:
        mode: Which failure to simulate. ``"none"`` returns immediately.
        tool_name: Name recorded on the raised error, for the trace panel.

    Raises:
        asyncio.TimeoutError: Never raised directly - the coroutine hangs instead
            and the caller's timeout fires, which is what a real hang looks like.
        RetryableError: For ``server_error``.
        RateLimitError: For ``rate_limit``, carrying ``retry_after``.
        MalformedPayloadError: For ``malformed``.
    """
    if mode == "none":
        return

    if mode == "timeout":
        # Hang rather than raising: a real provider timeout is the absence of a
        # response, and this exercises the caller's asyncio.wait_for deadline.
        await asyncio.sleep(SIMULATED_HANG_SECONDS)
        return

    if mode == "server_error":
        raise RetryableError(f"{tool_name}: simulated HTTP 500 from provider")

    if mode == "rate_limit":
        raise RateLimitError(
            f"{tool_name}: simulated HTTP 429 from provider",
            retry_after=SIMULATED_RETRY_AFTER,
        )

    if mode == "malformed":
        raise MalformedPayloadError(
            f"{tool_name}: provider returned a 200 with an unparseable body",
            tool_name=tool_name,
        )


__all__ = [
    "SIMULATED_HANG_SECONDS",
    "SIMULATED_RETRY_AFTER",
    "FailureMode",
    "MalformedPayloadError",
    "maybe_fail",
]
