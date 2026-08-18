"""Observability models: trace events, token accounting and tool errors.

These power the live "agent trace" panel in the UI. The panel is the fastest way
for a reviewer to see *what the graph actually did* on a request - which route it
picked and why, which tools fired, how long each node took, what was skipped.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EventKind = Literal[
    "route",  # a routing decision was made
    "tool",  # a tool was executed
    "llm",  # a model call happened
    "cache",  # a cache hit or miss
    "skip",  # a node was deliberately not run
    "error",  # something failed
    "timing",  # a measurement worth surfacing
    "info",  # anything else
]


class TraceEvent(BaseModel):
    """One thing that happened during a graph run.

    Attributes:
        node: Graph node that emitted the event.
        kind: Category used for icon and colour in the UI.
        message: Short human-readable description.
        duration_ms: Wall-clock duration, when the event describes work.
        data: Structured extras, e.g. similarity scores or tool arguments.
        timestamp: When the event was created (UTC).
    """

    model_config = ConfigDict(extra="forbid")

    node: str
    kind: EventKind = "info"
    message: str
    duration_ms: float | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolErrorRecord(BaseModel):
    """A tool failure, kept in state so the UI can show a warning banner.

    Attributes:
        tool: Registry name of the tool that failed.
        message: The exception message, already truncated for display.
        tool_call_id: Correlates the failure with the model's original tool call.
        recoverable: Whether the graph could continue without this tool's data.
    """

    model_config = ConfigDict(extra="forbid")

    tool: str
    message: str
    tool_call_id: str | None = None
    recoverable: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TokenUsage(BaseModel):
    """Accumulated LLM token counts for one request.

    Groq, OpenAI and Anthropic all report usage differently; each driver
    normalises into this shape. Cost is deliberately *not* stored as a dollar
    figure here because the three providers price differently - the UI shows the
    token counts and names the provider instead of printing a wrong number.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    model: str = ""

    def merged_with(self, other: TokenUsage) -> TokenUsage:
        """Return the field-wise sum of this usage and ``other``.

        Args:
            other: Usage recorded by another node, possibly on a parallel branch.

        Returns:
            A new :class:`TokenUsage` holding the combined totals.
        """
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
            model=other.model or self.model,
        )


class ParallelMetrics(BaseModel):
    """Evidence for the parallel fan-out claim.

    Attributes:
        branch_durations_ms: Duration of each branch that ran concurrently.
        sequential_equivalent_ms: What the same work would have cost run one after
            another - the sum of the branch durations.
        parallel_wall_clock_ms: What it actually cost, measured across the fan-out.
        speedup: ``sequential_equivalent_ms / parallel_wall_clock_ms``.
    """

    model_config = ConfigDict(extra="forbid")

    branch_durations_ms: dict[str, float] = Field(default_factory=dict)
    sequential_equivalent_ms: float = 0.0
    parallel_wall_clock_ms: float = 0.0
    speedup: float = 1.0


__all__ = ["EventKind", "ParallelMetrics", "TokenUsage", "ToolErrorRecord", "TraceEvent"]
