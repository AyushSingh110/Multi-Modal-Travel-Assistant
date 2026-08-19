"""The manual tool executor.

Parses the model's raw ``tool_calls``, runs each tool, and answers with a
``ToolMessage`` carrying the matching ``tool_call_id``.

LangGraph ships ``ToolNode``, which does all of this for you. We do it by
hand so the raw tool-calling protocol is visible, and so failures come back
as error messages the model can read instead of exceptions.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from travel_agent.logging_setup import Timer, get_logger
from travel_agent.schemas.knowledge import KnowledgeChunk
from travel_agent.schemas.response import ImageAsset, WeatherPayload
from travel_agent.schemas.state import TravelState
from travel_agent.schemas.tools import IMAGES_TOOL, WEATHER_TOOL, WEB_SEARCH_TOOL
from travel_agent.schemas.trace import ToolErrorRecord, TraceEvent
from travel_agent.tools.registry import ToolRegistry, ToolResult

logger = get_logger(__name__)

# Characters of tool output handed back to the model. Enough to summarise from,bounded so a large payload cannot dominate the context window.
MAX_TOOL_CONTENT_CHARS = 4000


def extract_tool_calls(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Return the tool calls on the last AI message, or an empty list."""
    if not messages:
        return []

    last = messages[-1]
    if not isinstance(last, AIMessage):
        return []

    return list(getattr(last, "tool_calls", []) or [])


class ManualToolExecutor:
    """Runs the tool calls the model asked for, without any prebuilt helper."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        node_name: str = "tool_executor",
        handles: set[str] | None = None,
    ) -> None:
        """Initialise the executor.

        Args:
            registry: Configured tool registry.
            node_name: Name recorded in trace events and the timings map.
            handles: Restrict execution to these tool names. ``None`` means every
                tool call in the message.
        """
        self.registry = registry
        self.node_name = node_name
        self.handles = handles

    async def __call__(self, state: TravelState) -> dict[str, Any]:
        """Run this node's tool calls concurrently and return the state update."""
        tool_calls = extract_tool_calls(state.get("messages", []))
        mine = [call for call in tool_calls if self._is_mine(call)]

        if not mine:
            logger.debug("%s: no matching tool calls", self.node_name)
            return {
                "trace": [
                    TraceEvent(
                        node=self.node_name,
                        kind="skip",
                        message="no tool calls addressed to this node",
                    )
                ],
                "skipped_nodes": [self.node_name],
            }
        with Timer() as timer:
            # Concurrent, and independent on purpose: return_exceptions keeps one
            # unexpected failure from cancelling its siblings mid-flight.
            results = await asyncio.gather(
                *(self._execute_one(call) for call in mine), return_exceptions=True
            )
        messages: list[ToolMessage] = []
        trace: list[TraceEvent] = []
        errors: list[ToolErrorRecord] = []
        timings: dict[str, float] = {self.node_name: timer.elapsed_ms}
        updates: dict[str, Any] = {}

        for call, outcome in zip(mine, results, strict=True):
            if isinstance(outcome, BaseException):
                # Defensive: ToolRegistry.execute is written not to raise, so
                # reaching here means a bug rather than a provider failure. It is
                # still reported as an error message instead of taking the graph
                # down with it.
                logger.exception("unexpected executor failure for %s", call.get("name"))
                messages.append(self._error_message(call, f"internal error: {outcome!r}"))
                errors.append(
                    ToolErrorRecord(
                        tool=str(call.get("name", "unknown")),
                        message=f"internal error: {outcome!r}"[:300],
                        tool_call_id=call.get("id"),
                        recoverable=False,
                    )
                )
                continue
            message, events, error, payload_updates = outcome
            messages.append(message)
            trace.extend(events)
            updates.update(payload_updates)
            if error is not None:
                errors.append(error)

        return {
            "messages": messages,
            "trace": trace,
            "errors": errors,
            "timings": timings,
            **updates,
        }

    # one call
    async def _execute_one(
        self, call: dict[str, Any]
    ) -> tuple[ToolMessage, list[TraceEvent], ToolErrorRecord | None, dict[str, Any]]:
        # Execute a single tool call and return the result, trace events, and any domain payload updates.
        name = str(call.get("name", ""))
        call_id = str(call.get("id") or "")
        raw_args = call.get("args", {})

        retries: list[str] = []

        def _note_retry(attempt: int, error: BaseException, delay: float) -> None:
            retries.append(
                f"attempt {attempt + 1} failed ({type(error).__name__}), retrying in {delay:.1f}s"
            )

        result: ToolResult = await self.registry.execute(name, raw_args, on_retry=_note_retry)

        events = [
            TraceEvent(
                node=self.node_name,
                kind="tool" if result.ok else "error",
                message=(
                    f"{name} -> {'ok' if result.ok else result.error_type}"
                    + (f" ({result.attempts} attempts)" if result.attempts > 1 else "")
                ),
                duration_ms=result.duration_ms,
                data={
                    "tool": name,
                    "tool_call_id": call_id,
                    "args": _jsonable(raw_args),
                    "provider": result.provider,
                    "attempts": result.attempts,
                    "retries": retries,
                    "error": result.error,
                },
            )
        ]
        if result.failed:
            return (
                self._error_message(call, result.error or "tool failed"),
                events,
                ToolErrorRecord(
                    tool=name,
                    message=result.error[:300],
                    tool_call_id=call_id,
                    # An argument or unknown-tool error is the model's mistake and
                    # it can correct itself; a dead provider it cannot.
                    recoverable=result.error_type in {"ToolArgumentError", "UnknownToolError"},
                ),
                {},
            )
        return (
            ToolMessage(
                content=self._summarise(name, result.payload),
                tool_call_id=call_id,
                name=name,
                status="success",
            ),
            events,
            None,
            self._state_updates(name, result.payload),
        )

    def _is_mine(self, call: dict[str, Any]) -> bool:
        """Decide whether this executor should run a given call.

        Args:
            call: One entry from the model's ``tool_calls`` payload.

        Returns:
            ``True`` when this node handles that tool name.
        """
        if self.handles is None:
            return True
        return str(call.get("name", "")) in self.handles

    def _error_message(self, call: dict[str, Any], detail: str) -> ToolMessage:
        """Build a failure reply the model can act on.

        ``status="error"`` is the point: it tells the model the call did not
        succeed, rather than letting it read the error text as data. For an
        unknown tool name the message also lists the tools that *do* exist, so the
        model can correct itself on the next turn instead of guessing again.

        Args:
            call: The originating tool call.
            detail: Human-readable failure description.

        Returns:
            A ``ToolMessage`` with ``status="error"`` and the matching id.
        """
        name = str(call.get("name", "unknown"))
        body = detail
        if name not in self.registry.tool_names:
            body = (
                f"{detail}. Available tools are: {', '.join(self.registry.tool_names)}. "
                f"Call one of those instead."
            )

        return ToolMessage(
            content=body[:MAX_TOOL_CONTENT_CHARS],
            tool_call_id=str(call.get("id") or ""),
            name=name,
            status="error",
        )

    # payload shaping
    @staticmethod
    def _summarise(tool_name: str, payload: Any) -> str:
        # Summarise a tool payload into a JSON string the model can read.
        if tool_name == WEATHER_TOOL and isinstance(payload, WeatherPayload):
            summary = {
                "city": payload.city,
                "current_temp_c": payload.current_temp_c,
                "current_condition": payload.current_condition,
                "forecast": [
                    {
                        "date": point.date.isoformat(),
                        "high_c": point.temp_max_c,
                        "low_c": point.temp_min_c,
                        "condition": point.condition,
                        "rain_pct": point.precipitation_chance,
                    }
                    for point in payload.forecast
                ],
            }
        elif tool_name == IMAGES_TOOL and isinstance(payload, list):
            summary = {
                "image_count": len(payload),
                "images": [{"url": asset.url, "caption": asset.caption} for asset in payload],
            }
        elif tool_name == WEB_SEARCH_TOOL and isinstance(payload, list):
            summary = {
                "result_count": len(payload),
                "results": [
                    {"title": hit.title, "snippet": hit.snippet, "url": hit.url} for hit in payload
                ],
            }
        else:
            summary = {"result": _jsonable(payload)}

        return json.dumps(summary, default=str)[:MAX_TOOL_CONTENT_CHARS]

    @staticmethod
    def _state_updates(tool_name: str, payload: Any) -> dict[str, Any]:
        """Map a tool payload onto the typed state keys it belongs in.

        The model gets text; the UI gets typed objects. Both come from the same
        call, and this is where the payload is filed into state so the renderer
        never has to parse the model's prose back into data.

        Args:
            tool_name: Which tool produced the payload.
            payload: The tool's return value.

        Returns:
            A partial state update, empty for tools with no domain payload.
        """
        if tool_name == WEATHER_TOOL and isinstance(payload, WeatherPayload):
            return {"weather": payload}

        if tool_name == IMAGES_TOOL and isinstance(payload, list):
            return {"images": [asset for asset in payload if isinstance(asset, ImageAsset)]}

        if tool_name == WEB_SEARCH_TOOL and isinstance(payload, list):
            # Web results become knowledge chunks so the summariser reads one
            # shape whether the facts came from the vector store or the web.
            return {
                "knowledge": [
                    KnowledgeChunk(
                        chunk_id=f"web::{index}",
                        city="",
                        section=hit.title,
                        text=f"{hit.title}. {hit.snippet}",
                        source=hit.url,
                    )
                    for index, hit in enumerate(payload)
                ]
            }

        return {}


def _jsonable(value: Any) -> Any:
    """Coerce a value into something ``json.dumps`` will accept.

    Args:
        value: Any value.

    Returns:
        The value if it is already JSON-friendly, otherwise its string form.
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def make_tool_executor(
    registry: ToolRegistry,
    *,
    node_name: str,
    handles: set[str] | None = None,
) -> ManualToolExecutor:
    """Build an executor node for the graph.

    Args:
        registry: Configured tool registry.
        node_name: Name for trace events and timings.
        handles: Tool names this node is responsible for.

    Returns:
        A callable suitable for ``StateGraph.add_node``.
    """
    return ManualToolExecutor(registry, node_name=node_name, handles=handles)


__all__ = [
    "MAX_TOOL_CONTENT_CHARS",
    "ManualToolExecutor",
    "extract_tool_calls",
    "make_tool_executor",
]
