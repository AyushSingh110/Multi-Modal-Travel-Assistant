"""The manual tool executor node - Distinction 1.

================================================================================
THE RAW TOOL-CALLING PROTOCOL, IN PLAIN ENGLISH
================================================================================

A language model cannot run code. It can only produce text. So when you want a
model to "use a tool", what actually happens is a strict, three-part exchange
between your program and the model, and this module implements the middle part
by hand.

**Part 1 - you advertise the tools.** Before asking anything, you send the model
a list of functions it is allowed to request, each with a name, a description,
and a JSON schema describing its arguments. In this project those schemas are
generated from Pydantic models in ``schemas/tools.py``.

**Part 2 - the model asks.** Instead of answering in prose, the model can reply
with a structured request. In LangChain that arrives as an ``AIMessage`` whose
``.tool_calls`` is a list, each entry looking like::

    {"id": "call_a1b2c3",              # the model's handle for THIS request
     "name": "get_weather_forecast",   # which advertised tool it wants
     "args": {"city": "Tokyo", "days": 7},
     "type": "tool_call"}

The model has not run anything. It has produced a request and stopped.

**Part 3 - you answer, and the id is the whole game.** Your program runs the
function and sends the result back as a ``ToolMessage``. That message MUST carry
``tool_call_id`` set to the *exact* id from the request it answers.

Why the pairing matters so much: a model can ask for several tools at once - this
project routinely requests weather and images in the same turn - and the results
come back as separate messages, potentially in a different order to the requests,
because the tools ran concurrently and finished when they finished. The id is the
only thing connecting an answer to its question. Order tells you nothing.

**What breaks if you get it wrong:**

* *Wrong id* - the model attributes the weather data to the image request. It
  will then confidently describe photographs of a 7-day forecast. Nothing raises;
  you just get nonsense.
* *Missing ToolMessage* - most providers reject the next request outright with an
  API error, because the conversation contains a question with no answer. OpenAI
  and Groq both do this. It is a hard failure, mid-conversation.
* *Reporting failure as success* - if a tool raises and you return the string
  ``"error: timeout"`` with the default status, the model reads that as the
  legitimate result of the call. It has no way to know the tool failed, so it
  will summarise "the weather is error: timeout". ``ToolMessage`` has a ``status``
  field for exactly this: setting ``status="error"`` tells the model the call did
  not succeed, so it can apologise, retry, or work around the gap honestly.

================================================================================
WHY THIS IS WRITTEN BY HAND
================================================================================

LangGraph ships ``langgraph.prebuilt.ToolNode``, which does all of the above in
one line. This project deliberately does not use it (and a test enforces that).
Writing it out buys three things the prebuilt node does not give:

1. **Per-tool error isolation.** One tool failing must never abort its siblings.
   Here each call is executed independently and a failure becomes an error
   ``ToolMessage`` while the other calls carry on.
2. **Selective execution.** One executor instance can be told to handle only
   *some* of the tool calls in a message. That is what lets the weather branch and
   the image branch of the parallel fan-out share this exact code while running
   concurrently in different graph nodes.
3. **Observability.** Every call emits a trace event and a timing, which is what
   the UI trace panel renders.
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

#: Characters of tool output handed back to the model. Enough to summarise from,
#: bounded so a large payload cannot dominate the context window.
MAX_TOOL_CONTENT_CHARS = 4000


def extract_tool_calls(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Read the pending tool calls from the most recent AI message.

    Args:
        messages: The conversation so far.

    Returns:
        The raw ``tool_calls`` payload, or an empty list when the last message is
        not an ``AIMessage`` or requested no tools.
    """
    if not messages:
        return []

    last = messages[-1]
    if not isinstance(last, AIMessage):
        return []

    return list(getattr(last, "tool_calls", []) or [])


class ManualToolExecutor:
    """Executes a model's tool calls and returns correctly paired tool messages.

    One instance is created per graph node. ``handles`` is what allows the same
    class to serve several concurrent branches: the weather branch executes only
    the weather call, the image branch only the image call, and neither touches
    the other's work.

    Attributes:
        registry: Where tool names are resolved and executed.
        node_name: Name used in trace events and timings.
        handles: Tool names this instance is responsible for, or ``None`` for all.
    """

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
        """Execute the pending tool calls this node is responsible for.

        Args:
            state: Current graph state.

        Returns:
            A partial state update: the ``ToolMessage`` replies, any domain
            payloads the tools produced, plus trace events, timings and error
            records. Never raises - a failing tool becomes an error message, not
            an exception, because the rest of the page must still render.
        """
        tool_calls = extract_tool_calls(state.get("messages", []))
        mine = [call for call in tool_calls if self._is_mine(call)]

        if not mine:
            # Not an error. On a follow-up turn the model may request only the
            # weather tool, leaving the image branch with nothing to do.
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
            # Concurrent, and crucially independent: return_exceptions keeps one
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

    # ------------------------------------------------------------ one call --
    async def _execute_one(
        self, call: dict[str, Any]
    ) -> tuple[ToolMessage, list[TraceEvent], ToolErrorRecord | None, dict[str, Any]]:
        """Run a single tool call and build its reply.

        Args:
            call: One entry from the model's ``tool_calls`` payload.

        Returns:
            A tuple of the ``ToolMessage`` to send back, trace events, an optional
            error record, and any domain state updates the payload produced.
        """
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

    # ------------------------------------------------------- payload shaping --
    @staticmethod
    def _summarise(tool_name: str, payload: Any) -> str:
        """Render a tool result as compact text for the model.

        Full payloads are verbose - a 7-day forecast with humidity and wind is
        several hundred tokens of JSON the model does not need. Each tool is
        summarised down to the fields that actually inform the answer.

        Args:
            tool_name: Which tool produced the payload.
            payload: The tool's return value.

        Returns:
            A compact JSON string, truncated to a sane length.
        """
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
