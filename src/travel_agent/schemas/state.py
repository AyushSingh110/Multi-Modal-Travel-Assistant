"""The typed graph state and its reducers.

WHAT STATE IS
    LangGraph runs a set of functions ("nodes") that each receive the current
    state and return a partial update. The state is a single typed dictionary
    that travels through the graph and is saved by the checkpointer between
    turns.

WHY REDUCERS EXIST
    When two nodes run *concurrently* in the same step and both write the same
    key, LangGraph does not silently pick a winner - it raises
    ``InvalidUpdateError: Can receive only one value per step``. A reducer is a
    function ``(existing, update) -> merged`` attached to a key via
    ``Annotated[...]`` that tells LangGraph how to combine those writes.

    This is not decoration. The parallel fan-out in this project (weather,
    images and knowledge retrieval running at once) is only legal *because* every
    key those branches touch carries a reducer. That is why each one below has a
    unit test proving it merges correctly.

RESET SEMANTICS
    The accumulating reducers treat an explicit ``None`` update as "clear this
    key". Without it there would be no way to empty a list that only ever grows,
    and each new turn would inherit the previous turn's trace.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict, TypeVar

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from travel_agent.schemas.intent import DateRange, Intent, MatchReason, RouteName
from travel_agent.schemas.knowledge import KnowledgeChunk
from travel_agent.schemas.response import ImageAsset, TravelResponse, WeatherPayload
from travel_agent.schemas.trace import ParallelMetrics, TokenUsage, ToolErrorRecord, TraceEvent

_T = TypeVar("_T")


def append_list(existing: list[_T] | None, update: list[_T] | None) -> list[_T]:
    """Concatenate list updates, with ``None`` meaning "reset to empty".

    Used for every key that several concurrent branches append to: the trace, the
    error log, the skipped-node list, cache hits.

    Args:
        existing: Value already in state, or ``None`` on the first write.
        update: The node's update, or ``None`` to clear the key.

    Returns:
        The merged list. Never ``None``, so downstream code can iterate freely.
    """
    if update is None:
        return []
    return list(existing or []) + list(update)


def merge_timings(
    existing: dict[str, float] | None, update: dict[str, float] | None
) -> dict[str, float]:
    """Merge per-node timing dictionaries written by concurrent branches.

    Parallel branches write disjoint keys (each node records its own name), so in
    practice this is a union. On a genuine key collision - the same node running
    twice, e.g. across turns - the newer measurement wins, because a stale
    duration is more misleading than a replaced one.

    Args:
        existing: Timings already in state, or ``None``.
        update: New timings, or ``None`` to clear.

    Returns:
        The merged mapping of node name to milliseconds.
    """
    if update is None:
        return {}
    return {**(existing or {}), **update}


def add_token_usage(existing: TokenUsage | None, update: TokenUsage | None) -> TokenUsage:
    """Sum token usage across every model call in a request.

    Several nodes call the LLM (intent classification, tool planning, synthesis)
    and some of them run in parallel, so usage has to accumulate rather than
    overwrite.

    Args:
        existing: Usage already in state, or ``None``.
        update: Usage from one model call, or ``None`` to reset the counter.

    Returns:
        The combined usage.
    """
    if update is None:
        return TokenUsage()
    if existing is None:
        return update
    return existing.merged_with(update)


def replace_value(existing: _T | None, update: _T | None) -> _T | None:
    """Overwrite a single-writer key, ignoring a ``None`` update.

    Weather, images and retrieved knowledge each have exactly one writer per
    turn, so they do not need to accumulate. They still carry a reducer for a
    different reason: on a follow-up turn only the weather branch runs, and this
    keeps the previous turn's images and summary in state instead of wiping them.

    Args:
        existing: Value already in state.
        update: Replacement value, or ``None`` to keep what is there.

    Returns:
        ``update`` when it carries data, otherwise ``existing``.
    """
    if update is None:
        return existing
    return update


class TravelState(TypedDict, total=False):
    """State passed between every node in the graph.

    ``total=False`` means no key is required up front: nodes fill in what they
    own. Every key written by more than one node, or by a node that runs
    concurrently with another, carries an ``Annotated`` reducer.
    """

    # --- conversation -------------------------------------------------------
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    thread_id: str
    turn_index: int

    # --- slots, preserved across turns by the checkpointer ------------------
    city: str | None
    previous_city: str | None
    date_range: DateRange
    intent: Intent

    # --- routing ------------------------------------------------------------
    route: RouteName | None
    route_score: float | None
    route_threshold: float
    route_reason: str
    route_match_reason: MatchReason
    route_all_scores: dict[str, float]
    matched_city: str | None

    # Set when the fan-out is dispatched and read by the join node. The gap
    # between the two is the superstep's wall-clock time, which is the measured
    # half of the parallelism claim.
    fanout_started_at: float

    # --- fan-out results (single writer each, reducer keeps prior turns) -----
    knowledge: Annotated[list[KnowledgeChunk] | None, replace_value]
    weather: Annotated[WeatherPayload | None, replace_value]
    images: Annotated[list[ImageAsset] | None, replace_value]

    # --- observability (written concurrently: reducers are mandatory) -------
    trace: Annotated[list[TraceEvent], append_list]
    errors: Annotated[list[ToolErrorRecord], append_list]
    skipped_nodes: Annotated[list[str], append_list]
    cache_hits: Annotated[list[str], append_list]
    timings: Annotated[dict[str, float], merge_timings]
    token_usage: Annotated[TokenUsage, add_token_usage]
    parallel_metrics: Annotated[ParallelMetrics | None, replace_value]

    # --- final answer -------------------------------------------------------
    response: Annotated[TravelResponse | None, replace_value]


def new_turn_updates(user_query: str, turn_index: int) -> dict[str, Any]:
    """Build the state update that starts a fresh turn.

    Clears the per-turn observability keys (via the reducers' ``None`` reset) so
    the trace panel shows this turn's activity rather than the whole session's,
    while leaving the slots and previous results intact for follow-ups.

    Args:
        user_query: The raw text the user typed.
        turn_index: Zero-based index of this turn within the thread.

    Returns:
        A partial state update.
    """
    return {
        "user_query": user_query,
        "turn_index": turn_index,
        "trace": None,
        "errors": None,
        "skipped_nodes": None,
        "cache_hits": None,
        "timings": None,
        "token_usage": None,
    }


__all__ = [
    "TravelState",
    "add_token_usage",
    "append_list",
    "merge_timings",
    "new_turn_updates",
    "replace_value",
]
