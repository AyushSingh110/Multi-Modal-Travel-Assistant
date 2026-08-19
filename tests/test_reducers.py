"""Tests for the state reducers."""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from travel_agent.schemas.state import (
    TravelState,
    add_token_usage,
    append_list,
    merge_timings,
    new_turn_updates,
    replace_value,
)
from travel_agent.schemas.trace import TokenUsage, TraceEvent


# --------------------------------------------------------------- append_list --
def test_append_list_concatenates_in_order() -> None:
    assert append_list([1, 2], [3]) == [1, 2, 3]


def test_append_list_handles_missing_existing_value() -> None:
    assert append_list(None, ["a"]) == ["a"]


def test_append_list_treats_none_update_as_reset() -> None:
    assert append_list([1, 2, 3], None) == []


def test_append_list_does_not_mutate_its_inputs() -> None:
    existing = [1]
    update = [2]
    merged = append_list(existing, update)
    merged.append(99)
    assert existing == [1]
    assert update == [2]


# ------------------------------------------------------------- merge_timings --
def test_merge_timings_unions_disjoint_keys() -> None:
    assert merge_timings({"weather": 10.0}, {"images": 20.0}) == {
        "weather": 10.0,
        "images": 20.0,
    }


def test_merge_timings_prefers_the_newer_measurement_on_collision() -> None:
    assert merge_timings({"weather": 10.0}, {"weather": 30.0}) == {"weather": 30.0}


def test_merge_timings_treats_none_update_as_reset() -> None:
    assert merge_timings({"weather": 10.0}, None) == {}


# ----------------------------------------------------------- add_token_usage --
def test_add_token_usage_sums_field_wise() -> None:
    left = TokenUsage(
        prompt_tokens=10, completion_tokens=5, total_tokens=15, llm_calls=1, model="a"
    )
    right = TokenUsage(
        prompt_tokens=20, completion_tokens=7, total_tokens=27, llm_calls=1, model="b"
    )

    merged = add_token_usage(left, right)

    assert merged.prompt_tokens == 30
    assert merged.completion_tokens == 12
    assert merged.total_tokens == 42
    assert merged.llm_calls == 2
    assert merged.model == "b"


def test_add_token_usage_handles_first_write() -> None:
    usage = TokenUsage(total_tokens=5, llm_calls=1)
    assert add_token_usage(None, usage) is usage


def test_add_token_usage_treats_none_update_as_reset() -> None:
    existing = TokenUsage(total_tokens=100, llm_calls=4)
    assert add_token_usage(existing, None).total_tokens == 0


# ------------------------------------------------------------ replace_value --
def test_replace_value_overwrites_with_the_update() -> None:
    assert replace_value("old", "new") == "new"


def test_replace_value_keeps_existing_when_update_is_none() -> None:
    """A follow-up turn runs only the weather branch; images must survive it."""
    assert replace_value(["image-a"], None) == ["image-a"]


# ------------------------------------------------------------- turn helpers --
def test_new_turn_updates_clears_per_turn_observability_keys() -> None:
    updates = new_turn_updates("Tell me about Kyoto", turn_index=2)

    assert updates["user_query"] == "Tell me about Kyoto"
    assert updates["turn_index"] == 2
    # None is the documented reset signal for the accumulating reducers.
    for key in ("trace", "errors", "skipped_nodes", "cache_hits", "timings", "token_usage"):
        assert updates[key] is None
    # Slots must NOT be cleared: that is what makes follow-up turns work.
    assert "city" not in updates
    assert "images" not in updates


# ----------------------------------------- the reducers under a real fan-out --
def test_reducers_merge_correctly_under_concurrent_writes() -> None:
    """Run three branches concurrently and assert every reducer merged cleanly.

    This is the test that makes the parallel design defensible: it does not stub
    LangGraph, it drives the real runtime and asserts on the merged state.
    """

    async def weather(state: TravelState) -> dict[str, object]:
        await asyncio.sleep(0.05)
        return {
            "trace": [TraceEvent(node="weather", message="fetched")],
            "timings": {"weather": 50.0},
            "token_usage": TokenUsage(prompt_tokens=10, total_tokens=10, llm_calls=1, model="m"),
            "cache_hits": ["weather:paris"],
        }

    async def images(state: TravelState) -> dict[str, object]:
        await asyncio.sleep(0.05)
        return {
            "trace": [TraceEvent(node="images", message="fetched")],
            "timings": {"images": 55.0},
            "token_usage": TokenUsage(prompt_tokens=4, total_tokens=4, llm_calls=1, model="m"),
        }

    async def knowledge(state: TravelState) -> dict[str, object]:
        await asyncio.sleep(0.05)
        return {
            "trace": [TraceEvent(node="knowledge", message="retrieved")],
            "timings": {"knowledge": 45.0},
            "token_usage": TokenUsage(prompt_tokens=6, total_tokens=6, llm_calls=1, model="m"),
        }

    builder: StateGraph = StateGraph(TravelState)
    builder.add_node("plan", lambda state: {"trace": [TraceEvent(node="plan", message="planned")]})
    builder.add_node("weather", weather)
    builder.add_node("images", images)
    builder.add_node("knowledge", knowledge)
    builder.add_node("join", lambda state: {})
    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan", lambda state: ["weather", "images", "knowledge"], ["weather", "images", "knowledge"]
    )
    for branch in ("weather", "images", "knowledge"):
        builder.add_edge(branch, "join")
    builder.add_edge("join", END)

    result = asyncio.run(builder.compile().ainvoke({"trace": [], "user_query": "x"}))

    # append_list: every branch's event survived, none overwrote another.
    assert len(result["trace"]) == 4
    assert {event.node for event in result["trace"]} == {"plan", "weather", "images", "knowledge"}

    # merge_timings: three disjoint keys unioned rather than clobbered.
    assert set(result["timings"]) == {"weather", "images", "knowledge"}

    # add_token_usage: summed across concurrent writers.
    assert result["token_usage"].total_tokens == 20
    assert result["token_usage"].llm_calls == 3

    # A key only one branch wrote is still present.
    assert result["cache_hits"] == ["weather:paris"]


def test_missing_reducer_would_have_failed() -> None:
    """Prove the reducers are load-bearing, not decorative.

    An identical fan-out over a key with no reducer raises. This is the failure
    the ``Annotated`` declarations in the state exist to prevent, and pinning it
    down in a test stops anyone "simplifying" them away later.
    """

    class UnreducedState(TypedDict, total=False):
        value: str
        counter: Annotated[int, operator.add]

    builder: StateGraph = StateGraph(UnreducedState)
    builder.add_node("start", lambda state: {})
    builder.add_node("a", lambda state: {"value": "a", "counter": 1})
    builder.add_node("b", lambda state: {"value": "b", "counter": 1})
    builder.add_edge(START, "start")
    builder.add_conditional_edges("start", lambda state: ["a", "b"], ["a", "b"])
    builder.add_edge("a", END)
    builder.add_edge("b", END)

    with pytest.raises(Exception, match="one value per step"):
        asyncio.run(builder.compile().ainvoke({}))
