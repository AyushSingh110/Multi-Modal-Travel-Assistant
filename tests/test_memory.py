"""Tests for the checkpointer and the follow-up partial update - Distinction 3.

The claim is that a follow-up like "what about next week?" re-runs *only* the
weather branch. That is the easiest of the three distinctions to fake - a graph
could re-run everything and quietly discard the result, and the user-visible
behaviour would be identical.

So these tests do not check the answer. They check which nodes executed, how long
the turn took, and that state genuinely crossed a process boundary when the
durable backend is used.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from travel_agent.config.settings import Settings
from travel_agent.graph import edges
from travel_agent.graph.builder import build_dependencies, build_graph
from travel_agent.graph.checkpointer import create_checkpointer
from travel_agent.schemas.response import TravelResponse

WEATHER_MS = 300
IMAGE_MS = 400
SEARCH_MS = 300


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "mock_weather_latency_ms": WEATHER_MS,
        "mock_image_latency_ms": IMAGE_MS,
        "mock_search_latency_ms": SEARCH_MS,
        "mock_latency_jitter": 0.0,
        "tool_timeout_seconds": 5.0,
        "tool_max_attempts": 1,
        "image_fallback_mode": "local",
        "router_similarity_threshold": 0.07,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


@pytest.fixture
def app() -> Any:
    """A graph with the default in-memory checkpointer."""
    return build_graph(build_dependencies(_settings()))


# ============================================== the follow-up, end to end ====
async def test_follow_up_runs_only_the_weather_branch(app: Any) -> None:
    """The heart of Distinction 3, checked by which nodes actually executed.

    ``timings`` is reset at the start of every turn, so its keys are a precise
    record of what ran *this* turn. A graph that re-ran the branches and discarded
    the results would still have their keys here, and this test would fail.
    """
    config = _config("follow-1")
    first = await app.ainvoke({"user_query": "Tell me about Tokyo"}, config=config)

    assert set(first["timings"]) >= {"retrieve_vector", "execute_weather", "execute_images"}

    second = await app.ainvoke({"user_query": "what about next week?"}, config=config)

    assert "execute_weather" in second["timings"], "the weather branch must re-run"
    assert "execute_images" not in second["timings"], "images must NOT re-run"
    assert "retrieve_vector" not in second["timings"], "the knowledge branch must NOT re-run"
    assert "web_search" not in second["timings"]


async def test_follow_up_avoids_real_work_and_is_no_slower(app: Any) -> None:
    """What skipping actually buys, stated honestly.

    The obvious assertion - "turn 2 is much faster on the clock" - overstates the
    case, and measuring it made that clear. The skipped branches ran *concurrently*
    with the weather branch on turn 1, so removing them takes almost nothing off
    the wall clock: the turn still costs whatever the weather branch costs.

    The real saving is **work avoided**, not latency: an entire image-provider
    round-trip and a knowledge read that would have been thrown away, plus the
    quota and money they cost against a live API. So that is what is asserted
    here, alongside the weaker claim that the follow-up is at least not slower.
    """
    config = _config("follow-2")

    started = time.perf_counter()
    await app.ainvoke({"user_query": "Tell me about Tokyo"}, config=config)
    first_turn = time.perf_counter() - started

    started = time.perf_counter()
    second = await app.ainvoke({"user_query": "what about next week?"}, config=config)
    second_turn = time.perf_counter() - started

    # The strong claim: measurable provider work was genuinely not done.
    assert second["skipped_ms_saved"] >= IMAGE_MS * 0.8, (
        f"only {second['skipped_ms_saved']:.0f} ms of work reported as avoided, "
        f"which is less than one image fetch"
    )
    # The weak claim: doing less must not somehow cost more.
    assert second_turn <= first_turn * 1.1, (
        f"follow-up took {second_turn * 1000:.0f} ms against {first_turn * 1000:.0f} ms "
        f"for a turn that did strictly more work"
    )


async def test_follow_up_carries_the_city_and_moves_the_dates(app: Any) -> None:
    config = _config("follow-3")
    first = await app.ainvoke({"user_query": "Tell me about Tokyo"}, config=config)
    second = await app.ainvoke({"user_query": "what about next week?"}, config=config)

    assert second["city"] == "Tokyo", "the city must come from checkpointed state"
    assert second["intent"] == "weather_only"
    assert second["date_range"].label == "next week"
    assert second["date_range"].start > first["date_range"].start


async def test_follow_up_reports_what_it_skipped_and_what_that_saved(app: Any) -> None:
    """The evidence that rules out re-running and discarding."""
    config = _config("follow-4")
    await app.ainvoke({"user_query": "Tell me about Tokyo"}, config=config)
    second = await app.ainvoke({"user_query": "what about next week?"}, config=config)

    assert set(second["skipped_nodes"]) == {"retrieve_vector", "execute_images"}
    assert second["skipped_ms_saved"] > IMAGE_MS * 0.5, (
        f"reported saving of {second['skipped_ms_saved']:.0f} ms looks too small to "
        f"be the image branch"
    )

    skip_events = [event for event in second["trace"] if event.kind == "skip"]
    assert skip_events, "the trace panel needs a skip event to render"
    assert "execute_images" in skip_events[0].message


async def test_follow_up_keeps_the_previous_images_and_summary(app: Any) -> None:
    """Skipping must preserve the earlier results, not blank them."""
    config = _config("follow-5")
    first = await app.ainvoke({"user_query": "Tell me about Tokyo"}, config=config)
    second = await app.ainvoke({"user_query": "what about next week?"}, config=config)

    response: TravelResponse = second["response"]
    assert response.image_urls == first["response"].image_urls
    assert response.city_summary == first["response"].city_summary
    assert len(response.weather_forecast) == 7


async def test_follow_up_actually_refreshes_the_forecast(app: Any) -> None:
    """The point of the turn: the weather must be for the new window."""
    config = _config("follow-6")
    first = await app.ainvoke({"user_query": "Tell me about Tokyo"}, config=config)
    second = await app.ainvoke({"user_query": "what about next week?"}, config=config)

    assert (
        second["response"].weather_forecast[0].date > first["response"].weather_forecast[0].date
    ), "the forecast window did not move"


async def test_only_the_weather_tool_is_offered_on_a_follow_up(app: Any) -> None:
    config = _config("follow-7")
    await app.ainvoke({"user_query": "Tell me about Tokyo"}, config=config)
    second = await app.ainvoke({"user_query": "what about next week?"}, config=config)

    plan_event = next(event for event in second["trace"] if event.node == "plan_tools")

    assert plan_event.data["tools_offered"] == ["get_weather_forecast"]
    assert plan_event.data["tools_requested"] == ["get_weather_forecast"]


async def test_a_new_city_on_an_existing_thread_runs_everything_again(app: Any) -> None:
    """Carrying context must not mean refusing to start fresh."""
    config = _config("follow-8")
    await app.ainvoke({"user_query": "Tell me about Tokyo"}, config=config)
    second = await app.ainvoke({"user_query": "Now tell me about Paris"}, config=config)

    assert second["city"] == "Paris"
    assert second["intent"] == "new_city"
    assert set(second["timings"]) >= {"retrieve_vector", "execute_weather", "execute_images"}
    assert not second["skipped_nodes"]


# ================================================== the guarded failure ====
async def test_a_bare_follow_up_on_a_fresh_thread_asks_for_clarification(app: Any) -> None:
    """No city in the question and none in memory: ask, do not guess."""
    result = await app.ainvoke({"user_query": "what about next week?"}, config=_config("cold-1"))

    assert result["intent"] == "clarify"
    assert result["city"] is None
    assert result["response"].is_clarification
    assert "which city" in result["response"].city_summary.lower()
    assert not result["timings"].get("execute_weather"), "no tools should have run"


async def test_clarification_does_not_invent_a_city_name(app: Any) -> None:
    result = await app.ainvoke({"user_query": "what about next week?"}, config=_config("cold-2"))

    assert result["response"].city == ""
    assert result["response"].warnings


async def test_the_thread_recovers_once_a_city_is_named(app: Any) -> None:
    config = _config("cold-3")
    await app.ainvoke({"user_query": "what about next week?"}, config=config)
    second = await app.ainvoke({"user_query": "Tell me about Paris"}, config=config)

    assert second["city"] == "Paris"
    assert second["response"].image_urls


# ======================================================= thread isolation ====
async def test_threads_do_not_leak_state_into_each_other(app: Any) -> None:
    """thread_id is the isolation boundary; two conversations stay separate."""
    await app.ainvoke({"user_query": "Tell me about Tokyo"}, config=_config("thread-a"))
    await app.ainvoke({"user_query": "Tell me about Paris"}, config=_config("thread-b"))

    follow_a = await app.ainvoke(
        {"user_query": "what about next week?"}, config=_config("thread-a")
    )
    follow_b = await app.ainvoke(
        {"user_query": "what about next week?"}, config=_config("thread-b")
    )

    assert follow_a["city"] == "Tokyo"
    assert follow_b["city"] == "Paris"


async def test_a_follow_up_on_an_unseen_thread_has_no_memory_to_borrow(app: Any) -> None:
    await app.ainvoke({"user_query": "Tell me about Tokyo"}, config=_config("thread-c"))

    stranger = await app.ainvoke(
        {"user_query": "what about next week?"}, config=_config("thread-d")
    )

    assert stranger["intent"] == "clarify", "state leaked across the thread boundary"


# ==================================================== durable persistence ====
async def test_state_survives_rebuilding_the_graph_from_a_sqlite_file(tmp_path: Path) -> None:
    """What proves the memory is real rather than a process-local dictionary.

    The first graph, its checkpointer and its database connection are all closed
    before the second graph is built. Only the file on disk connects them.
    """
    database = tmp_path / "checkpoints.sqlite"
    settings = _settings(checkpointer="sqlite", checkpoint_db_path=database)
    config = _config("durable-1")

    first_handle = await create_checkpointer(settings)
    assert first_handle.kind == "sqlite", "the durable backend must actually be in use"
    assert first_handle.is_durable

    first_app = build_graph(build_dependencies(settings), checkpointer=first_handle.saver)
    first = await first_app.ainvoke({"user_query": "Tell me about Tokyo"}, config=config)
    assert first["city"] == "Tokyo"

    await first_handle.close()
    del first_app, first_handle

    assert database.exists() and database.stat().st_size > 0

    # A second process, in effect: new connection, new saver, new graph.
    second_handle = await create_checkpointer(settings)
    second_app = build_graph(build_dependencies(settings), checkpointer=second_handle.saver)
    second = await second_app.ainvoke({"user_query": "what about next week?"}, config=config)

    assert second["city"] == "Tokyo", "the city did not survive the restart"
    assert second["intent"] == "weather_only"
    assert "execute_images" not in second["timings"]
    assert second["response"].image_urls, "the earlier images should still be in state"

    await second_handle.close()


async def test_memory_checkpointer_is_the_default(tmp_path: Path) -> None:
    handle = await create_checkpointer(_settings())

    assert handle.kind == "memory"
    assert not handle.is_durable
    assert "not durable" in handle.location
    await handle.close()


async def test_a_broken_sqlite_path_degrades_to_memory(tmp_path: Path) -> None:
    """Losing durability is survivable; failing to start is not."""
    unusable = tmp_path / "a-file-not-a-directory"
    unusable.write_text("blocking the path", encoding="utf-8")

    handle = await create_checkpointer(
        _settings(checkpointer="sqlite", checkpoint_db_path=unusable / "nested" / "cp.sqlite")
    )

    assert handle.kind == "memory"
    await handle.close()


# ============================================================= unit level ====
def test_planned_and_skipped_branches_are_complementary() -> None:
    """The edge dispatches one list and the node reports the other; they must agree."""
    full = {"intent": "new_city", "route": "vector"}
    follow_up = {"intent": "weather_only", "route": "vector"}

    assert edges.planned_branches(full) == [
        "retrieve_vector",
        "execute_weather",
        "execute_images",
    ]
    assert edges.skipped_branches(full) == []

    assert edges.planned_branches(follow_up) == ["execute_weather"]
    assert edges.skipped_branches(follow_up) == ["retrieve_vector", "execute_images"]


def test_skipped_branches_follows_the_web_route() -> None:
    assert edges.skipped_branches({"intent": "weather_only", "route": "web"}) == [
        "web_search",
        "execute_images",
    ]


def test_every_branch_is_either_planned_or_skipped() -> None:
    """No branch may fall through the gap between the two lists."""
    for intent in ("new_city", "weather_only"):
        for route in ("vector", "web"):
            state = {"intent": intent, "route": route}
            planned = set(edges.planned_branches(state))
            skipped = set(edges.skipped_branches(state))

            assert not planned & skipped, "a branch cannot be both run and skipped"
            knowledge = "web_search" if route == "web" else "retrieve_vector"
            assert planned | skipped == {knowledge, "execute_weather", "execute_images"}


def test_concurrent_turns_on_different_threads_do_not_interfere() -> None:
    """Two conversations running at once must stay separate."""
    app = build_graph(build_dependencies(_settings()))

    async def run() -> tuple[Any, Any]:
        return await asyncio.gather(
            app.ainvoke({"user_query": "Tell me about Tokyo"}, config=_config("par-a")),
            app.ainvoke({"user_query": "Tell me about Paris"}, config=_config("par-b")),
        )

    tokyo, paris = asyncio.run(run())

    assert tokyo["city"] == "Tokyo"
    assert paris["city"] == "Paris"
    assert tokyo["response"].image_urls != paris["response"].image_urls
