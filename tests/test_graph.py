"""Tests for graph assembly and the parallel fan-out - Distinction 2.

The parallelism claim is proved by measurement, not assertion: the fan-out is run
against providers with known latencies and the superstep's wall-clock time is
compared against the sum of the branch durations. A companion test removes a
reducer and shows the graph refuses to run, which is *why* the reducers exist.
"""

from __future__ import annotations

import asyncio
import operator
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from travel_agent.config.settings import Settings
from travel_agent.graph import edges
from travel_agent.graph.builder import GraphDependencies, build_dependencies, build_graph
from travel_agent.graph.diagram import annotate_mermaid, parse_edges
from travel_agent.schemas.response import TravelResponse, WeatherPayload
from travel_agent.schemas.trace import ParallelMetrics
from travel_agent.services.llm.mock import MockLLM
from travel_agent.tools.images.mock import MockImageProvider
from travel_agent.tools.registry import ToolRegistry
from travel_agent.tools.search.mock import MockSearchProvider
from travel_agent.tools.weather.base import WeatherProvider

REPO_ROOT = Path(__file__).resolve().parents[1]

# Latencies chosen so the branches differ and the arithmetic is unambiguous.
WEATHER_MS = 400
IMAGE_MS = 500


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "mock_weather_latency_ms": WEATHER_MS,
        "mock_image_latency_ms": IMAGE_MS,
        "mock_search_latency_ms": 300,
        "mock_latency_jitter": 0.0,
        "tool_timeout_seconds": 5.0,
        "tool_max_attempts": 1,
        "image_fallback_mode": "local",  # no network in tests
        "router_similarity_threshold": 0.07,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _dependencies(settings: Settings, weather: WeatherProvider | None = None) -> GraphDependencies:
    deps = build_dependencies(settings)
    if weather is not None:
        deps.registry = ToolRegistry(
            weather,
            MockImageProvider(settings),
            MockSearchProvider(settings),
            settings=settings,
        )
    return deps


def _app(settings: Settings | None = None, weather: WeatherProvider | None = None) -> Any:
    resolved = settings or _settings()
    return build_graph(_dependencies(resolved, weather))


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


# ================================================ Distinction 2: measured ====
async def test_fan_out_is_measurably_faster_than_sequential() -> None:
    """The parallelism claim, proved by the clock.

    Weather sleeps 400 ms and images 500 ms. Run one after another that is at
    least 900 ms; run together it should be about 500 ms. The assertion leaves
    generous headroom so a loaded machine does not produce a flaky failure - it
    is checking for concurrency, not for a specific speed.
    """
    result = await _app().ainvoke({"user_query": "Tell me about Tokyo"}, config=_config("par-1"))

    metrics: ParallelMetrics = result["parallel_metrics"]
    sequential = metrics.sequential_equivalent_ms
    parallel = metrics.parallel_wall_clock_ms

    assert sequential >= WEATHER_MS + IMAGE_MS - 50, f"branches did not both run: {metrics}"
    # The decisive check: the superstep took less time than its parts summed.
    assert parallel < sequential * 0.75, (
        f"fan-out looks sequential: {parallel:.0f} ms wall clock vs "
        f"{sequential:.0f} ms sequential-equivalent"
    )
    assert metrics.speedup > 1.3


async def test_wall_clock_is_close_to_the_slowest_branch() -> None:
    """Concurrent work costs the slowest branch, not the sum of all of them."""
    result = await _app().ainvoke({"user_query": "Tell me about Paris"}, config=_config("par-2"))

    metrics: ParallelMetrics = result["parallel_metrics"]
    slowest = max(metrics.branch_durations_ms.values())

    assert metrics.parallel_wall_clock_ms < slowest * 1.6, (
        f"wall clock {metrics.parallel_wall_clock_ms:.0f} ms is far above the "
        f"slowest branch {slowest:.0f} ms - the branches are not overlapping"
    )


async def test_all_three_branches_report_durations() -> None:
    result = await _app().ainvoke({"user_query": "Tell me about Tokyo"}, config=_config("par-3"))

    branches = result["parallel_metrics"].branch_durations_ms

    assert set(branches) == {"retrieve_vector", "execute_weather", "execute_images"}
    assert branches["execute_weather"] >= WEATHER_MS * 0.8
    assert branches["execute_images"] >= IMAGE_MS * 0.8


async def test_the_speed_up_is_recorded_in_the_trace_for_the_ui() -> None:
    result = await _app().ainvoke({"user_query": "Tell me about Paris"}, config=_config("par-4"))

    timing_events = [event for event in result["trace"] if event.kind == "timing"]

    assert timing_events, "the join node must publish its measurement"
    assert "sequential-equivalent" in timing_events[0].message
    assert timing_events[0].data["speedup"] > 1.0


# ======================================== why the reducers are mandatory ====
def test_removing_a_reducer_breaks_the_fan_out() -> None:
    """Pin down *why* the state annotations exist.

    This is the same shape as the real fan-out - one node scheduling several
    concurrent branches - over a state whose key has no reducer. LangGraph
    refuses to guess which write wins. Every key the real branches touch carries
    an ``Annotated`` reducer precisely because of this.
    """

    class UnreducedState(TypedDict, total=False):
        trace: Annotated[list[str], operator.add]
        weather: str  # deliberately un-reduced

    async def branch_a(state: UnreducedState) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"trace": ["a"], "weather": "from-a"}

    async def branch_b(state: UnreducedState) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"trace": ["b"], "weather": "from-b"}

    builder: StateGraph = StateGraph(UnreducedState)
    builder.add_node("plan", lambda state: {"trace": ["plan"]})
    builder.add_node("a", branch_a)
    builder.add_node("b", branch_b)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", lambda state: ["a", "b"], ["a", "b"])
    builder.add_edge("a", END)
    builder.add_edge("b", END)

    with pytest.raises(Exception, match="one value per step"):
        asyncio.run(builder.compile().ainvoke({}))


def test_the_same_fan_out_succeeds_once_the_key_has_a_reducer() -> None:
    """The control case: identical topology, one annotation added."""

    class ReducedState(TypedDict, total=False):
        trace: Annotated[list[str], operator.add]
        weather: Annotated[list[str], operator.add]

    async def branch_a(state: ReducedState) -> dict[str, Any]:
        return {"trace": ["a"], "weather": ["from-a"]}

    async def branch_b(state: ReducedState) -> dict[str, Any]:
        return {"trace": ["b"], "weather": ["from-b"]}

    builder: StateGraph = StateGraph(ReducedState)
    builder.add_node("plan", lambda state: {"trace": ["plan"]})
    builder.add_node("a", branch_a)
    builder.add_node("b", branch_b)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", lambda state: ["a", "b"], ["a", "b"])
    builder.add_edge("a", END)
    builder.add_edge("b", END)

    result = asyncio.run(builder.compile().ainvoke({}))

    assert sorted(result["weather"]) == ["from-a", "from-b"]


# ================================================== routing through the graph ==
async def test_in_store_city_routes_to_the_vector_store() -> None:
    result = await _app().ainvoke({"user_query": "Tell me about Tokyo"}, config=_config("route-1"))

    assert result["route"] == "vector"
    assert result["route_match_reason"] == "exact"
    assert result["matched_city"] == "Tokyo"
    assert result["knowledge"], "the vector branch should have retrieved passages"
    assert result["response"].knowledge_source == "vector_store"
    assert all(chunk.source.endswith(".md") for chunk in result["knowledge"])


async def test_out_of_store_city_routes_to_web_search() -> None:
    result = await _app().ainvoke({"user_query": "Tell me about Kyoto"}, config=_config("route-2"))

    assert result["route"] == "web"
    assert result["route_match_reason"] == "similarity"
    assert result["route_score"] < 0.07
    assert result["knowledge"], "the web branch should have produced knowledge chunks"
    assert result["response"].knowledge_source == "web_search"
    assert all(chunk.source.startswith("http") for chunk in result["knowledge"])


async def test_the_unused_knowledge_branch_does_not_run() -> None:
    """Routing means choosing, not running both and discarding one."""
    result = await _app().ainvoke({"user_query": "Tell me about Kyoto"}, config=_config("route-3"))

    assert "web_search" in result["timings"]
    assert "retrieve_vector" not in result["timings"]


async def test_the_web_search_tool_is_not_even_offered_for_a_known_city() -> None:
    result = await _app().ainvoke({"user_query": "Tell me about Paris"}, config=_config("route-4"))

    plan_events = [event for event in result["trace"] if event.node == "plan_tools"]
    offered = plan_events[0].data["tools_offered"]

    assert "web_search" not in offered
    assert "get_weather_forecast" in offered and "search_city_images" in offered


async def test_routing_decision_carries_score_and_reason_into_state() -> None:
    """The UI needs the number and the explanation, not just the verdict."""
    result = await _app().ainvoke({"user_query": "Tell me about Tokyo"}, config=_config("route-5"))

    assert result["route_score"] > 0.07
    assert result["route_threshold"] == pytest.approx(0.07)
    assert "Tokyo" in result["route_reason"]
    assert set(result["route_all_scores"]) == {"Paris", "Tokyo", "New York"}


# ========================================================== end to end ====
async def test_end_to_end_produces_a_populated_response() -> None:
    result = await _app().ainvoke({"user_query": "Tell me about Tokyo"}, config=_config("e2e-1"))

    response: TravelResponse = result["response"]
    assert isinstance(response, TravelResponse)
    assert response.city == "Tokyo"
    assert len(response.city_summary) > 40
    assert len(response.weather_forecast) == 7
    assert len(response.image_urls) == 4
    assert all(url.startswith("https://") for url in response.image_urls)
    assert response.highlights
    assert not response.warnings


async def test_end_to_end_for_an_unknown_city_also_completes() -> None:
    result = await _app().ainvoke({"user_query": "Tell me about Kyoto"}, config=_config("e2e-2"))

    response: TravelResponse = result["response"]
    assert response.city == "Kyoto"
    assert len(response.weather_forecast) == 7
    assert response.image_urls
    assert response.sources


async def test_tool_messages_are_paired_and_present_in_the_conversation() -> None:
    result = await _app().ainvoke({"user_query": "Tell me about Paris"}, config=_config("e2e-3"))

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and m.tool_calls]
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    requested_ids = {call["id"] for message in ai_messages for call in message.tool_calls}
    answered_ids = {message.tool_call_id for message in tool_messages}

    assert requested_ids, "the model should have requested tools"
    assert answered_ids == requested_ids, "every tool call must get exactly one reply"


async def test_state_is_fully_typed_after_a_run() -> None:
    result = await _app().ainvoke({"user_query": "Tell me about Tokyo"}, config=_config("e2e-4"))

    assert isinstance(result["weather"], WeatherPayload)
    assert isinstance(result["parallel_metrics"], ParallelMetrics)
    assert isinstance(result["response"], TravelResponse)
    assert isinstance(result["token_usage"].total_tokens, int)
    assert result["token_usage"].llm_calls >= 1


# ======================================================== degradation ====
async def test_a_dead_weather_tool_still_renders_summary_and_images() -> None:
    """The rubric's graceful-degradation line, end to end through the graph."""
    settings = _settings(force_weather_failure=True, weather_failure_mode="server_error")

    result = await build_graph(_dependencies(settings)).ainvoke(
        {"user_query": "Tell me about Tokyo"}, config=_config("degrade-1")
    )

    response: TravelResponse = result["response"]
    assert response.weather_forecast == []
    assert len(response.image_urls) == 4, "images must survive a weather failure"
    assert len(response.city_summary) > 40
    assert response.warnings, "the user must be told something is missing"
    assert response.is_degraded
    assert result["errors"]


async def test_a_timing_out_weather_tool_does_not_stall_the_graph() -> None:
    # The timeout applies to every tool, so the image latency is lowered to keep
    # it well inside the budget - otherwise this would test two timeouts, not
    # one failing branch alongside a healthy one.
    settings = _settings(
        force_weather_failure=True,
        weather_failure_mode="timeout",
        tool_timeout_seconds=0.3,
        mock_image_latency_ms=60,
        mock_search_latency_ms=60,
    )

    started = time.perf_counter()
    result = await build_graph(_dependencies(settings)).ainvoke(
        {"user_query": "Tell me about Paris"}, config=_config("degrade-2")
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0, "a hanging tool must not hold the graph open"
    assert result["response"].image_urls, "the healthy branch must still deliver"
    assert result["response"].weather_forecast == []
    warning = result["response"].warnings[0]
    assert warning.strip().endswith(("s", ")")), f"empty failure detail: {warning!r}"
    assert "TimeoutError" in warning


# ============================================================ intent ====
async def test_a_query_with_no_city_asks_for_clarification() -> None:
    result = await _app().ainvoke({"user_query": "what about the weather"}, config=_config("int-1"))

    assert result["intent"] == "clarify"
    assert result["response"] is not None
    assert "weather" not in result["timings"], "no tools should have run"


async def test_next_week_is_recognised_as_a_date_change() -> None:
    settings = _settings()
    app = build_graph(_dependencies(settings))

    await app.ainvoke({"user_query": "Tell me about Tokyo"}, config=_config("int-2"))
    follow_up = await app.ainvoke({"user_query": "what about next week?"}, config=_config("int-2"))

    assert follow_up["city"] == "Tokyo", "the city must carry over from memory"
    assert follow_up["intent"] == "weather_only"
    assert follow_up["date_range"].label == "next week"


# ============================================================ topology ====
def test_the_committed_diagram_matches_the_compiled_graph() -> None:
    """graph.png is a submission artifact - it must not drift from the code."""
    app = build_graph(_dependencies(_settings()))
    generated = annotate_mermaid(app.get_graph().draw_mermaid())
    committed = (REPO_ROOT / "graph.mmd").read_text(encoding="utf-8")

    assert parse_edges(generated) == parse_edges(
        committed
    ), "graph.mmd is out of date. Regenerate with: python scripts/export_graph.py --force"


def test_annotation_does_not_change_the_topology() -> None:
    """Labels are presentation; the edges must be exactly what LangGraph emitted."""
    app = build_graph(_dependencies(_settings()))
    raw = app.get_graph().draw_mermaid()

    assert parse_edges(annotate_mermaid(raw)) == parse_edges(raw)


def test_the_diagram_labels_the_concurrent_branches() -> None:
    """If the picture does not show the parallelism, the distinction is invisible."""
    committed = (REPO_ROOT / "graph.mmd").read_text(encoding="utf-8")

    assert "|concurrent| execute_weather" in committed
    assert "|concurrent| execute_images" in committed
    assert "knowledge: in store" in committed
    assert "knowledge: not in store" in committed


def test_both_diagram_artifacts_are_committed() -> None:
    assert (REPO_ROOT / "graph.png").exists(), "graph.png is a required submission artifact"
    assert (REPO_ROOT / "graph.mmd").exists()
    assert (REPO_ROOT / "graph.png").stat().st_size > 5000


# ============================================================== mock llm ====
async def test_mock_llm_emits_a_protocol_faithful_payload() -> None:
    from langchain_core.messages import HumanMessage

    from travel_agent.schemas.tools import openai_tool_schemas

    call = await MockLLM().plan(
        [HumanMessage(content="City: Tokyo\nIntent: new_city\nKnowledge source: vector_store\n")],
        openai_tool_schemas(["get_weather_forecast", "search_city_images"]),
    )

    assert isinstance(call.message, AIMessage)
    assert len(call.message.tool_calls) == 2
    for tool_call in call.message.tool_calls:
        assert set(tool_call) >= {"id", "name", "args", "type"}
        assert tool_call["type"] == "tool_call"
        assert isinstance(tool_call["args"], dict)
    assert len({c["id"] for c in call.message.tool_calls}) == 2, "ids must be unique"
    assert call.usage.llm_calls == 1


async def test_mock_llm_only_requests_tools_it_was_offered() -> None:
    from langchain_core.messages import HumanMessage

    from travel_agent.schemas.tools import openai_tool_schemas

    call = await MockLLM().plan(
        [HumanMessage(content="City: Kyoto\nIntent: new_city\nKnowledge source: web_search\n")],
        openai_tool_schemas(["get_weather_forecast"]),
    )

    assert [c["name"] for c in call.message.tool_calls] == ["get_weather_forecast"]


async def test_mock_llm_skips_images_on_a_weather_only_follow_up() -> None:
    from langchain_core.messages import HumanMessage

    from travel_agent.schemas.tools import openai_tool_schemas

    call = await MockLLM().plan(
        [
            HumanMessage(
                content="City: Tokyo\nIntent: weather_only\nKnowledge source: vector_store\n"
            )
        ],
        openai_tool_schemas(),
    )

    assert [c["name"] for c in call.message.tool_calls] == ["get_weather_forecast"]


# =============================================================== wiring ====
def test_graph_exposes_the_expected_nodes() -> None:
    nodes = set(build_graph(_dependencies(_settings())).get_graph().nodes)

    assert {
        "normalize_input",
        "classify_intent",
        "plan_tools",
        "retrieve_vector",
        "web_search",
        "execute_weather",
        "execute_images",
        "join",
        "synthesize",
    } <= nodes


def test_fan_out_edge_returns_a_list_of_targets() -> None:
    """The mechanism behind Distinction 2, checked directly."""
    targets = edges.route_and_fan_out({"intent": "new_city", "route": "vector"})

    assert isinstance(targets, list)
    assert targets == ["retrieve_vector", "execute_weather", "execute_images"]


def test_fan_out_switches_the_knowledge_branch_on_the_route() -> None:
    assert edges.route_and_fan_out({"intent": "new_city", "route": "web"})[0] == "web_search"


def test_fan_out_narrows_to_weather_on_a_follow_up() -> None:
    assert edges.route_and_fan_out({"intent": "weather_only", "route": "vector"}) == [
        "execute_weather"
    ]


def test_intent_edge_skips_the_work_when_nothing_changed() -> None:
    response = TravelResponse(
        city="Tokyo", city_summary="A summary long enough to satisfy the minimum length rule."
    )

    assert edges.route_after_intent({"intent": "refine", "response": response}) == "synthesize"
    assert edges.route_after_intent({"intent": "clarify"}) == "synthesize"
    assert edges.route_after_intent({"intent": "new_city"}) == "plan_tools"


# ================================================== plan completion (live) ====
def test_plan_completion_adds_a_tool_the_model_omitted() -> None:
    """Regression test for a divergence only a live provider exposed.

    Asked about Tokyo with both tools offered, Groq's gpt-oss-120b called only the
    weather tool and the gallery came back empty. The mock always called both, so
    nothing caught it. The interface has a fixed contract - summary, gallery,
    chart - so the graph completes a plan that is missing a required tool.
    """
    from travel_agent.graph.nodes.core import _complete_plan
    from travel_agent.schemas.intent import DateRange
    from travel_agent.schemas.tools import IMAGES_TOOL, WEATHER_TOOL

    model_reply = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": WEATHER_TOOL, "args": {"city": "Tokyo"}, "type": "tool_call"}
        ],
    )

    completed, added = _complete_plan(
        model_reply, [WEATHER_TOOL, IMAGES_TOOL], "Tokyo", DateRange()
    )

    assert added == [IMAGES_TOOL]
    assert [call["name"] for call in completed.tool_calls] == [WEATHER_TOOL, IMAGES_TOOL]
    assert completed.tool_calls[1]["args"]["city"] == "Tokyo"
    assert completed.tool_calls[1]["id"] != completed.tool_calls[0]["id"], "ids must stay unique"


def test_plan_completion_leaves_a_complete_plan_alone() -> None:
    from travel_agent.graph.nodes.core import _complete_plan
    from travel_agent.schemas.intent import DateRange
    from travel_agent.schemas.tools import IMAGES_TOOL, WEATHER_TOOL

    model_reply = AIMessage(
        content="",
        tool_calls=[
            {"id": "a", "name": WEATHER_TOOL, "args": {"city": "Tokyo"}, "type": "tool_call"},
            {"id": "b", "name": IMAGES_TOOL, "args": {"city": "Tokyo"}, "type": "tool_call"},
        ],
    )

    completed, added = _complete_plan(
        model_reply, [WEATHER_TOOL, IMAGES_TOOL], "Tokyo", DateRange()
    )

    assert added == []
    assert completed is model_reply, "an untouched plan should not be rebuilt"


def test_plan_completion_does_nothing_without_a_city() -> None:
    """With no city there is nothing sensible to synthesise arguments from."""
    from travel_agent.graph.nodes.core import _complete_plan
    from travel_agent.schemas.intent import DateRange
    from travel_agent.schemas.tools import WEATHER_TOOL

    completed, added = _complete_plan(AIMessage(content=""), [WEATHER_TOOL], "", DateRange())

    assert added == []
    assert completed.tool_calls == []


def test_plan_completion_passes_the_date_window_through() -> None:
    from datetime import date as date_type

    from travel_agent.graph.nodes.core import _complete_plan
    from travel_agent.schemas.intent import DateRange
    from travel_agent.schemas.tools import WEATHER_TOOL

    window = DateRange(start=date_type(2026, 9, 1), days=5, label="next week")

    completed, added = _complete_plan(AIMessage(content=""), [WEATHER_TOOL], "Tokyo", window)

    assert added == [WEATHER_TOOL]
    assert completed.tool_calls[0]["args"]["start_date"] == "2026-09-01"
    assert completed.tool_calls[0]["args"]["days"] == 5


async def test_the_graph_records_when_it_completed_a_plan() -> None:
    """The trace must disclose that the graph added a call the model did not ask for."""
    result = await _app().ainvoke(
        {"user_query": "Tell me about Tokyo"}, config=_config("plan-completion")
    )

    plan_event = next(event for event in result["trace"] if event.node == "plan_tools")

    assert "tools_added_by_graph" in plan_event.data
    # The mock plans completely, so nothing should need adding here.
    assert plan_event.data["tools_added_by_graph"] == []
