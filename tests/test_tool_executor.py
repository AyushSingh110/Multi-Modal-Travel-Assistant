"""Tests for the manual tool executor - Distinction 1."""

from __future__ import annotations

import ast
import asyncio
import io
import json
import tokenize
from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from travel_agent.config.settings import Settings
from travel_agent.exceptions import RetryableError
from travel_agent.graph.nodes.tool_executor import (
    ManualToolExecutor,
    extract_tool_calls,
    make_tool_executor,
)
from travel_agent.schemas.knowledge import KnowledgeChunk
from travel_agent.schemas.response import ImageAsset, WeatherPayload
from travel_agent.schemas.tools import IMAGES_TOOL, WEATHER_TOOL, WEB_SEARCH_TOOL
from travel_agent.tools.images.mock import MockImageProvider
from travel_agent.tools.registry import ToolRegistry
from travel_agent.tools.search.mock import MockSearchProvider
from travel_agent.tools.weather.base import WeatherProvider
from travel_agent.tools.weather.mock import MockWeatherProvider

REPO_ROOT = Path(__file__).resolve().parents[1]

# Symbols this project promises not to use. The promise is enforced, not trusted.
FORBIDDEN_SYMBOLS = {"ToolNode", "create_tool_calling_agent", "create_react_agent"}
FORBIDDEN_MODULES = {"langgraph.prebuilt"}


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "mock_weather_latency_ms": 10,
        "mock_image_latency_ms": 10,
        "mock_search_latency_ms": 10,
        "mock_latency_jitter": 0.0,
        "tool_timeout_seconds": 0.5,
        "tool_max_attempts": 1,
        "image_fallback_mode": "remote",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _registry(settings: Settings, weather: WeatherProvider | None = None) -> ToolRegistry:
    return ToolRegistry(
        weather or MockWeatherProvider(settings),
        MockImageProvider(settings),
        MockSearchProvider(settings),
        settings=settings,
    )


def _state(tool_calls: list[dict[str, object]]) -> dict[str, object]:
    """A conversation whose last message requests the given tools."""
    return {
        "messages": [
            HumanMessage(content="Tell me about Tokyo"),
            AIMessage(content="", tool_calls=tool_calls),
        ],
        "user_query": "Tell me about Tokyo",
    }


def _call(name: str, call_id: str, **args: object) -> dict[str, object]:
    """One entry shaped exactly like a model's tool_calls payload."""
    return {"id": call_id, "name": name, "args": args, "type": "tool_call"}


# ================================================== the enforced guarantee ====
def _python_files() -> list[Path]:
    return [path for folder in ("src", "scripts") for path in (REPO_ROOT / folder).rglob("*.py")]


def test_no_prebuilt_tool_calling_helpers_anywhere_in_the_source() -> None:
    """Distinction 1 requires hand-written tool execution - enforce it.

    Comments and strings are ignored deliberately: the executor's own docstring
    explains what ToolNode would have done, and documenting the decision must not
    trip the check. Only real code - imports and identifiers - counts.
    """
    offences: list[str] = []

    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(REPO_ROOT)

        # 1. Imports, via the AST: catches `from langgraph.prebuilt import X`
        #    and `import langgraph.prebuilt as p` alike.
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in FORBIDDEN_MODULES or module.startswith("langgraph.prebuilt"):
                    offences.append(f"{relative}:{node.lineno} imports from {module}")
                for alias in node.names:
                    if alias.name in FORBIDDEN_SYMBOLS:
                        offences.append(f"{relative}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in FORBIDDEN_MODULES:
                        offences.append(f"{relative}:{node.lineno} imports {alias.name}")

        # 2. Bare identifiers, via the tokeniser: catches usage even if the
        #    import were hidden behind an alias or a local import.
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.NAME and token.string in FORBIDDEN_SYMBOLS:
                offences.append(f"{relative}:{token.start[0]} uses {token.string}")

    assert not offences, "prebuilt tool-calling helpers found:\n  " + "\n  ".join(offences)


def test_the_guarantee_test_would_actually_catch_a_violation(tmp_path: Path) -> None:
    """A guard that cannot fail is not a guard. Prove this one can."""
    offender = tmp_path / "sneaky.py"
    offender.write_text(
        "from langgraph.prebuilt import ToolNode\n\nnode = ToolNode([])\n", encoding="utf-8"
    )

    source = offender.read_text(encoding="utf-8")
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "langgraph.prebuilt"
        ):
            found.append(node.module or "")
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NAME and token.string in FORBIDDEN_SYMBOLS:
            found.append(token.string)

    assert found, "the detection logic failed to flag a deliberate violation"


def test_documentation_mentions_of_toolnode_do_not_trip_the_guard() -> None:
    """The executor explains why it avoids ToolNode; that must stay legal."""
    module = REPO_ROOT / "src" / "travel_agent" / "graph" / "nodes" / "tool_executor.py"
    text = module.read_text(encoding="utf-8")

    assert "ToolNode" in text, "the module should document the decision it made"
    # And yet the guarantee test above passes, because it ignores strings.


# ============================================================== extraction ====
def test_extract_tool_calls_reads_the_last_ai_message() -> None:
    calls = extract_tool_calls(_state([_call(WEATHER_TOOL, "c1", city="Tokyo")])["messages"])  # type: ignore[arg-type]

    assert len(calls) == 1
    assert calls[0]["name"] == WEATHER_TOOL


def test_extract_tool_calls_ignores_a_non_ai_last_message() -> None:
    assert extract_tool_calls([HumanMessage(content="hello")]) == []


def test_extract_tool_calls_handles_an_empty_conversation() -> None:
    assert extract_tool_calls([]) == []


def test_extract_tool_calls_handles_an_ai_message_with_no_tools() -> None:
    assert extract_tool_calls([AIMessage(content="just prose")]) == []


# ============================================================= happy paths ====
async def test_single_tool_call_returns_one_paired_tool_message() -> None:
    executor = make_tool_executor(_registry(_settings()), node_name="tools")

    update = await executor(_state([_call(WEATHER_TOOL, "call_1", city="Tokyo", days=7)]))  # type: ignore[arg-type]

    messages = update["messages"]
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call_1"
    assert message.name == WEATHER_TOOL
    assert message.status == "success"
    assert "forecast" in json.loads(message.content)


async def test_multiple_tool_calls_in_one_message_all_execute() -> None:
    """The normal case: the model asks for weather and images together."""
    executor = make_tool_executor(_registry(_settings()), node_name="tools")

    update = await executor(
        _state(
            [
                _call(WEATHER_TOOL, "call_w", city="Paris", days=7),
                _call(IMAGES_TOOL, "call_i", city="Paris", count=3),
            ]
        )  # type: ignore[arg-type]
    )

    messages = update["messages"]
    assert len(messages) == 2
    assert {m.tool_call_id for m in messages} == {"call_w", "call_i"}
    assert all(m.status == "success" for m in messages)


async def test_payloads_are_filed_into_typed_state_keys() -> None:
    executor = make_tool_executor(_registry(_settings()), node_name="tools")

    update = await executor(
        _state(
            [
                _call(WEATHER_TOOL, "w", city="Paris"),
                _call(IMAGES_TOOL, "i", city="Paris", count=2),
                _call(WEB_SEARCH_TOOL, "s", query="Kyoto travel guide"),
            ]
        )  # type: ignore[arg-type]
    )

    assert isinstance(update["weather"], WeatherPayload)
    assert all(isinstance(asset, ImageAsset) for asset in update["images"])
    assert all(isinstance(chunk, KnowledgeChunk) for chunk in update["knowledge"])
    assert update["knowledge"][0].source.startswith("http")


async def test_trace_and_timings_are_emitted_for_the_panel() -> None:
    executor = make_tool_executor(_registry(_settings()), node_name="execute_weather")

    update = await executor(_state([_call(WEATHER_TOOL, "w", city="Tokyo")]))  # type: ignore[arg-type]

    assert "execute_weather" in update["timings"]
    events = update["trace"]
    assert len(events) == 1
    assert events[0].kind == "tool"
    assert events[0].data["tool"] == WEATHER_TOOL
    assert events[0].data["tool_call_id"] == "w"
    assert events[0].duration_ms is not None


# ============================================================= id pairing ====
class _SlowWeather(WeatherProvider):
    """Weather that finishes deliberately later than the image tool."""

    name = "slow"

    def __init__(self, settings: Settings, delay: float) -> None:
        self._inner = MockWeatherProvider(settings)
        self._delay = delay

    async def fetch_forecast(
        self, city: str, days: int = 7, start_date: date | None = None
    ) -> WeatherPayload:
        await asyncio.sleep(self._delay)
        return await self._inner.fetch_forecast(city, days=days, start_date=start_date)


async def test_ids_stay_paired_when_results_arrive_out_of_order() -> None:
    """The core protocol guarantee.

    Weather is made slower than images, so the calls complete in the opposite
    order to the request. Each reply must still carry the id of the request it
    answers - order carries no meaning at all.
    """
    settings = _settings(tool_timeout_seconds=5.0)
    executor = make_tool_executor(
        _registry(settings, weather=_SlowWeather(settings, delay=0.30)), node_name="tools"
    )

    update = await executor(
        _state(
            [
                _call(WEATHER_TOOL, "id_weather", city="Tokyo"),
                _call(IMAGES_TOOL, "id_images", city="Tokyo", count=2),
            ]
        )  # type: ignore[arg-type]
    )

    by_id = {message.tool_call_id: message for message in update["messages"]}
    assert by_id["id_weather"].name == WEATHER_TOOL
    assert by_id["id_images"].name == IMAGES_TOOL
    assert "forecast" in json.loads(by_id["id_weather"].content)
    assert "images" in json.loads(by_id["id_images"].content)


async def test_every_call_gets_exactly_one_reply() -> None:
    """A missing reply is an API error on the next turn with most providers."""
    executor = make_tool_executor(_registry(_settings()), node_name="tools")
    calls = [
        _call(WEATHER_TOOL, "a", city="Paris"),
        _call(IMAGES_TOOL, "b", city="Paris"),
        _call("not_a_tool", "c"),
    ]

    update = await executor(_state(calls))  # type: ignore[arg-type]

    assert len(update["messages"]) == len(calls)
    assert {m.tool_call_id for m in update["messages"]} == {"a", "b", "c"}


# ================================================================= errors ====
async def test_unknown_tool_returns_an_error_message_listing_real_tools() -> None:
    """The model must be able to correct itself, so tell it what exists."""
    executor = make_tool_executor(_registry(_settings()), node_name="tools")

    update = await executor(_state([_call("get_stock_price", "bad_1", ticker="AAPL")]))  # type: ignore[arg-type]

    message = update["messages"][0]
    assert message.status == "error"
    assert message.tool_call_id == "bad_1"
    assert WEATHER_TOOL in message.content
    assert IMAGES_TOOL in message.content
    assert update["errors"][0].recoverable is True


async def test_missing_required_argument_reports_the_validation_detail() -> None:
    """A generic 'invalid arguments' string would tell the model nothing."""
    executor = make_tool_executor(_registry(_settings()), node_name="tools")

    update = await executor(_state([_call(WEATHER_TOOL, "bad_2", days=7)]))  # type: ignore[arg-type]

    message = update["messages"][0]
    assert message.status == "error"
    assert "city" in message.content, "the model needs to know WHICH field was wrong"
    assert update["errors"][0].tool == WEATHER_TOOL


async def test_out_of_range_argument_is_rejected_with_detail() -> None:
    executor = make_tool_executor(_registry(_settings()), node_name="tools")

    update = await executor(_state([_call(WEATHER_TOOL, "bad_3", city="Paris", days=99)]))  # type: ignore[arg-type]

    assert update["messages"][0].status == "error"
    assert "days" in update["messages"][0].content


class _BrokenWeather(WeatherProvider):
    """Weather that always raises."""

    name = "broken"

    async def fetch_forecast(
        self, city: str, days: int = 7, start_date: date | None = None
    ) -> WeatherPayload:
        raise RetryableError("upstream weather service is down")


async def test_a_raising_tool_becomes_an_error_message_not_an_exception() -> None:
    settings = _settings()
    executor = make_tool_executor(_registry(settings, weather=_BrokenWeather()), node_name="tools")

    update = await executor(_state([_call(WEATHER_TOOL, "boom", city="Paris")]))  # type: ignore[arg-type]

    message = update["messages"][0]
    assert message.status == "error"
    assert message.tool_call_id == "boom"
    assert "down" in message.content
    assert "weather" not in update, "a failed tool must not write a payload into state"


class _HangingWeather(WeatherProvider):
    """Weather that never returns."""

    name = "hanging"

    async def fetch_forecast(
        self, city: str, days: int = 7, start_date: date | None = None
    ) -> WeatherPayload:
        await asyncio.sleep(30)
        raise AssertionError("should have timed out")


async def test_a_timing_out_tool_becomes_an_error_message() -> None:
    settings = _settings(tool_timeout_seconds=0.05)
    executor = make_tool_executor(_registry(settings, weather=_HangingWeather()), node_name="tools")

    update = await executor(_state([_call(WEATHER_TOOL, "slow", city="Paris")]))  # type: ignore[arg-type]

    assert update["messages"][0].status == "error"
    assert update["errors"][0].tool == WEATHER_TOOL


async def test_one_failing_tool_does_not_abort_its_siblings() -> None:
    """Per-call isolation: the rubric's graceful-degradation line, in the node."""
    settings = _settings()
    executor = make_tool_executor(_registry(settings, weather=_BrokenWeather()), node_name="tools")

    update = await executor(
        _state(
            [
                _call(WEATHER_TOOL, "w", city="Paris"),
                _call(IMAGES_TOOL, "i", city="Paris", count=3),
                _call(WEB_SEARCH_TOOL, "s", query="Paris travel guide"),
            ]
        )  # type: ignore[arg-type]
    )

    by_id = {message.tool_call_id: message for message in update["messages"]}
    assert by_id["w"].status == "error"
    assert by_id["i"].status == "success"
    assert by_id["s"].status == "success"
    assert len(update["images"]) == 3
    assert update["knowledge"]
    assert "weather" not in update


async def test_failures_are_recorded_for_the_ui_banner() -> None:
    settings = _settings()
    executor = make_tool_executor(_registry(settings, weather=_BrokenWeather()), node_name="tools")

    update = await executor(_state([_call(WEATHER_TOOL, "w", city="Paris")]))  # type: ignore[arg-type]

    error = update["errors"][0]
    assert error.tool == WEATHER_TOOL
    assert error.tool_call_id == "w"
    assert error.recoverable is False, "a dead provider is not something the model can fix"
    assert update["trace"][0].kind == "error"


# ================================================== selective execution ====
async def test_an_executor_only_runs_the_tools_it_handles() -> None:
    """This is what lets two parallel branches share one executor class."""
    settings = _settings()
    registry = _registry(settings)
    weather_branch = make_tool_executor(
        registry, node_name="execute_weather", handles={WEATHER_TOOL}
    )
    image_branch = make_tool_executor(registry, node_name="execute_images", handles={IMAGES_TOOL})

    state = _state(
        [_call(WEATHER_TOOL, "w", city="Tokyo"), _call(IMAGES_TOOL, "i", city="Tokyo", count=2)]
    )

    weather_update = await weather_branch(state)  # type: ignore[arg-type]
    image_update = await image_branch(state)  # type: ignore[arg-type]

    assert [m.tool_call_id for m in weather_update["messages"]] == ["w"]
    assert [m.tool_call_id for m in image_update["messages"]] == ["i"]
    assert "weather" in weather_update and "images" not in weather_update
    assert "images" in image_update and "weather" not in image_update


async def test_branches_can_run_concurrently_without_interfering() -> None:
    settings = _settings()
    registry = _registry(settings)
    state = _state(
        [_call(WEATHER_TOOL, "w", city="Tokyo"), _call(IMAGES_TOOL, "i", city="Tokyo", count=2)]
    )

    weather_update, image_update = await asyncio.gather(
        make_tool_executor(registry, node_name="w", handles={WEATHER_TOOL})(state),  # type: ignore[arg-type]
        make_tool_executor(registry, node_name="i", handles={IMAGES_TOOL})(state),  # type: ignore[arg-type]
    )

    assert weather_update["messages"][0].tool_call_id == "w"
    assert image_update["messages"][0].tool_call_id == "i"


async def test_a_branch_with_nothing_to_do_reports_a_skip() -> None:
    """On a follow-up turn the image branch legitimately has no work."""
    executor = make_tool_executor(
        _registry(_settings()), node_name="execute_images", handles={IMAGES_TOOL}
    )

    update = await executor(_state([_call(WEATHER_TOOL, "w", city="Tokyo")]))  # type: ignore[arg-type]

    assert update["skipped_nodes"] == ["execute_images"]
    assert update["trace"][0].kind == "skip"
    assert "messages" not in update


async def test_no_tool_calls_at_all_is_not_an_error() -> None:
    executor = make_tool_executor(_registry(_settings()), node_name="tools")

    update = await executor({"messages": [AIMessage(content="no tools needed")]})  # type: ignore[arg-type]

    assert update["skipped_nodes"] == ["tools"]


# ================================================================ plumbing ====
async def test_tool_content_is_bounded() -> None:
    """A huge payload must not be allowed to swamp the context window."""
    executor = make_tool_executor(_settings() and _registry(_settings()), node_name="tools")

    update = await executor(_state([_call(WEATHER_TOOL, "w", city="Paris", days=14)]))  # type: ignore[arg-type]

    from travel_agent.graph.nodes.tool_executor import MAX_TOOL_CONTENT_CHARS

    assert len(update["messages"][0].content) <= MAX_TOOL_CONTENT_CHARS


def test_string_arguments_cannot_reach_the_executor_via_an_ai_message() -> None:
    """Where the JSON-string argument case actually has to be handled.

    Providers really do return ``args`` as a JSON string sometimes, so the tool
    specs accept both. But langchain-core validates ``tool_calls[].args`` as a
    dictionary when the AIMessage is constructed, which means the string form is
    normalised one layer *below* this node and can never arrive here. The
    conversion therefore belongs in the provider adapter, not in the executor -
    pinned down here so nobody later adds dead defensive code for it.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="valid dictionary"):
        AIMessage(
            content="",
            tool_calls=[
                {"id": "s", "name": WEATHER_TOOL, "args": '{"city": "Paris"}', "type": "tool_call"}
            ],
        )


async def test_specs_still_accept_string_arguments_one_layer_down() -> None:
    """The registry handles the string form, because raw payloads can carry it."""
    from travel_agent.schemas.tools import TOOL_SPECS

    validated = TOOL_SPECS[WEATHER_TOOL].validate_args('{"city": "Paris", "days": 5}')

    assert validated.city == "Paris"
    assert validated.days == 5


def test_executor_is_constructed_with_an_explicit_registry() -> None:
    """No global state: each node is handed its dependencies."""
    executor = ManualToolExecutor(_registry(_settings()), node_name="x", handles={WEATHER_TOOL})

    assert executor.node_name == "x"
    assert executor.handles == {WEATHER_TOOL}
