"""Tests for the Streamlit bridge - Risk R2."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from travel_agent.config.settings import Settings
from travel_agent.ui.runner import AgentRuntime, build_runtime

TURN_TIMEOUT = 30.0


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "mock_weather_latency_ms": 60,
        "mock_image_latency_ms": 60,
        "mock_search_latency_ms": 60,
        "mock_latency_jitter": 0.0,
        "tool_timeout_seconds": 5.0,
        "tool_max_attempts": 1,
        "image_fallback_mode": "local",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture
def runtime() -> Any:
    """A runtime with the in-memory checkpointer, torn down after the test."""
    built = build_runtime(_settings())
    yield built
    built.shutdown()


# ================================================================ lifetime ====
def test_runtime_starts_with_a_live_loop(runtime: AgentRuntime) -> None:
    assert runtime.is_alive
    assert runtime.thread.name == "travel-agent-loop"
    assert runtime.thread.daemon, "the loop must not keep the process alive"


def test_a_turn_runs_and_returns_state(runtime: AgentRuntime) -> None:
    state = runtime.invoke("Tell me about Tokyo", thread_id="t1", timeout=TURN_TIMEOUT)

    assert state["city"] == "Tokyo"
    assert state["response"].image_urls
    assert runtime.turns_run == 1


def test_the_same_runtime_serves_repeated_turns(runtime: AgentRuntime) -> None:
    """A Streamlit rerun reuses the cached runtime - the loop must persist."""
    for index in range(4):
        state = runtime.invoke("Tell me about Paris", thread_id="rerun", timeout=TURN_TIMEOUT)
        assert state["city"] == "Paris"
        assert runtime.is_alive, f"loop died after turn {index + 1}"

    assert runtime.turns_run == 4


def test_a_follow_up_on_the_same_thread_keeps_context(runtime: AgentRuntime) -> None:
    runtime.invoke("Tell me about Tokyo", thread_id="ctx", timeout=TURN_TIMEOUT)
    second = runtime.invoke("what about next week?", thread_id="ctx", timeout=TURN_TIMEOUT)

    assert second["city"] == "Tokyo"
    assert second["intent"] == "weather_only"


def test_switching_threads_switches_conversations(runtime: AgentRuntime) -> None:
    """Changing the thread id in the sidebar must change conversation cleanly."""
    runtime.invoke("Tell me about Tokyo", thread_id="alpha", timeout=TURN_TIMEOUT)
    runtime.invoke("Tell me about Paris", thread_id="beta", timeout=TURN_TIMEOUT)

    back_to_alpha = runtime.invoke("what about next week?", thread_id="alpha", timeout=TURN_TIMEOUT)
    beta_state = runtime.get_state("beta")

    assert back_to_alpha["city"] == "Tokyo"
    assert beta_state["city"] == "Paris"


def test_reading_state_without_running_a_turn(runtime: AgentRuntime) -> None:
    runtime.invoke("Tell me about Tokyo", thread_id="peek", timeout=TURN_TIMEOUT)

    state = runtime.get_state("peek")

    assert state["city"] == "Tokyo"
    assert runtime.turns_run == 1, "reading state must not count as a turn"


def test_reading_an_unseen_thread_returns_empty(runtime: AgentRuntime) -> None:
    assert runtime.get_state("never-used") == {}


# =============================================================== survival ====
def test_an_exception_inside_a_node_does_not_kill_the_loop(runtime: AgentRuntime) -> None:
    """The critical one: a broken node must not leave the app permanently hung.

    A node is replaced with one that raises. The exception should surface on the
    calling thread, and the runtime must still serve the next request - if the
    loop died here, every later interaction would hang forever with no error.
    """

    async def exploding_node(state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("deliberate node failure")

    with pytest.raises(RuntimeError, match="deliberate node failure"):
        runtime.submit(exploding_node({}), timeout=TURN_TIMEOUT)

    assert runtime.is_alive, "the loop died with the exception"

    state = runtime.invoke("Tell me about Paris", thread_id="after-error", timeout=TURN_TIMEOUT)
    assert state["city"] == "Paris", "the runtime did not recover"


def test_a_cancelled_slow_call_leaves_the_runtime_usable(runtime: AgentRuntime) -> None:
    """A timeout cancels the work but must not take the loop with it."""

    async def far_too_slow() -> None:
        await asyncio.sleep(30)

    with pytest.raises(TimeoutError):
        runtime.submit(far_too_slow(), timeout=0.2)

    assert runtime.is_alive
    assert runtime.invoke("Tell me about Tokyo", thread_id="post-timeout", timeout=TURN_TIMEOUT)


def test_a_failing_tool_still_returns_a_rendered_answer(runtime: AgentRuntime) -> None:
    """The weather toggle path, through the bridge rather than the graph directly."""
    runtime.settings.force_weather_failure = True
    try:
        state = runtime.invoke("Tell me about Tokyo", thread_id="degraded", timeout=TURN_TIMEOUT)
    finally:
        runtime.settings.force_weather_failure = False

    response = state["response"]
    assert response.weather_forecast == []
    assert response.image_urls, "the healthy branch must still render"
    assert response.warnings


# ================================================= live settings mutation ====
def test_toggling_settings_takes_effect_without_a_rebuild(runtime: AgentRuntime) -> None:
    """Sidebar toggles must not tear down the loop or the conversation.

    The providers hold a reference to the runtime's Settings object, so mutating
    it changes behaviour on the next request while the loop, the graph and the
    checkpointed history stay exactly as they were.
    """
    healthy = runtime.invoke("Tell me about Tokyo", thread_id="toggle", timeout=TURN_TIMEOUT)
    assert healthy["response"].weather_forecast

    runtime.settings.force_weather_failure = True
    broken = runtime.invoke("Tell me about Paris", thread_id="toggle-2", timeout=TURN_TIMEOUT)
    assert broken["response"].weather_forecast == []

    runtime.settings.force_weather_failure = False
    recovered = runtime.invoke("Tell me about Paris", thread_id="toggle-3", timeout=TURN_TIMEOUT)
    assert recovered["response"].weather_forecast

    assert runtime.is_alive
    assert runtime.get_state("toggle")["city"] == "Tokyo", "history survived the toggles"


def test_image_fallback_mode_can_be_switched_live(runtime: AgentRuntime) -> None:
    runtime.settings.image_fallback_mode = "local"
    local = runtime.invoke("Tell me about Tokyo", thread_id="img-1", timeout=TURN_TIMEOUT)

    assert all(asset.prefer_local for asset in local["images"])


# ================================================ start-up diagnostics ====
def test_startup_checks_are_available_for_the_trace_panel(runtime: AgentRuntime) -> None:
    assert runtime.model_check is not None
    assert runtime.model_check.available
    assert runtime.threshold_check is not None
    assert runtime.threshold_check.status == "ok"


def test_runtime_reports_its_checkpointer_kind(runtime: AgentRuntime) -> None:
    assert runtime.checkpointer.kind == "memory"
    assert not runtime.checkpointer.is_durable


# ==================================================== durable checkpointer ====
def test_sqlite_connection_is_created_on_the_runtime_loop(tmp_path: Path) -> None:
    """The deadlock this design exists to prevent.

    An aiosqlite connection created on one loop and awaited on another never
    completes. Building the checkpointer *on the runtime loop* is what makes this
    work, and a regression would show up here as a timeout rather than an error.
    """
    settings = _settings(checkpointer="sqlite", checkpoint_db_path=tmp_path / "ui.sqlite")
    runtime = build_runtime(settings)

    try:
        assert runtime.checkpointer.kind == "sqlite"

        first = runtime.invoke("Tell me about Tokyo", thread_id="durable", timeout=TURN_TIMEOUT)
        assert first["city"] == "Tokyo"

        # A second turn re-reads the connection: this is where a cross-loop
        # connection would hang instead of returning.
        second = runtime.invoke("what about next week?", thread_id="durable", timeout=TURN_TIMEOUT)
        assert second["city"] == "Tokyo"
        assert second["intent"] == "weather_only"
    finally:
        runtime.shutdown()

    assert (tmp_path / "ui.sqlite").stat().st_size > 0


def test_shutdown_is_idempotent(tmp_path: Path) -> None:
    runtime = build_runtime(_settings())
    runtime.invoke("Tell me about Tokyo", thread_id="bye", timeout=TURN_TIMEOUT)

    runtime.shutdown()
    runtime.shutdown()

    assert not runtime.is_alive


def test_two_runtimes_can_coexist() -> None:
    """Rebuilding on a provider change must not disturb an existing runtime."""
    first = build_runtime(_settings())
    second = build_runtime(_settings())

    try:
        assert first.loop is not second.loop
        assert first.invoke("Tell me about Tokyo", thread_id="a", timeout=TURN_TIMEOUT)["city"]
        assert second.invoke("Tell me about Paris", thread_id="b", timeout=TURN_TIMEOUT)["city"]
    finally:
        first.shutdown()
        second.shutdown()
