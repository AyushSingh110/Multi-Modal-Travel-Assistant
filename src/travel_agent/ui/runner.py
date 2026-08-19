"""The bridge between Streamlit's synchronous reruns and the async graph.

THE PROBLEM
    Streamlit re-executes the entire script on every interaction. The graph is
    async. The obvious bridge - ``asyncio.run(app.ainvoke(...))`` inside the
    script - creates a brand new event loop on every rerun and destroys it
    immediately afterwards. That breaks two things badly:

    * Anything bound to a loop dies with it. The SQLite checkpointer holds an
      ``aiosqlite`` connection created on a particular loop; when that loop is
      gone the connection is unusable, and the next rerun either raises or hangs.
    * Every rerun pays the cost of building the loop, the providers and the graph
      again, so a UI toggle would re-read the vector store from disk.

THE SOLUTION
    One event loop, running forever on a dedicated background thread, created
    once per session. Work is submitted to it with
    ``asyncio.run_coroutine_threadsafe`` and waited on with a timeout. The loop
    outlives every rerun.

OWNERSHIP - THE TRAP THIS AVOIDS
    The loop, the compiled graph and the checkpointer's database connection all
    have to share one lifetime, and the connection has to be created **on the
    loop that will later use it**. Caching the loop while creating the connection
    per rerun is the specific pairing that deadlocks: the new connection belongs
    to a loop nobody is running, so the first ``await`` on it never returns and
    the UI hangs with no error.

    :class:`AgentRuntime` therefore owns all three, and
    :func:`build_runtime` constructs them in order - loop first, then the
    checkpointer *on that loop*, then the graph around it. Streamlit caches the
    whole object or none of it.

WHY SETTINGS ARE MUTATED RATHER THAN REBUILT
    The sidebar toggles - break the weather API, image fallback mode - change
    provider behaviour. Rebuilding the runtime for each of those would tear down
    the loop and the conversation history with it. Instead the runtime holds one
    ``Settings`` instance, the providers hold a reference to that same object, and
    a toggle mutates it in place. The next request sees the change; the loop, the
    graph and the memory are untouched.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeVar

from travel_agent.config.settings import Settings, get_settings
from travel_agent.graph.builder import GraphDependencies, build_dependencies, build_graph
from travel_agent.graph.checkpointer import CheckpointerHandle, create_checkpointer
from travel_agent.logging_setup import configure_logging, get_logger
from travel_agent.schemas.trace import ThresholdDiagnostics
from travel_agent.services.llm.base import ModelCheck

logger = get_logger(__name__)

#: How long a single graph turn may take before the UI gives up on it. Generous:
#: a live provider on a slow connection is not an error, it is just slow.
DEFAULT_TIMEOUT_SECONDS = 120.0

#: How long start-up work (opening the database, checking the model) may take.
STARTUP_TIMEOUT_SECONDS = 30.0

_T = TypeVar("_T")


@dataclass
class AgentRuntime:
    """Everything with a shared lifetime: the loop, the graph and the connection.

    Attributes:
        loop: The event loop, running on its own thread.
        thread: The thread running that loop.
        app: The compiled graph.
        checkpointer: The checkpointer handle, holding the database connection.
        dependencies: The graph's dependencies, kept for the trace panel.
        settings: The live settings object. Mutating it changes provider
            behaviour without rebuilding anything.
        model_check: Result of the provider's start-up model check.
        threshold_check: The router's threshold verdict, or ``None`` when no
            vector store is loaded.
    """

    loop: asyncio.AbstractEventLoop
    thread: threading.Thread
    app: Any
    checkpointer: CheckpointerHandle
    dependencies: GraphDependencies
    settings: Settings
    model_check: ModelCheck | None = None
    threshold_check: ThresholdDiagnostics | None = None
    _turn_counter: int = field(default=0, repr=False)

    # ------------------------------------------------------------- execution --
    def submit(self, coro: Coroutine[Any, Any, _T], timeout: float = DEFAULT_TIMEOUT_SECONDS) -> _T:
        """Run a coroutine on the runtime loop and wait for its result.

        Args:
            coro: The coroutine to run.
            timeout: Seconds to wait before giving up.

        Returns:
            Whatever the coroutine returned.

        Raises:
            TimeoutError: If the work did not finish in time. The loop keeps
                running, so the app stays usable.
            BaseException: Anything the coroutine raised, re-raised here on the
                calling thread. The loop survives it - a failing node must not
                leave the UI permanently hung.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            logger.warning("graph turn exceeded %.0fs and was cancelled", timeout)
            raise

    def invoke(
        self,
        query: str,
        thread_id: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Run one conversational turn.

        Args:
            query: What the user typed.
            thread_id: Conversation to run it against.
            timeout: Seconds to wait.

        Returns:
            The final graph state for the turn.
        """
        self._turn_counter += 1
        config = {"configurable": {"thread_id": thread_id}}
        logger.info("turn %d on thread %r: %r", self._turn_counter, thread_id, query)
        return dict(self.submit(self.app.ainvoke({"user_query": query}, config=config), timeout))

    def get_state(self, thread_id: str) -> dict[str, Any]:
        """Read a conversation's current state without running a turn.

        Args:
            thread_id: Conversation to read.

        Returns:
            The stored state, or an empty dictionary for an unseen thread.
        """
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.submit(self.app.aget_state(config), timeout=STARTUP_TIMEOUT_SECONDS)
        return dict(snapshot.values) if snapshot and snapshot.values else {}

    # ----------------------------------------------------------- diagnostics --
    @property
    def is_alive(self) -> bool:
        """Whether the loop thread is still running."""
        return self.thread.is_alive() and not self.loop.is_closed()

    @property
    def turns_run(self) -> int:
        """How many turns this runtime has executed."""
        return self._turn_counter

    def shutdown(self) -> None:
        """Close the database connection and stop the loop.

        Safe to call more than once. Streamlit gives no reliable teardown hook, so
        this exists mainly for tests and for an explicit runtime rebuild.
        """
        # A stopped loop is not a closed one, so is_closed() alone lets a second
        # shutdown submit a coroutine that will never be awaited. The thread is
        # the reliable signal that the loop is genuinely finished.
        if self.loop.is_closed() or not self.thread.is_alive():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.checkpointer.close(), self.loop).result(
                timeout=10
            )
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            logger.debug("checkpointer close failed during shutdown: %s", exc)

        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        logger.info("runtime shut down after %d turn(s)", self._turn_counter)


def _start_event_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Start an event loop on a dedicated daemon thread.

    Returns:
        The loop and the thread running it, once the loop is confirmed running.
    """
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=_run, name="travel-agent-loop", daemon=True)
    thread.start()

    # Wait until the loop is actually running. Submitting work to a loop that has
    # not started yet is another way to hang with no error message.
    if not ready.wait(timeout=10):
        raise RuntimeError("the agent event loop failed to start")

    return loop, thread


def build_runtime(settings: Settings | None = None) -> AgentRuntime:
    """Build the runtime: loop, checkpointer, graph, in that order.

    The order is the point. The checkpointer's database connection is created *on
    the runtime loop*, so every later ``await`` on it runs on the loop that owns
    it.

    Args:
        settings: Settings to build from. Defaults to the process singleton.

    Returns:
        A ready runtime.
    """
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    loop, thread = _start_event_loop()

    checkpointer = asyncio.run_coroutine_threadsafe(create_checkpointer(resolved), loop).result(
        timeout=STARTUP_TIMEOUT_SECONDS
    )

    dependencies = build_dependencies(resolved)
    app = build_graph(dependencies, checkpointer=checkpointer.saver)

    runtime = AgentRuntime(
        loop=loop,
        thread=thread,
        app=app,
        checkpointer=checkpointer,
        dependencies=dependencies,
        settings=resolved,
    )

    # Start-up checks, surfaced in the UI rather than buried in the logs: does the
    # configured model still exist, and is the routing threshold usable?
    try:
        runtime.model_check = runtime.submit(
            dependencies.llm.check_model(), timeout=STARTUP_TIMEOUT_SECONDS
        )
    except Exception as exc:  # noqa: BLE001 - an advisory check must never block start-up
        logger.warning("model check failed: %s", exc)

    if dependencies.router is not None:
        runtime.threshold_check = dependencies.router.diagnostics

    logger.info(
        "runtime ready: llm=%s checkpointer=%s vector_store=%s",
        dependencies.llm.name,
        checkpointer.kind,
        "loaded" if dependencies.retriever else "missing",
    )
    return runtime


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "AgentRuntime",
    "build_runtime",
]
