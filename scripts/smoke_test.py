"""Environment smoke test.

Imports every top-level dependency, prints its installed version, and asserts the
handful of LangGraph runtime behaviours this project's architecture depends on.

Run it immediately after creating the conda environment::

    python scripts/smoke_test.py

Exits non-zero if anything is missing or a required behaviour is absent, so it is
also usable as a CI gate.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata as metadata
import operator
import platform
import sys
import time
from typing import Annotated, TypedDict

# (import name, distribution name, why this project needs it)
DEPENDENCIES: list[tuple[str, str, str]] = [
    ("langgraph", "langgraph", "graph runtime (nodes, edges, supersteps)"),
    ("langgraph.checkpoint.memory", "langgraph-checkpoint", "in-memory thread checkpointer"),
    ("langgraph.checkpoint.sqlite", "langgraph-checkpoint-sqlite", "durable thread checkpointer"),
    ("langchain_core", "langchain-core", "AIMessage / ToolMessage types"),
    ("langchain_groq", "langchain-groq", "default demo LLM driver"),
    ("langchain_anthropic", "langchain-anthropic", "spec-named LLM provider"),
    ("langchain_openai", "langchain-openai", "spec-named LLM provider"),
    ("pydantic", "pydantic", "structured output contract"),
    ("pydantic_settings", "pydantic-settings", "centralised configuration"),
    ("faiss", "faiss-cpu", "vector index"),
    ("numpy", "numpy", "vectors + brute-force fallback store"),
    ("httpx", "httpx", "async HTTP with timeouts"),
    ("streamlit", "streamlit", "GUI"),
    ("plotly", "plotly", "forecast line chart"),
    ("pandas", "pandas", "chart dataframe"),
    ("ddgs", "ddgs", "optional live web search"),
    ("pytest", "pytest", "test runner"),
]

OK = "[ OK ]"
FAIL = "[FAIL]"


def check_imports() -> list[str]:
    """Import every dependency and print its version.

    Returns:
        A list of human-readable failure messages; empty when all imports succeed.
    """
    failures: list[str] = []
    print("=" * 78)
    print("DEPENDENCY IMPORTS")
    print("=" * 78)
    for module_name, dist_name, purpose in DEPENDENCIES:
        try:
            importlib.import_module(module_name)
            try:
                version = metadata.version(dist_name)
            except metadata.PackageNotFoundError:
                version = "unknown"
            print(f"{OK} {module_name:34s} {version:12s} {purpose}")
        except Exception as exc:  # noqa: BLE001 - smoke test reports everything
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
            print(f"{FAIL} {module_name:34s} {'-':12s} {exc}")
    return failures


async def check_graph_behaviours() -> list[str]:
    """Assert the LangGraph behaviours the architecture is built on.

    Three things must hold, and all three are load-bearing:

    1. A conditional edge that returns a *list* of node names fans out and runs
       those nodes concurrently in one superstep (Distinction 2).
    2. Concurrent writes to an un-reduced state key raise ``InvalidUpdateError``
       - which is why the typed state annotates every fan-out key with a reducer.
    3. A checkpointer preserves state between invocations on the same
       ``thread_id`` (Distinction 3).

    Returns:
        A list of human-readable failure messages; empty when all checks pass.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    failures: list[str] = []
    print()
    print("=" * 78)
    print("LANGGRAPH RUNTIME BEHAVIOURS")
    print("=" * 78)

    class FanOutState(TypedDict, total=False):
        trace: Annotated[list[str], operator.add]

    # NOTE: the node itself must be a coroutine *function*. A sync callable that
    # returns a coroutine is treated as a sync node and LangGraph rejects the
    # un-awaited coroutine as an invalid state update.
    async def _weather(state: FanOutState) -> dict[str, list[str]]:
        await asyncio.sleep(0.5)
        return {"trace": ["weather"]}

    async def _images(state: FanOutState) -> dict[str, list[str]]:
        await asyncio.sleep(0.5)
        return {"trace": ["images"]}

    builder: StateGraph = StateGraph(FanOutState)
    builder.add_node("plan", lambda state: {"trace": ["plan"]})
    builder.add_node("weather", _weather)
    builder.add_node("images", _images)
    builder.add_node("join", lambda state: {"trace": ["join"]})
    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan", lambda state: ["weather", "images"], ["weather", "images"]
    )
    builder.add_edge("weather", "join")
    builder.add_edge("images", "join")
    builder.add_edge("join", END)
    app = builder.compile(checkpointer=MemorySaver())

    # 1. parallel fan-out
    started = time.perf_counter()
    config = {"configurable": {"thread_id": "smoke"}}
    await app.ainvoke({"trace": []}, config=config)
    elapsed = time.perf_counter() - started
    if elapsed < 0.85:  # two 0.5s nodes in parallel ~= 0.5s; sequential would be ~1.0s
        print(f"{OK} parallel fan-out                  2x0.5s nodes -> {elapsed:.2f}s wall clock")
    else:
        failures.append(f"fan-out ran sequentially ({elapsed:.2f}s)")
        print(f"{FAIL} parallel fan-out                  {elapsed:.2f}s (expected < 0.85s)")

    # 2. reducers are mandatory for concurrent writes
    class UnreducedState(TypedDict, total=False):
        value: str

    plain: StateGraph = StateGraph(UnreducedState)
    plain.add_node("start", lambda state: {})
    plain.add_node("a", lambda state: {"value": "a"})
    plain.add_node("b", lambda state: {"value": "b"})
    plain.add_edge(START, "start")
    plain.add_conditional_edges("start", lambda state: ["a", "b"], ["a", "b"])
    plain.add_edge("a", END)
    plain.add_edge("b", END)
    try:
        await plain.compile().ainvoke({})
        failures.append("concurrent write to un-reduced key did not raise")
        print(f"{FAIL} reducer enforcement               no error raised")
    except Exception as exc:  # noqa: BLE001 - we assert on the type name only
        print(f"{OK} reducer enforcement               concurrent write -> {type(exc).__name__}")

    # 3. checkpointer persistence across turns
    second = await app.ainvoke({"trace": ["turn-2"]}, config=config)
    if len(second["trace"]) > 5:
        print(
            f"{OK} checkpointer persistence          turn 2 sees {len(second['trace'])} prior events"
        )
    else:
        failures.append("checkpointer did not preserve state across invocations")
        print(f"{FAIL} checkpointer persistence          state lost between turns")

    return failures


def main() -> int:
    """Run the smoke test and return a process exit code."""
    print(f"Python  : {sys.version.split()[0]}  ({platform.system()} {platform.release()})")
    print(f"Executable: {sys.executable}")
    print()

    failures = check_imports()
    failures += asyncio.run(check_graph_behaviours())

    print()
    print("=" * 78)
    if failures:
        print(f"SMOKE TEST FAILED - {len(failures)} problem(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("SMOKE TEST PASSED - environment is ready.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
