"""One live end-to-end request against the configured provider.

    python scripts/live_smoke_test.py

Deliberately minimal: a single query, one city, mock tools. What it proves is the
part that only a real provider can prove - that the live model returns a genuine
``tool_calls`` payload the manual executor can dispatch, and that its JSON reply
survives Pydantic validation.

Everything else in this project runs on mocks. This exists so the live path is
known to work before a demo, rather than during one.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from travel_agent.config.settings import get_settings  # noqa: E402
from travel_agent.graph.builder import build_dependencies, build_graph  # noqa: E402
from travel_agent.logging_setup import configure_logging  # noqa: E402


def safe_print(text: str) -> None:
    """Print text that may contain characters the console cannot encode.

    A live model emitted U+202F (a narrow no-break space) in its summary, which
    raised UnicodeEncodeError on a Windows cp1252 console and killed the script
    after the work had already succeeded.

    Args:
        text: Text to print.
    """
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run one live request against the real provider.")
    parser.add_argument("--query", default="Tell me about Tokyo", help="The single query to run.")
    parser.add_argument("--verbose", action="store_true", help="Show the full log output.")
    return parser.parse_args()


async def main() -> int:
    """Run the smoke test.

    Returns:
        A process exit code.
    """
    args = parse_args()
    configure_logging("INFO" if args.verbose else "ERROR")

    settings = get_settings()
    provider = settings.resolve_llm_provider()

    print("=" * 78)
    print("LIVE PROVIDER SMOKE TEST")
    print("=" * 78)
    print(f"provider : {provider}")
    print(f"model    : {settings.model_id_for(provider)}")
    print(f"tools    : weather={settings.weather_provider} images={settings.image_provider}")
    print(f"query    : {args.query}")
    print()

    if provider == "mock":
        print("No API key is configured, so this would only exercise the mock driver.")
        print("Set GROQ_API_KEY (or ANTHROPIC_API_KEY / OPENAI_API_KEY) in .env first.")
        return 1

    dependencies = build_dependencies(settings)

    check = await dependencies.llm.check_model()
    print(f"model check : {check.message}")

    app = build_graph(dependencies)
    started = time.perf_counter()
    state = await app.ainvoke(
        {"user_query": args.query}, config={"configurable": {"thread_id": "live-smoke"}}
    )
    elapsed = (time.perf_counter() - started) * 1000

    response = state["response"]
    usage = state.get("token_usage")

    print()
    print("-" * 78)
    print("RESULT")
    print("-" * 78)
    print(f"wall clock     : {elapsed:.0f} ms")
    print(f"city resolved  : {state.get('city')}")
    print(
        f"route          : {state.get('route')} "
        f"({state.get('route_match_reason')}, score {state.get('route_score'):.3f})"
    )

    tool_calls = [
        event.data.get("tool")
        for event in state.get("trace", [])
        if event.kind in {"tool", "error"}
    ]
    print(f"tools executed : {', '.join(filter(None, tool_calls))}")
    print(f"forecast points: {len(response.weather_forecast)}")
    print(f"images         : {len(response.image_urls)}")
    print(f"validated      : {type(response).__name__} passed Pydantic validation")

    if usage is not None:
        print(
            f"tokens         : {usage.total_tokens} "
            f"({usage.prompt_tokens} prompt + {usage.completion_tokens} completion) "
            f"across {usage.llm_calls} call(s) on {usage.model}"
        )

    metrics = state.get("parallel_metrics")
    if metrics is not None:
        print(
            f"parallel       : {metrics.sequential_equivalent_ms:.0f} ms sequential-equivalent "
            f"vs {metrics.parallel_wall_clock_ms:.0f} ms actual ({metrics.speedup:.2f}x)"
        )

    print()
    print("-" * 78)
    print("SUMMARY WRITTEN BY THE LIVE MODEL")
    print("-" * 78)
    safe_print(response.city_summary)
    if response.highlights:
        print()
        safe_print("highlights: " + "; ".join(response.highlights))

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
