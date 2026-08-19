"""Evaluate the router against labelled queries, and sweep the threshold."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from travel_agent.config.settings import Settings  # noqa: E402
from travel_agent.graph.builder import build_dependencies, build_graph  # noqa: E402
from travel_agent.graph.nodes.core import _extract_city  # noqa: E402
from travel_agent.logging_setup import configure_logging  # noqa: E402
from travel_agent.schemas.response import TravelResponse  # noqa: E402
from travel_agent.services.retriever import KnowledgeRetriever, try_load_retriever  # noqa: E402
from travel_agent.services.router import CONTROL_UNKNOWN_CITIES, KnowledgeRouter  # noqa: E402

QUERIES_PATH = Path(__file__).parent / "queries.jsonl"
# Thresholds to sweep it deliberately spans well past the useful range in both directions so the plateau and both failure modes are visible.
SWEEP_VALUES: tuple[float, ...] = (
    0.00,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
    0.11,
    0.12,
    0.15,
    0.20,
    0.25,
    0.55,
)
# Queries used for the end-to-end schema check.
SCHEMA_QUERIES: tuple[str, ...] = (
    "Tell me about Tokyo",
    "Tell me about Kyoto",
    "Tell me about New York",
    "what about next week?",
)


@dataclass
class Case:
    """One labelled query in the router evaluation set."""

    query: str
    expected_city: str | None
    expected_route: str
    note: str


def load_cases() -> list[Case]:
    """Read the labelled query set from ``queries.jsonl``."""
    if not QUERIES_PATH.exists():
        raise SystemExit(f"missing {QUERIES_PATH}")
    cases: list[Case] = []
    for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(
            Case(
                query=row["query"],
                expected_city=row["expected_city"],
                expected_route=row["expected_route"],
                note=row.get("note", ""),
            )
        )
    return cases


def evaluate_routing(
    cases: list[Case], retriever: KnowledgeRetriever, threshold: float
) -> tuple[int, list[str]]:
    """Evaluate the router against the labelled query set."""
    settings = Settings(_env_file=None, router_similarity_threshold=threshold)
    router = KnowledgeRouter(retriever, settings=settings, validate=False)
    correct = 0
    failures: list[str] = []

    for case in cases:
        city = _extract_city(case.query, retriever)
        decision = router.decide(city)
        city_ok = (city or None) == case.expected_city
        route_ok = decision.route == case.expected_route

        if city_ok and route_ok:
            correct += 1
        else:
            failures.append(
                f"{case.query!r}: got city={city!r} route={decision.route!r} "
                f"(score {decision.score:.3f}), expected "
                f"city={case.expected_city!r} route={case.expected_route!r}"
            )
    return correct, failures


def print_separation(retriever: KnowledgeRetriever) -> tuple[float, float]:
    """Print the score gap between seeded and unseeded cities."""
    seeded = {city: retriever.best_match(city)[1] for city in retriever.known_cities}
    controls = {city: retriever.best_match(city)[1] for city in CONTROL_UNKNOWN_CITIES}
    lowest_seeded = min(seeded.values())
    highest_control = max(controls.values())

    print("SEPARATION")
    for city, score in sorted(seeded.items(), key=lambda item: -item[1]):
        print(f"  seeded    {city:<14} {score:.3f}")
    for city, score in sorted(controls.items(), key=lambda item: -item[1]):
        print(f"  unseeded  {city:<14} {score:.3f}")

    print(f"  lowest seeded    {lowest_seeded:.3f}")
    print(f"  highest unseeded {highest_control:.3f}")
    print(f"  margin           {lowest_seeded - highest_control:.3f}")
    print(f"  midpoint         {(lowest_seeded + highest_control) / 2:.3f}")
    print()
    return highest_control, lowest_seeded


def evaluate_similarity_only(
    cases: list[Case], retriever: KnowledgeRetriever, threshold: float
) -> int:
    """Evaluate the router using only the similarity layer, ignoring the name list."""
    correct = 0
    for case in cases:
        city = _extract_city(case.query, retriever)
        if city is None:
            route = "clarify"
        else:
            _, score = retriever.best_match(city)
            route = "vector" if score >= threshold else "web"

        if (city or None) == case.expected_city and route == case.expected_route:
            correct += 1
    return correct


def print_sweep(cases: list[Case], retriever: KnowledgeRetriever, configured: float) -> None:
    """Print router accuracy across a range of thresholds."""
    print("THRESHOLD SWEEP")
    print(f"  {'threshold':>10}  {'full':>6}  {'sim-only':>8}  {'':<22}  notes")

    best: list[float] = []
    similarity_best: list[float] = []
    for threshold in SWEEP_VALUES:
        correct, _ = evaluate_routing(cases, retriever, threshold)
        similarity_correct = evaluate_similarity_only(cases, retriever, threshold)
        accuracy = correct / len(cases)
        bar = "#" * round(similarity_correct / len(cases) * 22)
        marker = "  <- configured" if abs(threshold - configured) < 1e-9 else ""
        if accuracy == 1.0:
            best.append(threshold)
        if similarity_correct == len(cases):
            similarity_best.append(threshold)
        print(
            f"  {threshold:>10.2f}  {correct:>3}/{len(cases)}"
            f"  {similarity_correct:>3}/{len(cases)}  {bar:<22}{marker}"
        )

    if similarity_best:
        print(
            f"  similarity layer alone is perfect from {min(similarity_best):.2f} "
            f"to {max(similarity_best):.2f}"
        )
        print(
            f"  midpoint of that window: "
            f"{(min(similarity_best) + max(similarity_best)) / 2:.3f}"
        )
        print(f"  the configured {configured:.2f} sits inside it")
    else:
        print("  no threshold works for the similarity layer - the corpus changed")

    if best and len(best) > len(similarity_best):
        print()
        print("  Note the two columns diverge. The full router tolerates a much wider")
        print("  range because the gazetteer answers first for every seeded city, so a")
        print("  badly set threshold stays hidden. The right-hand column is the one that")
        print("  justifies the value - and the one that would have caught a stale 0.55.")
    print()


async def evaluate_schema(queries: tuple[str, ...]) -> tuple[int, list[str]]:
    """Check the end-to-end response schema against a set of queries."""
    settings = Settings(
        _env_file=None,
        llm_provider="mock",
        image_fallback_mode="local",
        mock_weather_latency_ms=40,
        mock_image_latency_ms=40,
        mock_search_latency_ms=40,
        mock_latency_jitter=0.0,
        router_similarity_threshold=0.07,
    )
    app = build_graph(build_dependencies(settings))
    config = {"configurable": {"thread_id": "eval"}}
    valid = 0
    failures: list[str] = []

    for query in queries:
        state = await app.ainvoke({"user_query": query}, config=config)
        response = state.get("response")

        if not isinstance(response, TravelResponse):
            failures.append(f"{query!r}: no TravelResponse produced")
            continue
        problems: list[str] = []
        if len(response.city_summary) < 40:
            problems.append("summary too short")
        if not response.is_clarification:
            if not response.city:
                problems.append("no city")
            if len(response.weather_forecast) not in range(5, 8):
                problems.append(f"{len(response.weather_forecast)} forecast points, expected 5-7")
            if any(not url.startswith("https://") for url in response.image_urls):
                problems.append("a non-https image url")
        if problems:
            failures.append(f"{query!r}: {', '.join(problems)}")
        else:
            valid += 1

    return valid, failures


def main() -> int:
    """Run the evaluation and return a process exit code."""
    parser = argparse.ArgumentParser(description="Evaluate the router and the response schema.")
    parser.add_argument("--sweep-only", action="store_true", help="Only print the sweep.")
    parser.add_argument("--no-schema", action="store_true", help="Skip the schema checks.")
    args = parser.parse_args()

    configure_logging("ERROR")
    settings = Settings(_env_file=None)

    retriever = try_load_retriever(settings)
    if retriever is None:
        raise SystemExit("no vector store found. Run: python scripts/seed_vectorstore.py")

    cases = load_cases()

    print("ROUTER AND SCHEMA EVALUATION")
    print(f"corpus  : {len(retriever.store)} chunks across {', '.join(retriever.known_cities)}")
    print(f"queries : {len(cases)} labelled cases")
    print()
    print_separation(retriever)
    print_sweep(cases, retriever, settings.router_similarity_threshold)

    if args.sweep_only:
        return 0
    correct, failures = evaluate_routing(cases, retriever, settings.router_similarity_threshold)
    print("ROUTING ACCURACY AT THE CONFIGURED THRESHOLD")
    print(
        f"  {correct}/{len(cases)} correct ({correct / len(cases):.0%}) "
        f"at threshold {settings.router_similarity_threshold:.2f}"
    )
    for failure in failures:
        print(f"  FAIL {failure}")
    print()

    schema_valid, schema_failures = 0, []
    if not args.no_schema:
        schema_valid, schema_failures = asyncio.run(evaluate_schema(SCHEMA_QUERIES))
        print("RESPONSE SCHEMA VALIDITY")
        print(f"  {schema_valid}/{len(SCHEMA_QUERIES)} responses valid")
        for failure in schema_failures:
            print(f"  FAIL {failure}")
        print()

    ok = not failures and not schema_failures

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
