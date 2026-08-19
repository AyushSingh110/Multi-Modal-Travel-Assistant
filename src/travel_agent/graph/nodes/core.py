"""The graph's core nodes.

Input cleaning, intent classification, tool planning, knowledge retrieval and
the join that closes the parallel fan-out.
"""

from __future__ import annotations

import re
import time
from datetime import date, timedelta
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from travel_agent.graph import edges as graph_edges
from travel_agent.logging_setup import Timer, get_logger
from travel_agent.schemas.intent import DateRange, IntentDecision, RouteDecision
from travel_agent.schemas.state import TravelState, new_turn_updates
from travel_agent.schemas.tools import (
    IMAGES_TOOL,
    WEATHER_TOOL,
    WEB_SEARCH_TOOL,
    openai_tool_schemas,
)
from travel_agent.schemas.trace import ParallelMetrics, TraceEvent
from travel_agent.services.llm.base import BaseLLM
from travel_agent.services.retriever import KnowledgeRetriever, normalise_city_name
from travel_agent.services.router import KnowledgeRouter

logger = get_logger(__name__)

# Branch node names, defined once so edges, timings and metrics agree
KNOWLEDGE_VECTOR_NODE = "retrieve_vector"
KNOWLEDGE_WEB_NODE = "web_search"
WEATHER_NODE = "execute_weather"
IMAGES_NODE = "execute_images"
FAN_OUT_NODES = (KNOWLEDGE_VECTOR_NODE, KNOWLEDGE_WEB_NODE, WEATHER_NODE, IMAGES_NODE)

# Phrases that shift the forecast window without naming a new city.
_NEXT_WEEK = re.compile(r"\bnext\s+week\b", re.IGNORECASE)
_THIS_WEEKEND = re.compile(r"\b(this\s+)?weekend\b", re.IGNORECASE)
_IN_N_DAYS = re.compile(r"\bin\s+(\d{1,2})\s+days?\b", re.IGNORECASE)
_TOMORROW = re.compile(r"\btomorrow\b", re.IGNORECASE)

_PREPOSITION_CITY = re.compile(
    r"\b(?:about|in|for|to|visit|visiting|go\s+to|travelling\s+to|traveling\s+to)\s+"
    r"(?P<city>[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)*)"
)

# Words that look like city names but are not
_NON_CITY_WORDS = frozenset(
    {
        "tell",
        "what",
        "where",
        "how",
        "why",
        "when",
        "the",
        "about",
        "give",
        "show",
        "find",
        "please",
        "hello",
        "hi",
        "and",
        "for",
        "with",
        "next",
        "week",
        "weekend",
        "today",
        "tomorrow",
        "weather",
        "forecast",
        "images",
        "photos",
        "city",
        "now",
        "then",
        "also",
        "okay",
        "hey",
        "thanks",
        "some",
        "any",
        "good",
        "best",
        "should",
        "trip",
        "travel",
        "visit",
        "going",
        "like",
        "there",
    }
)


#  normalize
def normalize_input(state: TravelState) -> dict[str, Any]:
    """Clean the user query and start a fresh turn.

    This is the first node, so it seeds the state with a clean slate. Every later
    node can assume a single-line query and a correct turn index.
    """
    raw = str(state.get("user_query", "") or "")
    cleaned = re.sub(r"\s+", " ", raw).strip()
    turn_index = int(state.get("turn_index", -1)) + 1

    updates = new_turn_updates(cleaned, turn_index)
    updates["messages"] = [HumanMessage(content=cleaned)] if cleaned else []
    updates["trace"] = None  # reset first; the event below re-seeds the list
    return {
        **updates,
        "previous_city": state.get("city"),
    }


# classify intent
def make_classify_intent(retriever: KnowledgeRetriever | None) -> Any:
    """Build the intent-classification node."""

    async def classify_intent(state: TravelState) -> dict[str, Any]:
        # Classify the user's intent and extract the city and date range. This node is responsible for understanding what the user wants to do, which city they are asking about, and what date range they are interested in. It uses a combination of gazetteer lookups, regex patterns, and heuristics to extract this information from the user query.
        with Timer() as timer:
            query = str(state.get("user_query", ""))
            previous_city = state.get("city")
            previous_range = state.get("date_range") or DateRange()

            city = _extract_city(query, retriever) or previous_city
            date_range, date_changed = _extract_date_range(query, previous_range)

            if not city:
                decision = IntentDecision(
                    intent="clarify",
                    city=None,
                    city_changed=False,
                    date_changed=date_changed,
                    date_range=date_range,
                    reason="No city could be resolved from this turn or from memory.",
                )
            else:
                city_changed = normalise_city_name(city) != normalise_city_name(previous_city or "")
                if city_changed:
                    intent = "new_city"
                    reason = f"City resolved as {city}; nothing carried over."
                elif date_changed:
                    # The follow-up case: same city, different dates. Only the
                    # weather needs refreshing. Step 11 acts on this.
                    intent = "weather_only"
                    reason = f"City unchanged ({city}); only the date window moved."
                else:
                    intent = "refine"
                    reason = f"City unchanged ({city}) and no new date window."

                decision = IntentDecision(
                    intent=intent,
                    city=city,
                    city_changed=city_changed,
                    date_changed=date_changed,
                    date_range=date_range,
                    reason=reason,
                )

        logger.info("intent=%s city=%s (%s)", decision.intent, decision.city, decision.reason)
        return {
            "intent": decision.intent,
            "city": decision.city,
            "date_range": decision.date_range,
            "timings": {"classify_intent": timer.elapsed_ms},
            "trace": [
                TraceEvent(
                    node="classify_intent",
                    kind="route",
                    message=f"intent={decision.intent}: {decision.reason}",
                    duration_ms=timer.elapsed_ms,
                    data={
                        "intent": decision.intent,
                        "city": decision.city,
                        "city_changed": decision.city_changed,
                        "date_changed": decision.date_changed,
                        "date_range": decision.date_range.label,
                    },
                )
            ],
        }

    return classify_intent


def _extract_city(query: str, retriever: KnowledgeRetriever | None) -> str | None:
    # Extract a plausible city name from the user query. This function uses a combination of gazetteer lookups, regex patterns, and heuristics to identify a city name in the user's input. It prioritizes known cities from the retriever, then looks for prepositional phrases, capitalized phrases, and finally falls back to lowercase prepositions if necessary.
    if not query.strip():
        return None

    # 1. The gazetteer. A known city is a certainty, so it outranks any guess.
    if retriever is not None:
        # Longest windows first so "New York" beats "New".
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", query)
        for size in (3, 2, 1):
            for start in range(len(words) - size + 1):
                candidate = " ".join(words[start : start + size])
                match = retriever.find_city_by_name(candidate)
                if match:
                    return match

    prepositional = _PREPOSITION_CITY.search(query)
    if prepositional:
        candidate = prepositional.group("city").strip()
        if _is_plausible_city(candidate):
            return candidate

    tokens = [
        (match.group(1), match.start())
        for match in re.finditer(r"\b([A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)*)\b", query)
    ]
    plausible = [(token, start) for token, start in tokens if _is_plausible_city(token)]
    non_leading = [token for token, start in plausible if start > 0]
    if non_leading:
        return non_leading[0]
    if plausible:
        return plausible[0][0]

    # Last resort: a preposition in an all-lowercase query.
    tail = re.search(r"\b(?:about|in|for|to)\s+([a-z][a-z'-]{2,})\b", query, re.IGNORECASE)
    if tail and _is_plausible_city(tail.group(1)):
        return tail.group(1).title()
    return None


def _is_plausible_city(candidate: str) -> bool:
    # Decide whether a candidate string looks like a city name. This is a heuristic filter to avoid false positives from the regexes. It checks that the candidate is at least three characters long and does not contain any common English words that are unlikely to be city names.
    cleaned = candidate.strip()
    if len(cleaned) < 3:
        return False
    return all(word.lower() not in _NON_CITY_WORDS for word in cleaned.split())


def _extract_date_range(query: str, previous: DateRange) -> tuple[DateRange, bool]:
    # Extract a date range from the user query. This function looks for specific phrases that indicate a desired date range for the travel information. If no such phrases are found, it defaults to the previous date range or a standard "next 7 days" window.
    today = date.today()
    if _NEXT_WEEK.search(query):
        return DateRange(start=today + timedelta(days=7), days=7, label="next week"), True
    if _THIS_WEEKEND.search(query):
        days_until_saturday = (5 - today.weekday()) % 7
        return (
            DateRange(
                start=today + timedelta(days=days_until_saturday), days=2, label="this weekend"
            ),
            True,
        )
    if _TOMORROW.search(query):
        return DateRange(start=today + timedelta(days=1), days=3, label="from tomorrow"), True

    match = _IN_N_DAYS.search(query)
    if match:
        offset = min(int(match.group(1)), 14)
        return (
            DateRange(start=today + timedelta(days=offset), days=7, label=f"in {offset} days"),
            True,
        )

    # No date language: keep the default window, and treat a stale start as fresh.
    if previous.start != today and previous.label == "next 7 days":
        return DateRange(start=today, days=previous.days, label=previous.label), False
    return previous, False


# plan tools
def make_plan_tools(llm: BaseLLM, router: KnowledgeRouter | None) -> Any:
    """Build the tool-planning node."""

    async def plan_tools(state: TravelState) -> dict[str, Any]:
        # Plan which tools to call for this turn. This node is responsible for deciding which external tools (weather, images, web search) should be invoked based on the user's intent, the resolved city, and the date range. It uses the LLM to generate a plan and ensures that all required tools are included in the final plan.
        city = state.get("city")
        intent = state.get("intent", "new_city")
        date_range = state.get("date_range") or DateRange()

        with Timer() as timer:
            decision = _route(router, city)
            tool_names = _tools_for(decision, intent)
            schemas = openai_tool_schemas(tool_names)

            brief = (
                f"City: {city or ''}\n"
                f"Intent: {intent}\n"
                f"Knowledge source: {'web_search' if decision.route == 'web' else 'vector_store'}\n"
                f"Forecast days: {date_range.days}\n"
                f"Date window: {date_range.label}\n"
                f"Start date: {date_range.start.isoformat()}\n"
            )
            messages: list[AnyMessage] = [
                SystemMessage(
                    content=(
                        "You are the planning step of a travel assistant. Decide which "
                        "of the offered tools to call, and call them all in one reply.\n\n"
                        "A complete answer for this interface always contains three "
                        "things: a written summary, a photo gallery and a weather chart. "
                        "So call EVERY tool you are offered - if an image tool is "
                        "offered, the gallery needs it; if a web search tool is offered, "
                        "the summary needs it. Omitting one leaves a visibly empty panel "
                        "in the interface.\n\n"
                        "Reply with tool calls only. Do not write prose."
                    )
                ),
                HumanMessage(content=brief),
            ]

            call = await llm.plan(messages, schemas)
            planned_message, added_tools = _complete_plan(
                call.message, tool_names, city or "", date_range
            )
        requested = [entry.get("name") for entry in getattr(planned_message, "tool_calls", [])]

        # Which branches this turn will NOT run, and what that is worth.
        provisional: TravelState = {**state, "intent": intent, "route": decision.route}
        skipped = graph_edges.skipped_branches(provisional)
        previous_durations = state.get("last_branch_durations", {}) or {}
        saved_ms = sum(previous_durations.get(node, 0.0) for node in skipped)

        logger.info(
            "plan_tools: route=%s (%s, score=%.3f) tools=%s skipped=%s",
            decision.route,
            decision.match_reason,
            decision.score,
            requested or "none",
            skipped or "none",
        )
        return {
            "messages": [planned_message],
            "route": decision.route,
            "route_score": decision.score,
            "route_threshold": decision.threshold,
            "route_reason": decision.reason,
            "route_match_reason": decision.match_reason,
            "matched_city": decision.matched_city,
            "route_all_scores": decision.all_scores,
            "token_usage": call.usage,
            "timings": {"plan_tools": timer.elapsed_ms},
            "skipped_nodes": skipped,
            "skipped_ms_saved": round(saved_ms, 1),
            "fanout_started_at": time.perf_counter(),
            "trace": [
                TraceEvent(
                    node="plan_tools",
                    kind="route",
                    message=decision.reason,
                    duration_ms=timer.elapsed_ms,
                    data={
                        "route": decision.route,
                        "match_reason": decision.match_reason,
                        "score": round(decision.score, 4),
                        "threshold": decision.threshold,
                        "matched_city": decision.matched_city,
                        "all_scores": {k: round(v, 4) for k, v in decision.all_scores.items()},
                        "tools_offered": tool_names,
                        "tools_requested": requested,
                        "tools_added_by_graph": added_tools,
                        "skipped_nodes": skipped,
                        "skipped_ms_saved": round(saved_ms, 1),
                    },
                )
            ]
            + (
                [
                    TraceEvent(
                        node="plan_tools",
                        kind="skip",
                        message=(
                            f"skipped {', '.join(skipped)} - unchanged since the last turn"
                            + (f", saving about {saved_ms:.0f} ms" if saved_ms else "")
                        ),
                        data={"skipped": skipped, "ms_saved": round(saved_ms, 1)},
                    )
                ]
                if skipped
                else []
            ),
        }

    return plan_tools


def _complete_plan(
    message: AIMessage,
    offered: list[str],
    city: str,
    date_range: DateRange,
) -> tuple[AIMessage, list[str]]:
    # Ensure that all required tools are present in the model's plan. If the model omitted any tools, this function adds them with default arguments based on the resolved city and date range. It returns a new AIMessage with the complete list of tool calls and a list of any tools that were added by the graph.
    existing = list(getattr(message, "tool_calls", []) or [])
    requested = {entry.get("name") for entry in existing}
    missing = [name for name in offered if name not in requested]

    if not missing or not city:
        return message, []

    defaults: dict[str, dict[str, Any]] = {
        WEATHER_TOOL: {
            "city": city,
            "days": date_range.days,
            "start_date": date_range.start.isoformat(),
        },
        IMAGES_TOOL: {"city": city, "count": 4},
        WEB_SEARCH_TOOL: {"query": f"{city} travel guide overview", "max_results": 4},
    }

    added: list[str] = []
    for index, name in enumerate(missing):
        if name not in defaults:
            continue
        existing.append(
            {
                "id": f"call_graph_{index:03d}",
                "name": name,
                "args": defaults[name],
                "type": "tool_call",
            }
        )
        added.append(name)

    if not added:
        return message, []

    logger.info("plan completion: the model omitted %s; the graph added it", ", ".join(added))
    return AIMessage(content=message.content, tool_calls=existing), added


def _route(router: KnowledgeRouter | None, city: str | None) -> RouteDecision:
    # Decide which knowledge source to use for this turn. If a router is provided, it uses the router to decide between the vector store and web search based on the resolved city. If no router is available, it defaults to web search as the only source.
    if router is None:
        return RouteDecision(
            route="web",
            match_reason="similarity",
            score=0.0,
            threshold=0.0,
            reason="No vector store is loaded, so the web is the only source available.",
        )
    return router.decide(city)


def _tools_for(decision: RouteDecision, intent: str) -> list[str]:
    # Determine which tools to call based on the routing decision and user intent. The weather tool is always included, and the image tool is included unless the intent is "weather_only". If the routing decision indicates that web search should be used, the web search tool is also included.
    if intent == "weather_only":
        return [WEATHER_TOOL]

    tools = [WEATHER_TOOL, IMAGES_TOOL]
    if decision.route == "web":
        tools.append(WEB_SEARCH_TOOL)
    return tools


# retrieve vector
def make_retrieve_vector(retriever: KnowledgeRetriever | None) -> Any:
    """Build the vector-retrieval node."""

    async def retrieve_vector(state: TravelState) -> dict[str, Any]:
        # Retrieve knowledge chunks from the vector store for the resolved city. This node is responsible for fetching relevant information from the vector store based on the city identified in the user's query. If no retriever is available, it returns an empty list of chunks.
        city = state.get("city") or ""

        with Timer() as timer:
            chunks = retriever.chunks_for_city(city) if retriever is not None else []

        logger.info("retrieve_vector: %d chunk(s) for %s", len(chunks), city)
        return {
            "knowledge": chunks,
            "timings": {KNOWLEDGE_VECTOR_NODE: timer.elapsed_ms},
            "trace": [
                TraceEvent(
                    node=KNOWLEDGE_VECTOR_NODE,
                    kind="tool",
                    message=f"retrieved {len(chunks)} passage(s) from the vector store",
                    duration_ms=timer.elapsed_ms,
                    data={"city": city, "chunks": [chunk.chunk_id for chunk in chunks]},
                )
            ],
        }

    return retrieve_vector


# join
async def join(state: TravelState) -> dict[str, Any]:
    """Join the parallel branches and report the timing metrics.

    Collects what the concurrent branches produced, then records wall-clock time,
    sequential-equivalent time and the resulting speed-up.
    """
    started = state.get("fanout_started_at")
    wall_clock_ms = (time.perf_counter() - float(started)) * 1000.0 if started else 0.0

    timings = state.get("timings", {}) or {}
    branch_durations = {
        node: duration for node, duration in timings.items() if node in FAN_OUT_NODES
    }
    sequential_ms = sum(branch_durations.values())

    metrics = ParallelMetrics(
        branch_durations_ms={k: round(v, 1) for k, v in branch_durations.items()},
        sequential_equivalent_ms=round(sequential_ms, 1),
        parallel_wall_clock_ms=round(wall_clock_ms, 1),
        speedup=round(sequential_ms / wall_clock_ms, 2) if wall_clock_ms > 0 else 1.0,
    )

    logger.info(
        "join: %d branch(es), sequential %.0f ms vs parallel %.0f ms (%.2fx)",
        len(branch_durations),
        sequential_ms,
        wall_clock_ms,
        metrics.speedup,
    )

    return {
        "parallel_metrics": metrics,
        # Persisted deliberately across turns (not reset by new_turn_updates) so a
        # follow-up can report how much time skipping these branches saved.
        "last_branch_durations": {k: round(v, 1) for k, v in branch_durations.items()},
        "trace": [
            TraceEvent(
                node="join",
                kind="timing",
                message=(
                    f"{len(branch_durations)} branches in parallel: "
                    f"{sequential_ms:.0f} ms sequential-equivalent vs "
                    f"{wall_clock_ms:.0f} ms actual ({metrics.speedup:.2f}x)"
                ),
                duration_ms=wall_clock_ms,
                data=metrics.model_dump(),
            )
        ],
    }


# synthesize

__all__ = [
    "FAN_OUT_NODES",
    "IMAGES_NODE",
    "KNOWLEDGE_VECTOR_NODE",
    "KNOWLEDGE_WEB_NODE",
    "WEATHER_NODE",
    "join",
    "make_classify_intent",
    "make_plan_tools",
    "make_retrieve_vector",
    "normalize_input",
]
