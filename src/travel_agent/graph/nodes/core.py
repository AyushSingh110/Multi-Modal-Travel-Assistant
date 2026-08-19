"""The graph's non-tool nodes.

Each function here is one node: it receives the whole state and returns a partial
update. Nodes that do I/O are ``async def`` coroutine *functions* - not sync
functions returning a coroutine, which LangGraph would treat as a synchronous
node and then reject the un-awaited coroutine as an invalid state update.

Node responsibilities, in execution order:

``normalize_input``   tidy the raw query and start a fresh turn
``classify_intent``   resolve the city and date slots, decide what kind of turn
``plan_tools``        ask the model which tools to call, and route the knowledge
``retrieve_vector``   read the seeded corpus (the internal-knowledge branch)
``join``              barrier after the fan-out; measures the parallel speed-up
``synthesize``        assemble the validated response object
"""

from __future__ import annotations

import re
import time
from datetime import date, timedelta
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from travel_agent.graph import edges as graph_edges
from travel_agent.logging_setup import Timer, get_logger
from travel_agent.schemas.intent import DateRange, IntentDecision, RouteDecision
from travel_agent.schemas.response import TravelResponse
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

# Branch node names, defined once so edges, timings and metrics agree.
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

# "about Kyoto", "in New York", "visiting Osaka" - the grammar names the
# destination far more reliably than capitalisation does.
_PREPOSITION_CITY = re.compile(
    r"\b(?:about|in|for|to|visit|visiting|go\s+to|travelling\s+to|traveling\s+to)\s+"
    r"(?P<city>[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)*)"
)

# Words that look like city names but are not, so the extractor does not resolve
# "Tell" or "What" as a destination.
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


# ============================================================ normalize ======
def normalize_input(state: TravelState) -> dict[str, Any]:
    """Clean the incoming query and reset the per-turn observability keys.

    Args:
        state: Current graph state.

    Returns:
        A partial state update starting a fresh turn.
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


# ====================================================== classify intent ======
def make_classify_intent(retriever: KnowledgeRetriever | None) -> Any:
    """Build the intent classifier node.

    Slot extraction is deliberately deterministic rather than a model call. Two
    reasons: pulling a city name out of a sentence is a parsing problem, not a
    reasoning one, and a follow-up turn should not have to pay a model round-trip
    to work out that the city has not changed. The model is used where it earns
    its keep - planning tool calls and writing prose.

    Args:
        retriever: Retrieval service, used as a gazetteer for city names. May be
            ``None`` when the store has not been seeded.

    Returns:
        An async node function.
    """

    async def classify_intent(state: TravelState) -> dict[str, Any]:
        """Resolve the city and date slots and decide the kind of turn.

        Args:
            state: Current graph state.

        Returns:
            A partial state update carrying the intent and resolved slots.
        """
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
    """Pull a city name out of a sentence.

    Tries the gazetteer first - a known city is a certainty - then falls back to
    capitalised words, which is how an unknown city like "Kyoto" is picked up.

    Args:
        query: The user's text.
        retriever: Retrieval service used as a gazetteer, if available.

    Returns:
        The city name, or ``None`` when nothing usable was found.
    """
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

    # 2. A preposition names its object. "Now tell me about Kyoto" has two
    #    capitalised-ish candidates, and only the grammar says which one is the
    #    destination - an earlier version took the first non-filler token and
    #    confidently resolved the city as "Now".
    prepositional = _PREPOSITION_CITY.search(query)
    if prepositional:
        candidate = prepositional.group("city").strip()
        if _is_plausible_city(candidate):
            return candidate

    # 3. Capitalised phrases, preferring one that is not the first word of the
    #    sentence - an opening word is capitalised by convention, not by meaning.
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

    # 4. Last resort: a preposition in an all-lowercase query.
    tail = re.search(r"\b(?:about|in|for|to)\s+([a-z][a-z'-]{2,})\b", query, re.IGNORECASE)
    if tail and _is_plausible_city(tail.group(1)):
        return tail.group(1).title()
    return None


def _is_plausible_city(candidate: str) -> bool:
    """Reject sentence filler that happens to be capitalised.

    Args:
        candidate: A candidate city name.

    Returns:
        ``True`` when the candidate could be a place name.
    """
    cleaned = candidate.strip()
    if len(cleaned) < 3:
        return False
    return all(word.lower() not in _NON_CITY_WORDS for word in cleaned.split())


def _extract_date_range(query: str, previous: DateRange) -> tuple[DateRange, bool]:
    """Work out which forecast window the turn is asking about.

    Args:
        query: The user's text.
        previous: The window used on the previous turn.

    Returns:
        A ``(date_range, changed)`` pair.
    """
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


# ========================================================== plan tools ======
def make_plan_tools(llm: BaseLLM, router: KnowledgeRouter | None) -> Any:
    """Build the planning node.

    This node does two things that belong together: it decides *where the facts
    come from* (the routing decision) and then asks the model *which tools to
    call*, offering only the tools that suit that route. Deciding the route first
    means the web-search tool is never even advertised for a city the knowledge
    base already covers.

    Args:
        llm: The model driver.
        router: The knowledge router. ``None`` disables retrieval and forces the
            web path, which is what happens when the store is unseeded.

    Returns:
        An async node function.
    """

    async def plan_tools(state: TravelState) -> dict[str, Any]:
        """Route the knowledge source and ask the model for tool calls.

        Args:
            state: Current graph state.

        Returns:
            A partial state update carrying the routing decision, the model's
            reply with its ``tool_calls`` payload, and the fan-out start time.
        """
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
                # The start date must reach the tool, not just the label. Without
                # it a follow-up asking about "next week" refreshes the forecast
                # for today all over again - the window moves in the prose and
                # nowhere in the data.
                f"Start date: {date_range.start.isoformat()}\n"
            )
            messages = [
                SystemMessage(
                    content=(
                        "You are the planning step of a travel assistant. Decide which "
                        "of the offered tools to call and with what arguments. Call every "
                        "tool that is needed to answer fully, in one reply. Do not write "
                        "prose."
                    )
                ),
                HumanMessage(content=brief),
            ]

            call = await llm.plan(messages, schemas)

        requested = [entry.get("name") for entry in getattr(call.message, "tool_calls", [])]

        # Which branches this turn will NOT run, and what that is worth. The
        # durations come from the last full turn, so the figure is a measurement
        # of work genuinely avoided rather than an estimate.
        provisional = {**state, "intent": intent, "route": decision.route}
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
            "messages": [call.message],
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
            # Recorded here, read by the join node: the difference between this
            # and the join's own clock is the fan-out's wall time.
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


def _route(router: KnowledgeRouter | None, city: str | None) -> RouteDecision:
    """Ask the router where this city's facts should come from.

    Args:
        router: The knowledge router, or ``None`` when retrieval is unavailable.
        city: The resolved city.

    Returns:
        A routing decision. With no router the web path is chosen, and the reason
        says so rather than pretending a score was computed.
    """
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
    """Choose which tools to advertise for this turn.

    Args:
        decision: The routing decision.
        intent: The classified intent.

    Returns:
        Tool names to offer the model.
    """
    if intent == "weather_only":
        return [WEATHER_TOOL]

    tools = [WEATHER_TOOL, IMAGES_TOOL]
    if decision.route == "web":
        tools.append(WEB_SEARCH_TOOL)
    return tools


# ===================================================== retrieve vector ======
def make_retrieve_vector(retriever: KnowledgeRetriever | None) -> Any:
    """Build the internal-knowledge branch node.

    Args:
        retriever: Retrieval service, or ``None`` when the store is unseeded.

    Returns:
        An async node function.
    """

    async def retrieve_vector(state: TravelState) -> dict[str, Any]:
        """Read the seeded corpus for the resolved city.

        Unlike the weather and image branches this is not a tool call: it is an
        internal database read, so it does not go through the model's tool
        protocol at all.

        Args:
            state: Current graph state.

        Returns:
            A partial state update carrying the retrieved chunks.
        """
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


# ================================================================= join =====
async def join(state: TravelState) -> dict[str, Any]:
    """Barrier after the fan-out; measures what the parallelism actually bought.

    LangGraph schedules every branch in one superstep and only runs this node once
    all of them have finished. That makes it the right place to compare two
    numbers:

    * **sequential equivalent** - the sum of the branches' own durations, which is
      what the same work would have cost run one after another;
    * **parallel wall clock** - how long the superstep actually took, measured
      from the moment ``plan_tools`` finished.

    The ratio is the speed-up, and it is stored in state so the UI and the README
    quote a measured figure rather than a claim.

    Args:
        state: Current graph state.

    Returns:
        A partial state update carrying the parallel metrics.
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


# =========================================================== synthesize =====
async def synthesize(state: TravelState) -> dict[str, Any]:
    """Assemble the validated response object.

    Deliberately thin for now: it composes what the branches produced into a
    :class:`TravelResponse` so the graph is end-to-end runnable. Step 12 replaces
    the summary text with a model-written one and adds the validate-and-repair
    pass; the contract this returns does not change.

    Args:
        state: Current graph state.

    Returns:
        A partial state update carrying the response object.
    """
    with Timer() as timer:
        if state.get("intent") == "clarify":
            return _clarify_response(state, timer)

        city = state.get("city") or "Unknown"
        knowledge = state.get("knowledge") or []
        weather = state.get("weather")
        images = state.get("images") or []
        errors = state.get("errors") or []

        if knowledge:
            body = " ".join(chunk.text.split(". ", 1)[-1][:220].rstrip() for chunk in knowledge[:2])
            summary = f"{city}. {body}"
        else:
            summary = (
                f"{city} is the destination for this request, but no source material "
                f"was retrieved for it on this turn."
            )

        warnings = [f"{error.tool} unavailable: {error.message[:120]}" for error in errors]

        response = TravelResponse(
            city=city,
            city_summary=summary[:1800],
            weather_forecast=list(weather.forecast) if weather else [],
            image_urls=[asset.url for asset in images],
            highlights=[chunk.section for chunk in knowledge[:4]],
            knowledge_source="web_search" if state.get("route") == "web" else "vector_store",
            sources=[chunk.source for chunk in knowledge if chunk.source][:6],
            warnings=warnings,
        )

    return {
        "response": response,
        "timings": {"synthesize": timer.elapsed_ms},
        "trace": [
            TraceEvent(
                node="synthesize",
                kind="info",
                message=(
                    f"response built: {len(response.weather_forecast)} forecast points, "
                    f"{len(response.image_urls)} images, {len(response.warnings)} warning(s)"
                ),
                duration_ms=timer.elapsed_ms,
                data={"degraded": response.is_degraded},
            )
        ],
    }


def _clarify_response(state: TravelState, timer: Timer) -> dict[str, Any]:
    """Build the response for a turn that named no city.

    The guarded failure mode: a fresh thread asking "what about next week?" has no
    city in memory and no city in the question. Guessing one would be worse than
    asking - a confident answer about the wrong place is harder for a user to
    detect than a question.

    Args:
        state: Current graph state.
        timer: The synthesis timer, already running.

    Returns:
        A partial state update carrying a clarifying response.
    """
    query = state.get("user_query", "")
    response = TravelResponse(
        city="",
        city_summary=(
            "I could not tell which city you mean. This looks like a follow-up, but "
            "there is no earlier city in this conversation to carry forward. Name a "
            "city - for example 'Tell me about Tokyo' - and I will look it up."
        ),
        knowledge_source="memory",
        warnings=["No city could be resolved from this turn or from the conversation history."],
    )

    return {
        "response": response,
        "timings": {"synthesize": timer.elapsed_ms},
        "trace": [
            TraceEvent(
                node="synthesize",
                kind="skip",
                message="no city resolved - asked the user to clarify instead of guessing",
                data={"query": query},
            )
        ],
    }


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
    "synthesize",
]
