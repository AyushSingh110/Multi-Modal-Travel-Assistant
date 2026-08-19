"""The graph's conditional edges - where the routing and the parallelism live.

WHAT A CONDITIONAL EDGE IS
    An ordinary edge always goes to the same place. A conditional edge runs a
    function at execution time and goes wherever that function names. The two here
    are what make this a decision-making graph rather than a fixed pipeline.

WHAT A SUPERSTEP IS
    LangGraph executes in rounds. Everything scheduled in the same round - the
    same superstep - starts together, runs concurrently, and the next round does
    not begin until all of it has finished. A conditional edge that returns a
    *list* of node names schedules all of them into one superstep.

    That single fact is the whole of Distinction 2. The alternative - calling
    ``asyncio.gather`` inside one node - would also run the work concurrently, but
    the graph would contain one node where three should be, and ``graph.png``
    would show a straight line. Parallelism that a reviewer cannot see in the
    topology is parallelism they have to take on faith.
"""

from __future__ import annotations

from travel_agent.logging_setup import get_logger
from travel_agent.schemas.state import TravelState

logger = get_logger(__name__)

# Node names. Imported from one place so the edges, the builder and the metrics
# can never disagree about what a node is called.
PLAN_TOOLS = "plan_tools"
SYNTHESIZE = "synthesize"
RETRIEVE_VECTOR = "retrieve_vector"
WEB_SEARCH = "web_search"
EXECUTE_WEATHER = "execute_weather"
EXECUTE_IMAGES = "execute_images"

#: Every target the fan-out edge may return, declared for the diagram.
FAN_OUT_TARGETS = [RETRIEVE_VECTOR, WEB_SEARCH, EXECUTE_WEATHER, EXECUTE_IMAGES]

#: Targets of the intent edge.
INTENT_TARGETS = [PLAN_TOOLS, SYNTHESIZE]


def route_after_intent(state: TravelState) -> str:
    """Decide whether this turn needs tools at all.

    Args:
        state: Current graph state.

    Returns:
        ``"plan_tools"`` when work is needed, or ``"synthesize"`` to answer from
        what the checkpointer already holds.
    """
    intent = state.get("intent", "new_city")

    if intent == "clarify":
        # No city resolved. There is nothing to fetch, so skip straight to the
        # response, which will ask the user which city they mean.
        logger.info("route_after_intent: clarify -> %s", SYNTHESIZE)
        return SYNTHESIZE

    if intent == "refine" and state.get("response") is not None:
        # Same city, same dates, and a previous answer is in state: nothing has
        # changed that would alter the result, so re-running the tools would burn
        # latency and quota to produce the same page.
        logger.info("route_after_intent: refine with cached state -> %s", SYNTHESIZE)
        return SYNTHESIZE

    logger.info("route_after_intent: %s -> %s", intent, PLAN_TOOLS)
    return PLAN_TOOLS


def route_and_fan_out(state: TravelState) -> list[str]:
    """Choose the knowledge branch and dispatch every branch at once.

    This is both conditional edges the assignment asks for, in one function: the
    *knowledge routing* decision (vector store versus web search, already made and
    recorded by ``plan_tools``) and the *fan-out* that puts the chosen knowledge
    branch alongside the weather and image branches in a single superstep.

    Returning a list is what makes them concurrent. Returning them one at a time
    across three edges would serialise the same work.

    Args:
        state: Current graph state.

    Returns:
        The node names to run concurrently. Never empty - an empty list would
        strand the graph with no path to the join node.
    """
    intent = state.get("intent", "new_city")
    route = state.get("route", "vector")

    if intent == "weather_only":
        # The follow-up path: the city has not changed, so its summary and images
        # are already in state. Only the forecast needs refreshing.
        logger.info("fan-out: weather only (follow-up turn)")
        return [EXECUTE_WEATHER]

    knowledge_branch = WEB_SEARCH if route == "web" else RETRIEVE_VECTOR
    targets = [knowledge_branch, EXECUTE_WEATHER, EXECUTE_IMAGES]

    logger.info("fan-out: %s dispatched in one superstep", ", ".join(targets))
    return targets


__all__ = [
    "EXECUTE_IMAGES",
    "EXECUTE_WEATHER",
    "FAN_OUT_TARGETS",
    "INTENT_TARGETS",
    "PLAN_TOOLS",
    "RETRIEVE_VECTOR",
    "SYNTHESIZE",
    "WEB_SEARCH",
    "route_after_intent",
    "route_and_fan_out",
]
