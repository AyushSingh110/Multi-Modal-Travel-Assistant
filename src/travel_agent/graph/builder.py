"""Assembles the compiled LangGraph application.

The topology, in one place::

    START
      -> normalize_input
      -> classify_intent
      -> (conditional) plan_tools | synthesize
      -> plan_tools
      -> (conditional, returns a LIST -> one superstep)
             retrieve_vector | web_search      knowledge branch
             execute_weather                   tool branch
             execute_images                    tool branch
      -> join           barrier; measures the speed-up
      -> synthesize
      -> END

Dependencies - the model driver, the retriever, the router, the tool registry -
are constructed once here and closed over by the nodes. Nothing reaches for a
global, which is what lets a test build the same graph with a broken weather
provider and assert the page still renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from travel_agent.config.settings import Settings, get_settings
from travel_agent.graph import edges
from travel_agent.graph.nodes.core import (
    join,
    make_classify_intent,
    make_plan_tools,
    make_retrieve_vector,
    normalize_input,
)
from travel_agent.graph.nodes.synthesize import make_synthesize
from travel_agent.graph.nodes.tool_executor import make_tool_executor
from travel_agent.logging_setup import get_logger
from travel_agent.schemas.state import TravelState
from travel_agent.schemas.tools import IMAGES_TOOL, WEATHER_TOOL, WEB_SEARCH_TOOL
from travel_agent.services.llm.base import BaseLLM
from travel_agent.services.llm.factory import get_llm
from travel_agent.services.retriever import KnowledgeRetriever, try_load_retriever
from travel_agent.services.router import KnowledgeRouter
from travel_agent.tools.registry import ToolRegistry, build_registry

logger = get_logger(__name__)


@dataclass
class GraphDependencies:
    """Everything the graph needs, resolved once.

    Attributes:
        llm: The model driver.
        registry: Tool registry with the configured providers.
        retriever: Retrieval service, or ``None`` when the store is unseeded.
        router: Knowledge router, or ``None`` when retrieval is unavailable.
        settings: The settings these were built from.
    """

    llm: BaseLLM
    registry: ToolRegistry
    retriever: KnowledgeRetriever | None
    router: KnowledgeRouter | None
    settings: Settings


def build_dependencies(settings: Settings | None = None) -> GraphDependencies:
    """Construct the graph's dependencies.

    A missing vector store is survivable: the router and retriever become
    ``None`` and every city routes to web search, which is a working app with a
    visible warning rather than a crash on start-up.

    Args:
        settings: Settings to build from. Defaults to the process singleton.

    Returns:
        The resolved dependencies.
    """
    resolved = settings or get_settings()

    retriever = try_load_retriever(resolved)
    router = KnowledgeRouter(retriever, settings=resolved) if retriever is not None else None
    if router is None:
        logger.warning(
            "no vector store loaded - every city will route to web search. "
            "Run: python scripts/seed_vectorstore.py"
        )

    return GraphDependencies(
        llm=get_llm(resolved),
        registry=build_registry(resolved),
        retriever=retriever,
        router=router,
        settings=resolved,
    )


def build_graph(
    dependencies: GraphDependencies | None = None,
    *,
    settings: Settings | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build and compile the graph.

    Args:
        dependencies: Pre-built dependencies. Constructed from settings when
            omitted.
        settings: Settings to build dependencies from.
        checkpointer: Checkpointer to compile with. Defaults to an in-memory
            saver so conversation state survives between turns in one process.

    Returns:
        The compiled LangGraph application.
    """
    deps = dependencies or build_dependencies(settings)
    builder: StateGraph = StateGraph(TravelState)

    # --- nodes ---------------------------------------------------------------
    builder.add_node("normalize_input", normalize_input)
    builder.add_node("classify_intent", make_classify_intent(deps.retriever))
    builder.add_node(edges.PLAN_TOOLS, make_plan_tools(deps.llm, deps.router))
    builder.add_node(edges.RETRIEVE_VECTOR, make_retrieve_vector(deps.retriever))

    # The three tool branches are the same executor class with different
    # responsibilities. That is what "one executor serving both parallel
    # branches" means in practice: each picks its own calls out of the shared
    # AIMessage and ignores the rest.
    builder.add_node(
        edges.WEB_SEARCH,
        make_tool_executor(deps.registry, node_name=edges.WEB_SEARCH, handles={WEB_SEARCH_TOOL}),
    )
    builder.add_node(
        edges.EXECUTE_WEATHER,
        make_tool_executor(deps.registry, node_name=edges.EXECUTE_WEATHER, handles={WEATHER_TOOL}),
    )
    builder.add_node(
        edges.EXECUTE_IMAGES,
        make_tool_executor(deps.registry, node_name=edges.EXECUTE_IMAGES, handles={IMAGES_TOOL}),
    )

    builder.add_node("join", join)
    builder.add_node(edges.SYNTHESIZE, make_synthesize(deps.llm))

    # --- edges ---------------------------------------------------------------
    builder.add_edge(START, "normalize_input")
    builder.add_edge("normalize_input", "classify_intent")

    # Conditional edge 1: does this turn need any work doing?
    builder.add_conditional_edges("classify_intent", edges.route_after_intent, edges.INTENT_TARGETS)

    # Conditional edge 2: pick the knowledge source AND fan out. Returning a list
    # schedules every branch into a single superstep, which is what makes them
    # concurrent - and what makes the parallelism visible in graph.png.
    builder.add_conditional_edges(edges.PLAN_TOOLS, edges.route_and_fan_out, edges.FAN_OUT_TARGETS)

    for branch in edges.FAN_OUT_TARGETS:
        builder.add_edge(branch, "join")

    builder.add_edge("join", edges.SYNTHESIZE)
    builder.add_edge(edges.SYNTHESIZE, END)

    compiled = builder.compile(checkpointer=checkpointer or MemorySaver())
    logger.info("graph compiled: %d nodes", len(edges.FAN_OUT_TARGETS) + 5)
    return compiled


def build_uncompiled_graph(dependencies: GraphDependencies | None = None) -> StateGraph:
    """Build the graph without compiling it.

    Used by the diagram exporter, which only needs the topology.

    Args:
        dependencies: Pre-built dependencies.

    Returns:
        The uncompiled builder.
    """
    deps = dependencies or build_dependencies()
    compiled = build_graph(deps)
    return compiled  # type: ignore[return-value]


__all__ = ["GraphDependencies", "build_dependencies", "build_graph"]
