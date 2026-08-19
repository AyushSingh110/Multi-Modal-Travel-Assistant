"""Streamlit entry point: streamlit run src/travel_agent/ui/app.py."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# `streamlit run` runs this file as a plain script, so src/ is not on the import
# path the way it is for pytest or for scripts/. Put it there before the first
# travel_agent import, so the app runs from a clone with no install step.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import streamlit as st  # noqa: E402

from travel_agent.config.settings import Settings, get_settings  # noqa: E402
from travel_agent.logging_setup import get_logger  # noqa: E402
from travel_agent.ui.components.answer import render_answer  # noqa: E402
from travel_agent.ui.components.sidebar import render_sidebar  # noqa: E402
from travel_agent.ui.components.trace_panel import render_trace_panel  # noqa: E402
from travel_agent.ui.runner import AgentRuntime, build_runtime  # noqa: E402

logger = get_logger(__name__)

PRESETS: list[tuple[str, str, str]] = [
    (
        "In-store city",
        "Tell me about Tokyo",
        "Routes to the internal knowledge base by exact name match.",
    ),
    (
        "Out-of-store city",
        "Tell me about Kyoto",
        "Scores below the threshold, so it routes to web search instead.",
    ),
    (
        "Follow-up",
        "what about next week?",
        "Keeps the city from memory and re-runs only the weather branch.",
    ),
    (
        "No city named",
        "what about next week?",
        "On a fresh thread there is nothing to carry over, so it asks rather than guesses.",
    ),
]


@st.cache_resource(show_spinner=False)
def get_runtime(provider: str) -> AgentRuntime:
    """Build the runtime once per session and keep it across reruns.

    The cache key is the model provider, because changing that genuinely requires
    a new graph. Everything else the sidebar controls is applied by mutating the
    runtime's live settings, which leaves the loop and the conversation intact.

    Args:
        provider: Resolved model provider name, used as the cache key.

    Returns:
        The cached runtime.
    """
    logger.info("building the agent runtime (provider=%s)", provider)
    return build_runtime(get_settings())


def _configure_page() -> None:
    """Apply page configuration and a small amount of restrained styling."""
    st.set_page_config(
        page_title="Multi-Modal Travel Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2.2rem; max-width: 1250px; }
          h1 { font-size: 1.9rem; font-weight: 600; }
          h2 { font-size: 1.35rem; font-weight: 600; }
          h3 { font-size: 1.1rem; font-weight: 600; }
          [data-testid="stMetricValue"] { font-size: 1.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(runtime: AgentRuntime, settings: Settings) -> None:
    """Draw the page header and the state-of-the-system line.

    Args:
        runtime: The agent runtime.
        settings: Live settings.
    """
    st.title("Multi-Modal Travel Assistant")
    st.caption(
        "A LangGraph agent that decides where to source facts, fetches weather and "
        "images concurrently, and remembers the conversation."
    )

    if settings.force_weather_failure or settings.force_image_failure:
        broken = []
        if settings.force_weather_failure:
            broken.append(f"weather ({settings.weather_failure_mode})")
        if settings.force_image_failure:
            broken.append(f"images ({settings.image_failure_mode})")
        st.warning(
            f"Failure simulation is active for: {', '.join(broken)}. "
            f"The affected data will be missing and the rest of the answer should "
            f"still render."
        )

    if runtime.dependencies.retriever is None:
        st.error(
            "No vector store is loaded, so every city will route to web search. "
            "Run: python scripts/seed_vectorstore.py"
        )

    # A misconfigured threshold is the failure that hides itself: exact name
    # matches still work, so the app looks healthy while the similarity path is
    # dead and every unrecognised name goes to the web. It gets a banner on the
    # page, not just a line in the Start-up tab.
    threshold_check = runtime.threshold_check
    if threshold_check is not None and not threshold_check.is_healthy:
        st.warning(threshold_check.message)


def _render_query_form(thread_id: str) -> str | None:
    """Draw the query input and the preset buttons.

    Args:
        thread_id: The active conversation.

    Returns:
        The query to run, or ``None`` when the user has not asked for anything.
    """
    st.subheader("Ask about a destination")

    with st.form("query", clear_on_submit=False):
        query = st.text_input(
            "Question",
            placeholder="Tell me about Tokyo",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send", type="primary")

    st.caption("Demo presets, in the order that exercises every path:")
    columns = st.columns(len(PRESETS))
    for column, (label, preset_query, help_text) in zip(columns, PRESETS, strict=True):
        with column:
            if st.button(label, help=f"{preset_query} - {help_text}", width="stretch"):
                if label == "No city named":
                    # Deliberately switches to an unused conversation, because the
                    # point of this preset is having no history to fall back on.
                    st.session_state["thread_id"] = f"{thread_id}-fresh"
                return preset_query

    if submitted and query.strip():
        return query.strip()
    return None


def main() -> None:
    """Run one pass of the Streamlit script."""
    _configure_page()

    settings = get_settings()
    runtime = get_runtime(settings.resolve_llm_provider())

    sidebar = render_sidebar(runtime)
    if sidebar.reset_requested:
        st.session_state["thread_id"] = f"session-{st.session_state.get('reset_count', 0) + 1}"
        st.session_state["reset_count"] = st.session_state.get("reset_count", 0) + 1
        st.session_state.pop("last_state", None)
        st.rerun()

    st.session_state.setdefault("thread_id", sidebar.thread_id)
    if sidebar.thread_id != st.session_state["thread_id"]:
        st.session_state["thread_id"] = sidebar.thread_id
        st.session_state.pop("last_state", None)

    thread_id = st.session_state["thread_id"]

    _render_header(runtime, settings)
    query = _render_query_form(thread_id)

    if query:
        active_thread = st.session_state.get("thread_id", thread_id)
        with st.spinner(f"Running the graph for: {query}"):
            try:
                state: dict[str, Any] = runtime.invoke(query, thread_id=active_thread)
                st.session_state["last_state"] = state
                st.session_state["last_query"] = query
            except TimeoutError:
                st.error(
                    "The request took too long and was cancelled. The app is still "
                    "running - try again, or check whether a live provider is slow."
                )
            except Exception as exc:  # noqa: BLE001 - a banner beats a stack trace
                logger.exception("turn failed")
                st.error(f"The request failed: {type(exc).__name__}: {exc}")

    last_state: dict[str, Any] | None = st.session_state.get("last_state")
    if not last_state:
        st.info(
            "Ask about a city, or use one of the presets above. Tokyo, Paris and "
            "New York are in the knowledge base; anything else routes to web search."
        )
        return

    st.divider()
    answer_column, trace_column = st.columns([3, 2], gap="large")

    with answer_column:
        render_answer(last_state)

    with trace_column:
        render_trace_panel(last_state, runtime)


main()
