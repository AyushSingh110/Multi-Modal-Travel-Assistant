"""The agent trace panel.

This is the highest-leverage view in the application. Everything the graph
decided is otherwise invisible: a reviewer sees a summary, some photographs and a
chart, and has no way to tell whether the answer came from the knowledge base or
the web, whether the branches ran concurrently, or what a follow-up actually
skipped.

The panel makes each of those legible in a few seconds, and - importantly - shows
the *numbers behind the decisions* rather than just the verdicts. "Routed to the
vector store" is a claim. "Routed to the vector store; exact name match on
'Tokyo'; similarity 0.207 against a threshold of 0.07" is an explanation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from travel_agent.schemas.trace import ParallelMetrics, ThresholdDiagnostics, TraceEvent
from travel_agent.services.llm.base import ModelCheck

ROUTE_LABELS = {
    "vector": "Internal knowledge base",
    "web": "Live web search",
    "clarify": "Asked for clarification",
}

MATCH_REASON_LABELS = {
    "exact": "exact name match",
    "similarity": "similarity score",
    "none": "no city resolved",
}


def render_route_decision(state: dict[str, Any]) -> None:
    """Explain where the facts came from, with the score that decided it.

    Args:
        state: The final graph state for the turn.
    """
    route = state.get("route")
    if route is None:
        st.caption("No routing decision was needed on this turn.")
        return

    score = state.get("route_score") or 0.0
    threshold = state.get("route_threshold") or 0.0
    match_reason = state.get("route_match_reason", "similarity")
    matched_city = state.get("matched_city")

    st.markdown(f"**Source:** {ROUTE_LABELS.get(route, route)}")

    detail = f"Decided by {MATCH_REASON_LABELS.get(match_reason, match_reason)}"
    if match_reason == "exact" and matched_city:
        detail += f" on '{matched_city}'"
    detail += f". Similarity {score:.3f} against a threshold of {threshold:.2f}."
    st.caption(detail)

    scores = state.get("route_all_scores") or {}
    if scores:
        table = pd.DataFrame(
            [
                {
                    "City in knowledge base": city,
                    "Similarity": round(value, 3),
                    "Above threshold": "yes" if value >= threshold else "no",
                }
                for city, value in scores.items()
            ]
        )
        st.dataframe(table, hide_index=True, width="stretch")


def render_tool_activity(state: dict[str, Any]) -> None:
    """List the tools that fired, their durations and any failures.

    Args:
        state: The final graph state for the turn.
    """
    events = [event for event in state.get("trace", []) if event.kind in {"tool", "error"}]
    if not events:
        st.caption("No tools ran on this turn.")
        return

    rows = []
    for event in events:
        data = event.data or {}
        rows.append(
            {
                "Tool": data.get("tool", event.node),
                "Node": event.node,
                "Provider": data.get("provider", "-"),
                "Result": "error" if event.kind == "error" else "ok",
                "Duration (ms)": round(event.duration_ms or 0.0, 1),
                "Attempts": data.get("attempts", 1),
            }
        )

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    for event in events:
        if event.kind == "error":
            st.error(f"{event.data.get('tool', event.node)}: {event.data.get('error', '')}")


def render_parallel_metrics(metrics: ParallelMetrics | None) -> None:
    """Show the measured cost of the fan-out.

    Args:
        metrics: The join node's measurement, or ``None``.
    """
    if metrics is None or not metrics.branch_durations_ms:
        st.caption("No parallel branches ran on this turn.")
        return

    left, middle, right = st.columns(3)
    left.metric("Sequential equivalent", f"{metrics.sequential_equivalent_ms:.0f} ms")
    middle.metric("Actual wall clock", f"{metrics.parallel_wall_clock_ms:.0f} ms")
    right.metric("Speed-up", f"{metrics.speedup:.2f}x")

    st.caption(
        "The branches ran together in one superstep, so the turn cost roughly the "
        "slowest branch rather than the sum of all of them."
    )

    st.dataframe(
        pd.DataFrame(
            [
                {"Branch": node, "Duration (ms)": duration}
                for node, duration in metrics.branch_durations_ms.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def render_skipped_work(state: dict[str, Any]) -> None:
    """Show what a follow-up turn deliberately did not do.

    The framing here is deliberate and honest. Skipping these branches saves
    *work*, not much latency: they previously ran concurrently with the weather
    branch, so removing them barely shortens the turn. The win is the provider
    calls, quota and money not spent - which is the win that scales.

    Args:
        state: The final graph state for the turn.
    """
    skipped = state.get("skipped_nodes") or []
    if not skipped:
        st.caption("Nothing was skipped: this turn ran the full graph.")
        return

    saved = state.get("skipped_ms_saved") or 0.0
    st.markdown(f"**Skipped:** {', '.join(skipped)}")
    st.caption(
        f"About {saved:.0f} ms of provider work avoided, measured from the previous "
        f"turn. This is a saving in work and API cost rather than in wall clock - "
        f"those branches previously ran in parallel with the weather branch, so not "
        f"running them frees quota more than it frees time."
    )


def render_timeline(state: dict[str, Any]) -> None:
    """Show every node that ran, in order, with its duration.

    Args:
        state: The final graph state for the turn.
    """
    timings = state.get("timings") or {}
    if not timings:
        st.caption("No node timings were recorded.")
        return

    table = pd.DataFrame(
        [{"Node": node, "Duration (ms)": round(ms, 1)} for node, ms in timings.items()]
    ).sort_values("Duration (ms)", ascending=False)
    st.dataframe(table, hide_index=True, width="stretch")


def render_event_log(events: list[TraceEvent]) -> None:
    """Show the raw ordered event log.

    Args:
        events: Trace events from the turn.
    """
    if not events:
        st.caption("No events were recorded.")
        return

    rows = [
        {
            "Node": event.node,
            "Kind": event.kind,
            "Message": event.message,
            "Duration (ms)": round(event.duration_ms, 1) if event.duration_ms else None,
        }
        for event in events
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_startup_checks(
    model_check: ModelCheck | None,
    threshold_check: ThresholdDiagnostics | None,
    checkpointer_kind: str,
    checkpointer_location: str,
) -> None:
    """Show the start-up diagnostics that would otherwise sit in a log file.

    Args:
        model_check: Result of the provider's model availability check.
        threshold_check: The router's threshold verdict.
        checkpointer_kind: ``memory`` or ``sqlite``.
        checkpointer_location: Where conversation state is stored.
    """
    if model_check is not None:
        if model_check.available:
            st.success(f"Model check: {model_check.message}")
        else:
            st.warning(f"Model check: {model_check.message}")

    if threshold_check is not None:
        if threshold_check.is_healthy:
            st.success(f"Router threshold: {threshold_check.message}")
        else:
            st.warning(f"Router threshold: {threshold_check.message}")

    if checkpointer_kind == "sqlite":
        st.success(f"Memory: durable SQLite at {checkpointer_location}")
    else:
        st.info(
            "Memory: in-process only. Conversations are kept for this session and "
            "lost when the app restarts. Set CHECKPOINTER=sqlite for durable memory."
        )


def render_trace_panel(state: dict[str, Any], runtime: Any) -> None:
    """Render the whole panel.

    Args:
        state: The final graph state for the turn.
        runtime: The agent runtime, for start-up diagnostics.
    """
    st.subheader("Agent trace")
    st.caption("What the graph decided on this turn, and the measurements behind each decision.")

    routing, tools, parallel, memory, startup = st.tabs(
        ["Routing", "Tools", "Parallelism", "Memory", "Start-up"]
    )

    with routing:
        render_route_decision(state)

    with tools:
        render_tool_activity(state)
        st.markdown("**Node timings**")
        render_timeline(state)

    with parallel:
        render_parallel_metrics(state.get("parallel_metrics"))

    with memory:
        render_skipped_work(state)
        usage = state.get("token_usage")
        if usage is not None and usage.total_tokens:
            st.markdown("**Model usage this turn**")
            st.caption(
                f"{usage.total_tokens} tokens across {usage.llm_calls} call(s) "
                f"using {usage.model}. Token counts are shown rather than a cost, "
                f"because the three supported providers price differently."
            )

    with startup:
        render_startup_checks(
            runtime.model_check,
            runtime.threshold_check,
            runtime.checkpointer.kind,
            runtime.checkpointer.location,
        )

    with st.expander("Full event log"):
        render_event_log(state.get("trace", []))


__all__ = ["render_trace_panel"]
