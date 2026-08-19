"""The sidebar: configuration, conversation control and the failure toggle.

Every control here mutates the runtime's live ``Settings`` object rather than
rebuilding the runtime. That matters: rebuilding would tear down the event loop
and the conversation history along with it, so flipping "break the weather API"
would silently reset the demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st

FAILURE_MODES = ["server_error", "timeout", "rate_limit", "malformed"]
FAILURE_MODE_HELP = {
    "server_error": "HTTP 500 from the provider. Retried, then reported.",
    "timeout": "The provider never responds. The per-attempt deadline fires.",
    "rate_limit": "HTTP 429 with Retry-After. The advertised delay is honoured.",
    "malformed": "HTTP 200 with an unusable body. Deliberately not retried.",
}
IMAGE_MODES = ["auto", "remote", "local"]
IMAGE_MODE_HELP = {
    "auto": "Check once whether the image host is reachable, fall back if not.",
    "remote": "Always load from the remote host.",
    "local": "Always use the images bundled with this repository.",
}


@dataclass
class SidebarState:
    """What the user selected this rerun.

    Attributes:
        thread_id: The conversation to run against.
        reset_requested: Whether the user asked to start a new conversation.
    """

    thread_id: str
    reset_requested: bool = False


def render_sidebar(runtime: Any) -> SidebarState:
    """Draw the sidebar and apply its settings to the live runtime.

    Args:
        runtime: The agent runtime whose settings are being adjusted.

    Returns:
        The conversation selection for this rerun.
    """
    settings = runtime.settings

    with st.sidebar:
        st.subheader("Configuration")

        provider = runtime.dependencies.llm.name
        model = runtime.dependencies.llm.model_id
        st.text(f"Model provider: {provider}")
        st.text(f"Model: {model}")
        if provider == "mock":
            st.caption(
                "No API key is configured, so the deterministic mock model is in use. "
                "It emits real tool-calling payloads, so every downstream code path "
                "is the same one a live model would exercise."
            )

        st.text(f"Weather provider: {runtime.dependencies.registry.weather.name}")
        st.text(f"Image provider: {runtime.dependencies.registry.images.name}")
        st.text(f"Search provider: {runtime.dependencies.registry.search.name}")
        st.text(f"Vector store: {'loaded' if runtime.dependencies.retriever else 'not seeded'}")
        st.text(f"Memory: {runtime.checkpointer.kind}")

        st.divider()
        st.subheader("Conversation")

        thread_id = st.text_input(
            "Thread id",
            value=st.session_state.get("thread_id", "session-1"),
            help=(
                "Conversations are scoped by this value. Change it to start a "
                "separate conversation that shares no history with this one."
            ),
        )
        reset = st.button("Start a new conversation", width="stretch")

        st.divider()
        st.subheader("Failure simulation")
        st.caption(
            "The rubric asks whether the app survives a failing API. These toggles "
            "break one on demand so the answer can be demonstrated rather than claimed."
        )

        settings.force_weather_failure = st.toggle(
            "Break the weather API",
            value=settings.force_weather_failure,
            help="The weather tool will fail on every request until this is turned off.",
        )
        settings.weather_failure_mode = st.selectbox(  # type: ignore[assignment]
            "Weather failure mode",
            FAILURE_MODES,
            index=FAILURE_MODES.index(settings.weather_failure_mode),
            disabled=not settings.force_weather_failure,
            help="How the weather provider should fail.",
        )
        st.caption(FAILURE_MODE_HELP[settings.weather_failure_mode])

        settings.force_image_failure = st.toggle(
            "Break the image API",
            value=settings.force_image_failure,
            help="The image tool will fail on every request until this is turned off.",
        )

        st.divider()
        st.subheader("Images")
        settings.image_fallback_mode = st.selectbox(  # type: ignore[assignment]
            "Image source",
            IMAGE_MODES,
            index=IMAGE_MODES.index(settings.image_fallback_mode),
            help="Where gallery images are loaded from.",
        )
        st.caption(IMAGE_MODE_HELP[settings.image_fallback_mode])

        st.divider()
        st.subheader("Retrieval")
        # The configured value can sit outside the useful range - a stale .env
        # carrying 0.55 is exactly how the silent misrouting bug happened - so the
        # slider widens to include it rather than raising on an out-of-range value.
        configured = float(settings.router_similarity_threshold)
        settings.router_similarity_threshold = st.slider(
            "Router similarity threshold",
            min_value=0.0,
            max_value=max(0.30, round(configured + 0.05, 2)),
            value=configured,
            step=0.01,
            help=(
                "A city scoring below this is treated as unknown and routed to web "
                "search. Seeded cities score 0.10 to 0.21 and unseeded ones 0.00 to "
                "0.04, so 0.07 sits inside the gap."
            ),
        )
        if runtime.dependencies.router is not None:
            runtime.dependencies.router.threshold = settings.router_similarity_threshold

    return SidebarState(thread_id=thread_id, reset_requested=reset)


__all__ = ["SidebarState", "render_sidebar"]
