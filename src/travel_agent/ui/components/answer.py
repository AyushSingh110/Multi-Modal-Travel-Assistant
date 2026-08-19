"""Rendering of the answer itself: summary, gallery, forecast and warnings.

Everything here reads the validated :class:`TravelResponse`. No markdown from the
model is rendered directly, which is the practical benefit of the structured
output requirement: the layout is fixed and known, and a missing field is a
visible gap rather than a broken page.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from travel_agent.schemas.response import ImageAsset, TravelResponse
from travel_agent.ui.components.charts import (
    build_forecast_chart,
    build_precipitation_chart,
    forecast_dataframe,
)

GALLERY_COLUMNS = 4


def render_warnings(response: TravelResponse) -> None:
    """Show degradation notices as banners rather than stack traces.

    Args:
        response: The validated response.
    """
    for warning in response.warnings:
        st.warning(warning)


def render_summary(response: TravelResponse) -> None:
    """Render the city summary and its highlights.

    Args:
        response: The validated response.
    """
    st.markdown(response.city_summary)

    if response.highlights:
        st.markdown("**Covered in the source material**")
        st.markdown("\n".join(f"- {item}" for item in response.highlights))


def render_gallery(images: list[ImageAsset], response: TravelResponse) -> None:
    """Render the image gallery.

    Falls back to the bundled copies when the provider reported the remote host
    unreachable, so a blocked network produces a complete layout rather than a
    grid of broken images.

    Args:
        images: Image assets from state, carrying local fallbacks.
        response: The validated response, used when no assets are available.
    """
    if not images and not response.image_urls:
        st.info("No images were available for this destination on this turn.")
        return

    if images:
        sources = [(asset.display_source, asset.caption, asset.credit) for asset in images]
        using_local = any(asset.prefer_local and asset.local_path for asset in images)
    else:
        sources = [(url, "", "") for url in response.image_urls]
        using_local = False

    if using_local:
        st.caption(
            "Showing bundled images: the remote image host could not be reached "
            "from this machine."
        )

    columns = st.columns(min(GALLERY_COLUMNS, len(sources)))
    for column, (source, caption, credit) in zip(columns, sources, strict=False):
        with column:
            label = caption or ""
            if credit:
                label = f"{label} - {credit}" if label else credit
            if source.startswith("http"):
                st.image(source, caption=label, width="stretch")
            elif Path(source).exists():
                st.image(str(source), caption=label, width="stretch")


def render_forecast(response: TravelResponse) -> None:
    """Render the forecast chart and its underlying table.

    Args:
        response: The validated response.
    """
    if not response.weather_forecast:
        st.warning(
            "The forecast is unavailable for this turn. The rest of the answer is "
            "unaffected - the weather branch failed independently of the others."
        )
        return

    st.plotly_chart(
        build_forecast_chart(response.weather_forecast, response.city or "this destination"),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        build_precipitation_chart(response.weather_forecast),
        width="stretch",
        config={"displayModeBar": False},
    )

    with st.expander("Forecast data"):
        st.dataframe(
            forecast_dataframe(response.weather_forecast), hide_index=True, width="stretch"
        )


def render_sources(response: TravelResponse) -> None:
    """List where the summary's facts came from.

    Args:
        response: The validated response.
    """
    if not response.sources:
        return

    st.markdown("**Sources**")
    for source in response.sources:
        if source.startswith("http"):
            st.markdown(f"- [{source}]({source})")
        else:
            st.markdown(f"- {source}")


def render_answer(state: dict[str, Any]) -> None:
    """Render a complete answer from the graph state.

    Args:
        state: The final graph state for the turn.
    """
    response: TravelResponse | None = state.get("response")
    if response is None:
        st.info("Ask about a city to get started.")
        return

    if response.is_clarification:
        st.info(response.city_summary)
        return

    heading = response.city or "Destination"
    st.header(heading)

    source_label = {
        "vector_store": "Answered from the internal knowledge base",
        "web_search": "Answered from a live web search",
        "memory": "Answered from the conversation so far",
    }.get(response.knowledge_source, response.knowledge_source)
    st.caption(source_label)

    render_warnings(response)
    render_summary(response)

    st.divider()
    render_gallery(state.get("images") or [], response)

    st.divider()
    render_forecast(response)

    render_sources(response)


__all__ = ["render_answer", "render_forecast", "render_gallery", "render_summary"]
