"""The forecast chart.

Driven entirely by the validated :class:`TravelResponse`. The UI never parses
model prose - it reads typed fields, which is the point of the structured-output
requirement.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from travel_agent.schemas.response import ForecastPoint

# A restrained palette. The chart should read as a report, not a dashboard.
HIGH_COLOUR = "#1f4e79"
LOW_COLOUR = "#7fa8cc"
BAND_COLOUR = "rgba(31, 78, 121, 0.10)"
GRID_COLOUR = "rgba(0, 0, 0, 0.08)"


def forecast_dataframe(points: list[ForecastPoint]) -> pd.DataFrame:
    """Convert forecast points into a table for display.

    Args:
        points: Validated forecast points.

    Returns:
        A dataframe with one row per day.
    """
    return pd.DataFrame(
        [
            {
                "Date": point.date,
                "High (C)": point.temp_max_c,
                "Low (C)": point.temp_min_c,
                "Condition": point.condition,
                "Rain (%)": point.precipitation_chance,
                "Humidity (%)": point.humidity_pct,
                "Wind (km/h)": point.wind_kph,
            }
            for point in points
        ]
    )


def build_forecast_chart(points: list[ForecastPoint], city: str) -> go.Figure:
    """Build the daily temperature chart.

    The high and low are drawn as two lines with the range shaded between them,
    because a single average line hides exactly the thing a traveller wants to
    know - how cold it gets at night.

    Args:
        points: Validated forecast points.
        city: City name, used in the title.

    Returns:
        A configured Plotly figure.
    """
    dates = [point.date for point in points]
    highs = [point.temp_max_c for point in points]
    lows = [point.temp_min_c for point in points]
    conditions = [point.condition for point in points]
    rain = [point.precipitation_chance for point in points]

    figure = go.Figure()

    # The shaded band: lows drawn first, then highs filled back down to them.
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=lows,
            name="Low",
            mode="lines+markers",
            line={"color": LOW_COLOUR, "width": 2},
            marker={"size": 6},
            hovertemplate="%{x|%a %d %b}<br>Low %{y:.1f} C<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=highs,
            name="High",
            mode="lines+markers",
            line={"color": HIGH_COLOUR, "width": 2.5},
            marker={"size": 7},
            fill="tonexty",
            fillcolor=BAND_COLOUR,
            customdata=list(zip(conditions, rain, strict=True)),
            hovertemplate=(
                "%{x|%a %d %b}<br>High %{y:.1f} C<br>"
                "%{customdata[0]}<br>Rain %{customdata[1]}%<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title={"text": f"Seven-day forecast - {city}", "font": {"size": 16}},
        template="simple_white",
        height=380,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "xanchor": "right",
            "x": 1,
        },
        xaxis={
            "title": "",
            "tickformat": "%a %d %b",
            "showgrid": True,
            "gridcolor": GRID_COLOUR,
        },
        yaxis={
            "title": "Temperature (C)",
            "showgrid": True,
            "gridcolor": GRID_COLOUR,
            "zeroline": True,
            "zerolinecolor": "rgba(0,0,0,0.2)",
        },
    )
    return figure


def build_precipitation_chart(points: list[ForecastPoint]) -> go.Figure:
    """Build the precipitation-chance chart.

    Args:
        points: Validated forecast points.

    Returns:
        A configured Plotly figure.
    """
    figure = go.Figure(
        go.Bar(
            x=[point.date for point in points],
            y=[point.precipitation_chance for point in points],
            marker_color=LOW_COLOUR,
            hovertemplate="%{x|%a %d %b}<br>Rain %{y}%<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": "Chance of precipitation", "font": {"size": 14}},
        template="simple_white",
        height=220,
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
        xaxis={"title": "", "tickformat": "%a %d", "showgrid": False},
        yaxis={
            "title": "%",
            "range": [0, 100],
            "showgrid": True,
            "gridcolor": GRID_COLOUR,
        },
    )
    return figure


__all__ = ["build_forecast_chart", "build_precipitation_chart", "forecast_dataframe"]
