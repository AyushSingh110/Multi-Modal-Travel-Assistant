"""Pydantic models describing the agent's final answer.

The assignment requires the last node of the graph to emit a *structured object*
rather than markdown, containing ``city_summary``, ``weather_forecast`` and
``image_urls``. :class:`TravelResponse` is that object, and it is the only thing
the Streamlit UI is allowed to render from.

Keeping the contract in one place has a practical payoff: the UI never guesses
whether a field exists, and a malformed model answer is caught here (with a
useful error message) instead of surfacing as a blank panel.
"""

from __future__ import annotations

from datetime import UTC, datetime
from datetime import date as DateType  # noqa: N812 - avoids shadowing the 'date' field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

KnowledgeSource = Literal["vector_store", "web_search", "memory"]


class ForecastPoint(BaseModel):
    """A single day in the weather forecast.

    One instance becomes one x-axis point on the Streamlit line chart.
    """

    model_config = ConfigDict(extra="forbid")

    # Aliased import: a field literally named "date" would shadow the datetime.date
    # annotation and Pydantic cannot resolve it.
    date: DateType = Field(description="Calendar day this forecast point describes.")
    temp_max_c: float = Field(ge=-90, le=60, description="Daily high in degrees Celsius.")
    temp_min_c: float = Field(ge=-90, le=60, description="Daily low in degrees Celsius.")
    condition: str = Field(
        min_length=1, max_length=60, description="Short human label, e.g. 'Light rain'."
    )
    precipitation_chance: int = Field(
        ge=0, le=100, description="Chance of precipitation as a percentage."
    )
    humidity_pct: int | None = Field(default=None, ge=0, le=100, description="Relative humidity.")
    wind_kph: float | None = Field(default=None, ge=0, le=400, description="Wind speed in km/h.")

    @model_validator(mode="after")
    def _check_temperature_order(self) -> ForecastPoint:
        """Reject a forecast whose low exceeds its high.

        Returns:
            The validated model.

        Raises:
            ValueError: If ``temp_min_c`` is greater than ``temp_max_c``.
        """
        if self.temp_min_c > self.temp_max_c:
            raise ValueError(
                f"temp_min_c ({self.temp_min_c}) cannot exceed temp_max_c ({self.temp_max_c})"
            )
        return self


class WeatherPayload(BaseModel):
    """Everything the weather tool returns for one city."""

    model_config = ConfigDict(extra="forbid")

    city: str
    provider: str = Field(
        description="Which implementation produced this, e.g. 'mock' or 'openweather'."
    )
    current_temp_c: float | None = None
    current_condition: str | None = None
    forecast: list[ForecastPoint] = Field(default_factory=list)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_empty(self) -> bool:
        """Whether the payload carries no usable forecast data."""
        return not self.forecast


class ImageAsset(BaseModel):
    """One image returned by the image search tool."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="Direct https URL to the image.")
    caption: str = Field(default="", max_length=200)
    credit: str = Field(
        default="", max_length=120, description="Photographer or source attribution."
    )
    provider: str = "mock"

    @field_validator("url")
    @classmethod
    def _must_be_http_url(cls, value: str) -> str:
        """Ensure the URL is a real http(s) link.

        Args:
            value: Candidate URL.

        Returns:
            The validated URL.

        Raises:
            ValueError: If the URL does not use http or https.
        """
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"image url must start with http:// or https:// (got {value!r})")
        return value


class TravelResponse(BaseModel):
    """The structured object rendered by the UI.

    The three fields named by the assignment - ``city_summary``,
    ``weather_forecast`` and ``image_urls`` - are mandatory. The remaining fields
    exist so the interface can explain *how* the answer was produced (which
    knowledge source, which warnings) without the UI having to re-derive it.
    """

    model_config = ConfigDict(extra="forbid")

    city: str = Field(min_length=1, description="Resolved city name the answer is about.")
    city_summary: str = Field(
        min_length=40,
        description="Prose summary of the city. Minimum length guards against an empty answer.",
    )
    weather_forecast: list[ForecastPoint] = Field(
        default_factory=list,
        description="Five to seven daily forecast points. Empty when the weather tool failed.",
    )
    image_urls: list[str] = Field(
        default_factory=list,
        description="Direct image URLs for the gallery. Empty when the image tool failed.",
    )
    highlights: list[str] = Field(
        default_factory=list, description="Short bullet points, three to five entries."
    )
    knowledge_source: KnowledgeSource = Field(
        default="vector_store", description="Where the summary text came from."
    )
    sources: list[str] = Field(default_factory=list, description="Citations or document ids.")
    warnings: list[str] = Field(
        default_factory=list,
        description="User-facing degradation notices, e.g. 'weather unavailable'.",
    )
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("image_urls")
    @classmethod
    def _validate_image_urls(cls, value: list[str]) -> list[str]:
        """Drop anything that is not an http(s) URL.

        A hallucinated or relative URL would render as a broken image in the
        gallery, which looks worse than showing fewer pictures.

        Args:
            value: Candidate URL list.

        Returns:
            Only the well-formed URLs.
        """
        return [url for url in value if url.startswith(("http://", "https://"))]

    @property
    def is_degraded(self) -> bool:
        """Whether any part of the answer is missing because a tool failed."""
        return bool(self.warnings) or not self.weather_forecast or not self.image_urls


__all__ = [
    "ForecastPoint",
    "ImageAsset",
    "KnowledgeSource",
    "TravelResponse",
    "WeatherPayload",
]
