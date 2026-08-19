"""Tool argument schemas and their JSON-schema export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Registry names. Defined once here so the prompt, the executor and the tests can
# never disagree about what a tool is called.
WEATHER_TOOL = "get_weather_forecast"
IMAGES_TOOL = "search_city_images"
WEB_SEARCH_TOOL = "web_search"


class GetWeatherForecastArgs(BaseModel):
    """Arguments for the weather tool."""

    model_config = ConfigDict(extra="forbid")

    city: str = Field(description="City name to fetch the forecast for, e.g. 'Tokyo'.")
    days: int = Field(
        default=7,
        ge=1,
        le=14,
        description="How many days of forecast to return. The UI chart expects five to seven.",
    )
    start_date: str | None = Field(
        default=None,
        description="ISO date (YYYY-MM-DD) the forecast should start from. Omit for today.",
    )


class SearchCityImagesArgs(BaseModel):
    """Arguments for the image search tool."""

    model_config = ConfigDict(extra="forbid")

    city: str = Field(description="City name to find photographs of.")
    count: int = Field(default=4, ge=1, le=8, description="How many images to return.")


class WebSearchArgs(BaseModel):
    """Arguments for the web search tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Search query, usually '<city> travel guide overview'.")
    max_results: int = Field(default=5, ge=1, le=10, description="Maximum results to return.")


@dataclass(frozen=True)
class ToolSpec:
    """Everything the system knows about one tool.

    Attributes:
        name: Registry name advertised to the model.
        description: Natural-language description the model uses to choose it.
        args_model: Pydantic model validating the model's arguments.
    """

    name: str
    description: str
    args_model: type[BaseModel]

    def json_schema(self) -> dict[str, Any]:
        """Return the cleaned JSON schema for this tool's arguments.

        Returns:
            A JSON-schema object describing the argument shape.
        """
        return _clean_schema(self.args_model.model_json_schema())

    def to_openai_schema(self) -> dict[str, Any]:
        """Render this tool in the OpenAI/Groq function-calling wire format.

        Anthropic uses a slightly different envelope, but every LangChain chat
        model accepts this shape through ``bind_tools`` and adapts it, so one
        representation serves all four drivers.

        Returns:
            A ``{"type": "function", "function": {...}}`` dictionary.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }

    def validate_args(self, raw_args: dict[str, Any] | str) -> BaseModel:
        """Validate raw arguments from a model's tool call.

        Args:
            raw_args: The ``args`` payload. Usually a dictionary, but some
                providers hand back a JSON *string*, so both are accepted.

        Returns:
            A validated instance of :attr:`args_model`.

        Raises:
            ValueError: If a string payload is not valid JSON.
            pydantic.ValidationError: If the arguments do not fit the schema.
        """
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"tool arguments were not valid JSON: {exc}") from exc
        return self.args_model.model_validate(raw_args)


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip Pydantic bookkeeping that models do not need.

    Pydantic emits a ``title`` for the model and for every field. Those add tokens
    and occasionally confuse smaller models into treating the title as guidance,
    so they are removed recursively while descriptions are kept.

    Args:
        schema: Raw output of ``model_json_schema()``.

    Returns:
        A cleaned copy of the schema.
    """
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if isinstance(value, dict):
            cleaned[key] = _clean_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _clean_schema(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


TOOL_SPECS: dict[str, ToolSpec] = {
    WEATHER_TOOL: ToolSpec(
        name=WEATHER_TOOL,
        description=(
            "Get the daily weather forecast for a city. Returns high and low "
            "temperatures in Celsius, a condition label and precipitation chance "
            "for each day."
        ),
        args_model=GetWeatherForecastArgs,
    ),
    IMAGES_TOOL: ToolSpec(
        name=IMAGES_TOOL,
        description="Find high-quality photographs of a city for a visual gallery.",
        args_model=SearchCityImagesArgs,
    ),
    WEB_SEARCH_TOOL: ToolSpec(
        name=WEB_SEARCH_TOOL,
        description=(
            "Search the public web for facts about a place. Use this only when the "
            "internal knowledge base has no entry for the city."
        ),
        args_model=WebSearchArgs,
    ),
}


def openai_tool_schemas(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Export tool schemas in the format passed to a chat model.

    Args:
        names: Restrict the export to these tool names. ``None`` exports all of
            them.

    Returns:
        A list of function-schema dictionaries.

    Raises:
        KeyError: If an unknown tool name is requested.
    """
    selected = names if names is not None else list(TOOL_SPECS)
    return [TOOL_SPECS[name].to_openai_schema() for name in selected]


__all__ = [
    "IMAGES_TOOL",
    "TOOL_SPECS",
    "WEATHER_TOOL",
    "WEB_SEARCH_TOOL",
    "GetWeatherForecastArgs",
    "SearchCityImagesArgs",
    "ToolSpec",
    "WebSearchArgs",
    "openai_tool_schemas",
]
