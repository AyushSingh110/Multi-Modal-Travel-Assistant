"""Tests for tool argument schemas and their JSON-schema export."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from travel_agent.schemas.tools import (
    IMAGES_TOOL,
    TOOL_SPECS,
    WEATHER_TOOL,
    WEB_SEARCH_TOOL,
    GetWeatherForecastArgs,
    openai_tool_schemas,
)


def test_every_registered_tool_exports_openai_shape() -> None:
    schemas = openai_tool_schemas()

    assert len(schemas) == len(TOOL_SPECS)
    for schema in schemas:
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"] in TOOL_SPECS
        assert function["description"]
        assert function["parameters"]["type"] == "object"


def test_export_can_be_restricted_to_named_tools() -> None:
    schemas = openai_tool_schemas([WEATHER_TOOL])

    assert [schema["function"]["name"] for schema in schemas] == [WEATHER_TOOL]


def test_exported_schema_is_json_serialisable() -> None:
    """The schema is sent over the wire, so it must survive json.dumps."""
    payload = json.dumps(openai_tool_schemas())

    assert WEATHER_TOOL in payload
    assert IMAGES_TOOL in payload
    assert WEB_SEARCH_TOOL in payload


def test_pydantic_titles_are_stripped_but_descriptions_survive() -> None:
    parameters = TOOL_SPECS[WEATHER_TOOL].json_schema()

    assert "title" not in parameters
    for field_schema in parameters["properties"].values():
        assert "title" not in field_schema
    assert parameters["properties"]["city"]["description"]


def test_required_and_optional_fields_are_declared_correctly() -> None:
    parameters = TOOL_SPECS[WEATHER_TOOL].json_schema()

    assert parameters["required"] == ["city"]
    assert "days" in parameters["properties"]
    assert parameters["additionalProperties"] is False


def test_schema_round_trip_schema_to_llm_payload_to_validated_args() -> None:
    """Schema -> LLM-shaped tool call -> validated, typed arguments."""
    spec = TOOL_SPECS[WEATHER_TOOL]
    exported = spec.to_openai_schema()

    # What a model sends back, shaped by the schema above.
    llm_tool_call = {
        "id": "call_abc123",
        "name": exported["function"]["name"],
        "args": {"city": "Tokyo", "days": 5},
        "type": "tool_call",
    }

    validated = spec.validate_args(llm_tool_call["args"])

    assert isinstance(validated, GetWeatherForecastArgs)
    assert validated.city == "Tokyo"
    assert validated.days == 5
    assert validated.start_date is None  # default applied, not invented by the model


def test_arguments_arriving_as_a_json_string_are_accepted() -> None:
    """Some providers return args as a JSON string rather than an object."""
    validated = TOOL_SPECS[IMAGES_TOOL].validate_args('{"city": "Paris", "count": 3}')

    assert validated.city == "Paris"
    assert validated.count == 3


def test_malformed_json_string_raises_a_clear_error() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        TOOL_SPECS[IMAGES_TOOL].validate_args("{city: Paris")


def test_missing_required_argument_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TOOL_SPECS[WEATHER_TOOL].validate_args({"days": 7})


def test_out_of_range_argument_is_rejected() -> None:
    """The model asking for 99 days must fail here, not inside the tool."""
    with pytest.raises(ValidationError):
        TOOL_SPECS[WEATHER_TOOL].validate_args({"city": "Paris", "days": 99})


def test_unexpected_argument_is_rejected() -> None:
    """extra='forbid' catches a hallucinated parameter instead of ignoring it."""
    with pytest.raises(ValidationError):
        TOOL_SPECS[WEATHER_TOOL].validate_args({"city": "Paris", "units": "fahrenheit"})


def test_string_integer_is_coerced_to_int() -> None:
    """Models often send numbers as strings; Pydantic normalises that for us."""
    validated = TOOL_SPECS[WEATHER_TOOL].validate_args({"city": "Paris", "days": "6"})

    assert validated.days == 6
    assert isinstance(validated.days, int)
