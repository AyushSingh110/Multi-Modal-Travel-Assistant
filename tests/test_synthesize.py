"""Tests for the synthesis node - assignment section 2.C.

Grounding gets more attention here than prose quality, and deliberately so. The
"Now tell me about Kyoto" bug produced a complete, confident, well-written answer
about a city that does not exist. Nothing raised, every tool succeeded. The only
defence against that class of failure is refusing to write anything the retrieved
context does not support - so that is what these tests check.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AnyMessage

from travel_agent.graph.nodes.synthesize import SynthesisDraft, _validate, make_synthesize
from travel_agent.schemas.knowledge import KnowledgeChunk
from travel_agent.schemas.response import ForecastPoint, ImageAsset, TravelResponse, WeatherPayload
from travel_agent.schemas.trace import TokenUsage
from travel_agent.services.llm.base import BaseLLM, LLMCall
from travel_agent.services.llm.mock import MockLLM


# ============================================================ test doubles ====
class ScriptedLLM(BaseLLM):
    """Returns pre-arranged replies so each path can be driven exactly."""

    name = "scripted"
    model_id = "scripted-llm"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def plan(self, messages: list[AnyMessage], tools: list[dict[str, Any]]) -> LLMCall:
        return LLMCall(message=AIMessage(content=""), usage=TokenUsage())

    async def complete_json(self, messages: list[AnyMessage]) -> tuple[str, TokenUsage]:
        self.prompts.append(str(messages[-1].content))
        reply = self.replies.pop(0) if self.replies else "{}"
        return reply, TokenUsage(total_tokens=42, llm_calls=1, model=self.model_id)


class ExplodingLLM(BaseLLM):
    """Fails every call, standing in for a dead provider."""

    name = "exploding"
    model_id = "exploding-llm"

    async def plan(self, messages: list[AnyMessage], tools: list[dict[str, Any]]) -> LLMCall:
        raise RuntimeError("provider unavailable")

    async def complete_json(self, messages: list[AnyMessage]) -> tuple[str, TokenUsage]:
        raise RuntimeError("provider unavailable")


def _chunks(count: int = 4, city: str = "Tokyo") -> list[KnowledgeChunk]:
    sections = [
        ("Orientation and layout", "Tokyo is a chain of centres along the JR Yamanote loop line."),
        ("Getting around", "Rail does almost everything and trains stop around midnight."),
        ("Food and drink", "The everyday food is the story: ramen, standing soba, depachika."),
        ("Seasonality", "Cherry blossom peaks in late March and August is hot and humid."),
    ]
    return [
        KnowledgeChunk(
            chunk_id=f"{city.lower()}::{index}",
            city=city,
            section=section,
            text=f"{section}. {body}",
            source=f"{city.lower()}.md",
        )
        for index, (section, body) in enumerate(sections[:count])
    ]


def _weather(city: str = "Tokyo") -> WeatherPayload:
    return WeatherPayload(
        city=city,
        provider="mock",
        forecast=[
            ForecastPoint(
                date=date(2026, 8, 19 + offset),
                temp_max_c=30 - offset,
                temp_min_c=23 - offset,
                condition="Partly cloudy",
                precipitation_chance=40,
            )
            for offset in range(7)
        ],
    )


def _images(count: int = 4) -> list[ImageAsset]:
    return [
        ImageAsset(url=f"https://example.com/{index}.jpg", caption=f"image {index}")
        for index in range(count)
    ]


def _state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "city": "Tokyo",
        "intent": "new_city",
        "route": "vector",
        "knowledge": _chunks(),
        "weather": _weather(),
        "images": _images(),
        "errors": [],
    }
    base.update(overrides)
    return base


GOOD_REPLY = (
    '{"city_summary": "Tokyo is a chain of centres strung along the JR Yamanote loop '
    "line rather than a single downtown. Rail does almost everything, and the trains "
    'stop around midnight. The everyday food is the real story.", '
    '"highlights": ["Orientation", "Getting around", "Food and drink"]}'
)


# ================================================================ happy path ==
async def test_a_valid_reply_becomes_a_validated_response() -> None:
    node = make_synthesize(ScriptedLLM([GOOD_REPLY]))

    update = await node(_state())
    response: TravelResponse = update["response"]

    assert isinstance(response, TravelResponse)
    assert response.city == "Tokyo"
    assert "Yamanote" in response.city_summary
    assert len(response.weather_forecast) == 7
    assert len(response.image_urls) == 4
    assert response.highlights == ["Orientation", "Getting around", "Food and drink"]
    assert not response.warnings


async def test_the_forecast_and_images_are_not_model_output() -> None:
    """The model writes prose; the data comes from the tools that fetched it.

    Asking a model to echo structured values back would only create a chance for
    it to alter them, so the schema it fills in does not contain them at all.
    """
    llm = ScriptedLLM([GOOD_REPLY])
    node = make_synthesize(llm)

    update = await node(_state())

    assert update["response"].weather_forecast[0].temp_max_c == 30
    assert update["response"].image_urls == [asset.url for asset in _images()]
    assert "city_summary" in llm.prompts[0] or "FACTS" in llm.prompts[0]


async def test_the_prompt_carries_the_retrieved_passages() -> None:
    llm = ScriptedLLM([GOOD_REPLY])

    await make_synthesize(llm)(_state())

    prompt = llm.prompts[0]
    assert "FACTS:" in prompt
    assert "Yamanote" in prompt, "the retrieved passages must reach the model"
    assert "Getting around" in prompt


# ================================================================ grounding ==
async def test_sparse_context_is_flagged_to_the_model() -> None:
    """With almost nothing retrieved, the model is told to say so."""
    llm = ScriptedLLM([GOOD_REPLY])

    await make_synthesize(llm)(_state(knowledge=_chunks(1)))

    assert "SPARSE" in llm.prompts[0]
    assert "limited information" in llm.prompts[0].lower()


async def test_empty_context_degrades_rather_than_confabulating() -> None:
    """The defence against the confident-wrong-answer failure mode.

    With no retrieved material and a model that cannot help, the deterministic
    fallback must say there is limited information - not produce fluent prose
    about a place nothing is known about.
    """
    node = make_synthesize(ExplodingLLM())

    update = await node(_state(knowledge=[], city="Ulaanbaatar"))
    summary = update["response"].city_summary.lower()

    assert "limited information" in summary
    assert "ulaanbaatar" in summary
    assert len(update["response"].city_summary) < 400, "a fallback should be brief, not padded"


async def test_the_fallback_summary_only_uses_retrieved_sentences() -> None:
    """Every sentence in the deterministic summary must come from the corpus."""
    node = make_synthesize(ExplodingLLM())
    chunks = _chunks()

    update = await node(_state(knowledge=chunks))
    summary = update["response"].city_summary

    corpus = " ".join(chunk.text for chunk in chunks)
    for sentence in summary.split(". ")[1:]:
        core = sentence.strip().rstrip(".")
        if len(core) > 40:
            assert core in corpus, f"invented sentence in the fallback: {core!r}"


async def test_the_response_records_which_sources_it_used() -> None:
    update = await make_synthesize(ScriptedLLM([GOOD_REPLY]))(_state())

    assert update["response"].sources == ["tokyo.md"] * 4
    assert update["trace"][-1].data["grounded_in_chunks"] == 4


# ============================================================== validation ==
def test_valid_json_is_accepted() -> None:
    draft = _validate(GOOD_REPLY)

    assert isinstance(draft, SynthesisDraft)
    assert draft.city_summary.startswith("Tokyo")


def test_a_fenced_code_block_is_unwrapped() -> None:
    """Models wrap JSON in markdown fences more often than any prompt prevents."""
    assert _validate(f"```json\n{GOOD_REPLY}\n```") is not None


def test_prose_around_the_json_is_tolerated() -> None:
    assert (
        _validate(f"Here is the summary you asked for:\n{GOOD_REPLY}\nHope that helps.") is not None
    )


@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        "{}",
        '{"city_summary": "too short"}',
        '{"highlights": ["no summary field"]}',
        '{"city_summary": null}',
        "",
    ],
)
def test_unusable_replies_are_rejected(reply: str) -> None:
    assert _validate(reply) is None


# ================================================================== repair ==
async def test_an_invalid_reply_is_repaired_once() -> None:
    llm = ScriptedLLM(["{'not': 'json'}", GOOD_REPLY])

    update = await make_synthesize(llm)(_state())

    assert len(llm.prompts) == 2, "exactly one repair attempt"
    assert "Yamanote" in update["response"].city_summary
    assert any("repair attempt succeeded" in event.message for event in update["trace"])


async def test_the_repair_prompt_shows_the_model_its_own_output() -> None:
    llm = ScriptedLLM(["{'not': 'json'}", GOOD_REPLY])

    await make_synthesize(llm)(_state())

    assert "not valid for this schema" in llm.prompts[1]
    assert "not" in llm.prompts[1], "the failed reply itself must be quoted back"


async def test_two_failures_fall_back_to_a_deterministic_summary() -> None:
    """The user must never see a validation error."""
    llm = ScriptedLLM(["garbage", "still garbage"])

    update = await make_synthesize(llm)(_state())
    response = update["response"]

    assert len(llm.prompts) == 2, "the repair budget is one attempt, not a loop"
    assert isinstance(response, TravelResponse)
    assert len(response.city_summary) >= 40
    assert "Yamanote" in response.city_summary, "the fallback still uses the retrieved facts"
    assert any(
        "assembled the response from the tool payloads" in e.message for e in update["trace"]
    )


async def test_a_dead_model_still_produces_a_page() -> None:
    update = await make_synthesize(ExplodingLLM())(_state())

    assert isinstance(update["response"], TravelResponse)
    assert len(update["response"].weather_forecast) == 7, "tool data survives a dead model"
    assert any(event.kind == "error" for event in update["trace"])


# ============================================================= degradation ==
async def test_missing_weather_is_declared_in_the_prompt() -> None:
    llm = ScriptedLLM([GOOD_REPLY])

    await make_synthesize(llm)(_state(weather=None))

    assert "Weather: UNAVAILABLE" in llm.prompts[0]
    assert "Do not describe the weather" in llm.prompts[0]


async def test_missing_weather_is_stated_in_the_fallback_summary() -> None:
    update = await make_synthesize(ExplodingLLM())(_state(weather=None))

    assert "forecast is unavailable" in update["response"].city_summary.lower()
    assert update["response"].weather_forecast == []


async def test_missing_images_are_declared_in_the_prompt() -> None:
    llm = ScriptedLLM([GOOD_REPLY])

    await make_synthesize(llm)(_state(images=[]))

    assert "Images: UNAVAILABLE" in llm.prompts[0]


async def test_tool_errors_become_user_facing_warnings() -> None:
    from travel_agent.schemas.trace import ToolErrorRecord

    errors = [ToolErrorRecord(tool="get_weather_forecast", message="simulated HTTP 500")]

    update = await make_synthesize(ScriptedLLM([GOOD_REPLY]))(_state(weather=None, errors=errors))

    assert update["response"].warnings
    assert "get_weather_forecast unavailable" in update["response"].warnings[0]
    assert update["response"].is_degraded


# ================================================================ follow-up ==
async def test_a_follow_up_does_not_call_the_model_again() -> None:
    """Regenerating identical facts would cost a call and drift the wording."""
    previous = TravelResponse(
        city="Tokyo",
        city_summary="Tokyo is a chain of centres along the JR Yamanote loop line, and rail does the work.",
        highlights=["Orientation"],
    )
    llm = ScriptedLLM([GOOD_REPLY])

    update = await make_synthesize(llm)(_state(intent="weather_only", response=previous))

    assert llm.prompts == [], "no model call should have been made"
    assert update["response"].city_summary == previous.city_summary
    assert update["trace"][-1].data["regenerated"] is False


async def test_a_follow_up_still_refreshes_the_forecast() -> None:
    previous = TravelResponse(
        city="Tokyo",
        city_summary="Tokyo is a chain of centres along the JR Yamanote loop line, and rail does the work.",
    )

    update = await make_synthesize(ScriptedLLM([]))(
        _state(intent="weather_only", response=previous, weather=_weather())
    )

    assert len(update["response"].weather_forecast) == 7


async def test_a_follow_up_without_a_previous_summary_generates_one() -> None:
    """A weather-only intent on a thread with no stored answer must not blank out."""
    llm = ScriptedLLM([GOOD_REPLY])

    update = await make_synthesize(llm)(_state(intent="weather_only", response=None))

    assert len(llm.prompts) == 1
    assert "Yamanote" in update["response"].city_summary


# ================================================================= clarify ==
async def test_a_clarify_turn_asks_rather_than_answering() -> None:
    llm = ScriptedLLM([GOOD_REPLY])

    update = await make_synthesize(llm)(_state(intent="clarify", city=None, knowledge=[]))
    response = update["response"]

    assert llm.prompts == [], "no model call is needed to ask which city"
    assert response.is_clarification
    assert response.city == ""
    assert "which city" in response.city_summary.lower()


# ============================================================ with the mock ==
async def test_the_mock_driver_satisfies_the_same_contract() -> None:
    """The keyless path must produce a valid response like any other driver."""
    update = await make_synthesize(MockLLM())(_state())

    assert isinstance(update["response"], TravelResponse)
    assert len(update["response"].city_summary) >= 40
    assert update["response"].weather_forecast
