"""The final node.

Turns tool payloads and retrieved passages into a validated ``TravelResponse``.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from travel_agent.logging_setup import Timer, get_logger
from travel_agent.schemas.knowledge import KnowledgeChunk
from travel_agent.schemas.response import ImageAsset, TravelResponse, WeatherPayload
from travel_agent.schemas.state import TravelState
from travel_agent.schemas.trace import TraceEvent
from travel_agent.services.llm.base import BaseLLM

logger = get_logger(__name__)

#: Below this many retrieved passages the model is told to be explicit about how
#: little it has, rather than padding.
SPARSE_CONTEXT_THRESHOLD = 2

#: How much of each passage reaches the prompt. Enough to summarise from, bounded
#: so a large corpus cannot dominate the context window.
MAX_CHUNK_CHARS = 700

SYSTEM_PROMPT = """You write short factual travel summaries for a reference tool.

Rules, in order of importance:
1. Use ONLY the facts in the FACTS block. Do not add anything you happen to know
   about the place. If the facts are thin, say so plainly - write "There is
   limited information available about this destination" and summarise only what
   is there.
2. Never invent weather, temperatures, dates, prices or place names.
3. Plain prose. No brochure language, no superlatives you were not given, no
   second person, no exclamation marks.
4. Three to five sentences.

Reply with a JSON object and nothing else:
{"city_summary": "<the prose>", "highlights": ["<short phrase>", ...]}
The highlights are three to five short phrases naming topics the facts cover."""


class SynthesisDraft(BaseModel):
    """What the model is asked to produce.

    Deliberately small. The forecast and the image URLs are *not* model output -
    they are typed objects the tools already returned, and asking a model to copy
    them back would only create an opportunity to alter them.
    """

    model_config = ConfigDict(extra="ignore")

    city_summary: str = Field(min_length=40, max_length=2000)
    highlights: list[str] = Field(default_factory=list, max_length=6)


def make_synthesize(llm: BaseLLM) -> Any:
    """Build the synthesis node.

    Args:
        llm: The model driver used to write the summary.

    Returns:
        An async node function.
    """

    async def synthesize(state: TravelState) -> dict[str, Any]:
        """Assemble the validated response for this turn.

        Args:
            state: Current graph state.

        Returns:
            A partial state update carrying the response object.
        """
        with Timer() as timer:
            if state.get("intent") == "clarify":
                return _clarify_response(state, timer)

            carried = _carry_previous_summary(state)
            if carried is not None:
                return _response_update(state, carried, timer, regenerated=False, events=[])

            draft, events, usage = await _draft_summary(llm, state)
            update = _response_update(state, draft, timer, regenerated=True, events=events)
            if usage is not None:
                update["token_usage"] = usage
            return update

    return synthesize


# the model call
async def _draft_summary(
    llm: BaseLLM, state: TravelState
) -> tuple[SynthesisDraft, list[TraceEvent], Any]:
    """Ask the model for a summary, validating and repairing as needed.

    Args:
        llm: The model driver.
        state: Current graph state.

    Returns:
        A tuple of the validated draft, trace events describing what happened,
        and the token usage.
    """
    city = state.get("city") or "this destination"
    knowledge: list[KnowledgeChunk] = state.get("knowledge") or []
    prompt = _build_prompt(state, city, knowledge)
    events: list[TraceEvent] = []

    messages: list[AnyMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        raw, usage = await llm.complete_json(messages)
    except Exception as exc:  # noqa: BLE001 - a dead model must not blank the page
        logger.warning("synthesis model call failed: %s", exc)
        events.append(
            TraceEvent(
                node="synthesize",
                kind="error",
                message=f"model call failed ({type(exc).__name__}); using the deterministic summary",
            )
        )
        return _deterministic_draft(city, knowledge, state), events, None
    draft = _validate(raw)
    if draft is not None:
        events.append(
            TraceEvent(
                node="synthesize",
                kind="llm",
                message="summary generated and validated on the first attempt",
                data={"chars": len(draft.city_summary)},
            )
        )
        return draft, events, usage

    # Repair: hand the model its own error once. Models correct a malformed field
    # readily when told exactly what was wrong, and one retry is the point where
    # the cost stops being worth it.
    logger.info("synthesis output failed validation; attempting one repair")
    repair_messages: list[AnyMessage] = [
        *messages,
        HumanMessage(
            content=(
                f"Your previous reply was not valid for this schema:\n{raw[:600]}\n\n"
                f"Reply again with ONLY a JSON object of the form "
                f'{{"city_summary": "<at least 40 characters of prose>", '
                f'"highlights": ["<short phrase>"]}}'
            )
        ),
    ]
    try:
        repaired_raw, repair_usage = await llm.complete_json(repair_messages)
        repaired = _validate(repaired_raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("repair attempt failed: %s", exc)
        repaired, repair_usage = None, usage

    if repaired is not None:
        events.append(
            TraceEvent(
                node="synthesize",
                kind="llm",
                message="first reply failed validation; the repair attempt succeeded",
            )
        )
        return repaired, events, repair_usage

    events.append(
        TraceEvent(
            node="synthesize",
            kind="error",
            message=(
                "the model could not produce a valid summary after one repair; "
                "assembled the response from the tool payloads instead"
            ),
        )
    )
    return _deterministic_draft(city, knowledge, state), events, repair_usage


def _validate(raw: str) -> SynthesisDraft | None:
    """Parse and validate a model reply.

    Args:
        raw: The model's text.

    Returns:
        The validated draft, or ``None`` when the text was unusable.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Models wrap JSON in a fenced block more often than the prompt suggests.
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if text.lower().startswith("json") else text

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        logger.debug("no JSON object found in the model reply")
        return None

    try:
        payload = json.loads(text[start : end + 1])
        return SynthesisDraft.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.debug("synthesis validation failed: %s", exc)
        return None


# prompting
def _build_prompt(state: TravelState, city: str, knowledge: list[KnowledgeChunk]) -> str:
    """Build the synthesis prompt from what is actually in state.

    Args:
        state: Current graph state.
        city: The resolved city.
        knowledge: Retrieved passages.

    Returns:
        The prompt text, including the FACTS block the model must stay inside.
    """
    weather: WeatherPayload | None = state.get("weather")
    images: list[ImageAsset] = state.get("images") or []

    lines = [f"City: {city}"]

    if len(knowledge) < SPARSE_CONTEXT_THRESHOLD:
        lines.append(
            "Context: SPARSE. Very little was retrieved. Say plainly that there is "
            "limited information available rather than filling the gap."
        )

    if weather is None or not weather.forecast:
        lines.append(
            "Weather: UNAVAILABLE. The forecast could not be fetched. State that "
            "the forecast is unavailable. Do not describe the weather."
        )
    else:
        first = weather.forecast[0]
        lines.append(
            f"Weather: available, {len(weather.forecast)} days from {first.date}, "
            f"around {first.temp_min_c:.0f} to {first.temp_max_c:.0f} C, {first.condition}."
        )
    if not images:
        lines.append("Images: UNAVAILABLE. No photographs were retrieved.")
    lines.append("")
    lines.append("FACTS:")
    if knowledge:
        for chunk in knowledge[:6]:
            lines.append(f"- {chunk.text[:MAX_CHUNK_CHARS].strip()}")
    else:
        lines.append("- (nothing was retrieved for this destination)")

    return "\n".join(lines)


# assembling
def _deterministic_draft(
    city: str, knowledge: list[KnowledgeChunk], state: TravelState
) -> SynthesisDraft:
    """Build a summary without a model, from the retrieved passages.

    The fallback when the model is unavailable or its answer will not validate.
    It just lifts whole sentences from the source text. Boring, but true, and
    better than showing an error page.

    Args:
        city: The resolved city.
        knowledge: Retrieved passages.
        state: Current graph state, used to note missing data.

    Returns:
        A valid draft.
    """
    sentences: list[str] = []
    for chunk in knowledge[:3]:
        body = chunk.text.split(". ", 1)[-1] if ". " in chunk.text else chunk.text
        for sentence in body.replace("\n", " ").split(". "):
            cleaned = sentence.strip()
            if len(cleaned) > 40:
                sentences.append(cleaned.rstrip(".") + ".")
                break

    if sentences:
        summary = f"{city}. " + " ".join(sentences)
    else:
        summary = (
            f"There is limited information available about {city}. No source "
            f"material was retrieved for it on this turn."
        )

    weather = state.get("weather")
    if weather is None or not weather.forecast:
        summary += " The weather forecast is unavailable for this destination."

    return SynthesisDraft(
        city_summary=summary[:1800],
        highlights=[chunk.section for chunk in knowledge[:4]],
    )


def _carry_previous_summary(state: TravelState) -> SynthesisDraft | None:
    """Reuse the previous turn's summary on a follow-up.

    A follow-up that only moves the date window has not changed anything the
    summary describes. Regenerating it would cost a model call to produce
    different words for identical facts - and the wording drifting between turns
    would just look like the app had changed its mind.

    Args:
        state: Current graph state.

    Returns:
        The carried draft, or ``None`` when a fresh summary is needed.
    """
    if state.get("intent") != "weather_only":
        return None
    previous: TravelResponse | None = state.get("response")
    if previous is None or not previous.city_summary:
        return None
    logger.info("carrying the summary forward from the previous turn")
    return SynthesisDraft(city_summary=previous.city_summary, highlights=previous.highlights)


def _response_update(
    state: TravelState,
    draft: SynthesisDraft,
    timer: Timer,
    *,
    regenerated: bool,
    events: list[TraceEvent],
) -> dict[str, Any]:
    """Assemble the final response object and its state update.

    Args:
        state: Current graph state.
        draft: The validated summary.
        timer: The synthesis timer.
        regenerated: Whether the summary was written this turn.
        events: Trace events accumulated while drafting.

    Returns:
        A partial state update.
    """
    city = state.get("city") or ""
    weather: WeatherPayload | None = state.get("weather")
    images: list[ImageAsset] = state.get("images") or []
    knowledge: list[KnowledgeChunk] = state.get("knowledge") or []
    errors = state.get("errors") or []

    warnings = [f"{error.tool} unavailable: {error.message[:160]}" for error in errors]
    response = TravelResponse(
        city=city,
        city_summary=draft.city_summary,
        weather_forecast=list(weather.forecast) if weather else [],
        image_urls=[asset.url for asset in images],
        highlights=draft.highlights,
        knowledge_source="web_search" if state.get("route") == "web" else "vector_store",
        sources=[chunk.source for chunk in knowledge if chunk.source][:6],
        warnings=warnings,
    )
    return {
        "response": response,
        "timings": {"synthesize": timer.elapsed_ms},
        "trace": [
            *events,
            TraceEvent(
                node="synthesize",
                kind="info" if regenerated else "skip",
                message=(
                    f"response built: {len(response.weather_forecast)} forecast points, "
                    f"{len(response.image_urls)} images, {len(response.warnings)} warning(s)"
                    if regenerated
                    else "summary carried forward from the previous turn; no model call made"
                ),
                duration_ms=timer.elapsed_ms,
                data={
                    "degraded": response.is_degraded,
                    "regenerated": regenerated,
                    "grounded_in_chunks": len(knowledge),
                },
            ),
        ],
    }


def _clarify_response(state: TravelState, timer: Timer) -> dict[str, Any]:
    """Build the response for a turn that named no city.

    Better to ask than to guess. If we pick the wrong city the page still looks
    completely normal, so the mistake is easy to miss.

    Args:
        state: Current graph state.
        timer: The synthesis timer.

    Returns:
        A partial state update carrying a clarifying response.
    """
    response = TravelResponse(
        city="",
        city_summary=(
            "I could not tell which city you mean. This looks like a follow-up, but "
            "there is no earlier city in this conversation to carry forward. Name a "
            "city - for example 'Tell me about Tokyo' - and I will look it up."
        ),
        knowledge_source="memory",
        warnings=["No city could be resolved from this turn or from the conversation history."],
    )
    return {
        "response": response,
        "timings": {"synthesize": timer.elapsed_ms},
        "trace": [
            TraceEvent(
                node="synthesize",
                kind="skip",
                message="no city resolved - asked the user to clarify instead of guessing",
                data={"query": state.get("user_query", "")},
            )
        ],
    }


__all__ = ["SPARSE_CONTEXT_THRESHOLD", "SynthesisDraft", "make_synthesize"]
