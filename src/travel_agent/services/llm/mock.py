"""A deterministic LLM driver that speaks the real tool-calling protocol.

WHY THIS IS NOT A STUB
    Everything downstream of the model - the manual executor, the fan-out, the
    id pairing, the summariser - is exercised by whatever the model emits. If the
    mock returned a canned string, none of that code would ever be tested without
    an API key, and the assignment's promise that the app runs keyless would be
    hollow.

    So this driver emits a genuine ``AIMessage`` carrying a genuine ``tool_calls``
    payload: the same field names, the same argument shapes, the same unique call
    ids a real provider produces. The executor cannot tell the difference, which
    is exactly the point - the code path under test is the real one.

HOW IT DECIDES
    It reads the planning brief that the ``plan_tools`` node writes - city,
    intent, knowledge route - rather than trying to understand free text. A real
    model reasons about that brief; the mock parses it. Both produce the same
    payload, which is the only thing the rest of the system sees.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage

from travel_agent.logging_setup import get_logger
from travel_agent.schemas.tools import IMAGES_TOOL, WEATHER_TOOL, WEB_SEARCH_TOOL
from travel_agent.schemas.trace import TokenUsage
from travel_agent.services.llm.base import BaseLLM, LLMCall

logger = get_logger(__name__)

# Fields the planning brief carries. Parsed rather than reasoned about.
_FIELD_PATTERN = re.compile(r"^(?P<key>[A-Za-z_ ]+):\s*(?P<value>.+?)\s*$", re.MULTILINE)

#: Rough characters-per-token ratio, used so the mock still reports plausible
#: usage numbers and the cost counter has something to show offline.
CHARS_PER_TOKEN = 4


class MockLLM(BaseLLM):
    """Deterministic, offline, protocol-faithful model driver.

    Attributes:
        name: Always ``"mock"``.
        model_id: Always ``"mock-llm"``.
    """

    name = "mock"
    model_id = "mock-llm"

    def __init__(self, forecast_days: int = 7, image_count: int = 4) -> None:
        """Initialise the driver.

        Args:
            forecast_days: Days requested in the weather tool call.
            image_count: Images requested in the image tool call.
        """
        self._forecast_days = forecast_days
        self._image_count = image_count
        self._call_counter = 0

    async def plan(
        self,
        messages: list[AnyMessage],
        tools: list[dict[str, Any]],
    ) -> LLMCall:
        """Emit the tool calls a competent model would emit for this brief.

        Args:
            messages: Conversation. The last message is the planning brief.
            tools: Tool schemas available on this turn. Only tools present here
                are requested - the mock respects the offered set the way a real
                model does, so a branch that was not offered is never called.

        Returns:
            An ``AIMessage`` whose ``tool_calls`` is a real protocol payload.
        """
        brief = self._read_brief(messages)
        offered = {schema["function"]["name"] for schema in tools}

        city = brief.get("city", "").strip()
        intent = brief.get("intent", "new_city").strip()
        route = brief.get("knowledge source", "vector_store").strip()
        days = int(brief.get("forecast days", self._forecast_days) or self._forecast_days)

        tool_calls: list[dict[str, Any]] = []

        if WEATHER_TOOL in offered and city:
            tool_calls.append(self._tool_call(WEATHER_TOOL, {"city": city, "days": days}))

        # A follow-up asking only about dates must not re-fetch images or re-read
        # the knowledge base. That decision belongs to the planner, and the mock
        # makes it the same way a real model would when told the intent.
        if intent != "weather_only":
            if IMAGES_TOOL in offered and city:
                tool_calls.append(
                    self._tool_call(IMAGES_TOOL, {"city": city, "count": self._image_count})
                )
            if WEB_SEARCH_TOOL in offered and route == "web_search" and city:
                tool_calls.append(
                    self._tool_call(
                        WEB_SEARCH_TOOL,
                        {"query": f"{city} travel guide overview", "max_results": 4},
                    )
                )

        content = "" if tool_calls else "I need a city name before I can look anything up."
        message = AIMessage(content=content, tool_calls=tool_calls)

        logger.debug(
            "MockLLM planned %d tool call(s): %s",
            len(tool_calls),
            ", ".join(call["name"] for call in tool_calls) or "none",
        )
        return LLMCall(message=message, usage=self._usage(messages, content, tool_calls))

    async def complete_json(self, messages: list[AnyMessage]) -> tuple[str, TokenUsage]:
        """Return a JSON summary built from the facts in the prompt.

        The synthesis node hands over the retrieved passages and the tool results;
        this composes them into the same JSON envelope a real model would produce,
        so the validation and repair path downstream is genuinely exercised.

        Args:
            messages: Conversation whose last message carries the gathered facts.

        Returns:
            The raw JSON text and its token usage.
        """
        brief = self._read_brief(messages)
        city = brief.get("city", "this destination").strip() or "this destination"
        facts = self._read_facts(messages)

        summary = self._compose_summary(city, facts)
        payload = {
            "city_summary": summary,
            "highlights": self._compose_highlights(facts),
        }
        text = json.dumps(payload)
        return text, self._usage(messages, text, [])

    # ------------------------------------------------------------- internals --
    def _tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Build one tool call with a unique id.

        Ids must be unique within a turn: they are the only thing pairing a reply
        to its request, so a duplicate would silently mis-attribute results.

        Args:
            name: Tool name.
            args: Tool arguments.

        Returns:
            A tool call payload in the standard shape.
        """
        self._call_counter += 1
        return {
            "id": f"call_mock_{self._call_counter:03d}",
            "name": name,
            "args": args,
            "type": "tool_call",
        }

    @staticmethod
    def _read_brief(messages: list[AnyMessage]) -> dict[str, str]:
        """Parse the ``Key: value`` planning brief from the last message.

        Args:
            messages: Conversation.

        Returns:
            A lowercase-keyed mapping of the brief's fields.
        """
        if not messages:
            return {}
        content = messages[-1].content
        text = content if isinstance(content, str) else str(content)
        return {
            match.group("key").strip().lower(): match.group("value")
            for match in _FIELD_PATTERN.finditer(text)
        }

    @staticmethod
    def _read_facts(messages: list[AnyMessage]) -> list[str]:
        """Extract the fact block a synthesis prompt carries.

        Args:
            messages: Conversation.

        Returns:
            Non-empty fact lines, in prompt order.
        """
        if not messages:
            return []
        content = messages[-1].content
        text = content if isinstance(content, str) else str(content)
        if "FACTS:" not in text:
            return []
        block = text.split("FACTS:", 1)[1]
        return [line.strip("- ").strip() for line in block.splitlines() if line.strip("- ").strip()]

    @staticmethod
    def _compose_summary(city: str, facts: list[str]) -> str:
        """Write a summary from the supplied facts.

        Args:
            city: City the answer is about.
            facts: Retrieved passages.

        Returns:
            A prose summary of at least the schema's minimum length.
        """
        if not facts:
            return (
                f"{city} is the destination for this request. No detailed source "
                f"material was available, so this summary is limited to what the "
                f"tools returned."
            )

        opening = f"{city} at a glance, drawn from the retrieved source material."
        body = " ".join(fact.rstrip(".") + "." for fact in facts[:3])
        return f"{opening} {body}"[:1800]

    @staticmethod
    def _compose_highlights(facts: list[str]) -> list[str]:
        """Turn the first few facts into short bullet points.

        Args:
            facts: Retrieved passages.

        Returns:
            Between zero and four short strings.
        """
        return [fact[:140].rstrip(".") for fact in facts[:4]]

    def _usage(
        self,
        messages: list[AnyMessage],
        output: str,
        tool_calls: list[dict[str, Any]],
    ) -> TokenUsage:
        """Estimate token usage so the counter shows plausible numbers offline.

        Args:
            messages: The prompt that was sent.
            output: The text produced.
            tool_calls: Any tool calls produced.

        Returns:
            Estimated usage, marked with the mock model id so nobody mistakes it
            for a real measurement.
        """
        prompt_chars = sum(
            len(message.content if isinstance(message.content, str) else str(message.content))
            for message in messages
        )
        completion_chars = len(output) + len(json.dumps(tool_calls))
        prompt_tokens = prompt_chars // CHARS_PER_TOKEN
        completion_tokens = completion_chars // CHARS_PER_TOKEN
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            llm_calls=1,
            model=self.model_id,
        )


__all__ = ["MockLLM"]
