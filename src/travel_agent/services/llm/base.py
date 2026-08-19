"""LLM driver interface and the shared LangChain adapter.

Four drivers sit behind this interface - Groq, Anthropic, OpenAI and a
deterministic mock - and the graph never learns which one it is talking to. It
asks for two things only:

* :meth:`BaseLLM.plan` - "here are the tools, which do you want and with what
  arguments?", returning a raw ``AIMessage`` with a ``tool_calls`` payload;
* :meth:`BaseLLM.complete_json` - "answer with JSON matching this shape", used by
  the synthesis node.

Keeping the surface that small is what makes the mock genuinely equivalent rather
than a stub with a different shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage

from travel_agent.logging_setup import get_logger
from travel_agent.schemas.trace import TokenUsage

logger = get_logger(__name__)


@dataclass
class LLMCall:
    """The result of one model call.

    Attributes:
        message: The model's reply, carrying ``tool_calls`` when it asked for tools.
        usage: Normalised token accounting for this call.
    """

    message: AIMessage
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class ModelCheck:
    """Outcome of verifying that a configured model still exists.

    Attributes:
        requested: The model id from configuration.
        resolved: The model id actually used, which may differ after a fallback.
        available: Whether the requested id was found.
        message: Human-readable verdict for the logs and the trace panel.
    """

    requested: str
    resolved: str
    available: bool
    message: str = ""


class BaseLLM(ABC):
    """A language model the graph can plan tool calls with.

    Attributes:
        name: Driver name, e.g. ``"groq"``, recorded in the trace.
        model_id: The model identifier in use.
    """

    name: str = "base"
    model_id: str = ""

    @abstractmethod
    async def plan(
        self,
        messages: list[AnyMessage],
        tools: list[dict[str, Any]],
    ) -> LLMCall:
        """Ask the model which tools to call.

        Args:
            messages: Conversation to send.
            tools: Tool schemas in OpenAI function format.

        Returns:
            The model's reply. ``message.tool_calls`` holds the requests.
        """

    @abstractmethod
    async def complete_json(self, messages: list[AnyMessage]) -> tuple[str, TokenUsage]:
        """Ask the model for a JSON object.

        Args:
            messages: Conversation to send, whose system prompt describes the
                required shape.

        Returns:
            A ``(raw_text, usage)`` pair. The caller validates the text against a
            Pydantic model and repairs it if necessary - no driver is trusted to
            have produced valid JSON.
        """

    async def check_model(self) -> ModelCheck:
        """Verify the configured model exists, falling back if it does not.

        Overridden by drivers whose provider exposes a model listing. The default
        assumes the configured model is fine.

        Returns:
            The check result.
        """
        return ModelCheck(
            requested=self.model_id,
            resolved=self.model_id,
            available=True,
            message=f"{self.name}: model availability not checked",
        )


def usage_from_message(message: AIMessage, model_id: str) -> TokenUsage:
    """Normalise a provider's token accounting into :class:`TokenUsage`.

    Groq, OpenAI and Anthropic all report usage differently. ``langchain-core``
    exposes a common ``usage_metadata`` dictionary, which is what this reads, with
    a fall back to the raw ``response_metadata`` for drivers that do not populate
    it.

    Args:
        message: The model's reply.
        model_id: Model identifier to record.

    Returns:
        Populated usage, with zeroes when the provider reported nothing.
    """
    metadata = getattr(message, "usage_metadata", None) or {}
    prompt = int(metadata.get("input_tokens", 0))
    completion = int(metadata.get("output_tokens", 0))
    total = int(metadata.get("total_tokens", prompt + completion))

    if not total:
        raw = getattr(message, "response_metadata", {}).get("token_usage", {})
        prompt = int(raw.get("prompt_tokens", 0))
        completion = int(raw.get("completion_tokens", 0))
        total = int(raw.get("total_tokens", prompt + completion))

    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        llm_calls=1,
        model=model_id,
    )


class LangChainLLM(BaseLLM):
    """Shared implementation for every provider reached through LangChain.

    Groq, Anthropic and OpenAI differ only in which chat model class is
    constructed and how availability is checked, so the calling logic lives here
    once rather than three times.
    """

    def __init__(self, client: Any, model_id: str, name: str) -> None:
        """Initialise the driver.

        Args:
            client: A LangChain chat model instance.
            model_id: Model identifier, for logs and usage records.
            name: Driver name.
        """
        self._client = client
        self.model_id = model_id
        self.name = name

    async def plan(
        self,
        messages: list[AnyMessage],
        tools: list[dict[str, Any]],
    ) -> LLMCall:
        """Ask the model which tools to call.

        Args:
            messages: Conversation to send.
            tools: Tool schemas in OpenAI function format.

        Returns:
            The model's reply and its token usage.
        """
        bound = self._client.bind_tools(tools) if tools else self._client
        reply = await bound.ainvoke(messages)
        return LLMCall(message=reply, usage=usage_from_message(reply, self.model_id))

    async def complete_json(self, messages: list[AnyMessage]) -> tuple[str, TokenUsage]:
        """Ask the model for a JSON object.

        Uses JSON mode rather than schema-constrained decoding: not every provider
        supports strict schemas, and the project validates and repairs the result
        anyway, so the safer path is used uniformly.

        Args:
            messages: Conversation to send.

        Returns:
            The raw reply text and its token usage.
        """
        client = self._client
        try:
            client = self._client.bind(response_format={"type": "json_object"})
        except Exception as exc:  # noqa: BLE001 - not every provider accepts this
            logger.debug("%s does not accept response_format, sending plain: %s", self.name, exc)

        reply = await client.ainvoke(messages)
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        return text, usage_from_message(reply, self.model_id)


__all__ = ["BaseLLM", "LLMCall", "LangChainLLM", "ModelCheck", "usage_from_message"]
