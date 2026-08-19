"""Anthropic driver.

Named by the assignment and fully implemented. Selected with
``LLM_PROVIDER=anthropic`` or by supplying ``ANTHROPIC_API_KEY`` with no Groq key
present.
"""

from __future__ import annotations

from travel_agent.config.settings import Settings, get_settings
from travel_agent.exceptions import ConfigurationError
from travel_agent.logging_setup import get_logger
from travel_agent.services.llm.base import LangChainLLM

logger = get_logger(__name__)


class AnthropicLLM(LangChainLLM):
    """Chat driver backed by ``langchain-anthropic``."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the driver.

        Args:
            settings: Settings holding the API key, model id and timeouts.

        Raises:
            ConfigurationError: If the key is missing or the package is absent.
        """
        resolved = settings or get_settings()
        if not resolved.anthropic_api_key:
            raise ConfigurationError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set")

        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - pinned dependency
            raise ConfigurationError(f"langchain-anthropic is not installed: {exc}") from exc

        client = ChatAnthropic(
            model=resolved.anthropic_model,
            api_key=resolved.anthropic_api_key,
            temperature=resolved.llm_temperature,
            timeout=resolved.llm_timeout_seconds,
            max_retries=resolved.llm_max_retries,
        )
        super().__init__(client=client, model_id=resolved.anthropic_model, name="anthropic")


__all__ = ["AnthropicLLM"]
