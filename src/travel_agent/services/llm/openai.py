"""OpenAI driver.

Named by the assignment and fully implemented. Selected with
``LLM_PROVIDER=openai`` or by supplying ``OPENAI_API_KEY`` with no higher-priority
key present.
"""

from __future__ import annotations

from travel_agent.config.settings import Settings, get_settings
from travel_agent.exceptions import ConfigurationError
from travel_agent.logging_setup import get_logger
from travel_agent.services.llm.base import LangChainLLM

logger = get_logger(__name__)


class OpenAILLM(LangChainLLM):
    """Chat driver backed by ``langchain-openai``."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the driver.

        Args:
            settings: Settings holding the API key, model id and timeouts.

        Raises:
            ConfigurationError: If the key is missing or the package is absent.
        """
        resolved = settings or get_settings()
        if not resolved.openai_api_key:
            raise ConfigurationError("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set")

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - pinned dependency
            raise ConfigurationError(f"langchain-openai is not installed: {exc}") from exc

        client = ChatOpenAI(
            model=resolved.openai_model,
            api_key=resolved.openai_api_key,
            temperature=resolved.llm_temperature,
            timeout=resolved.llm_timeout_seconds,
            max_retries=resolved.llm_max_retries,
        )
        super().__init__(client=client, model_id=resolved.openai_model, name="openai")


__all__ = ["OpenAILLM"]
