"""LLM driver selection.

The one place that knows all four drivers exist. Everything else asks for "the
model" and gets whichever one the configuration and available keys imply.
"""

from __future__ import annotations

from travel_agent.config.settings import Settings, get_settings
from travel_agent.exceptions import ConfigurationError
from travel_agent.logging_setup import get_logger
from travel_agent.services.llm.base import BaseLLM
from travel_agent.services.llm.mock import MockLLM

logger = get_logger(__name__)


def get_llm(settings: Settings | None = None) -> BaseLLM:
    """Return the configured LLM driver.

    Selection order is implemented in
    :meth:`~travel_agent.config.settings.Settings.resolve_llm_provider`: an
    explicit ``LLM_PROVIDER`` wins, then the first key present in the order Groq,
    Anthropic, OpenAI, and with no keys at all the deterministic mock.

    A live driver that cannot be constructed - a missing key, an uninstalled
    package - falls back to the mock with a warning rather than taking the app
    down, because a demo that runs on mock data is better than one that does not
    start.

    Args:
        settings: Settings to read from. Defaults to the process singleton.

    Returns:
        A ready driver.
    """
    resolved = settings or get_settings()
    provider = resolved.resolve_llm_provider()

    if provider == "mock":
        logger.info("LLM provider: mock (no API keys configured)")
        return MockLLM()

    try:
        if provider == "groq":
            from travel_agent.services.llm.groq import GroqLLM

            driver: BaseLLM = GroqLLM(resolved)
        elif provider == "anthropic":
            from travel_agent.services.llm.anthropic import AnthropicLLM

            driver = AnthropicLLM(resolved)
        elif provider == "openai":
            from travel_agent.services.llm.openai import OpenAILLM

            driver = OpenAILLM(resolved)
        else:  # pragma: no cover - Literal type makes this unreachable
            raise ConfigurationError(f"unknown LLM provider {provider!r}")
    except ConfigurationError as exc:
        logger.warning("cannot use %s (%s); falling back to the mock driver", provider, exc)
        return MockLLM()

    logger.info("LLM provider: %s (model %s)", driver.name, driver.model_id)
    return driver


__all__ = ["get_llm"]
