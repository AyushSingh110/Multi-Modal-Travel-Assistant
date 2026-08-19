"""Groq driver - the default demo model.

Groq is the default because its free tier survives a day of iterative development
plus a live demo, and because its API is OpenAI-compatible and returns a genuine
``tool_calls`` payload, so the manual executor is exercised against the real wire
protocol rather than a shim.

The assignment names OpenAI and Anthropic; both are fully implemented alongside
this and switching is one line in ``.env``.

MODEL AVAILABILITY
    Model ids get retired. Rather than hard-coding one and hoping, this driver
    can query ``/v1/models`` at start-up, confirm the configured id is still
    listed, and fall back to another tool-capable model if it is not. The result
    is reported into the UI trace rather than buried in a log line, so the demo
    never dies on a deprecation the user cannot see.
"""

from __future__ import annotations

import httpx

from travel_agent.config.settings import Settings, get_settings
from travel_agent.exceptions import ConfigurationError
from travel_agent.logging_setup import get_logger
from travel_agent.services.llm.base import LangChainLLM, ModelCheck

logger = get_logger(__name__)

MODELS_URL = "https://api.groq.com/openai/v1/models"

#: Preference order when the configured model is unavailable. All are documented
#: by Groq as supporting tool calling.
FALLBACK_MODELS: tuple[str, ...] = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
)


class GroqLLM(LangChainLLM):
    """Chat driver backed by ``langchain-groq``."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the driver.

        Args:
            settings: Settings holding the API key, model id and timeouts.

        Raises:
            ConfigurationError: If Groq is selected without an API key, or the
                ``langchain-groq`` package is missing.
        """
        self._settings = settings or get_settings()
        if not self._settings.groq_api_key:
            raise ConfigurationError("LLM_PROVIDER=groq requires GROQ_API_KEY to be set")

        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:  # pragma: no cover - pinned dependency
            raise ConfigurationError(
                f"langchain-groq could not be imported - it is missing, or its version does not match langchain-core: {exc}"
            ) from exc

        client = ChatGroq(
            model=self._settings.groq_model,
            api_key=self._settings.groq_api_key,
            temperature=self._settings.llm_temperature,
            timeout=self._settings.llm_timeout_seconds,
            max_retries=self._settings.llm_max_retries,
        )
        super().__init__(client=client, model_id=self._settings.groq_model, name="groq")

    async def check_model(self) -> ModelCheck:
        """Confirm the configured model is still offered by Groq.

        Returns:
            The check result. A failed lookup is *not* fatal: if the endpoint
            cannot be reached the configured id is used unchanged, because a
            network hiccup should not block a demo.
        """
        requested = self.model_id

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.get(
                    MODELS_URL,
                    headers={"Authorization": f"Bearer {self._settings.groq_api_key}"},
                )
            response.raise_for_status()
            available = {item["id"] for item in response.json().get("data", [])}
        except Exception as exc:  # noqa: BLE001 - availability is advisory, not required
            logger.warning(
                "could not list Groq models (%s); using %s as configured", exc, requested
            )
            return ModelCheck(
                requested=requested,
                resolved=requested,
                available=True,
                message=f"Groq model list unavailable ({type(exc).__name__}); using {requested}",
            )

        if requested in available:
            return ModelCheck(
                requested=requested,
                resolved=requested,
                available=True,
                message=f"Groq model {requested} is available",
            )

        replacement = next((model for model in FALLBACK_MODELS if model in available), None)
        if replacement is None:
            replacement = sorted(available)[0] if available else requested

        logger.warning("Groq model %r is not available; falling back to %r", requested, replacement)
        self.model_id = replacement
        self._client = self._client.bind(model=replacement)

        return ModelCheck(
            requested=requested,
            resolved=replacement,
            available=False,
            message=(
                f"Groq model {requested} is no longer offered; using {replacement} instead. "
                f"Set GROQ_MODEL to silence this."
            ),
        )


__all__ = ["FALLBACK_MODELS", "GroqLLM"]
