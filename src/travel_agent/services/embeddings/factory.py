"""Embedder selection.

One function decides which embedding implementation the rest of the app gets,
based on configuration. Nothing else imports a concrete embedder class, so
changing the default is a one-line config change rather than a code change.
"""

from __future__ import annotations

from travel_agent.config.settings import Settings, get_settings
from travel_agent.logging_setup import get_logger
from travel_agent.services.embeddings.base import BaseEmbedder
from travel_agent.services.embeddings.hashed import HashedTfIdfEmbedder

logger = get_logger(__name__)


def get_embedder(settings: Settings | None = None) -> BaseEmbedder:
    """Return the configured embedder.

    ``EMBEDDING_PROVIDER=openai`` is accepted but falls back to the hashed
    embedder when no OpenAI key is configured, because silently producing zero
    vectors would break routing in a way that is hard to diagnose.

    Args:
        settings: Settings to read from. Defaults to the process singleton.

    Returns:
        A ready-to-use embedder. Call ``fit`` before indexing if the
        implementation needs corpus statistics.
    """
    settings = settings or get_settings()

    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            logger.warning(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is unset; "
                "falling back to the hashed embedder"
            )
        else:
            from travel_agent.services.embeddings.openai_embedder import OpenAIEmbedder

            return OpenAIEmbedder(
                api_key=settings.openai_api_key,
                timeout=settings.llm_timeout_seconds,
            )

    return HashedTfIdfEmbedder(dim=settings.embedding_dim)


__all__ = ["get_embedder"]
