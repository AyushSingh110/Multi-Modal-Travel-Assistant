"""OpenAI-backed embedder.

The live alternative to the hashed embedder. It exists so the provider
abstraction is real rather than theoretical: switching to genuine semantic
embeddings is ``EMBEDDING_PROVIDER=openai`` plus a key, with no other code change.

It is not the default because it requires network access and a paid key, and the
project's promise is that everything runs with zero keys.
"""

from __future__ import annotations

import numpy as np

from travel_agent.exceptions import ProviderError
from travel_agent.logging_setup import get_logger
from travel_agent.services.embeddings.base import BaseEmbedder, l2_normalise

logger = get_logger(__name__)

#: 1536-dimensional, cheapest of the OpenAI embedding models and more than enough
#: for a corpus of this size.
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIM = 1536


class OpenAIEmbedder(BaseEmbedder):
    """Embeds text with the OpenAI embeddings API.

    Attributes:
        dim: Vector dimensionality reported by the model.
        name: Identifier persisted in the store manifest.
    """

    def __init__(
        self,
        api_key: str | None,
        model: str = DEFAULT_MODEL,
        timeout: float = 30.0,
    ) -> None:
        """Initialise the embedder.

        Args:
            api_key: OpenAI API key.
            model: Embedding model id.
            timeout: Per-request timeout in seconds.

        Raises:
            ProviderError: If the ``openai`` package is not installed.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - openai is a pinned dependency
            raise ProviderError("the openai package is required for OpenAIEmbedder") from exc

        self._client = OpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        self.dim = DEFAULT_DIM
        self.name = f"openai-{model}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents.

        Args:
            texts: Passages to embed.

        Returns:
            An array of shape ``(len(texts), dim)``.

        Raises:
            ProviderError: If the API call fails.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query.

        Args:
            text: Query text.

        Returns:
            A vector of shape ``(dim,)``.

        Raises:
            ProviderError: If the API call fails.
        """
        return self._embed([text]).reshape(self.dim)

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Call the embeddings endpoint.

        Args:
            texts: Texts to embed.

        Returns:
            A normalised array of shape ``(len(texts), dim)``.

        Raises:
            ProviderError: If the API call fails.
        """
        try:
            response = self._client.embeddings.create(model=self._model, input=texts)
        except Exception as exc:  # noqa: BLE001 - normalise every SDK failure
            raise ProviderError(f"OpenAI embedding call failed: {exc}") from exc

        vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
        self.dim = vectors.shape[1]
        return l2_normalise(vectors)

    def state_dict(self) -> dict[str, object]:
        """Return the model identity so a store is never queried with a mismatched embedder.

        Returns:
            A JSON-serialisable dictionary.
        """
        return {"model": self._model, "dim": self.dim}


__all__ = ["OpenAIEmbedder"]
