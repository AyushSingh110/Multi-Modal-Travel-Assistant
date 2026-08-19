"""Embedding provider interface.

WHAT AN EMBEDDING IS
    A list of numbers that represents a piece of text, arranged so that texts
    about similar things end up close together. "Closeness" here is cosine
    similarity: 1.0 means identical direction, 0.0 means unrelated.

    This is what makes the router possible. Instead of matching city names with
    string equality, the graph asks "how close is this query to anything I know?"
    and gets back a number it can compare with a threshold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEmbedder(ABC):
    """Turns text into vectors.

    Two implementations exist: a dependency-free hashed embedder used by default,
    and an OpenAI-backed one for when embedding quality matters more than being
    able to run offline.
    """

    #: Dimensionality of the vectors produced by this embedder.
    dim: int

    #: Short identifier stored in the vector store manifest, so a store built with
    #: one embedder is never queried with a different one.
    name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents.

        Args:
            texts: Passages to embed.

        Returns:
            An array of shape ``(len(texts), dim)``, L2-normalised row-wise.
        """

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query.

        Args:
            text: The query text.

        Returns:
            A vector of shape ``(dim,)``, L2-normalised.
        """

    def fit(self, texts: list[str]) -> None:  # noqa: ARG002 - no-op hook for subclasses
        """Learn corpus statistics before embedding, if the implementation needs them.

        The hashed embedder uses this to compute inverse document frequencies.
        Implementations that need no fitting can leave this as a no-op.

        Args:
            texts: The full corpus that will be indexed.
        """
        return None

    def state_dict(self) -> dict[str, Any]:
        """Return any learned state that must be persisted alongside the index.

        Returns:
            A JSON-serialisable dictionary. Empty by default.
        """
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:  # noqa: ARG002 - no-op hook
        """Restore learned state produced by :meth:`state_dict`.

        Args:
            state: Previously persisted state.
        """
        return None


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length.

    With unit-length vectors, a dot product *is* the cosine similarity, which lets
    the FAISS inner-product index act as a cosine index without extra work.

    Args:
        matrix: Array of shape ``(n, dim)`` or ``(dim,)``.

    Returns:
        The normalised array. Zero rows are left as zeros rather than producing
        NaN, so an empty query cannot poison the index.
    """
    array = np.atleast_2d(matrix).astype(np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalised = array / norms
    return normalised.reshape(matrix.shape) if matrix.ndim == 1 else normalised


__all__ = ["BaseEmbedder", "l2_normalise"]
