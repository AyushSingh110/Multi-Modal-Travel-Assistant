"""Brute-force vector store built on NumPy.

The fallback backend. It keeps every vector in one matrix and answers a query
with a single matrix-vector product, which at this corpus size is exact and
instant. Its role is insurance: if the FAISS wheel will not import on a
reviewer's machine, the application keeps working with identical behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from travel_agent.exceptions import VectorStoreError
from travel_agent.services.vectorstore.base import BaseVectorStore

VECTORS_FILE = "vectors.npy"


class NumpyVectorStore(BaseVectorStore):
    """Exact cosine search over an in-memory matrix.

    Because all vectors are L2-normalised, the dot product of a query with the
    matrix *is* the vector of cosine similarities - no extra normalisation step
    is needed at query time.
    """

    backend = "numpy"

    def __init__(self, dim: int) -> None:
        """Initialise an empty store.

        Args:
            dim: Vector dimensionality.
        """
        super().__init__(dim)
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)

    def _add_vectors(self, vectors: np.ndarray) -> None:
        """Append vectors to the matrix.

        Args:
            vectors: Array of shape ``(n, dim)``.
        """
        self._matrix = vectors if self._matrix.size == 0 else np.vstack([self._matrix, vectors])

    def _all_vectors(self) -> np.ndarray:
        """Return the whole matrix.

        Returns:
            An array of shape ``(n, dim)``.
        """
        return self._matrix

    def _search_vectors(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``k`` highest cosine similarities.

        Args:
            query: Array of shape ``(1, dim)``.
            k: Number of neighbours.

        Returns:
            A ``(scores, indices)`` pair sorted by descending score.
        """
        similarities = (self._matrix @ query.reshape(-1)).astype(np.float32)
        # argpartition finds the top k without sorting everything, then the slice
        # is sorted properly. At this size it is habit rather than necessity.
        top = np.argpartition(-similarities, kth=min(k, len(similarities) - 1))[:k]
        ordered = top[np.argsort(-similarities[top])]
        return similarities[ordered], ordered

    def _save_vectors(self, directory: Path) -> None:
        """Write the matrix to ``vectors.npy``.

        Args:
            directory: Destination directory.
        """
        np.save(directory / VECTORS_FILE, self._matrix)

    def _load_vectors(self, directory: Path) -> None:
        """Read the matrix from ``vectors.npy``.

        Args:
            directory: Source directory.

        Raises:
            VectorStoreError: If the vector file is missing.
        """
        path = directory / VECTORS_FILE
        if not path.exists():
            raise VectorStoreError(f"missing {VECTORS_FILE} in {directory}")
        self._matrix = np.load(path).astype(np.float32)


__all__ = ["NumpyVectorStore"]
