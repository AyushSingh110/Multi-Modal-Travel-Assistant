"""FAISS-backed vector store.

Uses ``IndexFlatIP`` - a flat (exhaustive) index scored by inner product. Two
deliberate choices:

* **Flat, not IVF/HNSW.** Approximate indexes trade recall for speed on millions
  of vectors. With a few dozen chunks they would be slower to build, harder to
  reason about, and less accurate. Flat is exact.
* **Inner product, not L2.** Every vector is L2-normalised before it reaches the
  index, and for unit vectors the inner product equals the cosine similarity.
  That means the score FAISS returns is directly comparable with the router
  threshold, with no conversion step to get wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from travel_agent.exceptions import VectorStoreError
from travel_agent.services.vectorstore.base import BaseVectorStore

INDEX_FILE = "index.faiss"


class FaissVectorStore(BaseVectorStore):
    """Exact cosine search backed by ``faiss.IndexFlatIP``."""

    backend = "faiss"

    def __init__(self, dim: int) -> None:
        """Initialise an empty index.

        Args:
            dim: Vector dimensionality.

        Raises:
            VectorStoreError: If FAISS cannot be imported. Callers should use the
                factory, which catches this and falls back to NumPy.
        """
        super().__init__(dim)
        self._faiss = _import_faiss()
        self._index: Any = self._faiss.IndexFlatIP(dim)

    def _add_vectors(self, vectors: np.ndarray) -> None:
        """Add vectors to the index.

        Args:
            vectors: Array of shape ``(n, dim)``.
        """
        self._index.add(vectors)

    def _all_vectors(self) -> np.ndarray:
        """Rebuild the stored vectors from the flat index.

        ``IndexFlatIP`` keeps the original vectors verbatim, so reconstruction is
        exact rather than approximate.

        Returns:
            An array of shape ``(n, dim)``.
        """
        return np.asarray(self._index.reconstruct_n(0, self._index.ntotal), dtype=np.float32)

    def _search_vectors(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search the index.

        Args:
            query: Array of shape ``(1, dim)``.
            k: Number of neighbours.

        Returns:
            A ``(scores, indices)`` pair, flattened from the FAISS batch shape.
        """
        scores, indices = self._index.search(query, k)
        return scores[0], indices[0]

    def _save_vectors(self, directory: Path) -> None:
        """Write the index to disk.

        Args:
            directory: Destination directory.
        """
        self._faiss.write_index(self._index, str(directory / INDEX_FILE))

    def _load_vectors(self, directory: Path) -> None:
        """Read the index from disk.

        Args:
            directory: Source directory.

        Raises:
            VectorStoreError: If the index file is missing.
        """
        path = directory / INDEX_FILE
        if not path.exists():
            raise VectorStoreError(f"missing {INDEX_FILE} in {directory}")
        self._index = self._faiss.read_index(str(path))


def _import_faiss() -> Any:
    """Import FAISS, converting an import failure into a domain error.

    Isolated in a function so the factory has a single seam to catch, and so
    tests can force the fallback path by patching this name.

    Returns:
        The imported ``faiss`` module.

    Raises:
        VectorStoreError: If FAISS is not installed or its native library fails
            to load.
    """
    try:
        import faiss
    except Exception as exc:  # noqa: BLE001 - a broken native wheel raises OSError, not ImportError
        raise VectorStoreError(f"faiss is unavailable: {exc}") from exc
    return faiss


__all__ = ["FaissVectorStore"]
