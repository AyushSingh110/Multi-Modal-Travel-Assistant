"""Vector store interface and shared persistence logic.

WHAT A VECTOR STORE IS
    A place to keep embeddings so you can ask "which of my stored passages is
    most similar to this query?" and get an answer quickly, together with the
    original text.

WHY THERE ARE TWO IMPLEMENTATIONS
    FAISS is the right default: it is the library the assignment names, it is a
    single wheel, and its inner-product index is exact at this corpus size. But a
    native wheel is exactly the kind of thing that fails to install on somebody
    else's machine, and this project has to run on a reviewer's laptop.
    :class:`~travel_agent.services.vectorstore.numpy_store.NumpyVectorStore`
    implements the identical interface with a brute-force dot product, so an
    install failure degrades to "slightly slower" instead of "does not run".

    With a few dozen chunks, brute force is not even measurably slower - the
    honest justification for FAISS here is that it is the right shape for a
    corpus that grows, not that it is faster today.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from travel_agent.exceptions import VectorStoreError
from travel_agent.schemas.knowledge import KnowledgeChunk, SearchHit

CHUNKS_FILE = "chunks.json"
MANIFEST_FILE = "manifest.json"
EMBEDDER_FILE = "embedder.json"


class BaseVectorStore(ABC):
    """Common behaviour for every vector store backend.

    Subclasses only have to say how vectors are stored, searched and written to
    disk; chunk bookkeeping and the manifest are handled here.

    Attributes:
        backend: Short backend name recorded in the manifest.
        dim: Dimensionality of the vectors this store holds.
    """

    backend: str = "base"

    def __init__(self, dim: int) -> None:
        """Initialise an empty store.

        Args:
            dim: Vector dimensionality.
        """
        self.dim = dim
        self._chunks: list[KnowledgeChunk] = []

    def __len__(self) -> int:
        """Return how many chunks are indexed."""
        return len(self._chunks)

    @property
    def chunks(self) -> list[KnowledgeChunk]:
        """All indexed chunks, in insertion order."""
        return list(self._chunks)

    @property
    def cities(self) -> list[str]:
        """Distinct city names present in the store, sorted alphabetically."""
        return sorted({chunk.city for chunk in self._chunks})

    # ------------------------------------------------------------- indexing --
    def add(self, chunks: list[KnowledgeChunk], vectors: np.ndarray) -> None:
        """Add chunks and their embeddings to the index.

        Args:
            chunks: Passages being indexed.
            vectors: Array of shape ``(len(chunks), dim)``, already L2-normalised.

        Raises:
            VectorStoreError: If the counts or dimensions do not line up. This is
                a programming error, so it fails loudly rather than silently
                indexing mismatched data.
        """
        if len(chunks) != vectors.shape[0]:
            raise VectorStoreError(f"got {len(chunks)} chunks but {vectors.shape[0]} vectors")
        if vectors.shape[1] != self.dim:
            raise VectorStoreError(
                f"vector dimension {vectors.shape[1]} does not match store dimension {self.dim}"
            )

        self._chunks.extend(chunks)
        self._add_vectors(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[SearchHit]:
        """Return the closest chunks to a query vector.

        Args:
            query_vector: Query embedding of shape ``(dim,)``, L2-normalised.
            top_k: Maximum number of hits to return.

        Returns:
            Hits ordered by descending cosine similarity. Empty when the store is
            empty, so callers never have to special-case an unseeded store.
        """
        if not self._chunks:
            return []

        k = min(top_k, len(self._chunks))
        scores, indices = self._search_vectors(
            np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32), k
        )

        hits: list[SearchHit] = []
        for score, index in zip(scores, indices, strict=True):
            if index < 0:  # FAISS pads with -1 when fewer than k results exist
                continue
            chunk = self._chunks[int(index)].model_copy(update={"score": float(score)})
            hits.append(SearchHit(chunk=chunk, score=float(score)))
        return hits

    # ---------------------------------------------------------- persistence --
    def save(
        self,
        directory: Path,
        *,
        embedder_name: str,
        embedder_state: dict[str, Any] | None = None,
        fingerprint: str = "",
    ) -> None:
        """Persist the store to a directory.

        Args:
            directory: Target directory; created if missing.
            embedder_name: Identifier of the embedder used to build the vectors.
            embedder_state: Learned embedder state (e.g. the idf table) that must
                be reused when embedding queries later.
            fingerprint: Hash of the source corpus, used to decide whether a
                rebuild is needed.
        """
        directory.mkdir(parents=True, exist_ok=True)

        (directory / CHUNKS_FILE).write_text(
            json.dumps([chunk.model_dump(mode="json") for chunk in self._chunks], indent=2),
            encoding="utf-8",
        )
        (directory / EMBEDDER_FILE).write_text(json.dumps(embedder_state or {}), encoding="utf-8")
        manifest = {
            "backend": self.backend,
            "dim": self.dim,
            "chunk_count": len(self._chunks),
            "cities": self.cities,
            "embedder": embedder_name,
            "fingerprint": fingerprint,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        self._save_vectors(directory)

    def load(self, directory: Path) -> None:
        """Load a previously saved store in place.

        Args:
            directory: Directory written by :meth:`save`.

        Raises:
            VectorStoreError: If the directory is missing or incomplete.
        """
        if not (directory / MANIFEST_FILE).exists():
            raise VectorStoreError(f"no vector store manifest found in {directory}")

        raw_chunks = json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))
        self._chunks = [KnowledgeChunk.model_validate(item) for item in raw_chunks]

        manifest = read_manifest(directory)
        self.dim = int(manifest["dim"])
        self._load_vectors(directory)

    @property
    def vectors(self) -> np.ndarray:
        """Every indexed vector, in insertion order.

        Needed to build per-city centroid vectors for the router. Exposed on the
        base class so both backends answer it the same way.

        Returns:
            An array of shape ``(len(self), dim)``.
        """
        if not self._chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._all_vectors()

    # ------------------------------------------------------------- subclass --
    @abstractmethod
    def _all_vectors(self) -> np.ndarray:
        """Return every stored vector.

        Returns:
            An array of shape ``(n, dim)`` in insertion order.
        """

    @abstractmethod
    def _add_vectors(self, vectors: np.ndarray) -> None:
        """Append vectors to the backend index.

        Args:
            vectors: Contiguous float32 array of shape ``(n, dim)``.
        """

    @abstractmethod
    def _search_vectors(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Run the nearest-neighbour search.

        Args:
            query: Contiguous float32 array of shape ``(1, dim)``.
            k: Number of neighbours to return.

        Returns:
            A ``(scores, indices)`` pair, each a flat array of length ``k``.
        """

    @abstractmethod
    def _save_vectors(self, directory: Path) -> None:
        """Write the backend index to disk.

        Args:
            directory: Destination directory, already created.
        """

    @abstractmethod
    def _load_vectors(self, directory: Path) -> None:
        """Read the backend index from disk.

        Args:
            directory: Directory written by :meth:`_save_vectors`.
        """


def read_manifest(directory: Path) -> dict[str, Any]:
    """Read a store manifest.

    Args:
        directory: Directory containing the store.

    Returns:
        The manifest contents.

    Raises:
        VectorStoreError: If the manifest is missing or unreadable.
    """
    path = directory / MANIFEST_FILE
    if not path.exists():
        raise VectorStoreError(f"no vector store manifest found in {directory}")
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise VectorStoreError(f"vector store manifest is corrupt: {exc}") from exc


def read_embedder_state(directory: Path) -> dict[str, Any]:
    """Read persisted embedder state.

    Args:
        directory: Directory containing the store.

    Returns:
        The embedder state, or an empty dictionary when none was saved.
    """
    path = directory / EMBEDDER_FILE
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "CHUNKS_FILE",
    "EMBEDDER_FILE",
    "MANIFEST_FILE",
    "BaseVectorStore",
    "read_embedder_state",
    "read_manifest",
]
