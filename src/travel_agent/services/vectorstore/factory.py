"""Vector store construction and loading, with automatic backend fallback.

This is the only module that knows both backends exist. Everything else asks for
"a vector store" and gets whichever one works on this machine.
"""

from __future__ import annotations

from pathlib import Path

from travel_agent.config.settings import Settings, get_settings
from travel_agent.exceptions import VectorStoreError
from travel_agent.logging_setup import get_logger
from travel_agent.services.vectorstore.base import BaseVectorStore, read_manifest
from travel_agent.services.vectorstore.numpy_store import NumpyVectorStore

logger = get_logger(__name__)


def create_vector_store(
    dim: int,
    backend: str = "faiss",
    *,
    settings: Settings | None = None,
) -> BaseVectorStore:
    """Create an empty vector store, falling back to NumPy when FAISS is unusable.

    The fallback is silent by design at the call site but loud in the logs: a
    reviewer whose FAISS wheel is broken should still get a working app, and
    should be able to see why the backend changed.

    Args:
        dim: Vector dimensionality.
        backend: Requested backend, ``"faiss"`` or ``"numpy"``.
        settings: Unused today; accepted so callers can pass configuration
            without the signature changing later.

    Returns:
        A ready, empty store.
    """
    del settings  # reserved for future backend options

    if backend == "numpy":
        return NumpyVectorStore(dim)

    try:
        from travel_agent.services.vectorstore.faiss_store import FaissVectorStore

        return FaissVectorStore(dim)
    except VectorStoreError as exc:
        logger.warning("FAISS unavailable (%s); falling back to the NumPy store", exc)
        return NumpyVectorStore(dim)


def load_vector_store(
    directory: Path | None = None,
    *,
    settings: Settings | None = None,
) -> BaseVectorStore:
    """Load a persisted vector store from disk.

    The backend recorded in the manifest is preferred, but a store saved with
    FAISS on one machine can still be read here: the chunks are backend-neutral
    JSON, and only the vector file differs.

    Args:
        directory: Store directory. Defaults to the configured location.
        settings: Settings to read from. Defaults to the process singleton.

    Returns:
        The loaded store.

    Raises:
        VectorStoreError: If the store has not been seeded, or if it was saved
            with a backend whose vector file cannot be read here.
    """
    settings = settings or get_settings()
    directory = directory or settings.vector_store_path

    if not directory.exists():
        raise VectorStoreError(
            f"vector store not found at {directory}. Run: python scripts/seed_vectorstore.py"
        )

    manifest = read_manifest(directory)
    store = create_vector_store(int(manifest["dim"]), str(manifest.get("backend", "faiss")))

    try:
        store.load(directory)
    except VectorStoreError as exc:
        # Saved with FAISS but FAISS is missing here (or vice versa). Rebuilding is
        # cheap and always correct, so say so plainly instead of half-failing.
        raise VectorStoreError(
            f"could not load the {manifest.get('backend')} index ({exc}). "
            "Re-seed with: python scripts/seed_vectorstore.py --force"
        ) from exc

    logger.info(
        "loaded vector store: backend=%s chunks=%d cities=%s",
        store.backend,
        len(store),
        ", ".join(store.cities),
    )
    return store


__all__ = ["create_vector_store", "load_vector_store"]
