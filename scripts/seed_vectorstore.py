"""Build the vector store from the seed corpus of city facts.

Run once before starting the app::

    python scripts/seed_vectorstore.py            # build if needed
    python scripts/seed_vectorstore.py --force    # always rebuild
    python scripts/seed_vectorstore.py --backend numpy

The script is idempotent: it fingerprints the corpus and skips the rebuild when
the store on disk already matches, so putting it in a start-up script costs
nothing.

It also prints a similarity matrix. That output is not decoration - it is the
evidence that retrieval separates the three seeded cities from everything else,
and it is where the router threshold comes from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from travel_agent.config.settings import get_settings  # noqa: E402
from travel_agent.logging_setup import configure_logging, get_logger  # noqa: E402
from travel_agent.services.corpus import (  # noqa: E402
    chunk_counts_by_city,
    corpus_fingerprint,
    load_corpus,
)
from travel_agent.services.embeddings.factory import get_embedder  # noqa: E402
from travel_agent.services.retriever import KnowledgeRetriever  # noqa: E402
from travel_agent.services.vectorstore.base import read_manifest  # noqa: E402
from travel_agent.services.vectorstore.factory import create_vector_store  # noqa: E402

logger = get_logger("seed")

# Probes for the sanity matrix. These are CITY NAMES, not sentences, because the
# router scores the city slot extracted by the intent node rather than the raw
# query - see KnowledgeRetriever.best_match for why that distinction matters.
PROBE_QUERIES: list[str] = [
    "Paris",
    "Tokyo",
    "New York",
    "Kyoto",
    "Snohomish",
    "Reykjavik",
    "Bogota",
    "Ulaanbaatar",
]

IN_STORE_PROBES = {"Paris", "Tokyo", "New York"}

# Demonstrates the failure mode the slot extraction avoids: generic words in a
# raw sentence match the corpus even when the city in it is unknown.
RAW_QUERY_PROBES: list[str] = [
    "Tell me about Paris",
    "what is the weather in Kyoto next week",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even if the store is already current."
    )
    parser.add_argument(
        "--backend",
        choices=["faiss", "numpy"],
        default=None,
        help="Override the configured vector store backend.",
    )
    parser.add_argument(
        "--no-matrix", action="store_true", help="Skip the similarity sanity matrix."
    )
    return parser.parse_args()


def needs_rebuild(directory: Path, fingerprint: str, *, force: bool) -> bool:
    """Decide whether the store must be rebuilt.

    Args:
        directory: Vector store directory.
        fingerprint: Fingerprint of the current corpus.
        force: Whether the user asked for an unconditional rebuild.

    Returns:
        ``True`` when the store is missing, stale or a rebuild was forced.
    """
    if force:
        return True
    if not (directory / "manifest.json").exists():
        return True
    try:
        manifest = read_manifest(directory)
    except Exception:  # noqa: BLE001 - a corrupt manifest just means "rebuild"
        return True
    return str(manifest.get("fingerprint", "")) != fingerprint


def print_similarity_matrix(retriever: KnowledgeRetriever) -> None:
    """Print a query-by-city similarity matrix and a threshold recommendation.

    For each probe query the score shown is the *best* cosine similarity against
    any chunk of that city, which is exactly what the router compares against its
    threshold.

    Args:
        retriever: A retriever loaded from the freshly built store.
    """
    cities = retriever.known_cities
    header = f"{'city probe':<20}" + "".join(f"{city:>13}" for city in cities) + f"{'best':>9}"
    print()
    print("SIMILARITY MATRIX (cosine, higher is closer)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    in_store_best: list[float] = []
    out_of_store_best: list[float] = []

    for query in PROBE_QUERIES:
        per_city = retriever.city_scores(query)
        best = max(per_city.values()) if per_city else 0.0
        marker = "  (in store)" if query in IN_STORE_PROBES else ""
        (in_store_best if query in IN_STORE_PROBES else out_of_store_best).append(best)

        row = f"{query:<20}" + "".join(f"{per_city.get(city, 0.0):>13.3f}" for city in cities)
        print(f"{row}{best:>9.3f}{marker}")

    print("-" * len(header))
    if in_store_best and out_of_store_best:
        lowest_known = min(in_store_best)
        highest_unknown = max(out_of_store_best)
        print(f"  lowest score for a KNOWN city    : {lowest_known:.3f}")
        print(f"  highest score for an UNKNOWN city: {highest_unknown:.3f}")
        print(f"  separation margin                : {lowest_known - highest_unknown:.3f}")
        midpoint = (lowest_known + highest_unknown) / 2
        print(f"  midpoint (suggested threshold)   : {midpoint:.3f}")
        print()
        print("  The router threshold should sit inside that gap. evals/run_eval.py")
        print("  sweeps it across a labelled query set to pick the final value.")

    print()
    print("WHY THE ROUTER SCORES THE CITY SLOT, NOT THE RAW QUERY")
    print("-" * 62)
    for query in RAW_QUERY_PROBES:
        scores = retriever.city_scores(query)
        city, score = next(iter(scores.items()))
        print(f"  {query:<42} -> {city:<10} {score:.3f}")
    print("  A raw sentence drags in ordinary words that match every city, which")
    print("  is why the intent node extracts the city first and the router scores")
    print("  only that.")


def main() -> int:
    """Build the vector store and report on it.

    Returns:
        A process exit code.
    """
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    directory = settings.vector_store_path
    backend = args.backend or settings.vector_store_backend

    print("=" * 78)
    print("SEEDING VECTOR STORE")
    print("=" * 78)
    print(f"corpus   : {settings.city_facts_dir}")
    print(f"store    : {directory}")
    print(f"backend  : {backend}")
    print(f"embedder : {settings.embedding_provider} (dim={settings.embedding_dim})")
    print()

    chunks = load_corpus(settings.city_facts_dir)
    fingerprint = corpus_fingerprint(chunks)

    counts = chunk_counts_by_city(chunks)
    print("CHUNKS PER CITY")
    for city, count in counts.items():
        print(f"  {city:<14} {count:>3} chunks")
    print(f"  {'TOTAL':<14} {len(chunks):>3} chunks   fingerprint={fingerprint}")
    print()

    if not needs_rebuild(directory, fingerprint, force=args.force):
        print("Store is already up to date with this corpus - skipping rebuild.")
        print("(Use --force to rebuild anyway.)")
    else:
        embedder = get_embedder(settings)
        # Fit before embedding: the idf table has to be learned from the corpus
        # that is about to be indexed, and it is persisted so queries later use
        # exactly the same statistics.
        embedder.fit([chunk.text for chunk in chunks])

        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        store = create_vector_store(embedder.dim, backend, settings=settings)
        store.add(chunks, vectors)
        store.save(
            directory,
            embedder_name=embedder.name,
            embedder_state=embedder.state_dict(),
            fingerprint=fingerprint,
        )
        print(f"Built and saved {len(store)} vectors using the {store.backend} backend.")

    if not args.no_matrix:
        retriever = KnowledgeRetriever.from_disk(directory, settings=settings)
        print_similarity_matrix(retriever)

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
