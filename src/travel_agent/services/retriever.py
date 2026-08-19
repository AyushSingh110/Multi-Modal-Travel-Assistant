"""Retrieval service: the embedder and the vector store working as one unit.

The router and the retrieval node both need the same two things - "how well does
the knowledge base cover this city?" and "give me the passages". Keeping them in
one service guarantees both questions are answered with the *same* embedder
state, which is what makes the similarity threshold meaningful.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np

from travel_agent.config.settings import Settings, get_settings
from travel_agent.exceptions import VectorStoreError
from travel_agent.logging_setup import get_logger
from travel_agent.schemas.knowledge import KnowledgeChunk, SearchHit
from travel_agent.services.embeddings.base import BaseEmbedder
from travel_agent.services.embeddings.factory import get_embedder
from travel_agent.services.vectorstore.base import BaseVectorStore, read_embedder_state
from travel_agent.services.vectorstore.factory import load_vector_store

logger = get_logger(__name__)


# Lexical shortcuts for names the corpus does not spell the way people type them.
# This is a gazetteer, not intelligence: it exists so "NYC" and "New York City"
# reach the same profile as "New York" without relying on a similarity score to
# make a decision that is really a naming convention.
def normalise_city_name(text: str) -> str:
    """Fold a city name to a comparable form.

    Strips accents, punctuation and case so that "Zurich", "zürich" and
    "ZÜRICH" are one key. Unicode decomposition (NFKD) splits an accented
    character into its base letter plus a combining mark, and the marks are then
    discarded - which is why this works for any accented name, not just ones
    someone remembered to add to a lookup table.

    Args:
        text: Raw city name as typed.

    Returns:
        The folded form, or an empty string when nothing usable remains.
    """
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", without_accents)
    return re.sub(r"\s+", " ", cleaned).strip()


CITY_ALIASES: dict[str, str] = {
    "nyc": "New York",
    "new york city": "New York",
    "ny": "New York",
    "the big apple": "New York",
    "tokyo japan": "Tokyo",
    "paris france": "Paris",
}


class KnowledgeRetriever:
    """Similarity search over the seeded city corpus.

    Attributes:
        store: The loaded vector store.
        embedder: The embedder used to build that store.
    """

    def __init__(self, store: BaseVectorStore, embedder: BaseEmbedder) -> None:
        """Initialise the retriever.

        Args:
            store: A loaded vector store.
            embedder: An embedder whose state matches the store.
        """
        self.store = store
        self.embedder = embedder
        self._city_profiles: dict[str, np.ndarray] | None = None

    @classmethod
    def from_disk(
        cls,
        directory: Path | None = None,
        *,
        settings: Settings | None = None,
    ) -> KnowledgeRetriever:
        """Load the retriever from a seeded vector store directory.

        Args:
            directory: Store directory. Defaults to the configured location.
            settings: Settings to read from. Defaults to the process singleton.

        Returns:
            A ready retriever.

        Raises:
            VectorStoreError: If the store has not been seeded yet.
        """
        settings = settings or get_settings()
        directory = directory or settings.vector_store_path

        store = load_vector_store(directory, settings=settings)
        embedder = get_embedder(settings)
        # Critical: restore the idf table saved at seed time. Embedding a query
        # with different corpus statistics would produce scores that are not
        # comparable with the ones the threshold was calibrated against.
        embedder.load_state_dict(read_embedder_state(directory))
        return cls(store, embedder)

    @property
    def known_cities(self) -> list[str]:
        """Cities the knowledge base covers."""
        return self.store.cities

    def search(self, query: str, top_k: int = 4) -> list[SearchHit]:
        """Return the passages most similar to a query.

        Args:
            query: Natural-language query, usually the resolved city name.
            top_k: Maximum number of hits.

        Returns:
            Hits ordered by descending similarity.
        """
        return self.store.search(self.embedder.embed_query(query), top_k=top_k)

    # ------------------------------------------------------------- routing --
    @property
    def city_profiles(self) -> dict[str, np.ndarray]:
        """One vector per city: the normalised mean of that city's chunk vectors.

        WHY NOT JUST USE THE BEST-MATCHING CHUNK
            The first version of this router scored a query against individual
            chunks and took the highest. It ranked "Kyoto" *above* "Paris",
            because the Tokyo day-trips passage mentions Kyoto once, and a single
            strong sentence in one chunk beat a city that is discussed across
            nine. Routing on that signal would have sent a city we know nothing
            about to the vector store.

            Averaging a city's chunks fixes it. A word that appears throughout a
            city's documents survives the mean; a word mentioned once in nine
            passages is diluted to roughly a ninth of its weight. That is exactly
            the distinction the router needs: "is this city a subject of my
            knowledge base?" rather than "is this word anywhere in it?".

        Returns:
            A mapping of city name to a unit-length profile vector, computed once
            and cached.
        """
        if self._city_profiles is None:
            vectors = self.store.vectors
            profiles: dict[str, np.ndarray] = {}
            for city in self.store.cities:
                rows = [
                    index for index, chunk in enumerate(self.store.chunks) if chunk.city == city
                ]
                centroid = vectors[rows].mean(axis=0)
                norm = float(np.linalg.norm(centroid))
                profiles[city] = centroid / norm if norm else centroid
            self._city_profiles = profiles
        return self._city_profiles

    def city_scores(self, query: str) -> dict[str, float]:
        """Score a query against every city profile.

        Args:
            query: Natural-language query, usually the resolved city name.

        Returns:
            A mapping of city name to cosine similarity, highest first.
        """
        query_vector = self.embedder.embed_query(query)
        scores = {
            city: float(np.dot(profile, query_vector))
            for city, profile in self.city_profiles.items()
        }
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    def best_match(self, query: str) -> tuple[str | None, float]:
        """Return the closest city in the store and its similarity score.

        IMPORTANT - pass the *resolved city name*, not the raw user sentence.
        Embedding "what is the weather in Kyoto next week" scores 0.089 against
        Paris, because "weather", "next" and "week" are ordinary words that appear
        throughout the corpus, while "Kyoto" contributes nothing. Embedding the
        extracted slot value "Kyoto" scores 0.040 and routes correctly. The slot
        extraction happens in the intent node precisely so this stays clean.

        This is the number the conditional edge routes on. It is deliberately a
        *score*, not a boolean: showing the reviewer "0.34 against a 0.15
        threshold" explains the decision in a way that "found: true" cannot.

        Args:
            query: Natural-language query, usually the resolved city name.

        Returns:
            A ``(city, score)`` pair. ``(None, 0.0)`` when the store is empty.
        """
        scores = self.city_scores(query)
        if not scores:
            return None, 0.0
        city, score = next(iter(scores.items()))
        return city, score

    def find_city_by_name(self, text: str) -> str | None:
        """Resolve a city name to a store city by exact or alias match.

        The router uses this before it looks at any similarity score. Whether the
        knowledge base has documents for "New York" is a question about names,
        and answering a naming question with a cosine threshold is how you end up
        explaining to a panel why "NYC" went to web search.

        Args:
            text: A city name, e.g. ``"nyc"`` or ``"New York City"``.

        Returns:
            The matching store city, or ``None`` when there is no exact match.
        """
        normalised = normalise_city_name(text)
        if not normalised:
            return None

        # Candidate forms, most literal first: the whole string, the part before a
        # comma ("Tokyo, Japan"), and the string without a trailing "city".
        candidates = [normalised]
        if "," in text:
            head = normalise_city_name(text.split(",", 1)[0])
            if head:
                candidates.append(head)
        if normalised.endswith(" city"):
            candidates.append(normalised[: -len(" city")].strip())

        known = {normalise_city_name(city): city for city in self.known_cities}
        for candidate in candidates:
            if candidate in known:
                return known[candidate]
            alias_target = CITY_ALIASES.get(candidate)
            if alias_target and alias_target in self.known_cities:
                return alias_target
        return None

    def chunks_for_city(self, city: str, limit: int = 8) -> list[KnowledgeChunk]:
        """Return every indexed chunk belonging to one city.

        Used once the router has committed to the vector-store path: at that
        point the city is known, so exact filtering beats similarity ranking.

        Args:
            city: City name, matched case-insensitively.
            limit: Maximum chunks to return.

        Returns:
            The city's chunks, in corpus order.
        """
        wanted = city.strip().lower()
        return [chunk for chunk in self.store.chunks if chunk.city.lower() == wanted][:limit]


def try_load_retriever(settings: Settings | None = None) -> KnowledgeRetriever | None:
    """Load the retriever, returning ``None`` instead of raising when unseeded.

    The UI uses this so a reviewer who forgot to run the seeder gets a clear
    banner rather than a stack trace.

    Args:
        settings: Settings to read from. Defaults to the process singleton.

    Returns:
        The retriever, or ``None`` if the store is missing or unreadable.
    """
    try:
        return KnowledgeRetriever.from_disk(settings=settings)
    except VectorStoreError as exc:
        logger.warning("vector store unavailable: %s", exc)
        return None


__all__ = ["CITY_ALIASES", "KnowledgeRetriever", "normalise_city_name", "try_load_retriever"]
