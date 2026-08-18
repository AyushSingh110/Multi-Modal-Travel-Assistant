"""A dependency-free TF-IDF embedder using the hashing trick.

WHY NOT A NEURAL EMBEDDER BY DEFAULT
    ``sentence-transformers`` pulls in PyTorch - hundreds of megabytes, a slow
    cold start, and a download that can fail on a reviewer's machine. OpenAI
    embeddings need an API key, which breaks the "runs with zero keys" promise.

    This project's retrieval problem is narrow: decide whether a query names one
    of three seeded cities. That is a *lexical* question, and TF-IDF answers it
    well. The trade-off is real and worth stating out loud: this embedder has no
    idea that "the French capital" means Paris. If that mattered, the interface
    lets a neural embedder be swapped in without touching the store or the router.

HOW IT WORKS
    1. Split text into lowercase word tokens, drop stopwords.
    2. Hash each token to a column index (the "hashing trick": no vocabulary
       needs to be stored, and unseen words at query time are handled naturally).
    3. Weight each token by ``tf * idf`` - term frequency times inverse document
       frequency, so a rare word like "shinjuku" counts for far more than "city".
    4. L2-normalise, so a dot product equals cosine similarity.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from travel_agent.services.embeddings.base import BaseEmbedder, l2_normalise

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Deliberately small: only words that carry no discriminative signal in travel
# text. Large stopword lists drop words like "no" and "not", which can flip the
# meaning of a fact.
_STOPWORDS: frozenset[str] = frozenset(  # noqa: SIM905 - kept as prose for readability
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "there",
        "here",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "into",
        "about",
        "over",
        "under",
        "between",
        "it",
        "its",
        "as",
        "also",
        "very",
        "more",
        "most",
        "much",
        "many",
        "some",
        "any",
        "each",
        "every",
        "i",
        "you",
        "he",
        "she",
        "they",
        "we",
        "me",
        "him",
        "her",
        "them",
        "us",
        "your",
        "our",
        "their",
        "his",
        "do",
        "does",
        "did",
        "done",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
    ]
)


class HashedTfIdfEmbedder(BaseEmbedder):
    """TF-IDF vectoriser with hashed feature indices.

    Attributes:
        dim: Vector dimensionality.
        name: Identifier persisted in the store manifest.
    """

    name = "hashed-tfidf-v1"

    def __init__(self, dim: int = 512) -> None:
        """Initialise the embedder.

        Args:
            dim: Number of hash buckets, i.e. the vector dimensionality. 512 is
                ample for a corpus of a few dozen chunks; collisions are rare and
                the sign trick below cancels most of their effect.
        """
        self.dim = dim
        self._idf: dict[str, float] = {}
        self._default_idf: float = 1.0
        self._document_count: int = 0

    # ------------------------------------------------------------------ fit --
    def fit(self, texts: list[str]) -> None:
        """Compute inverse document frequencies over the corpus.

        Args:
            texts: Every chunk that will be indexed.
        """
        self._document_count = len(texts)
        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(self._tokenise(text)))

        # Smoothed idf, the standard form: rare terms score high, terms present
        # in every document score near zero.
        self._idf = {
            token: math.log((1 + self._document_count) / (1 + count)) + 1.0
            for token, count in document_frequency.items()
        }
        # Weight used for a token that was never seen while fitting. It applies
        # to documents only - see _embed for why queries drop unknown tokens.
        self._default_idf = math.log(1 + self._document_count) + 1.0

    # ------------------------------------------------------------- embedding --
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents.

        Args:
            texts: Passages to embed.

        Returns:
            An array of shape ``(len(texts), dim)``.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query.

        Args:
            text: Query text.

        Returns:
            A vector of shape ``(dim,)``. All-zero when the query shares no
            vocabulary with the indexed corpus, which is the correct answer for a
            city the knowledge base has never heard of.
        """
        return self._embed(text, drop_unknown=bool(self._idf)).reshape(self.dim)

    def _embed(self, text: str, *, drop_unknown: bool = False) -> np.ndarray:
        """Vectorise one piece of text.

        Args:
            text: Text to embed.
            drop_unknown: Discard tokens that were not seen during fitting.

        Returns:
            A normalised array of shape ``(1, dim)``.
        """
        vector = np.zeros((1, self.dim), dtype=np.float32)
        tokens = self._tokenise(text)
        if drop_unknown:
            # WHY QUERIES DROP OUT-OF-VOCABULARY TOKENS
            #   With the hashing trick a token maps to a bucket whether or not the
            #   corpus contains it, so an unknown word lands on some bucket that
            #   real content also occupies. Giving it the maximum idf weight - the
            #   textbook treatment for a rare term - made that collision the
            #   loudest component of the query vector: "Reykjavik" scored 0.10
            #   against Paris purely by hash accident, higher than several genuine
            #   matches. An unseen token cannot carry real signal here, so it is
            #   dropped and contributes nothing.
            tokens = [token for token in tokens if token in self._idf]
        if not tokens:
            return vector

        counts = Counter(tokens)
        for token, count in counts.items():
            # Sublinear term frequency: the tenth mention of "temple" says much
            # less than the second.
            tf = 1.0 + math.log(count)
            idf = self._idf.get(token, self._default_idf)
            index, sign = self._hash(token)
            vector[0, index] += sign * tf * idf

        return l2_normalise(vector)

    # -------------------------------------------------------------- internals --
    def _tokenise(self, text: str) -> list[str]:
        """Split text into meaningful lowercase tokens.

        Args:
            text: Raw text.

        Returns:
            Tokens with stopwords and single characters removed.
        """
        return [
            token
            for token in _TOKEN_PATTERN.findall(text.lower())
            if len(token) > 1 and token not in _STOPWORDS
        ]

    def _hash(self, token: str) -> tuple[int, float]:
        """Map a token to a column index and a sign.

        The sign trick (one bit of the same hash decides +1 or -1) makes hash
        collisions cancel out on average instead of always adding, which keeps
        similarity scores honest without storing a vocabulary.

        Args:
            token: Token to hash.

        Returns:
            An ``(index, sign)`` pair.
        """
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % self.dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    # ------------------------------------------------------------ persistence --
    def state_dict(self) -> dict[str, object]:
        """Return the learned idf table for persistence.

        Returns:
            A JSON-serialisable dictionary.
        """
        return {
            "dim": self.dim,
            "idf": self._idf,
            "default_idf": self._default_idf,
            "document_count": self._document_count,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore a persisted idf table.

        Queries must be embedded with the same statistics used to build the index,
        otherwise the scores are not comparable and the router threshold is
        meaningless.

        Args:
            state: Dictionary produced by :meth:`state_dict`.
        """
        self.dim = int(state.get("dim", self.dim))  # type: ignore[arg-type]
        self._idf = dict(state.get("idf", {}))  # type: ignore[arg-type]
        self._default_idf = float(state.get("default_idf", 1.0))  # type: ignore[arg-type]
        self._document_count = int(state.get("document_count", 0))  # type: ignore[arg-type]


__all__ = ["HashedTfIdfEmbedder"]
