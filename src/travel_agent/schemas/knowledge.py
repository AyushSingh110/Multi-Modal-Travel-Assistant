"""Models for retrieved knowledge."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeChunk(BaseModel):
    """A retrievable passage of text about a city.

    Attributes:
        chunk_id: Stable identifier, ``"<city-slug>::<section-slug>"``.
        city: City the chunk describes, in display form, e.g. ``"New York"``.
        section: Section heading the chunk came from, e.g. ``"Getting around"``.
        text: The passage itself.
        source: Where it came from - a filename for seeded facts, a URL for web
            search results.
        score: Similarity score assigned at retrieval time; ``None`` when the chunk
            was not produced by a similarity search.
    """

    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    city: str
    section: str
    text: str = Field(min_length=1)
    source: str = ""
    score: float | None = None

    def as_context_block(self) -> str:
        """Render the chunk for inclusion in an LLM prompt.

        Returns:
            A labelled text block that keeps the section heading attached, so the
            model can attribute facts to a topic.
        """
        return f"[{self.city} / {self.section}]\n{self.text}"


class SearchHit(BaseModel):
    """A scored result returned by a vector store query.

    Attributes:
        chunk: The retrieved chunk.
        score: Cosine similarity in ``[-1, 1]``; in practice ``[0, 1]`` here
            because all vectors are non-negative and L2-normalised.
    """

    model_config = ConfigDict(extra="forbid")

    chunk: KnowledgeChunk
    score: float


__all__ = ["KnowledgeChunk", "SearchHit"]
