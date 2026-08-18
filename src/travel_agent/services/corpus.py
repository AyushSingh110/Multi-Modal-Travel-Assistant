"""Loading and chunking the seed corpus of city facts.

WHAT CHUNKING IS AND WHY IT MATTERS
    Embeddings represent a whole passage as one vector. Feed in an entire
    document and the vector becomes an average of everything it says, which
    matches nothing well. Split it too finely and each piece loses the context
    that made it meaningful.

    The corpus here is written in markdown with one ``##`` section per topic -
    transit, food, etiquette - so the document structure already marks the
    natural boundaries. One section becomes one chunk. That gives passages of
    roughly 100 to 150 words, each self-contained and about a single subject,
    with the heading kept as metadata so retrieved text can be attributed.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from travel_agent.exceptions import VectorStoreError
from travel_agent.logging_setup import get_logger
from travel_agent.schemas.knowledge import KnowledgeChunk

logger = get_logger(__name__)

_TITLE_PATTERN = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)
_SECTION_PATTERN = re.compile(r"^##\s+(?P<heading>.+?)\s*$", re.MULTILINE)


def slugify(text: str) -> str:
    """Convert a heading into an identifier-safe slug.

    Args:
        text: Arbitrary text, e.g. ``"Getting around"``.

    Returns:
        A lowercase, hyphen-separated slug, e.g. ``"getting-around"``.
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def parse_city_document(path: Path) -> list[KnowledgeChunk]:
    """Parse one city markdown file into chunks.

    Args:
        path: Path to a markdown file whose first line is a ``#`` city title and
            whose body is a series of ``##`` sections.

    Returns:
        One chunk per section, in document order.

    Raises:
        VectorStoreError: If the file has no city title or no sections.
    """
    text = path.read_text(encoding="utf-8")

    title_match = _TITLE_PATTERN.search(text)
    if title_match is None:
        raise VectorStoreError(f"{path.name} has no '# City' title line")
    city = title_match.group("title").strip()

    matches = list(_SECTION_PATTERN.finditer(text))
    if not matches:
        raise VectorStoreError(f"{path.name} has no '## Section' headings to chunk on")

    chunks: list[KnowledgeChunk] = []
    for index, match in enumerate(matches):
        heading = match.group("heading").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            logger.warning("skipping empty section %r in %s", heading, path.name)
            continue

        chunks.append(
            KnowledgeChunk(
                chunk_id=f"{slugify(city)}::{slugify(heading)}",
                city=city,
                section=heading,
                # The heading is prepended to the embedded text as well as kept as
                # metadata: "Getting around" is a strong retrieval signal for a
                # transit question, and dropping it would throw that away.
                text=f"{heading}. {body}",
                source=path.name,
            )
        )

    return chunks


def load_corpus(directory: Path) -> list[KnowledgeChunk]:
    """Load and chunk every city file in a directory.

    Args:
        directory: Directory containing ``*.md`` city fact files.

    Returns:
        All chunks from all files, ordered by filename.

    Raises:
        VectorStoreError: If the directory is missing or contains no markdown.
    """
    if not directory.exists():
        raise VectorStoreError(f"city facts directory not found: {directory}")

    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise VectorStoreError(f"no markdown files found in {directory}")

    chunks: list[KnowledgeChunk] = []
    for path in paths:
        chunks.extend(parse_city_document(path))
    return chunks


def corpus_fingerprint(chunks: list[KnowledgeChunk]) -> str:
    """Hash the corpus so the seeder can tell whether a rebuild is needed.

    Args:
        chunks: The parsed corpus.

    Returns:
        A short hex digest covering every chunk id and its text.
    """
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(chunk.text.encode("utf-8"))
    return digest.hexdigest()[:16]


def chunk_counts_by_city(chunks: list[KnowledgeChunk]) -> dict[str, int]:
    """Count chunks per city.

    Args:
        chunks: The parsed corpus.

    Returns:
        A mapping of city name to chunk count, ordered by city name.
    """
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.city] = counts.get(chunk.city, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "chunk_counts_by_city",
    "corpus_fingerprint",
    "load_corpus",
    "parse_city_document",
    "slugify",
]
