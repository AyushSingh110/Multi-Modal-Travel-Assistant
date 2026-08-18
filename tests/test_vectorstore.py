"""Tests for the corpus, embedder, vector stores and the FAISS fallback path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from travel_agent.config.settings import get_settings
from travel_agent.exceptions import VectorStoreError
from travel_agent.schemas.knowledge import KnowledgeChunk
from travel_agent.services.corpus import (
    chunk_counts_by_city,
    corpus_fingerprint,
    load_corpus,
    parse_city_document,
    slugify,
)
from travel_agent.services.embeddings.hashed import HashedTfIdfEmbedder
from travel_agent.services.retriever import KnowledgeRetriever
from travel_agent.services.vectorstore import factory as store_factory
from travel_agent.services.vectorstore.numpy_store import NumpyVectorStore

CITY_FACTS_DIR = get_settings().city_facts_dir


# ------------------------------------------------------------------ corpus --
def test_slugify_produces_identifier_safe_text() -> None:
    assert slugify("Getting around") == "getting-around"
    assert slugify("Money, safety and practicalities") == "money-safety-and-practicalities"


def test_corpus_covers_exactly_the_three_seeded_cities() -> None:
    counts = chunk_counts_by_city(load_corpus(CITY_FACTS_DIR))

    assert set(counts) == {"Paris", "Tokyo", "New York"}


def test_every_city_has_enough_chunks_to_be_useful() -> None:
    for city, count in chunk_counts_by_city(load_corpus(CITY_FACTS_DIR)).items():
        assert count >= 8, f"{city} has only {count} chunks"


def test_chunks_are_substantial_and_carry_their_heading() -> None:
    for chunk in load_corpus(CITY_FACTS_DIR):
        assert len(chunk.text) > 200, f"{chunk.chunk_id} is too thin to retrieve well"
        assert chunk.text.startswith(chunk.section)
        assert chunk.chunk_id.count("::") == 1


def test_fingerprint_changes_when_the_corpus_changes() -> None:
    chunks = load_corpus(CITY_FACTS_DIR)
    edited = [*chunks[:-1], chunks[-1].model_copy(update={"text": chunks[-1].text + " extra"})]

    assert corpus_fingerprint(chunks) != corpus_fingerprint(edited)
    assert corpus_fingerprint(chunks) == corpus_fingerprint(load_corpus(CITY_FACTS_DIR))


def test_document_without_a_title_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.md"
    path.write_text("## Only a section\n\nbody", encoding="utf-8")

    with pytest.raises(VectorStoreError, match="no '# City' title"):
        parse_city_document(path)


def test_document_without_sections_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.md"
    path.write_text("# Somewhere\n\njust prose, no headings", encoding="utf-8")

    with pytest.raises(VectorStoreError, match="no '## Section'"):
        parse_city_document(path)


# ---------------------------------------------------------------- embedder --
def test_embeddings_are_unit_length() -> None:
    embedder = HashedTfIdfEmbedder(dim=256)
    embedder.fit(["paris is a city", "tokyo is a city"])

    vector = embedder.embed_query("paris")

    assert vector.shape == (256,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_empty_text_gives_a_zero_vector_rather_than_nan() -> None:
    embedder = HashedTfIdfEmbedder(dim=64)
    embedder.fit(["something"])

    vector = embedder.embed_query("   ")

    assert not np.isnan(vector).any()
    assert np.allclose(vector, 0.0)


def test_embedder_state_survives_a_save_and_load_cycle() -> None:
    original = HashedTfIdfEmbedder(dim=128)
    original.fit(["paris museums and bakeries", "tokyo trains and ramen"])
    restored = HashedTfIdfEmbedder(dim=128)
    restored.load_state_dict(original.state_dict())

    assert np.allclose(original.embed_query("paris trains"), restored.embed_query("paris trains"))


# ------------------------------------------------------------ vector store --
def _tiny_store() -> NumpyVectorStore:
    embedder = HashedTfIdfEmbedder(dim=128)
    chunks = [
        KnowledgeChunk(chunk_id="a::1", city="A", section="s", text="paris bakeries and museums"),
        KnowledgeChunk(chunk_id="b::1", city="B", section="s", text="tokyo ramen and trains"),
    ]
    embedder.fit([chunk.text for chunk in chunks])
    store = NumpyVectorStore(embedder.dim)
    store.add(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    return store


def test_search_ranks_the_matching_chunk_first() -> None:
    store = _tiny_store()
    embedder = HashedTfIdfEmbedder(dim=128)
    embedder.fit(["paris bakeries and museums", "tokyo ramen and trains"])

    hits = store.search(embedder.embed_query("ramen"), top_k=2)

    assert hits[0].chunk.city == "B"
    assert hits[0].score > hits[1].score


def test_search_on_an_empty_store_returns_no_hits() -> None:
    assert NumpyVectorStore(16).search(np.zeros(16, dtype=np.float32)) == []


def test_mismatched_chunk_and_vector_counts_are_rejected() -> None:
    store = NumpyVectorStore(8)
    chunk = KnowledgeChunk(chunk_id="a::1", city="A", section="s", text="text")

    with pytest.raises(VectorStoreError, match="1 chunks but 2 vectors"):
        store.add([chunk], np.zeros((2, 8), dtype=np.float32))


def test_mismatched_dimensions_are_rejected() -> None:
    store = NumpyVectorStore(8)
    chunk = KnowledgeChunk(chunk_id="a::1", city="A", section="s", text="text")

    with pytest.raises(VectorStoreError, match="does not match store dimension"):
        store.add([chunk], np.zeros((1, 16), dtype=np.float32))


def test_store_round_trips_through_disk(tmp_path: Path) -> None:
    store = _tiny_store()
    store.save(tmp_path, embedder_name="test", embedder_state={"dim": 128}, fingerprint="abc")

    reloaded = NumpyVectorStore(128)
    reloaded.load(tmp_path)

    assert len(reloaded) == len(store)
    assert reloaded.cities == store.cities


def test_loading_an_unseeded_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(VectorStoreError, match="not found|manifest"):
        store_factory.load_vector_store(tmp_path / "missing")


# -------------------------------------------------------- backend fallback --
def test_faiss_backend_is_used_when_available() -> None:
    store = store_factory.create_vector_store(32, "faiss")

    assert store.backend == "faiss"


def test_falls_back_to_numpy_when_faiss_cannot_be_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the FAISS import to fail and assert the app keeps working.

    A native wheel failing on a reviewer's machine must degrade to a slower
    backend, never to a crash.
    """
    import travel_agent.services.vectorstore.faiss_store as faiss_module

    def explode() -> None:
        raise VectorStoreError("simulated: faiss native library failed to load")

    monkeypatch.setattr(faiss_module, "_import_faiss", explode)

    store = store_factory.create_vector_store(32, "faiss")

    assert store.backend == "numpy"
    assert isinstance(store, NumpyVectorStore)


def test_numpy_backend_can_be_requested_explicitly() -> None:
    assert store_factory.create_vector_store(32, "numpy").backend == "numpy"


# --------------------------------------------------- retriever end-to-end --
def _build_retriever(tmp_path: Path) -> KnowledgeRetriever:
    chunks = load_corpus(CITY_FACTS_DIR)
    embedder = HashedTfIdfEmbedder(dim=512)
    embedder.fit([chunk.text for chunk in chunks])
    store = store_factory.create_vector_store(embedder.dim, "numpy")
    store.add(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    store.save(tmp_path, embedder_name=embedder.name, embedder_state=embedder.state_dict())
    return KnowledgeRetriever.from_disk(tmp_path)


# The threshold the router ships with. Kept here so a change to the corpus that
# breaks the separation fails a test rather than silently degrading routing.
ROUTER_THRESHOLD = 0.07


def test_retriever_matches_seeded_cities_above_the_threshold(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path)

    for city in ("Paris", "Tokyo", "New York"):
        matched, score = retriever.best_match(city)
        assert matched == city
        assert score > ROUTER_THRESHOLD, f"{city} scored only {score:.4f}"


def test_retriever_scores_unknown_cities_below_the_threshold(tmp_path: Path) -> None:
    """The routing decision depends on this gap existing."""
    retriever = _build_retriever(tmp_path)

    for city in ("Snohomish", "Reykjavik", "Bogota", "Ulaanbaatar", "Kyoto"):
        _, score = retriever.best_match(city)
        assert score < ROUTER_THRESHOLD, f"{city} scored {score:.4f}, above the threshold"


def test_seeded_cities_outscore_unseeded_ones_with_a_real_margin(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path)

    seeded = [retriever.best_match(city)[1] for city in ("Paris", "Tokyo", "New York")]
    unseeded = [retriever.best_match(city)[1] for city in ("Kyoto", "Snohomish", "Bogota")]

    assert min(seeded) > max(unseeded)
    assert min(seeded) - max(unseeded) > 0.05, "separation margin has collapsed"


def test_unknown_vocabulary_scores_exactly_zero(tmp_path: Path) -> None:
    """Regression test: out-of-vocabulary tokens must not produce hash noise.

    Before queries dropped unseen tokens, "Reykjavik" scored 0.10 against Paris
    purely through a hash collision - higher than several genuine matches.
    """
    retriever = _build_retriever(tmp_path)

    _, score = retriever.best_match("Ulaanbaatar")

    assert score == 0.0


def test_city_centroid_beats_naive_max_chunk_scoring(tmp_path: Path) -> None:
    """Regression test for the Kyoto bug.

    The Tokyo corpus mentions Kyoto once, in the day-trips section. Scoring a
    query against individual chunks and taking the maximum therefore ranked
    Kyoto above Paris, and would have routed an unknown city to the vector
    store. Averaging each city's chunks into a profile fixes it.
    """
    retriever = _build_retriever(tmp_path)

    _, kyoto_centroid = retriever.best_match("Kyoto")
    _, paris_centroid = retriever.best_match("Paris")
    kyoto_max_chunk = max(hit.score for hit in retriever.search("Kyoto", top_k=5))

    assert kyoto_centroid < paris_centroid, "centroid scoring must rank Paris above Kyoto"
    assert kyoto_max_chunk > kyoto_centroid, "max-chunk scoring is the weaker signal"


def test_city_scores_cover_every_known_city(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path)

    scores = retriever.city_scores("Tokyo")

    assert set(scores) == {"Paris", "Tokyo", "New York"}
    assert list(scores)[0] == "Tokyo", "results must be ordered best-first"


def test_gazetteer_resolves_aliases_and_casing(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path)

    assert retriever.find_city_by_name("paris") == "Paris"
    assert retriever.find_city_by_name("NYC") == "New York"
    assert retriever.find_city_by_name("New York City") == "New York"
    assert retriever.find_city_by_name("  tokyo  ") == "Tokyo"


def test_gazetteer_returns_none_for_unknown_cities(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path)

    assert retriever.find_city_by_name("Kyoto") is None
    assert retriever.find_city_by_name("") is None


def test_chunks_for_city_filters_exactly(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path)

    chunks = retriever.chunks_for_city("tokyo")

    assert chunks
    assert all(chunk.city == "Tokyo" for chunk in chunks)
