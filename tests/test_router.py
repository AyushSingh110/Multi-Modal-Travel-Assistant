"""Tests for the layered knowledge router and its threshold guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from travel_agent.config.settings import Settings, get_settings
from travel_agent.schemas.knowledge import KnowledgeChunk
from travel_agent.services.corpus import load_corpus
from travel_agent.services.embeddings.hashed import HashedTfIdfEmbedder
from travel_agent.services.retriever import KnowledgeRetriever, normalise_city_name
from travel_agent.services.router import KnowledgeRouter
from travel_agent.services.vectorstore.factory import create_vector_store

CITY_FACTS_DIR = get_settings().city_facts_dir


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"_env_file": None}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture
def retriever(tmp_path: Path) -> KnowledgeRetriever:
    """A retriever over the real seeded corpus, built in a temp directory."""
    chunks = load_corpus(CITY_FACTS_DIR)
    embedder = HashedTfIdfEmbedder(dim=512)
    embedder.fit([chunk.text for chunk in chunks])
    store = create_vector_store(embedder.dim, "numpy")
    store.add(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    store.save(tmp_path, embedder_name=embedder.name, embedder_state=embedder.state_dict())
    return KnowledgeRetriever.from_disk(tmp_path)


@pytest.fixture
def router(retriever: KnowledgeRetriever) -> KnowledgeRouter:
    return KnowledgeRouter(retriever, settings=_settings(router_similarity_threshold=0.07))


# ------------------------------------------------------------ name folding --
def test_normalisation_strips_accents_case_and_punctuation() -> None:
    assert normalise_city_name("Zürich") == "zurich"
    assert normalise_city_name("ZÜRICH") == "zurich"
    assert normalise_city_name("  São Paulo!  ") == "sao paulo"
    assert normalise_city_name("Málaga") == "malaga"
    assert normalise_city_name("") == ""


def test_normalisation_is_idempotent() -> None:
    once = normalise_city_name("Düsseldorf")
    assert normalise_city_name(once) == once


# --------------------------------------------------- layer 1: the gazetteer --
@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("Tokyo", "Tokyo"),
        ("tokyo", "Tokyo"),
        ("TOKYO", "Tokyo"),
        ("  Tokyo  ", "Tokyo"),
        ("Tokyo, Japan", "Tokyo"),
        ("NYC", "New York"),
        ("nyc", "New York"),
        ("New York City", "New York"),
        ("new york city", "New York"),
        ("Paris, France", "Paris"),
    ],
)
def test_gazetteer_resolves_aliases_and_variants(
    router: KnowledgeRouter, typed: str, expected: str
) -> None:
    decision = router.decide(typed)

    assert decision.route == "vector"
    assert decision.match_reason == "exact"
    assert decision.matched_city == expected


def test_exact_match_still_records_the_similarity_score(router: KnowledgeRouter) -> None:
    """Layering must not hide the measurement - the UI shows both."""
    decision = router.decide("Tokyo")

    assert decision.match_reason == "exact"
    assert decision.score > 0.0
    assert decision.threshold == pytest.approx(0.07)
    assert set(decision.all_scores) == {"Paris", "Tokyo", "New York"}
    assert "Tokyo" in decision.reason and "similarity" in decision.reason


def test_exact_match_wins_even_when_similarity_is_ambiguous(
    retriever: KnowledgeRetriever,
) -> None:
    """The case a bare threshold fumbles.

    A synthetic embedding is forced into the ambiguous 0.04-0.10 band - below the
    0.07 threshold - for a city the store genuinely covers. Similarity alone
    would send it to web search; the gazetteer layer catches it.
    """
    target_score = 0.06  # inside the 0.04-0.10 band, below the 0.07 threshold
    profile = retriever.city_profiles["New York"]

    # Build a unit vector whose cosine with the profile is exactly target_score:
    # take random noise, remove its component along the profile so what is left is
    # perpendicular, then recombine the two in a right-angled triangle.
    rng = np.random.default_rng(0)
    noise = rng.normal(size=profile.shape).astype(np.float32)
    perpendicular = noise - float(np.dot(noise, profile)) * profile
    perpendicular /= np.linalg.norm(perpendicular)
    blended = target_score * profile + np.sqrt(1 - target_score**2) * perpendicular
    retriever.embedder.embed_query = lambda text: blended  # type: ignore[method-assign]

    ambiguous_score = retriever.city_scores("New York City")["New York"]
    assert 0.04 < ambiguous_score < 0.10, f"fixture drifted: score was {ambiguous_score:.4f}"

    router = KnowledgeRouter(
        retriever, settings=_settings(router_similarity_threshold=0.07), validate=False
    )
    decision = router.decide("New York City")

    assert decision.route == "vector", "gazetteer must catch what the threshold misses"
    assert decision.match_reason == "exact"
    assert decision.score == pytest.approx(ambiguous_score)


# ------------------------------------------------- layer 2: similarity gate --
def test_unknown_city_routes_to_web_search(router: KnowledgeRouter) -> None:
    decision = router.decide("Kyoto")

    assert decision.route == "web"
    assert decision.match_reason == "similarity"
    assert decision.score < 0.07
    assert "below" in decision.reason


@pytest.mark.parametrize("city", ["Snohomish", "Reykjavik", "Bogota", "Ulaanbaatar"])
def test_cities_outside_the_corpus_all_route_to_web(router: KnowledgeRouter, city: str) -> None:
    assert router.decide(city).route == "web"


def test_similarity_layer_can_route_to_the_vector_store(retriever: KnowledgeRetriever) -> None:
    """With a low enough threshold, a near-miss name is accepted on score alone."""
    router = KnowledgeRouter(
        retriever, settings=_settings(router_similarity_threshold=0.01), validate=False
    )

    decision = router.decide("Kyoto")

    assert decision.route == "vector"
    assert decision.match_reason == "similarity"


# ------------------------------------------------------- layer 3: clarify --
@pytest.mark.parametrize("value", [None, "", "   "])
def test_no_resolved_city_asks_for_clarification(
    router: KnowledgeRouter, value: str | None
) -> None:
    decision = router.decide(value)

    assert decision.route == "clarify"
    assert decision.match_reason == "none"


# ------------------------------------------------- the start-up guard --
def test_healthy_threshold_reports_ok(router: KnowledgeRouter) -> None:
    diagnostics = router.diagnostics

    assert diagnostics.status == "ok"
    assert diagnostics.is_healthy
    assert diagnostics.highest_unknown_score < 0.07 < diagnostics.lowest_known_score


def test_threshold_above_every_seeded_score_is_diagnosed_loudly(
    retriever: KnowledgeRetriever,
) -> None:
    """The exact failure that went unnoticed during development.

    0.55 was the planned value. No seeded city can reach it, so every request
    would have routed to web search while the app looked healthy.
    """
    router = KnowledgeRouter(
        retriever, settings=_settings(router_similarity_threshold=0.55), validate=False
    )

    diagnostics = router.check_threshold()

    assert diagnostics.status == "too_high"
    assert not diagnostics.is_healthy
    assert "TOO HIGH" in diagnostics.message
    assert "WEB SEARCH" in diagnostics.message
    assert "ROUTER_SIMILARITY_THRESHOLD" in diagnostics.message
    assert f"{diagnostics.lowest_known_score:.3f}" in diagnostics.message


def test_threshold_below_the_noise_floor_is_diagnosed(retriever: KnowledgeRetriever) -> None:
    router = KnowledgeRouter(
        retriever, settings=_settings(router_similarity_threshold=0.001), validate=False
    )

    diagnostics = router.check_threshold()

    assert diagnostics.status == "too_low"
    assert "TOO LOW" in diagnostics.message


def test_guard_warns_through_the_logger(
    retriever: KnowledgeRetriever, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        KnowledgeRouter(retriever, settings=_settings(router_similarity_threshold=0.55))

    assert any("TOO HIGH" in record.message for record in caplog.records)


def test_diagnostics_are_measured_once_and_cached(router: KnowledgeRouter) -> None:
    assert router.check_threshold() is router.check_threshold()


def test_empty_store_is_reported_rather_than_crashing(tmp_path: Path) -> None:
    embedder = HashedTfIdfEmbedder(dim=64)
    embedder.fit(["placeholder text for the idf table"])
    store = create_vector_store(embedder.dim, "numpy")
    store.add(
        [KnowledgeChunk(chunk_id="x::1", city="X", section="s", text="placeholder")],
        embedder.embed_documents(["placeholder"]),
    )
    store._chunks.clear()  # simulate an index that holds no chunks
    store.save(tmp_path, embedder_name=embedder.name, embedder_state=embedder.state_dict())

    router = KnowledgeRouter(KnowledgeRetriever.from_disk(tmp_path), settings=_settings())

    assert router.diagnostics.status == "unknown"
    assert router.decide("Paris").route == "web"
