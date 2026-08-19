"""Tests for slot extraction: which city, which dates, which kind of turn."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from travel_agent.config.settings import Settings
from travel_agent.graph.nodes.core import _extract_city, _extract_date_range, _is_plausible_city
from travel_agent.schemas.intent import DateRange
from travel_agent.services.retriever import KnowledgeRetriever, try_load_retriever


@pytest.fixture(scope="module")
def retriever() -> KnowledgeRetriever | None:
    """The seeded retriever, used by the extractor as a gazetteer."""
    return try_load_retriever(Settings(_env_file=None))


# ================================================================ cities ====
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Tell me about Tokyo", "Tokyo"),
        ("tell me about tokyo", "Tokyo"),
        ("Tokyo", "Tokyo"),
        ("What about Paris?", "Paris"),
        ("Tell me about New York City", "New York"),
        ("nyc please", "New York"),
        ("how is the weather in Paris", "Paris"),
    ],
)
def test_known_cities_resolve_through_the_gazetteer(
    retriever: KnowledgeRetriever | None, query: str, expected: str
) -> None:
    assert _extract_city(query, retriever) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Tell me about Kyoto", "Kyoto"),
        ("I am visiting Osaka next week", "Osaka"),
        ("how is the weather in Reykjavik", "Reykjavik"),
        ("Should I go to Lisbon in March", "Lisbon"),
        ("Hey, tell me about Snohomish", "Snohomish"),
    ],
)
def test_unknown_cities_are_still_extracted(
    retriever: KnowledgeRetriever | None, query: str, expected: str
) -> None:
    """An unknown city must be recognised as a city - that is what routes it to the web."""
    assert _extract_city(query, retriever) == expected


def test_a_leading_capitalised_filler_word_is_not_mistaken_for_a_city(
    retriever: KnowledgeRetriever | None,
) -> None:
    """Regression test for a bug caught in a demo run.

    "Now tell me about Kyoto" resolved the city as **"Now"**: the extractor took
    the first capitalised token that was not in the filler list. The fix reads the
    grammar - a preposition names its object - and prefers a candidate that is not
    the opening word of the sentence, since an opening capital means nothing.
    """
    assert _extract_city("Now tell me about Kyoto", retriever) == "Kyoto"
    assert _extract_city("Okay, tell me about Kyoto", retriever) == "Kyoto"
    assert _extract_city("So what about Osaka", retriever) == "Osaka"


@pytest.mark.parametrize(
    "query",
    [
        "what about next week?",
        "and the weekend?",
        "how about tomorrow",
        "",
        "   ",
        "hello",
    ],
)
def test_queries_with_no_city_return_none(retriever: KnowledgeRetriever | None, query: str) -> None:
    """Returning None is what triggers the clarify path instead of a guess."""
    assert _extract_city(query, retriever) is None


def test_extraction_works_without_a_gazetteer() -> None:
    """An unseeded vector store must not break slot extraction."""
    assert _extract_city("Tell me about Kyoto", None) == "Kyoto"


def test_filler_words_are_rejected_as_cities() -> None:
    assert not _is_plausible_city("Now")
    assert not _is_plausible_city("What")
    assert not _is_plausible_city("Ok")
    assert _is_plausible_city("Kyoto")
    assert _is_plausible_city("New York")


# ================================================================= dates ====
def test_default_window_is_a_week_from_today() -> None:
    window, changed = _extract_date_range("Tell me about Tokyo", DateRange())

    assert window.days == 7
    assert window.start == date.today()
    assert not changed


def test_next_week_moves_the_window_forward(retriever: KnowledgeRetriever | None) -> None:
    window, changed = _extract_date_range("what about next week?", DateRange())

    assert changed
    assert window.label == "next week"
    assert window.start == date.today() + timedelta(days=7)


def test_tomorrow_is_recognised() -> None:
    window, changed = _extract_date_range("what about tomorrow", DateRange())

    assert changed
    assert window.start == date.today() + timedelta(days=1)


def test_in_n_days_is_recognised() -> None:
    window, changed = _extract_date_range("what about in 3 days", DateRange())

    assert changed
    assert window.start == date.today() + timedelta(days=3)


def test_an_absurd_offset_is_clamped() -> None:
    """The forecast schema allows at most 14 days; a request beyond it must not crash."""
    window, changed = _extract_date_range("what about in 99 days", DateRange())

    assert changed
    assert window.start <= date.today() + timedelta(days=14)


def test_the_weekend_produces_a_short_window() -> None:
    window, changed = _extract_date_range("what about the weekend", DateRange())

    assert changed
    assert window.days == 2
    assert window.label == "this weekend"


def test_a_stale_window_is_refreshed_to_today() -> None:
    """A conversation resumed the next day must not forecast from yesterday."""
    stale = DateRange(start=date.today() - timedelta(days=3), days=7, label="next 7 days")

    window, changed = _extract_date_range("Tell me about Tokyo", stale)

    assert window.start == date.today()
    assert not changed, "refreshing a stale start is not a user-requested change"


def test_date_range_end_is_inclusive() -> None:
    window = DateRange(start=date(2026, 8, 19), days=7)

    assert window.end == date(2026, 8, 25)


def test_shifting_a_range_keeps_its_width() -> None:
    original = DateRange(start=date(2026, 8, 19), days=5, label="next 7 days")

    shifted = original.shifted(weeks=1)

    assert shifted.days == 5
    assert shifted.start == date(2026, 8, 26)
