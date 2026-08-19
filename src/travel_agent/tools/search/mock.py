"""Mock web search provider.

Written so the out-of-store demo path produces a genuinely good answer rather
than lorem ipsum. Kyoto and Snohomish are the two cities the assignment names as
examples, so both have real, specific material; any other city gets plausible
generic results that still exercise the full summarisation path.

The point of the mock is to prove the *plumbing*: the router chose web search,
the tool ran, its results reached the summariser, and the citations reached the
UI. Whether the text came from DuckDuckGo or from this file changes nothing about
that path, which is why swapping in the live provider is a config change.
"""

from __future__ import annotations

import asyncio
import random

from travel_agent.config.settings import Settings, get_settings
from travel_agent.logging_setup import get_logger
from travel_agent.tools.search.base import SearchProvider, SearchResult

logger = get_logger(__name__)

CURATED_RESULTS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "kyoto": (
        (
            "Kyoto - the imperial capital for a thousand years",
            "Kyoto was Japan's capital from 794 until 1868 and escaped the worst of "
            "wartime bombing, leaving roughly 1,600 Buddhist temples and 400 Shinto "
            "shrines standing. Seventeen sites form a single UNESCO World Heritage "
            "listing. The city sits in a basin ringed by mountains on three sides, "
            "which makes summers humid and winters sharp.",
            "https://en.wikipedia.org/wiki/Kyoto",
        ),
        (
            "What to see in Kyoto",
            "Fushimi Inari's vermilion torii gates climb the mountain behind the "
            "shrine and are quietest before eight in the morning. Kinkaku-ji, the "
            "Golden Pavilion, and Ginkaku-ji sit at opposite ends of the city. "
            "Arashiyama's bamboo grove is best paired with the Sagano railway. "
            "Gion remains the geiko district; photography is restricted on its "
            "private lanes.",
            "https://www.japan-guide.com/e/e2158.html",
        ),
        (
            "Getting around Kyoto",
            "Kyoto's subway has only two lines, so buses do most of the work and a "
            "day pass is good value. The city is laid out on a grid inherited from "
            "the Heian period, which makes it unusually easy to navigate by street "
            "name. Cycling is popular and flat. Kyoto Station is 15 minutes from "
            "Osaka and 2 hours 15 from Tokyo by shinkansen.",
            "https://www.insidekyoto.com/getting-around-kyoto",
        ),
        (
            "When to visit Kyoto",
            "Cherry blossom in early April and autumn colour in the second half of "
            "November are spectacular and extremely crowded, with accommodation "
            "priced accordingly. June brings the rainy season and August is hot and "
            "humid. Late May and early October are the quiet compromises.",
            "https://www.kyoto.travel/en/season",
        ),
    ),
    "snohomish": (
        (
            "Snohomish, Washington",
            "Snohomish is a city of about 10,000 people in Snohomish County, "
            "Washington, roughly 30 miles north-east of Seattle where the Pikeen "
            "and Snohomish rivers meet. Founded in 1859, its First Street historic "
            "district is lined with Victorian commercial buildings and it markets "
            "itself as the antique capital of the north-west.",
            "https://en.wikipedia.org/wiki/Snohomish,_Washington",
        ),
        (
            "Things to do in Snohomish",
            "The historic downtown is the main draw, with antique malls, cafes and "
            "a riverfront trail. Harvey Field runs skydiving and scenic flights. "
            "The Centennial Trail follows a former railway grade for 30 miles "
            "north towards Arlington. Nearby valley farms open for pumpkin season "
            "in October.",
            "https://www.snohomishwa.gov/visitors",
        ),
        (
            "Snohomish climate",
            "The climate is temperate oceanic, like most of western Washington: "
            "cool wet winters with highs near 8 C, dry mild summers averaging 25 C "
            "in July and August, and most of the annual rainfall between October "
            "and March. Snow is occasional and rarely lasts.",
            "https://www.weather.gov/sew/",
        ),
    ),
}


class MockSearchProvider(SearchProvider):
    """Returns curated results for the demo cities, generic ones otherwise.

    Attributes:
        name: Always ``"mock"``.
    """

    name = "mock"

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the provider.

        Args:
            settings: Settings to read the simulated latency from.
        """
        self._settings = settings or get_settings()

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run a simulated search.

        Args:
            query: Search query. The city is recovered from it by matching against
                the curated keys, so "Kyoto travel guide overview" still works.
            max_results: Maximum number of results.

        Returns:
            Search results, curated where available.
        """
        base = self._settings.mock_search_latency_ms / 1000.0
        jitter = self._settings.mock_latency_jitter
        await asyncio.sleep(random.uniform(base * (1 - jitter), base * (1 + jitter)))

        lowered = query.strip().lower()
        for city, results in CURATED_RESULTS.items():
            if city in lowered:
                return [
                    SearchResult(title=title, snippet=snippet, url=url)
                    for title, snippet, url in results[:max_results]
                ]

        return self._generic_results(query, max_results)

    @staticmethod
    def _generic_results(query: str, max_results: int) -> list[SearchResult]:
        """Build plausible results for a city with no curated material.

        Args:
            query: The original query, used verbatim in the titles.
            max_results: Maximum number of results.

        Returns:
            Generic but structurally realistic search results.
        """
        city = query.replace("travel guide", "").replace("overview", "").strip() or "the city"
        templates = (
            (
                f"{city} - travel overview",
                f"{city} is a destination with its own centre, transport links and "
                f"local character. Visitors typically arrive by rail or road and "
                f"base themselves near the historic core, where most restaurants "
                f"and accommodation are concentrated.",
                f"https://en.wikipedia.org/wiki/{city.replace(' ', '_')}",
            ),
            (
                f"Things to do in {city}",
                f"Local highlights in {city} usually include the central district, "
                f"a museum or gallery covering regional history, markets on set days "
                f"of the week, and green space within walking distance of the centre.",
                "https://www.lonelyplanet.com/",
            ),
            (
                f"{city} weather and best time to visit",
                f"Shoulder seasons - late spring and early autumn - tend to offer "
                f"the best balance of comfortable temperatures and lower visitor "
                f"numbers in {city}. Check a current forecast before travelling.",
                "https://www.timeanddate.com/weather/",
            ),
        )
        return [
            SearchResult(title=title, snippet=snippet, url=url)
            for title, snippet, url in templates[:max_results]
        ]


__all__ = ["CURATED_RESULTS", "MockSearchProvider"]
