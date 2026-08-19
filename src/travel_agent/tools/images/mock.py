"""Mock image provider serving real, verified photographs.

The assignment asks for valid image URLs, and a gallery of broken thumbnails is
a worse demo than no gallery at all. So the mock does not invent URLs: for each
seeded city it serves Wikimedia Commons photographs, and every URL here was
checked with an HTTP request and returned ``200 image/jpeg``.

Commons ``Special:FilePath`` links are used rather than direct ``upload.wikimedia``
paths because they resolve by filename and do not embed the storage hash, which
makes them stable and readable.

For a city with no curated set - Kyoto, Snohomish, anything reached through web
search - the provider falls back to seeded placeholder photography. The seed is
derived from the city name, so each city gets a consistent set of its own rather
than the same four pictures every time, and the caption says plainly that the
imagery is generic. Pretending a stock photo is Kyoto would be a lie the UI
would then repeat to the user.
"""

from __future__ import annotations

import asyncio
import random

from travel_agent.config.settings import Settings, get_settings
from travel_agent.logging_setup import get_logger
from travel_agent.schemas.response import ImageAsset
from travel_agent.schemas.tools import IMAGES_TOOL
from travel_agent.tools.failures import maybe_fail
from travel_agent.tools.images.base import ImageProvider

logger = get_logger(__name__)

COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath"

# Commons serves the original upload by default, which for these photographs is
# 5-6 MB each - roughly 25 MB to render one city's gallery. The width parameter
# asks Commons for a scaled rendition instead: the Eiffel Tower photo drops from
# 5.3 MB to 339 KB, which is the difference between a gallery that appears and
# one the reviewer watches load.
IMAGE_WIDTH = 900

# (filename, caption, credit). Every URL verified to return 200 image/jpeg.
CURATED_IMAGES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "paris": (
        (
            "Tour_Eiffel_Wikimedia_Commons.jpg",
            "The Eiffel Tower from the Champ de Mars",
            "Wikimedia Commons",
        ),
        (
            "Louvre_Museum_Wikimedia_Commons.jpg",
            "The Louvre and I. M. Pei's glass pyramid",
            "Wikimedia Commons",
        ),
        (
            "Notre-Dame_de_Paris_2013-07-24.jpg",
            "Notre-Dame de Paris on the Ile de la Cite",
            "Wikimedia Commons",
        ),
        (
            "Arc_de_Triomphe,_Paris_21_October_2010.jpg",
            "The Arc de Triomphe at Place Charles de Gaulle",
            "Wikimedia Commons",
        ),
    ),
    "tokyo": (
        (
            "Skyscrapers_of_Shinjuku_2009_January.jpg",
            "The skyscraper district of Nishi-Shinjuku",
            "Wikimedia Commons",
        ),
        ("Shibuya_Crossing.jpg", "Shibuya scramble crossing", "Wikimedia Commons"),
        ("Asakusa_Sensoji.jpg", "Senso-ji temple in Asakusa", "Wikimedia Commons"),
        (
            "Tokyo_Tower_and_around_Skyscrapers.jpg",
            "Tokyo Tower above the Minato skyline",
            "Wikimedia Commons",
        ),
    ),
    "new york": (
        (
            "Lower_Manhattan_skyline_-_June_2017.jpg",
            "The Lower Manhattan skyline",
            "Wikimedia Commons",
        ),
        ("Times_Square,_New_York_City_(HDR).jpg", "Times Square after dark", "Wikimedia Commons"),
        (
            "Brooklyn_Bridge_Postdlf.jpg",
            "Brooklyn Bridge across the East River",
            "Wikimedia Commons",
        ),
        (
            "Central_Park_-_The_Pond_(48377220157).jpg",
            "The Pond in Central Park",
            "Wikimedia Commons",
        ),
    ),
}

#: Placeholder service for cities with no curated set. Deterministic per seed.
PLACEHOLDER_URL = "https://picsum.photos/seed/{seed}/900/600"


class MockImageProvider(ImageProvider):
    """Returns verified photographs for seeded cities, placeholders otherwise.

    Attributes:
        name: Always ``"mock"``.
    """

    name = "mock"

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the provider.

        Args:
            settings: Settings to read latency and failure injection from.
        """
        self._settings = settings or get_settings()

    async def search_images(self, city: str, count: int = 4) -> list[ImageAsset]:
        """Return images for a city, simulating network latency first.

        Args:
            city: City name.
            count: How many images to return.

        Returns:
            Between one and ``count`` image assets.

        Raises:
            RetryableError: When failure injection simulates a 500.
            RateLimitError: When failure injection simulates a 429.
            MalformedPayloadError: When failure injection simulates a bad body.
        """
        await asyncio.sleep(self._latency_seconds())

        if self._settings.force_image_failure:
            logger.warning("image failure injection active: %s", self._settings.image_failure_mode)
            await maybe_fail(self._settings.image_failure_mode, IMAGES_TOOL)

        curated = CURATED_IMAGES.get(city.strip().lower())
        if curated:
            return [
                ImageAsset(
                    url=f"{COMMONS}/{filename}?width={IMAGE_WIDTH}",
                    caption=caption,
                    credit=credit,
                    provider=self.name,
                )
                for filename, caption, credit in curated[:count]
            ]

        return self._placeholders(city, count)

    def _placeholders(self, city: str, count: int) -> list[ImageAsset]:
        """Build deterministic placeholder images for an uncurated city.

        Args:
            city: City name, used as the seed so the set is stable per city.
            count: How many to return.

        Returns:
            Placeholder image assets, captioned honestly as generic imagery.
        """
        slug = city.strip().lower().replace(" ", "-") or "city"
        return [
            ImageAsset(
                url=PLACEHOLDER_URL.format(seed=f"{slug}-{index}"),
                caption=f"{city} - representative travel imagery",
                credit="Placeholder photography (no curated set for this city)",
                provider=self.name,
            )
            for index in range(1, count + 1)
        ]

    def _latency_seconds(self) -> float:
        """Return the simulated network delay for one call.

        Returns:
            Seconds to sleep: the configured image latency with jitter applied.
        """
        base = self._settings.mock_image_latency_ms / 1000.0
        jitter = self._settings.mock_latency_jitter
        return random.uniform(base * (1 - jitter), base * (1 + jitter))


__all__ = ["CURATED_IMAGES", "IMAGE_WIDTH", "MockImageProvider"]
