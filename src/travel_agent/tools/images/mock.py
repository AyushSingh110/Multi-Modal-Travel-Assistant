"""Mock image provider serving real, verified photographs with an offline fallback.

The assignment asks for valid image URLs, and a gallery of broken thumbnails is a
worse demo than no gallery at all. So the mock does not invent URLs: for each
seeded city it serves Wikimedia Commons photographs, and every URL here was
checked with an HTTP request that returned ``200 image/jpeg``.

Commons ``Special:FilePath`` links are used rather than direct ``upload.wikimedia``
paths because they resolve by filename and do not embed the storage hash, which
makes them stable and readable.

WHY THERE IS ALSO A LOCAL COPY
    Serving remote images means the gallery depends on Commons being reachable at
    the exact moment of the demo. A blocked network, captive-portal wifi, or a
    corporate proxy would turn the one screenshot that matters into a grid of
    broken-image icons. So every curated image also names a small bundled PNG in
    ``data/images/``, and the provider decides which to use:

        IMAGE_FALLBACK_MODE=auto    probe Commons once per process, use the
                                    bundled files if it is unreachable (default)
        IMAGE_FALLBACK_MODE=remote  always use Commons, never fall back
        IMAGE_FALLBACK_MODE=local   always use the bundled files, no network

    The bundled files are generated placeholders, not copies of the photographs:
    redistributing someone's photograph would mean shipping their licence
    obligations with the repository. They exist to keep the layout intact, and
    they say so on their face.

For a city with no curated set - Kyoto, Snohomish, anything reached through web
search - the provider falls back to seeded placeholder photography. The seed is
derived from the city name, so each city gets a consistent set of its own rather
than the same four pictures every time, and the caption says plainly that the
imagery is generic. Pretending a stock photo is Kyoto would be a lie the UI would
then repeat to the user.

ATTRIBUTION
    Photographer and licence for every Commons photograph are recorded in
    ``data/images/ATTRIBUTION.md``, read from the Commons API rather than guessed,
    and the credit string travels with each asset so the interface can show it.
"""

from __future__ import annotations

import asyncio
import random

import httpx

from travel_agent.config.settings import PROJECT_ROOT, Settings, get_settings
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

#: Directory holding the committed offline fallback images.
LOCAL_IMAGE_DIR = PROJECT_ROOT / "data" / "images"

#: How long the Commons reachability probe may take. Deliberately short: it runs
#: before the gallery renders, and a slow answer is as bad as no answer.
PROBE_TIMEOUT_SECONDS = 2.5

#: Wikimedia's user-agent policy rejects generic library agents. Without this the
#: probe gets HTTP 403 even on a perfectly healthy network, which would silently
#: downgrade the gallery to placeholders - a false negative that is worse than no
#: probe at all, because it looks like it worked.
PROBE_HEADERS = {"User-Agent": "multi-modal-travel-assistant/1.0 (educational project)"}

# (commons filename, caption, credit, bundled fallback file).
# Credits are photographer plus licence, read from the Commons API.
CURATED_IMAGES: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "paris": (
        (
            "Tour_Eiffel_Wikimedia_Commons.jpg",
            "The Eiffel Tower from the Champ de Mars",
            "Benh LIEU SONG / Public domain (Wikimedia Commons)",
            "paris-1.png",
        ),
        (
            "Louvre_Museum_Wikimedia_Commons.jpg",
            "The Louvre and its glass pyramid",
            "Benh LIEU SONG / CC BY-SA 3.0 (Wikimedia Commons)",
            "paris-2.png",
        ),
        (
            "Notre-Dame_de_Paris_2013-07-24.jpg",
            "Notre-Dame de Paris on the Ile de la Cite",
            "P e z i / CC BY-SA 3.0 (Wikimedia Commons)",
            "paris-3.png",
        ),
        (
            "Arc_de_Triomphe,_Paris_21_October_2010.jpg",
            "The Arc de Triomphe at Place Charles de Gaulle",
            "Jiuguang Wang / CC BY-SA 2.0 (Wikimedia Commons)",
            "paris-4.png",
        ),
    ),
    "tokyo": (
        (
            "Skyscrapers_of_Shinjuku_2009_January.jpg",
            "The skyscraper district of Nishi-Shinjuku",
            "Morio / CC BY-SA 3.0 (Wikimedia Commons)",
            "tokyo-1.png",
        ),
        (
            "Shibuya_Crossing.jpg",
            "Shibuya scramble crossing",
            "Landry Miguel / CC BY-SA 4.0 (Wikimedia Commons)",
            "tokyo-2.png",
        ),
        (
            "Asakusa_Sensoji.jpg",
            "Senso-ji temple in Asakusa",
            "ElHeineken / CC BY 4.0 (Wikimedia Commons)",
            "tokyo-3.png",
        ),
        (
            "Tokyo_Tower_and_around_Skyscrapers.jpg",
            "Tokyo Tower above the Minato skyline",
            "Volfgang / CC BY-SA 3.0 (Wikimedia Commons)",
            "tokyo-4.png",
        ),
    ),
    "new york": (
        (
            "Lower_Manhattan_skyline_-_June_2017.jpg",
            "The Lower Manhattan skyline",
            "MusikAnimal / CC BY-SA 4.0 (Wikimedia Commons)",
            "new-york-1.png",
        ),
        (
            "Times_Square,_New_York_City_(HDR).jpg",
            "Times Square after dark",
            "Francisco Diez / CC BY 2.0 (Wikimedia Commons)",
            "new-york-2.png",
        ),
        (
            "Brooklyn_Bridge_Postdlf.jpg",
            "Brooklyn Bridge across the East River",
            "User:Postdlf / CC BY-SA 3.0 (Wikimedia Commons)",
            "new-york-3.png",
        ),
        (
            "Central_Park_-_The_Pond_(48377220157).jpg",
            "The Pond in Central Park",
            "Ajay Suresh / CC BY 2.0 (Wikimedia Commons)",
            "new-york-4.png",
        ),
    ),
}

#: Placeholder service for cities with no curated set. Deterministic per seed.
PLACEHOLDER_URL = "https://picsum.photos/seed/{seed}/900/600"

#: Bundled fallbacks used when an uncurated city cannot reach the network either.
GENERIC_FALLBACKS = ("generic-1.png", "generic-2.png", "generic-3.png", "generic-4.png")

# Process-wide cache for the reachability probe: None means "not yet checked".
_remote_reachable: bool | None = None


def reset_reachability_cache() -> None:
    """Forget the cached Commons probe result.

    Used by tests, and by the UI when the user asks to re-check connectivity.
    """
    global _remote_reachable
    _remote_reachable = None


async def commons_reachable(timeout: float = PROBE_TIMEOUT_SECONDS) -> bool:
    """Check once whether Wikimedia Commons can be reached.

    The result is cached for the life of the process: probing on every request
    would pay the latency every time, and connectivity rarely changes mid-demo.

    Args:
        timeout: Seconds to wait before giving up on the probe.

    Returns:
        ``True`` if Commons answered, ``False`` on any failure at all.
    """
    global _remote_reachable
    if _remote_reachable is not None:
        return _remote_reachable

    probe_url = f"{COMMONS}/{CURATED_IMAGES['paris'][0][0]}?width=64"
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers=PROBE_HEADERS
        ) as client:
            response = await client.head(probe_url)
        _remote_reachable = response.is_success
    except Exception as exc:  # noqa: BLE001 - any failure means "use the local copies"
        logger.warning("Wikimedia Commons unreachable (%s); using bundled images", exc)
        _remote_reachable = False

    logger.info("image source: %s", "remote (Commons)" if _remote_reachable else "bundled local")
    return _remote_reachable


class MockImageProvider(ImageProvider):
    """Returns verified photographs for seeded cities, placeholders otherwise.

    Attributes:
        name: Always ``"mock"``.
    """

    name = "mock"

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the provider.

        Args:
            settings: Settings to read latency, fallback mode and failure
                injection from.
        """
        self._settings = settings or get_settings()

    async def search_images(self, city: str, count: int = 4) -> list[ImageAsset]:
        """Return images for a city, simulating network latency first.

        Args:
            city: City name.
            count: How many images to return.

        Returns:
            Between one and ``count`` image assets, each carrying both a remote
            URL and a bundled fallback path where one exists.

        Raises:
            RetryableError: When failure injection simulates a 500.
            RateLimitError: When failure injection simulates a 429.
            MalformedPayloadError: When failure injection simulates a bad body.
        """
        await asyncio.sleep(self._latency_seconds())

        if self._settings.force_image_failure:
            logger.warning("image failure injection active: %s", self._settings.image_failure_mode)
            await maybe_fail(self._settings.image_failure_mode, IMAGES_TOOL)

        prefer_local = await self._should_use_local()
        curated = CURATED_IMAGES.get(city.strip().lower())

        if curated:
            return [
                ImageAsset(
                    url=f"{COMMONS}/{filename}?width={IMAGE_WIDTH}",
                    caption=caption,
                    credit=credit,
                    provider=self.name,
                    local_path=self._local_path(local_file),
                    prefer_local=prefer_local,
                )
                for filename, caption, credit, local_file in curated[:count]
            ]

        return self._placeholders(city, count, prefer_local=prefer_local)

    # ------------------------------------------------------------- internals --
    async def _should_use_local(self) -> bool:
        """Decide whether the bundled images should be preferred.

        Returns:
            ``True`` when configuration forces local files, or when the
            reachability probe says Commons is unavailable.
        """
        mode = self._settings.image_fallback_mode
        if mode == "local":
            return True
        if mode == "remote":
            return False
        return not await commons_reachable()

    def _local_path(self, filename: str) -> str | None:
        """Return the absolute path of a bundled image if it exists.

        Args:
            filename: Bundled file name, e.g. ``"paris-1.png"``.

        Returns:
            A path string, or ``None`` when the file is missing, so the UI never
            tries to render something that is not there.
        """
        path = LOCAL_IMAGE_DIR / filename
        return str(path) if path.exists() else None

    def _placeholders(self, city: str, count: int, *, prefer_local: bool) -> list[ImageAsset]:
        """Build deterministic placeholder images for an uncurated city.

        Args:
            city: City name, used as the seed so the set is stable per city.
            count: How many to return.
            prefer_local: Whether the bundled files should be used.

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
                local_path=self._local_path(
                    GENERIC_FALLBACKS[(index - 1) % len(GENERIC_FALLBACKS)]
                ),
                prefer_local=prefer_local,
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


__all__ = [
    "CURATED_IMAGES",
    "GENERIC_FALLBACKS",
    "IMAGE_WIDTH",
    "LOCAL_IMAGE_DIR",
    "MockImageProvider",
    "commons_reachable",
    "reset_reachability_cache",
]
