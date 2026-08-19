"""Live image provider backed by the Unsplash API."""

from __future__ import annotations

import httpx

from travel_agent.config.settings import Settings, get_settings
from travel_agent.exceptions import ConfigurationError, ImageToolError, RetryableError
from travel_agent.logging_setup import get_logger
from travel_agent.schemas.response import ImageAsset
from travel_agent.schemas.tools import IMAGES_TOOL
from travel_agent.tools.images.base import ImageProvider
from travel_agent.tools.retry import rate_limit_from_response

logger = get_logger(__name__)

SEARCH_URL = "https://api.unsplash.com/search/photos"


class UnsplashImageProvider(ImageProvider):
    """Searches Unsplash for city photography.

    Attributes:
        name: Always ``"unsplash"``.
    """

    name = "unsplash"

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the provider.

        Args:
            settings: Settings holding the access key and timeouts.

        Raises:
            ConfigurationError: If the provider is selected without a key.
        """
        self._settings = settings or get_settings()
        if not self._settings.unsplash_access_key:
            raise ConfigurationError("IMAGE_PROVIDER=live requires UNSPLASH_ACCESS_KEY to be set")
        self._access_key = self._settings.unsplash_access_key

    async def search_images(self, city: str, count: int = 4) -> list[ImageAsset]:
        """Search Unsplash for photographs of a city.

        Args:
            city: City name.
            count: Maximum number of images.

        Returns:
            Image assets carrying photographer attribution, which Unsplash's
            terms require.

        Raises:
            ImageToolError: If the response is an error or unusable.
            RetryableError: On a transient HTTP failure.
            RateLimitError: On HTTP 429.
        """
        timeout = httpx.Timeout(self._settings.tool_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                SEARCH_URL,
                params={"query": f"{city} city", "per_page": count, "orientation": "landscape"},
                headers={"Authorization": f"Client-ID {self._access_key}"},
            )

        if not response.is_success:
            rate_limited = rate_limit_from_response(response.status_code, dict(response.headers))
            if rate_limited is not None:
                raise rate_limited
            if response.status_code >= 500:
                raise RetryableError(f"Unsplash returned HTTP {response.status_code}")
            raise ImageToolError(
                f"Unsplash returned HTTP {response.status_code}: {response.text[:200]}",
                IMAGES_TOOL,
            )

        results = response.json().get("results", [])
        return [
            ImageAsset(
                url=item["urls"]["regular"],
                caption=(item.get("description") or item.get("alt_description") or city)[:200],
                credit=f"{item['user']['name']} / Unsplash",
                provider=self.name,
            )
            for item in results[:count]
        ]


__all__ = ["UnsplashImageProvider"]
