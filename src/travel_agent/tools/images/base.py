"""Image provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from travel_agent.schemas.response import ImageAsset


class ImageProvider(ABC):
    """Finds photographs of a city.

    Attributes:
        name: Provider identifier recorded on each asset and in the trace.
    """

    name: str = "base"

    @abstractmethod
    async def search_images(self, city: str, count: int = 4) -> list[ImageAsset]:
        """Search for images.

        Args:
            city: City name to find photographs of.
            count: Maximum number of images to return.

        Returns:
            Image assets, possibly fewer than ``count``.

        Raises:
            ImageToolError: If the provider fails in a way worth reporting.
            RetryableError: If the failure is transient.
        """


__all__ = ["ImageProvider"]
