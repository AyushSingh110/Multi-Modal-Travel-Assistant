"""Weather provider interface.

Two implementations sit behind this: a mock that generates climate-plausible
data offline, and a live OpenWeatherMap client. The graph never learns which one
it is talking to - it asks for a forecast and gets a
:class:`~travel_agent.schemas.response.WeatherPayload`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from travel_agent.schemas.response import WeatherPayload


class WeatherProvider(ABC):
    """Fetches a daily forecast for a city.

    Attributes:
        name: Provider identifier recorded on the payload and in the trace.
    """

    name: str = "base"

    @abstractmethod
    async def fetch_forecast(
        self,
        city: str,
        days: int = 7,
        start_date: date | None = None,
    ) -> WeatherPayload:
        """Fetch the forecast.

        Args:
            city: City name to look up.
            days: How many daily points to return.
            start_date: First day of the window. Defaults to today.

        Returns:
            A populated weather payload.

        Raises:
            WeatherToolError: If the provider fails in a way the caller should
                report rather than retry.
            RetryableError: If the failure is transient and worth retrying.
        """


__all__ = ["WeatherProvider"]
