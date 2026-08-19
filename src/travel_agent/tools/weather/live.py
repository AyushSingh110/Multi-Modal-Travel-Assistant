"""Live weather provider backed by OpenWeatherMap.

Kept deliberately small. Its purpose is to prove the provider abstraction is
real: switching from the mock to this is ``WEATHER_PROVIDER=live`` plus a key,
with no change anywhere else in the application.

Two calls are needed because the free tier has no "forecast by city name"
endpoint that returns daily aggregates: geocode the city, then pull the 3-hourly
forecast and roll it up into days.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime

import httpx

from travel_agent.config.settings import Settings, get_settings
from travel_agent.exceptions import ConfigurationError, RetryableError, WeatherToolError
from travel_agent.logging_setup import get_logger
from travel_agent.schemas.response import ForecastPoint, WeatherPayload
from travel_agent.schemas.tools import WEATHER_TOOL
from travel_agent.tools.retry import rate_limit_from_response
from travel_agent.tools.weather.base import WeatherProvider

logger = get_logger(__name__)

GEOCODE_URL = "https://api.openweathermap.org/geo/1.0/direct"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


class OpenWeatherProvider(WeatherProvider):
    """Fetches forecasts from OpenWeatherMap.

    Attributes:
        name: Always ``"openweather"``.
    """

    name = "openweather"

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the provider.

        Args:
            settings: Settings holding the API key and timeouts.

        Raises:
            ConfigurationError: If the provider is selected without a key.
        """
        self._settings = settings or get_settings()
        if not self._settings.openweather_api_key:
            raise ConfigurationError("WEATHER_PROVIDER=live requires OPENWEATHER_API_KEY to be set")
        self._api_key = self._settings.openweather_api_key

    async def fetch_forecast(
        self,
        city: str,
        days: int = 7,
        start_date: date | None = None,
    ) -> WeatherPayload:
        """Fetch and aggregate the forecast.

        Args:
            city: City name to geocode and look up.
            days: Maximum number of daily points to return. The free tier covers
                five days, so fewer points than requested is normal.
            start_date: Ignored - the API only forecasts forward from now. The
                argument is kept so the interface matches the mock.

        Returns:
            A populated weather payload.

        Raises:
            WeatherToolError: If the city cannot be found or the response is
                unusable.
            RetryableError: On a transient HTTP failure.
            RateLimitError: On HTTP 429.
        """
        del start_date  # not supported upstream; documented above

        timeout = httpx.Timeout(self._settings.tool_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            latitude, longitude, resolved = await self._geocode(client, city)
            payload = await self._forecast(client, latitude, longitude)

        points = self._to_daily_points(payload)[:days]
        if not points:
            raise WeatherToolError(
                f"OpenWeather returned no usable forecast for {resolved}", WEATHER_TOOL
            )

        return WeatherPayload(
            city=resolved,
            provider=self.name,
            current_temp_c=points[0].temp_max_c,
            current_condition=points[0].condition,
            forecast=points,
        )

    async def _geocode(self, client: httpx.AsyncClient, city: str) -> tuple[float, float, str]:
        """Resolve a city name to coordinates.

        Args:
            client: An open HTTP client.
            city: City name.

        Returns:
            A ``(latitude, longitude, resolved_name)`` tuple.

        Raises:
            WeatherToolError: If the city is unknown to the geocoder.
        """
        response = await client.get(
            GEOCODE_URL, params={"q": city, "limit": 1, "appid": self._api_key}
        )
        self._raise_for_status(response)

        results = response.json()
        if not results:
            raise WeatherToolError(f"OpenWeather could not geocode {city!r}", WEATHER_TOOL)

        first = results[0]
        return float(first["lat"]), float(first["lon"]), str(first.get("name", city))

    async def _forecast(
        self, client: httpx.AsyncClient, latitude: float, longitude: float
    ) -> dict[str, object]:
        """Fetch the 3-hourly forecast.

        Args:
            client: An open HTTP client.
            latitude: Latitude in degrees.
            longitude: Longitude in degrees.

        Returns:
            The decoded JSON body.
        """
        response = await client.get(
            FORECAST_URL,
            params={
                "lat": latitude,
                "lon": longitude,
                "units": "metric",
                "appid": self._api_key,
            },
        )
        self._raise_for_status(response)
        return dict(response.json())

    @staticmethod
    def _to_daily_points(payload: dict[str, object]) -> list[ForecastPoint]:
        """Roll 3-hourly readings up into one point per day.

        Args:
            payload: The decoded forecast response.

        Returns:
            Daily forecast points in chronological order.
        """
        buckets: dict[date, list[dict[str, object]]] = defaultdict(list)
        for entry in payload.get("list", []):  # type: ignore[union-attr]
            moment = datetime.fromtimestamp(
                int(entry["dt"])
            )  # noqa: DTZ006 - local day is intended
            buckets[moment.date()].append(entry)

        points: list[ForecastPoint] = []
        for day, entries in sorted(buckets.items()):
            temps = [float(item["main"]["temp"]) for item in entries]  # type: ignore[index]
            conditions = [str(item["weather"][0]["main"]) for item in entries]  # type: ignore[index]
            humidity = [float(item["main"]["humidity"]) for item in entries]  # type: ignore[index]
            winds = [float(item["wind"]["speed"]) for item in entries]  # type: ignore[index]
            rain = [float(item.get("pop", 0.0)) for item in entries]  # type: ignore[union-attr]

            points.append(
                ForecastPoint(
                    date=day,
                    temp_max_c=round(max(temps), 1),
                    temp_min_c=round(min(temps), 1),
                    # The most frequent label across the day, not the first one.
                    condition=max(set(conditions), key=conditions.count),
                    precipitation_chance=int(round(max(rain) * 100)),
                    humidity_pct=int(round(sum(humidity) / len(humidity))),
                    wind_kph=round(max(winds) * 3.6, 1),
                )
            )
        return points

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """Convert an HTTP error into the project's exception vocabulary.

        Args:
            response: The response to inspect.

        Raises:
            RateLimitError: On HTTP 429, carrying any ``Retry-After`` value.
            RetryableError: On 5xx, which is worth another attempt.
            WeatherToolError: On any other error status.
        """
        if response.is_success:
            return

        rate_limited = rate_limit_from_response(response.status_code, dict(response.headers))
        if rate_limited is not None:
            raise rate_limited

        if response.status_code >= 500:
            raise RetryableError(f"OpenWeather returned HTTP {response.status_code}")

        raise WeatherToolError(
            f"OpenWeather returned HTTP {response.status_code}: {response.text[:200]}",
            WEATHER_TOOL,
        )


__all__ = ["OpenWeatherProvider"]
