"""Mock weather provider with climate-plausible data.

WHY THIS IS NOT `random.uniform(10, 25)`
    The forecast feeds a line chart the reviewer will look at. Uniform noise
    produces a jagged mess that reads as fake; a constant produces a flat line
    that reads as broken. Real weather has three visible properties, and this
    generator reproduces all three:

    1. **A seasonal baseline.** Tokyo in August is not Tokyo in January, so each
       city carries a real monthly climate table and the forecast starts from the
       right level for the date being asked about.
    2. **Day-to-day persistence.** Tomorrow resembles today. The generator walks
       a smooth curve rather than redrawing each day independently, so warm and
       cool spells last several days the way weather actually does.
    3. **Correlated conditions.** Precipitation chance follows the city's rainy
       season, and the text label ("Light rain", "Clear") is derived from that
       number rather than picked separately - so the chart and the labels agree.

    Output is deterministic for a given city and start date. The same demo run
    twice looks the same, which matters when a panel asks you to do it again.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
from datetime import date, timedelta

from travel_agent.config.settings import Settings, get_settings
from travel_agent.logging_setup import get_logger
from travel_agent.schemas.response import ForecastPoint, WeatherPayload
from travel_agent.schemas.tools import WEATHER_TOOL
from travel_agent.tools.failures import maybe_fail
from travel_agent.tools.weather.base import WeatherProvider

logger = get_logger(__name__)

# Monthly average high and low in Celsius, January to December. Real climate
# normals, rounded - enough to make the seasonal shape correct.
CLIMATE_TABLES: dict[str, dict[str, tuple[float, ...]]] = {
    "paris": {
        "high": (7, 8, 12, 16, 20, 23, 25, 25, 21, 16, 11, 8),
        "low": (3, 3, 5, 7, 11, 14, 16, 16, 13, 10, 6, 4),
        "rain": (45, 42, 40, 40, 40, 35, 32, 33, 38, 45, 48, 50),
    },
    "tokyo": {
        "high": (10, 10, 14, 19, 23, 26, 30, 31, 27, 22, 17, 12),
        "low": (2, 3, 6, 10, 15, 19, 23, 24, 21, 16, 10, 5),
        "rain": (25, 28, 40, 45, 48, 65, 60, 50, 62, 55, 35, 25),
    },
    "new york": {
        "high": (4, 6, 11, 17, 22, 27, 29, 28, 24, 18, 12, 7),
        "low": (-3, -2, 2, 7, 13, 18, 21, 20, 17, 10, 5, 0),
        "rain": (35, 33, 38, 40, 40, 38, 40, 38, 35, 33, 36, 38),
    },
    # Used for any city the table does not cover, nudged per city so two unknown
    # cities do not produce identical charts.
    "_default": {
        "high": (9, 11, 14, 18, 22, 26, 28, 27, 23, 18, 13, 10),
        "low": (2, 3, 5, 9, 13, 17, 19, 19, 15, 11, 6, 3),
        "rain": (38, 36, 38, 40, 42, 40, 38, 38, 40, 42, 42, 40),
    },
}

CONDITION_BANDS: tuple[tuple[int, str], ...] = (
    (15, "Clear"),
    (30, "Sunny spells"),
    (45, "Partly cloudy"),
    (60, "Overcast"),
    (75, "Light rain"),
    (100, "Rain"),
)


class MockWeatherProvider(WeatherProvider):
    """Generates realistic offline forecasts.

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

    async def fetch_forecast(
        self,
        city: str,
        days: int = 7,
        start_date: date | None = None,
    ) -> WeatherPayload:
        """Generate a forecast, simulating network latency first.

        Args:
            city: City name.
            days: Number of daily points.
            start_date: First day of the window. Defaults to today.

        Returns:
            A populated weather payload.

        Raises:
            RetryableError: When failure injection simulates a 500.
            RateLimitError: When failure injection simulates a 429.
            MalformedPayloadError: When failure injection simulates a bad body.
        """
        await asyncio.sleep(self._latency_seconds())

        if self._settings.force_weather_failure:
            logger.warning(
                "weather failure injection active: %s", self._settings.weather_failure_mode
            )
            await maybe_fail(self._settings.weather_failure_mode, WEATHER_TOOL)

        start = start_date or date.today()
        points = [self._point_for(city, start + timedelta(days=offset)) for offset in range(days)]
        current = points[0]

        return WeatherPayload(
            city=city,
            provider=self.name,
            # "Now" sits between today's low and high rather than at either end.
            current_temp_c=round((current.temp_max_c + current.temp_min_c) / 2, 1),
            current_condition=current.condition,
            forecast=points,
        )

    # ------------------------------------------------------------ generation --
    def _latency_seconds(self) -> float:
        """Return the simulated network delay for one call.

        Returns:
            Seconds to sleep: the configured weather latency with jitter applied.
        """
        base = self._settings.mock_weather_latency_ms / 1000.0
        jitter = self._settings.mock_latency_jitter
        return random.uniform(base * (1 - jitter), base * (1 + jitter))

    def _point_for(self, city: str, day: date) -> ForecastPoint:
        """Build one day of forecast.

        Args:
            city: City name, used to pick the climate table and the random seed.
            day: The day to generate.

        Returns:
            A validated forecast point.
        """
        table = CLIMATE_TABLES.get(city.strip().lower(), CLIMATE_TABLES["_default"])
        month_index = day.month - 1

        # Interpolate between this month and the next so the baseline moves
        # smoothly through the month instead of stepping on the 1st.
        progress = (day.day - 1) / 31.0
        next_month = (month_index + 1) % 12
        base_high = self._blend(table["high"], month_index, next_month, progress)
        base_low = self._blend(table["low"], month_index, next_month, progress)
        base_rain = self._blend(table["rain"], month_index, next_month, progress)

        # Two sine waves of different periods, offset per city, produce spells
        # that persist for several days rather than independent daily noise.
        seed = self._city_seed(city)
        phase = (day.toordinal() + seed) / 3.7
        swing = math.sin(phase) * 3.0 + math.sin(phase / 2.6) * 1.8
        rain_swing = math.sin(phase / 2.2 + 1.1) * 18.0

        high = base_high + swing
        low = base_low + swing * 0.7
        # A wet day runs cooler; keeping this correlation stops the chart from
        # showing a 30 C downpour.
        rain_chance = max(0, min(100, base_rain + rain_swing))
        if rain_chance > 60:
            high -= 1.5

        return ForecastPoint(
            date=day,
            temp_max_c=round(high, 1),
            temp_min_c=round(min(low, high - 1.0), 1),
            condition=self._condition_for(rain_chance, high),
            precipitation_chance=int(round(rain_chance)),
            humidity_pct=int(round(min(95, 45 + rain_chance * 0.45))),
            wind_kph=round(6 + abs(math.sin(phase / 1.9)) * 18, 1),
        )

    @staticmethod
    def _blend(values: tuple[float, ...], first: int, second: int, progress: float) -> float:
        """Linearly interpolate between two monthly values.

        Args:
            values: The twelve monthly figures.
            first: Index of the current month.
            second: Index of the following month.
            progress: How far through the month, in ``[0, 1]``.

        Returns:
            The interpolated value.
        """
        return values[first] * (1 - progress) + values[second] * progress

    @staticmethod
    def _city_seed(city: str) -> int:
        """Derive a stable per-city offset.

        Args:
            city: City name.

        Returns:
            A small integer, identical across runs for the same city.
        """
        digest = hashlib.blake2b(city.strip().lower().encode("utf-8"), digest_size=2).digest()
        return int.from_bytes(digest, "big") % 97

    @staticmethod
    def _condition_for(rain_chance: float, high: float) -> str:
        """Derive the condition label from the numbers already generated.

        Args:
            rain_chance: Precipitation chance as a percentage.
            high: The day's high temperature in Celsius.

        Returns:
            A short human label consistent with the data.
        """
        for ceiling, label in CONDITION_BANDS:
            if rain_chance < ceiling:
                # Heavy rain plus real heat reads as a summer storm.
                if label == "Rain" and high > 26:
                    return "Thunderstorms"
                if label == "Clear" and high < 4:
                    return "Cold and clear"
                return label
        return "Rain"


__all__ = ["CLIMATE_TABLES", "MockWeatherProvider"]
