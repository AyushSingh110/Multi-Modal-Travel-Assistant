"""Tests for the tool layer: providers, failure injection, retry and degradation.

The failure-mode tests matter as much as the happy path. The rubric asks whether
the app survives a dead weather API, and the answer has to be demonstrable, not
asserted - so every failure shape the demo toggle can produce is exercised here.
"""

from __future__ import annotations

import asyncio
import statistics
from datetime import date, timedelta

import pytest

from travel_agent.config.settings import Settings
from travel_agent.exceptions import RateLimitError, RetryableError
from travel_agent.schemas.response import ImageAsset, WeatherPayload
from travel_agent.schemas.tools import IMAGES_TOOL, WEATHER_TOOL, WEB_SEARCH_TOOL
from travel_agent.tools.failures import MalformedPayloadError, maybe_fail
from travel_agent.tools.images.mock import CURATED_IMAGES, MockImageProvider
from travel_agent.tools.registry import ToolRegistry, build_registry
from travel_agent.tools.search.base import SearchProvider, SearchResult
from travel_agent.tools.search.mock import MockSearchProvider
from travel_agent.tools.weather.base import WeatherProvider
from travel_agent.tools.weather.mock import MockWeatherProvider


def _settings(**overrides: object) -> Settings:
    """Settings with fast latencies, so the suite stays quick."""
    base: dict[str, object] = {
        "_env_file": None,
        "mock_weather_latency_ms": 20,
        "mock_image_latency_ms": 20,
        "mock_search_latency_ms": 20,
        "mock_latency_jitter": 0.0,
        "tool_timeout_seconds": 0.5,
        "tool_max_attempts": 3,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# ============================================================ weather: data ==
async def test_weather_returns_the_requested_number_of_days() -> None:
    payload = await MockWeatherProvider(_settings()).fetch_forecast("Paris", days=7)

    assert isinstance(payload, WeatherPayload)
    assert len(payload.forecast) == 7
    assert payload.provider == "mock"
    assert payload.city == "Paris"


@pytest.mark.parametrize("days", [5, 6, 7])
async def test_weather_honours_the_days_argument(days: int) -> None:
    payload = await MockWeatherProvider(_settings()).fetch_forecast("Tokyo", days=days)

    assert len(payload.forecast) == days


async def test_forecast_days_are_consecutive_from_the_start_date() -> None:
    start = date(2026, 3, 15)
    payload = await MockWeatherProvider(_settings()).fetch_forecast(
        "Paris", days=7, start_date=start
    )

    assert [point.date for point in payload.forecast] == [
        start + timedelta(days=offset) for offset in range(7)
    ]


async def test_forecast_is_not_a_flat_line() -> None:
    """The chart has to look like weather, so the series must actually vary."""
    payload = await MockWeatherProvider(_settings()).fetch_forecast("Paris", days=7)
    highs = [point.temp_max_c for point in payload.forecast]

    assert len(set(highs)) > 1
    assert statistics.pstdev(highs) > 0.5, f"series is too flat: {highs}"


async def test_forecast_is_not_uniform_noise_either() -> None:
    """Consecutive days should resemble each other - real weather persists."""
    payload = await MockWeatherProvider(_settings()).fetch_forecast("Tokyo", days=7)
    highs = [point.temp_max_c for point in payload.forecast]
    jumps = [abs(second - first) for first, second in zip(highs, highs[1:], strict=False)]

    assert max(jumps) < 8.0, f"day-to-day swing is implausible: {highs}"


async def test_seasonal_variation_is_reflected_per_city() -> None:
    """August in Tokyo must be much warmer than January in Tokyo."""
    provider = MockWeatherProvider(_settings())

    summer = await provider.fetch_forecast("Tokyo", days=5, start_date=date(2026, 8, 1))
    winter = await provider.fetch_forecast("Tokyo", days=5, start_date=date(2026, 1, 10))

    summer_mean = statistics.mean(point.temp_max_c for point in summer.forecast)
    winter_mean = statistics.mean(point.temp_max_c for point in winter.forecast)

    assert summer_mean - winter_mean > 12, f"summer {summer_mean:.1f} vs winter {winter_mean:.1f}"


async def test_different_cities_have_different_climates() -> None:
    """New York in January is colder than Tokyo in January."""
    provider = MockWeatherProvider(_settings())
    january = date(2026, 1, 15)

    new_york = await provider.fetch_forecast("New York", days=5, start_date=january)
    tokyo = await provider.fetch_forecast("Tokyo", days=5, start_date=january)

    assert statistics.mean(p.temp_max_c for p in new_york.forecast) < statistics.mean(
        p.temp_max_c for p in tokyo.forecast
    )


async def test_unknown_city_still_gets_a_plausible_forecast() -> None:
    payload = await MockWeatherProvider(_settings()).fetch_forecast("Snohomish", days=7)

    assert len(payload.forecast) == 7
    assert all(-40 < point.temp_max_c < 55 for point in payload.forecast)


async def test_forecast_is_deterministic_for_the_same_inputs() -> None:
    """A demo repeated in front of a panel must produce the same chart."""
    provider = MockWeatherProvider(_settings())
    start = date(2026, 5, 1)

    first = await provider.fetch_forecast("Paris", days=7, start_date=start)
    second = await provider.fetch_forecast("Paris", days=7, start_date=start)

    assert [p.temp_max_c for p in first.forecast] == [p.temp_max_c for p in second.forecast]


async def test_conditions_agree_with_precipitation_numbers() -> None:
    payload = await MockWeatherProvider(_settings()).fetch_forecast("Tokyo", days=7)

    for point in payload.forecast:
        if point.precipitation_chance >= 75:
            assert point.condition in {"Rain", "Thunderstorms"}
        if point.precipitation_chance < 15:
            assert point.condition in {"Clear", "Cold and clear"}


async def test_simulated_latency_is_close_to_the_configured_value() -> None:
    """Latency is what makes the parallel speed-up measurable rather than noise."""
    settings = _settings(mock_weather_latency_ms=300, mock_latency_jitter=0.0)
    started = asyncio.get_event_loop().time()

    await MockWeatherProvider(settings).fetch_forecast("Paris", days=5)

    elapsed = asyncio.get_event_loop().time() - started
    assert 0.25 < elapsed < 0.6, f"took {elapsed:.3f}s"


# ============================================================= images: data ==
async def test_images_returns_curated_assets_for_seeded_cities() -> None:
    assets = await MockImageProvider(_settings()).search_images("Paris", count=4)

    assert len(assets) == 4
    assert all(isinstance(asset, ImageAsset) for asset in assets)
    assert all(asset.url.startswith("https://") for asset in assets)
    assert all(asset.caption for asset in assets)
    assert all(asset.credit for asset in assets)


@pytest.mark.parametrize("city", ["Paris", "Tokyo", "New York"])
async def test_every_seeded_city_has_four_curated_images(city: str) -> None:
    assert len(CURATED_IMAGES[city.lower()]) >= 4
    assert len(await MockImageProvider(_settings()).search_images(city, count=4)) == 4


async def test_image_count_is_respected() -> None:
    assets = await MockImageProvider(_settings()).search_images("Tokyo", count=2)

    assert len(assets) == 2


async def test_uncurated_city_gets_honest_placeholder_imagery() -> None:
    """A stock photo must not be captioned as if it were the real city."""
    assets = await MockImageProvider(_settings()).search_images("Kyoto", count=3)

    assert len(assets) == 3
    assert all(asset.url.startswith("https://") for asset in assets)
    assert all("laceholder" in asset.credit for asset in assets)


async def test_placeholder_urls_are_stable_per_city() -> None:
    provider = MockImageProvider(_settings())

    first = await provider.search_images("Snohomish", count=3)
    second = await provider.search_images("Snohomish", count=3)

    assert [a.url for a in first] == [a.url for a in second]


async def test_different_uncurated_cities_get_different_images() -> None:
    provider = MockImageProvider(_settings())

    kyoto = await provider.search_images("Kyoto", count=2)
    lagos = await provider.search_images("Lagos", count=2)

    assert {a.url for a in kyoto}.isdisjoint({a.url for a in lagos})


# ============================================================= search: data ==
async def test_search_returns_curated_results_for_demo_cities() -> None:
    results = await MockSearchProvider(_settings()).search("Kyoto travel guide overview")

    assert len(results) >= 3
    assert all(isinstance(result, SearchResult) for result in results)
    assert any("temple" in result.snippet.lower() for result in results)
    assert all(result.url.startswith("http") for result in results)


async def test_search_falls_back_to_generic_results() -> None:
    results = await MockSearchProvider(_settings()).search("Ulaanbaatar travel guide")

    assert results
    assert all(result.title for result in results)


async def test_search_respects_max_results() -> None:
    results = await MockSearchProvider(_settings()).search("Kyoto", max_results=2)

    assert len(results) == 2


# ====================================================== failure injection ====
async def test_failure_injection_is_off_by_default() -> None:
    payload = await MockWeatherProvider(_settings()).fetch_forecast("Paris")

    assert payload.forecast


async def test_injected_server_error_is_retryable() -> None:
    with pytest.raises(RetryableError, match="500"):
        await maybe_fail("server_error", WEATHER_TOOL)


async def test_injected_rate_limit_carries_retry_after() -> None:
    with pytest.raises(RateLimitError) as caught:
        await maybe_fail("rate_limit", WEATHER_TOOL)

    assert caught.value.retry_after == 0.5


async def test_injected_malformed_payload_is_not_retryable() -> None:
    """Retrying a successful-but-unusable response would waste the user's time."""
    with pytest.raises(MalformedPayloadError):
        await maybe_fail("malformed", WEATHER_TOOL)

    assert not issubclass(MalformedPayloadError, RetryableError)


async def test_injected_timeout_hangs_until_the_caller_gives_up() -> None:
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(maybe_fail("timeout", WEATHER_TOOL), timeout=0.05)


@pytest.mark.parametrize("mode", ["server_error", "rate_limit", "malformed"])
async def test_weather_provider_honours_the_failure_flag(mode: str) -> None:
    settings = _settings(force_weather_failure=True, weather_failure_mode=mode)

    with pytest.raises(Exception) as caught:
        await MockWeatherProvider(settings).fetch_forecast("Paris")

    assert "simulated" in str(caught.value) or "unparseable" in str(caught.value)


async def test_image_provider_honours_the_failure_flag() -> None:
    settings = _settings(force_image_failure=True, image_failure_mode="server_error")

    with pytest.raises(RetryableError):
        await MockImageProvider(settings).search_images("Paris")


# ============================================== registry: retry + degrade ====
def _registry(settings: Settings, **providers: object) -> ToolRegistry:
    return ToolRegistry(
        providers.get("weather") or MockWeatherProvider(settings),  # type: ignore[arg-type]
        providers.get("images") or MockImageProvider(settings),  # type: ignore[arg-type]
        providers.get("search") or MockSearchProvider(settings),  # type: ignore[arg-type]
        settings=settings,
    )


async def test_registry_happy_path_returns_a_payload() -> None:
    settings = _settings()

    result = await _registry(settings).execute(WEATHER_TOOL, {"city": "Paris", "days": 7})

    assert result.ok
    assert not result.failed
    assert len(result.payload.forecast) == 7
    assert result.provider == "mock"
    assert result.duration_ms > 0


async def test_registry_executes_every_registered_tool() -> None:
    settings = _settings()
    registry = _registry(settings)

    weather = await registry.execute(WEATHER_TOOL, {"city": "Tokyo"})
    images = await registry.execute(IMAGES_TOOL, {"city": "Tokyo", "count": 2})
    search = await registry.execute(WEB_SEARCH_TOOL, {"query": "Kyoto travel guide"})

    assert weather.ok and images.ok and search.ok
    assert len(images.payload) == 2


async def test_unknown_tool_degrades_instead_of_raising() -> None:
    result = await _registry(_settings()).execute("get_stock_price", {"ticker": "AAPL"})

    assert result.failed
    assert result.error_type == "UnknownToolError"
    assert "not registered" in result.error


async def test_invalid_arguments_degrade_instead_of_raising() -> None:
    result = await _registry(_settings()).execute(WEATHER_TOOL, {"days": 7})

    assert result.failed
    assert result.error_type == "ToolArgumentError"


async def test_out_of_range_arguments_are_caught_before_the_provider_runs() -> None:
    result = await _registry(_settings()).execute(WEATHER_TOOL, {"city": "Paris", "days": 99})

    assert result.failed
    assert result.error_type == "ToolArgumentError"


class _FlakyWeather(WeatherProvider):
    """Fails a set number of times, then succeeds."""

    name = "flaky"

    def __init__(self, failures: int, settings: Settings) -> None:
        self.failures = failures
        self.calls = 0
        self._inner = MockWeatherProvider(settings)

    async def fetch_forecast(
        self, city: str, days: int = 7, start_date: date | None = None
    ) -> WeatherPayload:
        self.calls += 1
        if self.calls <= self.failures:
            raise RetryableError(f"transient failure {self.calls}")
        return await self._inner.fetch_forecast(city, days=days, start_date=start_date)


async def test_retry_then_succeed() -> None:
    settings = _settings(tool_max_attempts=3)
    provider = _FlakyWeather(failures=2, settings=settings)

    result = await _registry(settings, weather=provider).execute(WEATHER_TOOL, {"city": "Paris"})

    assert result.ok
    assert provider.calls == 3
    assert result.attempts == 3


async def test_retry_exhausted_then_degrades() -> None:
    """The graph must still be able to render the rest of the page."""
    settings = _settings(tool_max_attempts=3)
    provider = _FlakyWeather(failures=99, settings=settings)

    result = await _registry(settings, weather=provider).execute(WEATHER_TOOL, {"city": "Paris"})

    assert result.failed
    assert provider.calls == 3, "attempt budget must be bounded"
    assert result.error_type == "RetryableError"
    assert result.tool == WEATHER_TOOL


async def test_retry_callback_reports_each_attempt() -> None:
    settings = _settings(tool_max_attempts=3)
    provider = _FlakyWeather(failures=2, settings=settings)
    observed: list[tuple[int, str, float]] = []

    await _registry(settings, weather=provider).execute(
        WEATHER_TOOL,
        {"city": "Paris"},
        on_retry=lambda attempt, error, delay: observed.append(
            (attempt, type(error).__name__, delay)
        ),
    )

    assert len(observed) == 2
    assert all(entry[1] == "RetryableError" for entry in observed)


async def test_rate_limited_tool_waits_the_advertised_time_then_degrades() -> None:
    settings = _settings(
        force_weather_failure=True, weather_failure_mode="rate_limit", tool_max_attempts=2
    )
    delays: list[float] = []

    result = await _registry(settings).execute(
        WEATHER_TOOL,
        {"city": "Paris"},
        on_retry=lambda attempt, error, delay: delays.append(delay),
    )

    assert result.failed
    assert result.error_type == "RateLimitError"
    assert delays == [0.5], "Retry-After must win over the backoff curve"


async def test_timeout_failure_degrades_within_the_configured_budget() -> None:
    settings = _settings(
        force_weather_failure=True,
        weather_failure_mode="timeout",
        tool_timeout_seconds=0.05,
        tool_max_attempts=2,
    )

    result = await _registry(settings).execute(WEATHER_TOOL, {"city": "Paris"})

    assert result.failed
    assert result.error_type == "TimeoutError"


async def test_malformed_payload_is_not_retried() -> None:
    settings = _settings(
        force_weather_failure=True, weather_failure_mode="malformed", tool_max_attempts=3
    )
    retries: list[int] = []

    result = await _registry(settings).execute(
        WEATHER_TOOL, {"city": "Paris"}, on_retry=lambda a, e, d: retries.append(a)
    )

    assert result.failed
    assert retries == [], "a deterministic failure must not be retried"


async def test_one_tool_failing_leaves_the_others_working() -> None:
    """The graceful-degradation requirement, at the registry level."""
    settings = _settings(
        force_weather_failure=True, weather_failure_mode="server_error", tool_max_attempts=1
    )
    registry = _registry(settings)

    weather = await registry.execute(WEATHER_TOOL, {"city": "Paris"})
    images = await registry.execute(IMAGES_TOOL, {"city": "Paris", "count": 4})
    search = await registry.execute(WEB_SEARCH_TOOL, {"query": "Paris travel guide"})

    assert weather.failed
    assert images.ok and len(images.payload) == 4
    assert search.ok


# ================================================== provider construction ====
def test_registry_defaults_to_mock_providers() -> None:
    registry = build_registry(_settings())

    assert [registry.provider_for(name) for name in registry.tool_names] == ["mock"] * 3


def test_live_weather_without_a_key_is_a_configuration_error() -> None:
    from travel_agent.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="OPENWEATHER_API_KEY"):
        build_registry(_settings(weather_provider="live"))


def test_live_images_without_a_key_is_a_configuration_error() -> None:
    from travel_agent.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="UNSPLASH_ACCESS_KEY"):
        build_registry(_settings(image_provider="live"))


def test_live_search_picks_duckduckgo_when_tavily_has_no_key() -> None:
    registry = build_registry(_settings(search_provider="live"))

    assert registry.provider_for(WEB_SEARCH_TOOL) == "duckduckgo"


def test_providers_conform_to_their_interfaces() -> None:
    settings = _settings()

    assert isinstance(MockWeatherProvider(settings), WeatherProvider)
    assert isinstance(MockSearchProvider(settings), SearchProvider)
