"""Tests for configuration resolution and the retry/backoff policy."""

from __future__ import annotations

import asyncio

import pytest

from travel_agent.config.settings import Settings
from travel_agent.exceptions import RateLimitError, RetryableError
from travel_agent.schemas.response import ForecastPoint, TravelResponse
from travel_agent.tools.retry import backoff_delay, call_with_retry, rate_limit_from_response


def _settings(**overrides: object) -> Settings:
    """Build settings without reading the developer's real .env file."""
    base: dict[str, object] = {
        "llm_provider": None,
        "groq_api_key": None,
        "anthropic_api_key": None,
        "openai_api_key": None,
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# ------------------------------------------------------ provider selection --
def test_defaults_to_mock_when_no_keys_are_present() -> None:
    assert _settings().resolve_llm_provider() == "mock"


def test_groq_wins_when_several_keys_are_present() -> None:
    resolved = _settings(
        groq_api_key="g", anthropic_api_key="a", openai_api_key="o"
    ).resolve_llm_provider()

    assert resolved == "groq"


def test_anthropic_is_second_in_the_order() -> None:
    assert (
        _settings(anthropic_api_key="a", openai_api_key="o").resolve_llm_provider() == "anthropic"
    )


def test_openai_is_third_in_the_order() -> None:
    assert _settings(openai_api_key="o").resolve_llm_provider() == "openai"


def test_explicit_provider_overrides_auto_detection() -> None:
    """LLM_PROVIDER must win even when a higher-priority key exists."""
    resolved = _settings(llm_provider="anthropic", groq_api_key="g").resolve_llm_provider()

    assert resolved == "anthropic"


def test_blank_environment_values_are_treated_as_unset() -> None:
    """.env.example ships keys as empty strings; they must not look configured."""
    assert _settings(groq_api_key="   ", llm_provider="").resolve_llm_provider() == "mock"


def test_model_id_lookup_matches_the_resolved_provider() -> None:
    settings = _settings(groq_api_key="g", groq_model="openai/gpt-oss-120b")

    assert settings.model_id_for(settings.resolve_llm_provider()) == "openai/gpt-oss-120b"


def test_relative_paths_resolve_against_the_repository_root() -> None:
    settings = _settings()

    assert settings.vector_store_path.is_absolute()
    assert settings.city_facts_dir.name == "city_facts"


# ------------------------------------------------------------------ retry --
def test_backoff_grows_and_stays_within_the_ceiling() -> None:
    for attempt in range(5):
        delay = backoff_delay(attempt, base_delay=0.5, max_delay=8.0)
        assert 0.0 <= delay <= min(8.0, 0.5 * 2**attempt)


def test_successful_call_is_not_retried() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    assert asyncio.run(call_with_retry(operation, label="test")) == "ok"
    assert calls == 1


def test_transient_failure_is_retried_then_succeeds() -> None:
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RetryableError("temporary")
        return "recovered"

    result = asyncio.run(call_with_retry(flaky, label="test", base_delay=0.001, max_delay=0.002))

    assert result == "recovered"
    assert calls == 3


def test_attempts_are_bounded_and_the_last_error_propagates() -> None:
    calls = 0

    async def always_fails() -> str:
        nonlocal calls
        calls += 1
        raise RetryableError("still broken")

    with pytest.raises(RetryableError, match="still broken"):
        asyncio.run(
            call_with_retry(
                always_fails, label="test", attempts=3, base_delay=0.001, max_delay=0.002
            )
        )

    assert calls == 3


def test_non_retryable_failure_is_raised_immediately() -> None:
    calls = 0

    async def bad_arguments() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("this will never succeed")

    with pytest.raises(ValueError):
        asyncio.run(call_with_retry(bad_arguments, label="test", attempts=3))

    assert calls == 1, "a deterministic failure must not be retried"


def test_timeout_is_enforced_per_attempt() -> None:
    async def too_slow() -> str:
        await asyncio.sleep(5)
        return "never"

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(call_with_retry(too_slow, label="test", attempts=1, timeout=0.05))


def test_rate_limit_retry_after_is_honoured_over_backoff() -> None:
    """Groq's free tier returns 429 with Retry-After; the server's number wins."""
    observed: list[float] = []
    calls = 0

    async def limited() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimitError("429", retry_after=0.01)
        return "ok"

    def record(attempt: int, error: BaseException, delay: float) -> None:
        observed.append(delay)

    result = asyncio.run(
        call_with_retry(limited, label="test", base_delay=5.0, max_delay=10.0, on_retry=record)
    )

    assert result == "ok"
    assert observed == [0.01], "backoff should have been overridden by Retry-After"


def test_rate_limit_is_detected_from_an_http_response() -> None:
    error = rate_limit_from_response(429, {"retry-after": "2.5"})

    assert isinstance(error, RateLimitError)
    assert error.retry_after == 2.5


def test_non_429_response_is_not_a_rate_limit() -> None:
    assert rate_limit_from_response(500, {}) is None


def test_unparseable_retry_after_falls_back_to_backoff() -> None:
    error = rate_limit_from_response(429, {"retry-after": "soon"})

    assert isinstance(error, RateLimitError)
    assert error.retry_after is None


# --------------------------------------------------------- response schema --
def test_forecast_rejects_a_low_above_its_high() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        ForecastPoint(
            date="2026-08-19",
            temp_max_c=20,
            temp_min_c=25,
            condition="Sunny",
            precipitation_chance=0,
        )


def test_response_drops_malformed_image_urls() -> None:
    response = TravelResponse(
        city="Paris",
        city_summary="A long enough summary of Paris to satisfy the minimum length rule.",
        image_urls=["https://example.com/a.jpg", "not-a-url", "/relative.png"],
    )

    assert response.image_urls == ["https://example.com/a.jpg"]


def test_response_flags_itself_as_degraded_when_data_is_missing() -> None:
    response = TravelResponse(
        city="Paris",
        city_summary="A long enough summary of Paris to satisfy the minimum length rule.",
        image_urls=["https://example.com/a.jpg"],
        warnings=["weather unavailable"],
    )

    assert response.is_degraded
