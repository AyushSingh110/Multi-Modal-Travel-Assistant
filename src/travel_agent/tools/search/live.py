"""Live web search providers: DuckDuckGo and Tavily.

DuckDuckGo needs no API key, which makes it the natural live default. Its client
library is synchronous, so calls are pushed to a worker thread with
``asyncio.to_thread`` - blocking inside an async node would stall the whole event
loop and silently destroy the parallel fan-out.
"""

from __future__ import annotations

import asyncio

import httpx

from travel_agent.config.settings import Settings, get_settings
from travel_agent.exceptions import ConfigurationError, RetryableError, SearchToolError
from travel_agent.logging_setup import get_logger
from travel_agent.schemas.tools import WEB_SEARCH_TOOL
from travel_agent.tools.retry import rate_limit_from_response
from travel_agent.tools.search.base import SearchProvider, SearchResult

logger = get_logger(__name__)

TAVILY_URL = "https://api.tavily.com/search"


class DuckDuckGoSearchProvider(SearchProvider):
    """Searches the web through the keyless ``ddgs`` client.

    Attributes:
        name: Always ``"duckduckgo"``.
    """

    name = "duckduckgo"

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the provider.

        Args:
            settings: Settings to read timeouts from.
        """
        self._settings = settings or get_settings()

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run a DuckDuckGo search on a worker thread.

        Args:
            query: Search query.
            max_results: Maximum number of results.

        Returns:
            Search results.

        Raises:
            SearchToolError: If the client is missing or the search fails.
        """
        try:
            from ddgs import DDGS
        except ImportError as exc:  # pragma: no cover - ddgs is a pinned dependency
            raise SearchToolError(
                f"the ddgs package is not installed: {exc}", WEB_SEARCH_TOOL
            ) from exc

        def _run() -> list[dict[str, str]]:
            with DDGS() as client:
                return list(client.text(query, max_results=max_results))

        try:
            raw = await asyncio.to_thread(_run)
        except Exception as exc:  # noqa: BLE001 - the client raises a wide variety
            raise RetryableError(f"DuckDuckGo search failed: {exc}") from exc

        return [
            SearchResult(
                title=item.get("title") or query,
                snippet=item.get("body") or "",
                url=item.get("href") or "",
            )
            for item in raw[:max_results]
        ]


class TavilySearchProvider(SearchProvider):
    """Searches the web through Tavily's answer-oriented API.

    Attributes:
        name: Always ``"tavily"``.
    """

    name = "tavily"

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the provider.

        Args:
            settings: Settings holding the API key and timeouts.

        Raises:
            ConfigurationError: If Tavily is selected without a key.
        """
        self._settings = settings or get_settings()
        if not self._settings.tavily_api_key:
            raise ConfigurationError("Tavily search requires TAVILY_API_KEY to be set")
        self._api_key = self._settings.tavily_api_key

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run a Tavily search.

        Args:
            query: Search query.
            max_results: Maximum number of results.

        Returns:
            Search results.

        Raises:
            SearchToolError: If the response is an error.
            RetryableError: On a transient HTTP failure.
            RateLimitError: On HTTP 429.
        """
        timeout = httpx.Timeout(self._settings.tool_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                TAVILY_URL,
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
            )

        if not response.is_success:
            rate_limited = rate_limit_from_response(response.status_code, dict(response.headers))
            if rate_limited is not None:
                raise rate_limited
            if response.status_code >= 500:
                raise RetryableError(f"Tavily returned HTTP {response.status_code}")
            raise SearchToolError(
                f"Tavily returned HTTP {response.status_code}: {response.text[:200]}",
                WEB_SEARCH_TOOL,
            )

        return [
            SearchResult(
                title=item.get("title") or query,
                snippet=item.get("content") or "",
                url=item.get("url") or "",
            )
            for item in response.json().get("results", [])[:max_results]
        ]


__all__ = ["DuckDuckGoSearchProvider", "TavilySearchProvider"]
