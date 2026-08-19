"""Web search provider interface and result model.

This is the tool the graph reaches for when the router decides the knowledge base
has nothing about a city. Three implementations exist: a mock with hand-written
material for the demo cities, DuckDuckGo, and Tavily.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field


class SearchResult(BaseModel):
    """One web search hit.

    Attributes:
        title: Page title.
        snippet: Extract used as context for the summary.
        url: Source link, shown as a citation in the UI.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    snippet: str = ""
    url: str = ""


class SearchProvider(ABC):
    """Searches the public web.

    Attributes:
        name: Provider identifier recorded in the trace.
    """

    name: str = "base"

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run a search.

        Args:
            query: Search query.
            max_results: Maximum number of results to return.

        Returns:
            Search results, possibly fewer than requested.

        Raises:
            SearchToolError: If the provider fails in a way worth reporting.
            RetryableError: If the failure is transient.
        """


__all__ = ["SearchProvider", "SearchResult"]
