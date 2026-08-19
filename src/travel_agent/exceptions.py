"""The application's exception hierarchy."""

from __future__ import annotations


class TravelAgentError(Exception):
    """Base class for every error raised by this application."""


class ConfigurationError(TravelAgentError):
    """Settings are missing or mutually inconsistent.

    Raised at start-up rather than mid-request, e.g. a provider is set to ``live``
    but the matching API key is absent.
    """


class ProviderError(TravelAgentError):
    """A downstream provider (LLM, weather, images, search) misbehaved."""


class LLMError(ProviderError):
    """The language model call failed or returned an unusable payload."""


class RetryableError(ProviderError):
    """A transient failure that is worth retrying (timeout, 5xx, connection reset)."""


class RateLimitError(RetryableError):
    """The provider rejected the call with a rate limit response.

    Attributes:
        retry_after: Seconds the provider asked us to wait, parsed from the
            ``Retry-After`` header when present. ``None`` means the provider gave
            no hint and the caller should fall back to exponential backoff.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description of the rate limit response.
            retry_after: Optional server-provided cool-off period in seconds.
        """
        super().__init__(message)
        self.retry_after = retry_after


class ToolExecutionError(ProviderError):
    """A tool invoked by the agent failed.

    This is the error the manual tool executor converts into a ``ToolMessage``
    with ``status="error"`` so the model sees a genuine failure rather than a
    success-shaped string.

    Attributes:
        tool_name: Registry name of the tool that failed.
        tool_call_id: Identifier from the model's ``tool_calls`` payload, needed to
            correlate the error back to the request that caused it.
    """

    def __init__(
        self,
        message: str,
        tool_name: str,
        tool_call_id: str | None = None,
    ) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description of the failure.
            tool_name: Registry name of the failing tool.
            tool_call_id: The ``id`` field of the originating tool call, if known.
        """
        super().__init__(message)
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id


class WeatherToolError(ToolExecutionError):
    """The weather provider failed."""


class ImageToolError(ToolExecutionError):
    """The image provider failed."""


class SearchToolError(ToolExecutionError):
    """The web search provider failed."""


class UnknownToolError(ToolExecutionError):
    """The model asked for a tool that is not in the registry (a hallucinated name)."""


class ToolArgumentError(ToolExecutionError):
    """The model's tool arguments failed validation against the tool's schema."""


class VectorStoreError(TravelAgentError):
    """The vector store could not be built, loaded or queried."""


class SchemaValidationError(TravelAgentError):
    """The model's final answer did not satisfy the response schema after repair."""
