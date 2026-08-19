"""Centralised application configuration.

Everything tunable lives here: provider selection, model ids, latency knobs,
retrieval thresholds, paths. No other module reads ``os.environ`` and no other
module hard-codes a path or a magic number.

Values come from environment variables (and a ``.env`` file when present) via
``pydantic-settings``, which means every setting is *typed and validated at
start-up* rather than blowing up halfway through a request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["groq", "anthropic", "openai", "mock"]
ToolProvider = Literal["live", "mock"]
EmbeddingProvider = Literal["hashed", "openai"]
FailureMode = Literal["none", "timeout", "server_error", "malformed", "rate_limit"]
VectorStoreBackend = Literal["faiss", "numpy"]
CheckpointerKind = Literal["memory", "sqlite"]

# <repo root>/src/travel_agent/config/settings.py -> parents[3] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Typed application settings loaded from the environment.

    Attributes are grouped to mirror ``.env.example`` so the two stay readable
    side by side.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------------- LLM ----
    llm_provider: LLMProvider | None = Field(
        default=None,
        description="Explicit provider override. When None the provider is auto-detected.",
    )
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3

    # -------------------------------------------------------------- tools ----
    weather_provider: ToolProvider = "mock"
    image_provider: ToolProvider = "mock"
    search_provider: ToolProvider = "mock"
    openweather_api_key: str | None = None
    unsplash_access_key: str | None = None
    tavily_api_key: str | None = None
    # Per-tool mock latency. Weather and images differ on purpose: the parallel
    # fan-out speed-up is measured against these numbers, and two branches with
    # identical durations would make the measurement look staged.
    mock_weather_latency_ms: int = 900
    mock_image_latency_ms: int = 1100
    mock_search_latency_ms: int = 800
    mock_latency_jitter: float = 0.15

    # Failure injection - powers the "break the weather API" demo toggle.
    force_weather_failure: bool = False
    weather_failure_mode: FailureMode = "server_error"
    force_image_failure: bool = False
    image_failure_mode: FailureMode = "server_error"

    # Per-attempt tool timeout and attempt budget.
    tool_timeout_seconds: float = 12.0
    tool_max_attempts: int = 3

    # ---------------------------------------------------------- retrieval ----
    embedding_provider: EmbeddingProvider = "hashed"
    embedding_dim: int = 512
    vector_store_dir: Path = Path("data/vector_store")
    vector_store_backend: VectorStoreBackend = "faiss"
    # Calibrated from the measured separation between seeded and unseeded
    # cities - see scripts/seed_vectorstore.py output and evals/run_eval.py.
    router_similarity_threshold: float = 0.07
    retrieval_top_k: int = 4

    # ------------------------------------------------------------- memory ----
    checkpointer: CheckpointerKind = "memory"
    checkpoint_db_path: Path = Path("data/checkpoints.sqlite")

    # ------------------------------------------------------------ runtime ----
    log_level: str = "INFO"
    enable_cache: bool = True
    cache_ttl_seconds: int = 900

    @field_validator(
        "llm_provider",
        "groq_api_key",
        "anthropic_api_key",
        "openai_api_key",
        "openweather_api_key",
        "unsplash_access_key",
        "tavily_api_key",
        mode="before",
    )
    @classmethod
    def _empty_string_to_none(cls, value: Any) -> Any:
        """Treat a blank environment variable as unset.

        ``.env.example`` ships keys as ``GROQ_API_KEY=`` so reviewers can see every
        option. Without this, an empty string would look like a configured key and
        the app would try to call a live API with no credential.

        Args:
            value: Raw value straight from the environment.

        Returns:
            ``None`` for blank/whitespace-only strings, otherwise the value unchanged.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def project_root(self) -> Path:
        """Absolute path to the repository root."""
        return PROJECT_ROOT

    @property
    def vector_store_path(self) -> Path:
        """Absolute path to the vector store directory."""
        return self._absolute(self.vector_store_dir)

    @property
    def checkpoint_db(self) -> Path:
        """Absolute path to the SQLite checkpoint database."""
        return self._absolute(self.checkpoint_db_path)

    @property
    def city_facts_dir(self) -> Path:
        """Absolute path to the seed corpus of city fact files."""
        return PROJECT_ROOT / "data" / "city_facts"

    def _absolute(self, path: Path) -> Path:
        """Resolve a possibly-relative configured path against the repo root.

        Args:
            path: Configured path, absolute or relative.

        Returns:
            An absolute path.
        """
        return path if path.is_absolute() else PROJECT_ROOT / path

    def resolve_llm_provider(self) -> LLMProvider:
        """Decide which LLM driver to use.

        Selection order, highest priority first:

        1. An explicit ``LLM_PROVIDER`` value - always honoured, even when other
           keys are present. This is the escape hatch for "I have three keys but I
           want to demo on Anthropic today".
        2. ``GROQ_API_KEY`` - the default demo driver: fast, and its free tier
           survives a day of iterative development plus a live demo.
        3. ``ANTHROPIC_API_KEY`` - a provider named by the assignment spec.
        4. ``OPENAI_API_KEY`` - a provider named by the assignment spec.
        5. Nothing configured - the deterministic mock, so the app still runs
           end-to-end with zero API keys.

        Returns:
            The provider name to instantiate.
        """
        if self.llm_provider is not None:
            return self.llm_provider
        if self.groq_api_key:
            return "groq"
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return "mock"

    def model_id_for(self, provider: LLMProvider) -> str:
        """Return the configured model id for a provider.

        Args:
            provider: Provider name.

        Returns:
            The model identifier string, or ``"mock-llm"`` for the mock provider.
        """
        return {
            "groq": self.groq_model,
            "anthropic": self.anthropic_model,
            "openai": self.openai_model,
            "mock": "mock-llm",
        }[provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that Streamlit's script re-runs do not re-parse ``.env`` on every
    widget interaction.

    Returns:
        The loaded :class:`Settings` instance.
    """
    return Settings()


__all__ = ["PROJECT_ROOT", "Settings", "get_settings"]
