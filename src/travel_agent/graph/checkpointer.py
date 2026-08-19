"""Checkpointer construction - Distinction 3's storage layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from travel_agent.config.settings import Settings, get_settings
from travel_agent.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class CheckpointerHandle:
    """A checkpointer plus whatever is needed to shut it down.

    Attributes:
        saver: The checkpointer to compile the graph with.
        kind: ``"memory"`` or ``"sqlite"``, for the trace panel.
        location: Human-readable description of where state is stored.
        _connection: The database connection, when there is one.
    """

    saver: Any
    kind: str
    location: str
    _connection: Any = None

    @property
    def is_durable(self) -> bool:
        """Whether state survives a restart of the process."""
        return self.kind == "sqlite"

    async def close(self) -> None:
        """Release the database connection, if any."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None


async def create_checkpointer(settings: Settings | None = None) -> CheckpointerHandle:
    """Build the configured checkpointer.

    Args:
        settings: Settings to read from. Defaults to the process singleton.

    Returns:
        A handle holding the saver and its connection. A SQLite failure degrades
        to the in-memory saver with a warning rather than preventing start-up:
        losing durability is survivable, failing to start is not.
    """
    resolved = settings or get_settings()

    if resolved.checkpointer != "sqlite":
        logger.info("checkpointer: in-memory (state is lost when the process exits)")
        return CheckpointerHandle(
            saver=MemorySaver(), kind="memory", location="process memory (not durable)"
        )

    path = resolved.checkpoint_db

    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        # Inside the guard on purpose: creating the directory can fail too (a file
        # already occupying the path, a read-only volume), and that must degrade
        # to memory like any other storage failure rather than crash start-up.
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(str(path))
        saver = AsyncSqliteSaver(connection)
        await saver.setup()
    except (
        Exception
    ) as exc:  # noqa: BLE001 - we can live without durable storage, not without starting
        logger.warning("could not open the SQLite checkpointer at %s (%s); using memory", path, exc)
        return CheckpointerHandle(
            saver=MemorySaver(), kind="memory", location=f"process memory (sqlite failed: {exc})"
        )

    logger.info("checkpointer: sqlite at %s (durable across restarts)", path)
    return CheckpointerHandle(
        saver=saver, kind="sqlite", location=str(path), _connection=connection
    )


__all__ = ["CheckpointerHandle", "create_checkpointer"]
