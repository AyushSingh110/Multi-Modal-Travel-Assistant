"""Checkpointer construction - Distinction 3's storage layer.

WHAT A CHECKPOINTER IS
    After every superstep LangGraph hands the current state to a checkpointer,
    which saves it under a ``thread_id``. When a later request arrives on the same
    thread id, the graph starts from that saved state instead of from nothing.

    That is the whole of "memory" here. There is no separate memory system and no
    summarisation step: the typed state *is* the memory, and the checkpointer is
    where it lives between turns.

WHAT ``thread_id`` SCOPES
    One conversation. Two browser tabs with different thread ids are two separate
    conversations that cannot see each other's cities, images or forecasts. It is
    the isolation boundary, and a test asserts state does not leak across it.

WHICH BACKEND
    ``MemorySaver`` is the default: zero setup, and fast. Its limitation is real
    and worth saying out loud - **it dies with the process**. Restart the app and
    every conversation is gone.

    ``CHECKPOINTER=sqlite`` swaps in a durable file-backed saver, which is what
    proves the memory is genuinely persisted rather than a process-local
    dictionary that happens to survive between two calls in one test.

A VERIFIED DETAIL
    The synchronous ``SqliteSaver`` cannot be used here. This graph runs through
    ``ainvoke``, and the sync saver raises
    ``NotImplementedError: The SqliteSaver does not support async methods`` on the
    first checkpoint write. ``AsyncSqliteSaver`` is the one that works, and its
    connection is owned by this module so the saver outlives a single ``async
    with`` block.
"""

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
    except Exception as exc:  # noqa: BLE001 - durability is a nice-to-have, starting is not
        logger.warning("could not open the SQLite checkpointer at %s (%s); using memory", path, exc)
        return CheckpointerHandle(
            saver=MemorySaver(), kind="memory", location=f"process memory (sqlite failed: {exc})"
        )

    logger.info("checkpointer: sqlite at %s (durable across restarts)", path)
    return CheckpointerHandle(
        saver=saver, kind="sqlite", location=str(path), _connection=connection
    )


__all__ = ["CheckpointerHandle", "create_checkpointer"]
