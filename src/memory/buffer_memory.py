"""ConversationBufferMemory — Short-term, in-memory conversation history.

Stores the most recent N messages for the current conversation thread.
Acts as the sliding-window short-term memory.
"""

from __future__ import annotations

import tiktoken
from collections import defaultdict

from src.config import BUFFER_MEMORY_MAX_MESSAGES, OPENAI_MODEL
from src.memory.base import MemoryBackend, MemoryEntry, MemoryType


class ConversationBufferMemory(MemoryBackend):
    """In-memory conversation buffer with a configurable sliding window."""

    def __init__(self, max_messages: int | None = None):
        self._max_messages = max_messages or BUFFER_MEMORY_MAX_MESSAGES
        # thread_id -> list[MemoryEntry]
        self._buffers: dict[str, list[MemoryEntry]] = defaultdict(list)
        self._current_thread: str = "default"
        try:
            self._encoder = tiktoken.encoding_for_model(OPENAI_MODEL)
        except KeyError:
            self._encoder = tiktoken.get_encoding("cl100k_base")

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.SHORT_TERM

    def set_thread(self, thread_id: str) -> None:
        """Switch the active conversation thread."""
        self._current_thread = thread_id

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    async def store(self, entry: MemoryEntry) -> None:
        """Append an entry to the current thread's buffer."""
        entry.token_count = self._count_tokens(entry.content)
        entry.memory_type = MemoryType.SHORT_TERM
        buf = self._buffers[self._current_thread]
        buf.append(entry)
        # Trim oldest messages if over capacity
        while len(buf) > self._max_messages:
            buf.pop(0)

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Return the most recent entries from the current thread."""
        buf = self._buffers[self._current_thread]
        return buf[-top_k:]

    async def clear(self) -> None:
        """Clear the current thread's buffer."""
        self._buffers[self._current_thread].clear()

    async def clear_all(self) -> None:
        """Clear all threads."""
        self._buffers.clear()

    async def get_all(self) -> list[MemoryEntry]:
        """Return all entries in the current thread."""
        return list(self._buffers[self._current_thread])

    async def get_thread_ids(self) -> list[str]:
        """Return all thread IDs with stored messages."""
        return list(self._buffers.keys())
