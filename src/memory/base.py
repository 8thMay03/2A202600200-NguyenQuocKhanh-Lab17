"""Abstract base class and shared types for memory backends."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memory backends."""
    SHORT_TERM = "short_term"       # ConversationBufferMemory
    LONG_TERM = "long_term"         # Redis
    EPISODIC = "episodic"           # JSON episodic log
    SEMANTIC = "semantic"           # Chroma


class MemoryEntry(BaseModel):
    """A single memory entry that can be stored in any backend."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    memory_type: MemoryType
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 2  # 0=critical, 1=high, 2=medium, 3=low
    token_count: int = 0

    model_config = {"use_enum_values": True}


class MemoryBackend(ABC):
    """Abstract base class for all memory backends."""

    @property
    @abstractmethod
    def memory_type(self) -> MemoryType:
        """Return the type of this memory backend."""
        ...

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry."""
        ...

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Retrieve relevant memory entries for a query."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored memories."""
        ...

    @abstractmethod
    async def get_all(self) -> list[MemoryEntry]:
        """Get all stored memory entries."""
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about this memory backend."""
        entries = await self.get_all()
        return {
            "type": self.memory_type.value,
            "entry_count": len(entries),
            "total_tokens": sum(e.token_count for e in entries),
        }
