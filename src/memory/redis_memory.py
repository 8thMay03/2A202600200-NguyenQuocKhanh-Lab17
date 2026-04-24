"""Redis Long-Term Memory — Persistent cross-session storage.

Uses fakeredis for local development so no external Redis server is required.
Stores user preferences, facts, and long-term knowledge as key-value pairs.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import tiktoken

from src.config import OPENAI_MODEL
from src.memory.base import MemoryBackend, MemoryEntry, MemoryType


class RedisLongTermMemory(MemoryBackend):
    """Long-term memory backed by Redis (or fakeredis for local dev)."""

    NAMESPACE = "ltm"

    def __init__(self, redis_url: str | None = None):
        try:
            import fakeredis
            self._redis = fakeredis.FakeRedis(decode_responses=True)
        except ImportError:
            import redis as _redis
            from src.config import REDIS_URL
            self._redis = _redis.Redis.from_url(
                redis_url or REDIS_URL, decode_responses=True
            )

        try:
            self._encoder = tiktoken.encoding_for_model(OPENAI_MODEL)
        except KeyError:
            self._encoder = tiktoken.get_encoding("cl100k_base")

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.LONG_TERM

    def _key(self, entry_id: str) -> str:
        return f"{self.NAMESPACE}:{entry_id}"

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _serialize(self, entry: MemoryEntry) -> str:
        data = entry.model_dump()
        data["timestamp"] = entry.timestamp.isoformat()
        return json.dumps(data)

    def _deserialize(self, raw: str) -> MemoryEntry:
        data = json.loads(raw)
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return MemoryEntry(**data)

    async def store(self, entry: MemoryEntry) -> None:
        """Store entry in Redis."""
        entry.token_count = self._count_tokens(entry.content)
        entry.memory_type = MemoryType.LONG_TERM
        self._redis.set(self._key(entry.id), self._serialize(entry))
        # Maintain an index set of all entry IDs
        self._redis.sadd(f"{self.NAMESPACE}:index", entry.id)

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Simple keyword-match retrieval across stored entries.

        For production, this should use Redis Search or vector similarity.
        Here we do a basic substring match for simplicity.
        """
        all_entries = await self.get_all()
        query_lower = query.lower()
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in all_entries:
            content_lower = entry.content.lower()
            # Simple relevance: count keyword overlaps
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored.append((overlap, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    async def clear(self) -> None:
        """Remove all long-term memory entries."""
        ids = self._redis.smembers(f"{self.NAMESPACE}:index")
        for entry_id in ids:
            self._redis.delete(self._key(entry_id))
        self._redis.delete(f"{self.NAMESPACE}:index")

    async def get_all(self) -> list[MemoryEntry]:
        """Return all stored entries."""
        ids = self._redis.smembers(f"{self.NAMESPACE}:index")
        entries: list[MemoryEntry] = []
        for entry_id in ids:
            raw = self._redis.get(self._key(entry_id))
            if raw:
                entries.append(self._deserialize(raw))
        entries.sort(key=lambda e: e.timestamp)
        return entries

    async def store_preference(self, key: str, value: str) -> None:
        """Convenience: store a user preference."""
        entry = MemoryEntry(
            content=f"User preference — {key}: {value}",
            memory_type=MemoryType.LONG_TERM,
            metadata={"category": "preference", "pref_key": key, "pref_value": value},
            priority=1,
        )
        await self.store(entry)

    async def get_preferences(self) -> dict[str, str]:
        """Retrieve all stored user preferences."""
        all_entries = await self.get_all()
        prefs: dict[str, str] = {}
        for e in all_entries:
            if e.metadata.get("category") == "preference":
                prefs[e.metadata["pref_key"]] = e.metadata["pref_value"]
        return prefs
