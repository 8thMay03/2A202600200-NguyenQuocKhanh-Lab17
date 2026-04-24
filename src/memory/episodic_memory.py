"""JSON Episodic Memory — Append-only log of significant interactions.

Each episode captures a complete interaction with metadata like topic,
outcome, and sentiment. Stored as JSON files on disk.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken

from src.config import EPISODIC_LOG_DIR, OPENAI_MODEL
from src.memory.base import MemoryBackend, MemoryEntry, MemoryType


class EpisodicMemory(MemoryBackend):
    """File-based episodic memory using JSON logs."""

    def __init__(self, log_dir: Path | str | None = None):
        self._log_dir = Path(log_dir or EPISODIC_LOG_DIR)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "episodes.jsonl"
        try:
            self._encoder = tiktoken.encoding_for_model(OPENAI_MODEL)
        except KeyError:
            self._encoder = tiktoken.get_encoding("cl100k_base")

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.EPISODIC

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    async def store(self, entry: MemoryEntry) -> None:
        """Append an episode to the JSONL log."""
        entry.token_count = self._count_tokens(entry.content)
        entry.memory_type = MemoryType.EPISODIC
        data = entry.model_dump()
        data["timestamp"] = entry.timestamp.isoformat()
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    async def log_episode(
        self,
        user_query: str,
        agent_response: str,
        topic: str = "",
        outcome: str = "",
        sentiment: str = "neutral",
    ) -> None:
        """Convenience: log a full episode with structured metadata."""
        content = f"User: {user_query}\nAgent: {agent_response}"
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.EPISODIC,
            metadata={
                "topic": topic,
                "outcome": outcome,
                "sentiment": sentiment,
                "user_query": user_query,
                "agent_response": agent_response,
            },
            priority=3,
        )
        await self.store(entry)

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Retrieve episodes matching the query by keyword overlap."""
        all_eps = await self.get_all()
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored: list[tuple[float, MemoryEntry]] = []
        for ep in all_eps:
            content_words = set(ep.content.lower().split())
            # Also check topic metadata
            topic = ep.metadata.get("topic", "").lower()
            topic_words = set(topic.split())
            overlap = len(query_words & (content_words | topic_words))
            if overlap > 0:
                scored.append((overlap, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    async def clear(self) -> None:
        """Clear the episode log file."""
        if self._log_file.exists():
            self._log_file.unlink()

    async def get_all(self) -> list[MemoryEntry]:
        """Read all episodes from the log file."""
        if not self._log_file.exists():
            return []
        entries: list[MemoryEntry] = []
        with open(self._log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                    entries.append(MemoryEntry(**data))
                except (json.JSONDecodeError, ValueError):
                    continue
        return entries

    async def get_episodes_by_topic(self, topic: str) -> list[MemoryEntry]:
        """Filter episodes by topic."""
        all_eps = await self.get_all()
        return [
            e for e in all_eps
            if topic.lower() in e.metadata.get("topic", "").lower()
        ]
