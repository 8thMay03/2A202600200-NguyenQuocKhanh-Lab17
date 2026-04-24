"""Context Window Manager — Auto-trim and priority-based eviction.

Uses a 4-level priority hierarchy for eviction:
  0 (Critical): System prompt, current query — never evict
  1 (High):     Recent conversation turns (last 3)
  2 (Medium):   Retrieved memories, semantic context
  3 (Low):      Older conversation history, episodic logs
"""

from __future__ import annotations
from dataclasses import dataclass, field
import tiktoken
from src.config import (
    MAX_CONTEXT_TOKENS, OPENAI_MODEL,
    PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_LOW, PRIORITY_MEDIUM, TRIM_THRESHOLD,
)
from src.memory.base import MemoryEntry


@dataclass
class ContextBlock:
    content: str
    priority: int
    token_count: int = 0
    source: str = ""
    entry_id: str = ""


@dataclass
class ContextWindowState:
    blocks: list[ContextBlock] = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = MAX_CONTEXT_TOKENS
    trimmed_count: int = 0
    evicted_sources: list[str] = field(default_factory=list)


class ContextWindowManager:
    def __init__(self, max_tokens: int | None = None, trim_threshold: float | None = None):
        self._max_tokens = max_tokens or MAX_CONTEXT_TOKENS
        self._trim_threshold = trim_threshold or TRIM_THRESHOLD
        self._trim_target = int(self._max_tokens * self._trim_threshold)
        try:
            self._encoder = tiktoken.encoding_for_model(OPENAI_MODEL)
        except KeyError:
            self._encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def build_context(
        self, system_prompt: str, current_query: str,
        recent_turns: list[dict[str, str]], memory_entries: list[MemoryEntry],
    ) -> ContextWindowState:
        blocks: list[ContextBlock] = []
        blocks.append(ContextBlock(
            content=system_prompt, priority=PRIORITY_CRITICAL,
            token_count=self.count_tokens(system_prompt), source="system_prompt",
        ))
        blocks.append(ContextBlock(
            content=current_query, priority=PRIORITY_CRITICAL,
            token_count=self.count_tokens(current_query), source="current_query",
        ))
        for i, turn in enumerate(recent_turns):
            content = f"{turn['role']}: {turn['content']}"
            tokens = self.count_tokens(content)
            is_recent = i >= len(recent_turns) - 3
            blocks.append(ContextBlock(
                content=content,
                priority=PRIORITY_HIGH if is_recent else PRIORITY_LOW,
                token_count=tokens,
                source="recent_turn" if is_recent else "older_turn",
            ))
        for entry in memory_entries:
            priority = PRIORITY_LOW if entry.memory_type == "episodic" else PRIORITY_MEDIUM
            blocks.append(ContextBlock(
                content=entry.content, priority=priority,
                token_count=entry.token_count or self.count_tokens(entry.content),
                source=f"memory_{entry.memory_type}", entry_id=entry.id,
            ))
        total_tokens = sum(b.token_count for b in blocks)
        state = ContextWindowState(blocks=blocks, total_tokens=total_tokens, max_tokens=self._max_tokens)
        if total_tokens > self._trim_target:
            state = self._evict(state)
        return state

    def _evict(self, state: ContextWindowState) -> ContextWindowState:
        evictable = [(i, b) for i, b in enumerate(state.blocks) if b.priority > PRIORITY_CRITICAL]
        evictable.sort(key=lambda x: (-x[1].priority, x[0]))
        indices_to_remove: set[int] = set()
        current_total = state.total_tokens
        for idx, block in evictable:
            if current_total <= self._trim_target:
                break
            indices_to_remove.add(idx)
            current_total -= block.token_count
            state.evicted_sources.append(block.source)
            state.trimmed_count += 1
        state.blocks = [b for i, b in enumerate(state.blocks) if i not in indices_to_remove]
        state.total_tokens = sum(b.token_count for b in state.blocks)
        return state

    def format_context_for_llm(self, state: ContextWindowState) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for block in state.blocks:
            if block.source == "system_prompt":
                messages.append({"role": "system", "content": block.content})
            elif block.source == "current_query":
                continue
            elif block.source in ("recent_turn", "older_turn"):
                if block.content.startswith("user:"):
                    messages.append({"role": "user", "content": block.content[5:].strip()})
                elif block.content.startswith("assistant:"):
                    messages.append({"role": "assistant", "content": block.content[10:].strip()})
            elif block.source.startswith("memory_"):
                messages.append({"role": "system", "content": f"[Retrieved Memory ({block.source})]: {block.content}"})
        for block in state.blocks:
            if block.source == "current_query":
                messages.append({"role": "user", "content": block.content})
                break
        return messages

    def get_utilization(self, state: ContextWindowState) -> dict:
        return {
            "total_tokens": state.total_tokens,
            "max_tokens": state.max_tokens,
            "utilization_pct": round(state.total_tokens / state.max_tokens * 100, 1) if state.max_tokens > 0 else 0,
            "blocks_count": len(state.blocks),
            "trimmed_count": state.trimmed_count,
            "evicted_sources": state.evicted_sources,
            "by_priority": {
                "critical": sum(b.token_count for b in state.blocks if b.priority == PRIORITY_CRITICAL),
                "high": sum(b.token_count for b in state.blocks if b.priority == PRIORITY_HIGH),
                "medium": sum(b.token_count for b in state.blocks if b.priority == PRIORITY_MEDIUM),
                "low": sum(b.token_count for b in state.blocks if b.priority == PRIORITY_LOW),
            },
        }
