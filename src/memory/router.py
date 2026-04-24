"""Memory Router — Selects the appropriate memory backend based on query intent.

Uses LLM-based classification to determine whether a query is about:
- user_preference  → Redis long-term memory
- factual_recall   → ConversationBuffer short-term memory
- experience_recall → JSON episodic memory
- semantic_search   → Chroma semantic memory
"""

from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.memory.base import MemoryBackend, MemoryEntry, MemoryType
from src.memory.buffer_memory import ConversationBufferMemory
from src.memory.redis_memory import RedisLongTermMemory
from src.memory.episodic_memory import EpisodicMemory
from src.memory.semantic_memory import ChromaSemanticMemory


ROUTING_PROMPT = """You are a memory router. Given a user query, classify which type of memory
should be consulted to best answer it. Choose exactly ONE of the following categories:

1. "short_term" — The user is asking about something from the CURRENT conversation,
   recent messages, or immediate context. E.g., "What did I just say?", "Summarize our chat",
   "Can you repeat that?"

2. "long_term" — The user is asking about their PREFERENCES, personal facts, or
   persistent information that should be remembered across sessions. E.g., "What's my
   favorite language?", "Remember I prefer dark mode", "What's my name?"

3. "episodic" — The user is referencing a PAST EXPERIENCE or previous conversation/session.
   E.g., "Last time we talked about...", "Remember when I asked about...",
   "What happened in our previous session?"

4. "semantic" — The user wants to find SIMILAR content, topics, or concepts.
   E.g., "Find topics related to...", "What do you know about X?",
   "Search for information about..."

If the query doesn't clearly match any category, default to "short_term".

Respond with ONLY a JSON object: {{"memory_type": "<type>", "reason": "<brief reason>"}}

User query: {query}"""


class MemoryRouter:
    """Routes queries to the appropriate memory backend based on intent."""

    def __init__(
        self,
        buffer_memory: ConversationBufferMemory,
        redis_memory: RedisLongTermMemory,
        episodic_memory: EpisodicMemory,
        semantic_memory: ChromaSemanticMemory,
        use_llm_routing: bool = True,
    ):
        self._backends: dict[MemoryType, MemoryBackend] = {
            MemoryType.SHORT_TERM: buffer_memory,
            MemoryType.LONG_TERM: redis_memory,
            MemoryType.EPISODIC: episodic_memory,
            MemoryType.SEMANTIC: semantic_memory,
        }
        self._use_llm_routing = use_llm_routing
        if use_llm_routing and OPENAI_API_KEY:
            self._llm = ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0,
                max_tokens=100,
            )
        else:
            self._llm = None

    def get_backend(self, memory_type: MemoryType) -> MemoryBackend:
        """Get a specific memory backend by type."""
        return self._backends[memory_type]

    @property
    def all_backends(self) -> dict[MemoryType, MemoryBackend]:
        return dict(self._backends)

    async def route(self, query: str) -> tuple[MemoryType, str]:
        """Determine which memory type to use for the given query.

        Returns:
            Tuple of (MemoryType, reason)
        """
        if self._llm and self._use_llm_routing:
            return await self._llm_route(query)
        return self._rule_based_route(query)

    async def _llm_route(self, query: str) -> tuple[MemoryType, str]:
        """Use LLM to classify query intent."""
        try:
            response = await self._llm.ainvoke(
                ROUTING_PROMPT.format(query=query)
            )
            content = response.content.strip()
            # Parse JSON from response
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            memory_type = MemoryType(result["memory_type"])
            reason = result.get("reason", "LLM classification")
            return memory_type, reason
        except Exception:
            # Fallback to rule-based
            return self._rule_based_route(query)

    def _rule_based_route(self, query: str) -> tuple[MemoryType, str]:
        """Simple keyword-based routing as fallback."""
        q = query.lower()

        # Long-term / preference keywords
        pref_keywords = [
            "prefer", "favorite", "my name", "remember that i",
            "always", "usually", "i like", "i hate", "i want",
            "my preference", "dark mode", "light mode", "save this",
        ]
        if any(kw in q for kw in pref_keywords):
            return MemoryType.LONG_TERM, "Query contains preference-related keywords"

        # Episodic keywords
        episodic_keywords = [
            "last time", "previously", "before", "past session",
            "remember when", "earlier", "history", "we discussed",
            "past conversation", "last session",
        ]
        if any(kw in q for kw in episodic_keywords):
            return MemoryType.EPISODIC, "Query references past experiences"

        # Semantic keywords
        semantic_keywords = [
            "similar", "related", "search for", "find",
            "what do you know about", "information about",
            "topics like", "knowledge about",
        ]
        if any(kw in q for kw in semantic_keywords):
            return MemoryType.SEMANTIC, "Query requests semantic search"

        # Default: short-term
        return MemoryType.SHORT_TERM, "Default to current conversation context"

    async def retrieve_from_all(
        self, query: str, top_k_per_backend: int = 3
    ) -> dict[MemoryType, list[MemoryEntry]]:
        """Retrieve from ALL memory backends (used for comprehensive context)."""
        results: dict[MemoryType, list[MemoryEntry]] = {}
        for mem_type, backend in self._backends.items():
            try:
                entries = await backend.retrieve(query, top_k=top_k_per_backend)
                if entries:
                    results[mem_type] = entries
            except Exception:
                continue
        return results

    async def retrieve_routed(
        self, query: str, top_k: int = 5
    ) -> tuple[MemoryType, list[MemoryEntry], str]:
        """Route query and retrieve from the chosen backend.

        Returns:
            Tuple of (chosen_type, entries, reason)
        """
        mem_type, reason = await self.route(query)
        backend = self._backends[mem_type]
        entries = await backend.retrieve(query, top_k=top_k)
        return mem_type, entries, reason

    async def store_to_appropriate(self, entry: MemoryEntry) -> None:
        """Store an entry in the backend matching its memory_type."""
        mem_type = MemoryType(entry.memory_type)
        backend = self._backends[mem_type]
        await backend.store(entry)

    async def get_all_stats(self) -> dict[str, Any]:
        """Get statistics from all backends."""
        stats = {}
        for mem_type, backend in self._backends.items():
            stats[mem_type.value] = await backend.get_stats()
        return stats
