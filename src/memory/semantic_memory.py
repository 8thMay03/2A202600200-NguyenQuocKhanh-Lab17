"""Chroma Semantic Memory — Vector-based similarity search.

Uses OpenAI embeddings and ChromaDB for semantic retrieval
of past context and knowledge.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import tiktoken

from src.config import CHROMA_DB_DIR, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_MODEL
from src.memory.base import MemoryBackend, MemoryEntry, MemoryType


class ChromaSemanticMemory(MemoryBackend):
    """Semantic memory using ChromaDB with OpenAI embeddings."""

    COLLECTION_NAME = "semantic_memory"

    def __init__(self, persist_dir: str | None = None):
        import chromadb
        from chromadb.config import Settings

        self._persist_dir = persist_dir or str(CHROMA_DB_DIR)
        self._client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=self._persist_dir,
        ))
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Embedding function — uses OpenAI if key is available, else falls back
        self._use_openai_embeddings = bool(OPENAI_API_KEY)

        try:
            self._encoder = tiktoken.encoding_for_model(OPENAI_MODEL)
        except KeyError:
            self._encoder = tiktoken.get_encoding("cl100k_base")

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.SEMANTIC

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI API."""
        if not self._use_openai_embeddings:
            # Fallback: simple hash-based pseudo-embeddings for testing
            import hashlib
            embeddings = []
            for text in texts:
                h = hashlib.sha256(text.encode()).hexdigest()
                vec = [int(h[i:i+2], 16) / 255.0 for i in range(0, min(len(h), 64), 2)]
                # Pad to 32 dimensions
                vec = (vec + [0.0] * 32)[:32]
                embeddings.append(vec)
            return embeddings

        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def store(self, entry: MemoryEntry) -> None:
        """Store entry with its embedding in ChromaDB."""
        entry.token_count = self._count_tokens(entry.content)
        entry.memory_type = MemoryType.SEMANTIC

        embeddings = self._get_embeddings([entry.content])

        metadata = {
            "timestamp": entry.timestamp.isoformat(),
            "priority": entry.priority,
            "token_count": entry.token_count,
        }
        # Flatten user metadata (Chroma only supports primitive types)
        for k, v in entry.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                metadata[k] = v

        self._collection.upsert(
            ids=[entry.id],
            documents=[entry.content],
            embeddings=embeddings,
            metadatas=[metadata],
        )

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Retrieve semantically similar entries."""
        if self._collection.count() == 0:
            return []

        query_embedding = self._get_embeddings([query])
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        entries: list[MemoryEntry] = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0

                entry = MemoryEntry(
                    id=results["ids"][0][i],
                    content=doc,
                    memory_type=MemoryType.SEMANTIC,
                    metadata={**meta, "similarity_distance": distance},
                    timestamp=datetime.fromisoformat(
                        meta.get("timestamp", datetime.now(timezone.utc).isoformat())
                    ),
                    priority=int(meta.get("priority", 2)),
                    token_count=int(meta.get("token_count", 0)),
                )
                entries.append(entry)

        return entries

    async def clear(self) -> None:
        """Delete the collection and recreate it."""
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    async def get_all(self) -> list[MemoryEntry]:
        """Get all stored entries."""
        if self._collection.count() == 0:
            return []

        results = self._collection.get(include=["documents", "metadatas"])
        entries: list[MemoryEntry] = []
        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            entry = MemoryEntry(
                id=results["ids"][i],
                content=doc,
                memory_type=MemoryType.SEMANTIC,
                metadata=meta,
                timestamp=datetime.fromisoformat(
                    meta.get("timestamp", datetime.utcnow().isoformat())
                ),
                priority=int(meta.get("priority", 2)),
                token_count=int(meta.get("token_count", 0)),
            )
            entries.append(entry)
        return entries
