"""Tests for memory backends and routing."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.memory.base import MemoryEntry, MemoryType
from src.memory.buffer_memory import ConversationBufferMemory
from src.memory.redis_memory import RedisLongTermMemory
from src.memory.episodic_memory import EpisodicMemory
from src.memory.semantic_memory import ChromaSemanticMemory
from src.memory.router import MemoryRouter
from src.context_manager import ContextWindowManager


@pytest.mark.asyncio
async def test_buffer_memory():
    mem = ConversationBufferMemory(max_messages=5)
    for i in range(7):
        await mem.store(MemoryEntry(content=f"Message {i}", memory_type=MemoryType.SHORT_TERM))
    entries = await mem.get_all()
    assert len(entries) == 5, f"Expected 5 entries after overflow, got {len(entries)}"
    assert entries[0].content == "Message 2"  # Oldest kept
    recent = await mem.retrieve("test", top_k=3)
    assert len(recent) == 3


@pytest.mark.asyncio
async def test_buffer_memory_threads():
    mem = ConversationBufferMemory()
    mem.set_thread("thread_a")
    await mem.store(MemoryEntry(content="Thread A msg", memory_type=MemoryType.SHORT_TERM))
    mem.set_thread("thread_b")
    await mem.store(MemoryEntry(content="Thread B msg", memory_type=MemoryType.SHORT_TERM))
    entries_b = await mem.get_all()
    assert len(entries_b) == 1
    assert entries_b[0].content == "Thread B msg"
    mem.set_thread("thread_a")
    entries_a = await mem.get_all()
    assert len(entries_a) == 1
    assert entries_a[0].content == "Thread A msg"


@pytest.mark.asyncio
async def test_redis_memory():
    mem = RedisLongTermMemory()
    await mem.clear()
    await mem.store_preference("language", "Python")
    await mem.store_preference("theme", "dark")
    prefs = await mem.get_preferences()
    assert prefs["language"] == "Python"
    assert prefs["theme"] == "dark"
    results = await mem.retrieve("Python language")
    assert len(results) > 0


@pytest.mark.asyncio
async def test_episodic_memory(tmp_path):
    mem = EpisodicMemory(log_dir=tmp_path / "episodes")
    await mem.clear()
    await mem.log_episode(
        user_query="How to use Docker?",
        agent_response="Docker is a containerization platform...",
        topic="docker", outcome="success",
    )
    await mem.log_episode(
        user_query="Explain Kubernetes",
        agent_response="Kubernetes orchestrates containers...",
        topic="kubernetes", outcome="success",
    )
    all_eps = await mem.get_all()
    assert len(all_eps) == 2
    docker_eps = await mem.get_episodes_by_topic("docker")
    assert len(docker_eps) == 1
    results = await mem.retrieve("Docker containers")
    assert len(results) > 0


@pytest.mark.asyncio
async def test_semantic_memory(tmp_path):
    mem = ChromaSemanticMemory(persist_dir=str(tmp_path / "chroma"))
    await mem.clear()
    await mem.store(MemoryEntry(
        content="Python is great for data science and machine learning",
        memory_type=MemoryType.SEMANTIC,
    ))
    await mem.store(MemoryEntry(
        content="JavaScript is used for web development",
        memory_type=MemoryType.SEMANTIC,
    ))
    results = await mem.retrieve("data science programming", top_k=2)
    assert len(results) >= 1


def test_rule_based_routing():
    buf = ConversationBufferMemory()
    redis = RedisLongTermMemory()
    episodic = EpisodicMemory()
    semantic = ChromaSemanticMemory()
    router = MemoryRouter(buf, redis, episodic, semantic, use_llm_routing=False)

    mem_type, _ = router._rule_based_route("What's my favorite color?")
    assert mem_type == MemoryType.LONG_TERM

    mem_type, _ = router._rule_based_route("Last time we discussed Docker")
    assert mem_type == MemoryType.EPISODIC

    mem_type, _ = router._rule_based_route("Find similar topics to Python")
    assert mem_type == MemoryType.SEMANTIC

    mem_type, _ = router._rule_based_route("Hello, how are you?")
    assert mem_type == MemoryType.SHORT_TERM


def test_context_window_manager():
    cwm = ContextWindowManager(max_tokens=100, trim_threshold=0.8)
    state = cwm.build_context(
        system_prompt="You are helpful.",
        current_query="Hello",
        recent_turns=[
            {"role": "user", "content": "Hi there " * 20},
            {"role": "assistant", "content": "Hello! " * 20},
        ],
        memory_entries=[],
    )
    assert state.total_tokens <= 100
    util = cwm.get_utilization(state)
    assert util["utilization_pct"] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
