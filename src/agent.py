"""Multi-Memory LangGraph Agent.

Builds a LangGraph StateGraph that integrates all 4 memory backends,
memory routing, and context window management.
"""

from __future__ import annotations
import operator
from typing import Annotated, Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.context_manager import ContextWindowManager
from src.memory.base import MemoryEntry, MemoryType
from src.memory.buffer_memory import ConversationBufferMemory
from src.memory.redis_memory import RedisLongTermMemory
from src.memory.episodic_memory import EpisodicMemory
from src.memory.semantic_memory import ChromaSemanticMemory
from src.memory.router import MemoryRouter

SYSTEM_PROMPT = """You are a helpful AI assistant with access to multiple memory systems.
You can remember user preferences, recall past conversations, and find semantically
related information. Use the provided context to give personalized, relevant responses.
If you remember something about the user, use it naturally in your response.
Always be helpful, accurate, and context-aware."""


class AgentState(TypedDict):
    messages: Annotated[list[dict[str, str]], operator.add]
    current_query: str
    memory_context: list[MemoryEntry]
    routed_memory_type: str
    routing_reason: str
    response: str
    context_utilization: dict[str, Any]
    turn_count: int


class MultiMemoryAgent:
    """LangGraph agent with full memory stack."""

    def __init__(self, use_memory: bool = True, use_llm_routing: bool = True):
        self.use_memory = use_memory
        self.buffer_memory = ConversationBufferMemory()
        self.redis_memory = RedisLongTermMemory()
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = ChromaSemanticMemory()
        self.router = MemoryRouter(
            buffer_memory=self.buffer_memory,
            redis_memory=self.redis_memory,
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
            use_llm_routing=use_llm_routing,
        )
        self.context_manager = ContextWindowManager()
        self.llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.7)
        self._turn_count = 0
        self._conversation_history: list[dict[str, str]] = []
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve_memory", self._retrieve_memory_node)
        workflow.add_node("build_context", self._build_context_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("store_memory", self._store_memory_node)
        workflow.set_entry_point("retrieve_memory")
        workflow.add_edge("retrieve_memory", "build_context")
        workflow.add_edge("build_context", "generate_response")
        workflow.add_edge("generate_response", "store_memory")
        workflow.add_edge("store_memory", END)
        return workflow.compile()

    async def _retrieve_memory_node(self, state: AgentState) -> dict:
        query = state["current_query"]
        if not self.use_memory:
            return {"memory_context": [], "routed_memory_type": "none", "routing_reason": "Memory disabled"}
        mem_type, entries, reason = await self.router.retrieve_routed(query, top_k=5)
        all_memories = await self.router.retrieve_from_all(query, top_k_per_backend=2)
        combined = list(entries)
        for mt, ents in all_memories.items():
            if mt != mem_type:
                for e in ents:
                    if e.id not in {c.id for c in combined}:
                        combined.append(e)
        return {"memory_context": combined, "routed_memory_type": mem_type.value, "routing_reason": reason}

    async def _build_context_node(self, state: AgentState) -> dict:
        ctx_state = self.context_manager.build_context(
            system_prompt=SYSTEM_PROMPT, current_query=state["current_query"],
            recent_turns=self._conversation_history[-10:],
            memory_entries=state.get("memory_context", []),
        )
        utilization = self.context_manager.get_utilization(ctx_state)
        messages = self.context_manager.format_context_for_llm(ctx_state)
        return {"messages": messages, "context_utilization": utilization}

    async def _generate_response_node(self, state: AgentState) -> dict:
        messages = []
        for msg in state.get("messages", []):
            if msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        if not messages:
            messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=state["current_query"])]
        response = await self.llm.ainvoke(messages)
        return {"response": response.content}

    async def _store_memory_node(self, state: AgentState) -> dict:
        query = state["current_query"]
        response = state.get("response", "")
        self._conversation_history.append({"role": "user", "content": query})
        self._conversation_history.append({"role": "assistant", "content": response})
        self._turn_count += 1
        if not self.use_memory:
            return {"turn_count": self._turn_count}
        # Store in short-term
        await self.buffer_memory.store(MemoryEntry(content=f"User: {query}", memory_type=MemoryType.SHORT_TERM, priority=1))
        await self.buffer_memory.store(MemoryEntry(content=f"Assistant: {response}", memory_type=MemoryType.SHORT_TERM, priority=1))
        # Store in episodic
        await self.episodic_memory.log_episode(user_query=query, agent_response=response[:200], topic=self._extract_topic(query))
        # Store in semantic
        await self.semantic_memory.store(MemoryEntry(content=f"Q: {query}\nA: {response[:200]}", memory_type=MemoryType.SEMANTIC, priority=2))
        # Detect and store preferences
        await self._detect_and_store_preferences(query, response)
        return {"turn_count": self._turn_count}

    async def _detect_and_store_preferences(self, query: str, response: str):
        pref_indicators = ["i prefer", "i like", "my favorite", "i always", "i want", "call me", "my name is", "i use", "i hate"]
        q_lower = query.lower()
        if any(ind in q_lower for ind in pref_indicators):
            await self.redis_memory.store(MemoryEntry(
                content=f"User stated: {query}", memory_type=MemoryType.LONG_TERM,
                metadata={"category": "preference", "source_query": query}, priority=1,
            ))

    def _extract_topic(self, query: str) -> str:
        words = query.lower().split()
        stop_words = {"i", "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "should", "what", "how", "why", "when", "where", "who", "my", "me", "you", "your", "it", "this", "that", "to", "of", "in", "for", "on", "with", "at", "by", "from", "about"}
        topic_words = [w for w in words if w not in stop_words and len(w) > 2]
        return " ".join(topic_words[:3]) if topic_words else "general"

    async def chat(self, user_message: str) -> dict[str, Any]:
        initial_state: AgentState = {
            "messages": [], "current_query": user_message, "memory_context": [],
            "routed_memory_type": "", "routing_reason": "", "response": "",
            "context_utilization": {}, "turn_count": self._turn_count,
        }
        result = await self.graph.ainvoke(initial_state)
        return {
            "response": result.get("response", ""),
            "routed_memory_type": result.get("routed_memory_type", ""),
            "routing_reason": result.get("routing_reason", ""),
            "context_utilization": result.get("context_utilization", {}),
            "turn_count": result.get("turn_count", self._turn_count),
        }

    async def reset(self):
        self._conversation_history.clear()
        self._turn_count = 0
        await self.buffer_memory.clear_all()
        await self.redis_memory.clear()
        await self.episodic_memory.clear()
        await self.semantic_memory.clear()

    def set_thread(self, thread_id: str):
        self.buffer_memory.set_thread(thread_id)

    async def get_memory_stats(self) -> dict:
        return await self.router.get_all_stats()
