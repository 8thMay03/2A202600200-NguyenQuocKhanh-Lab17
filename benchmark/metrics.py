"""Evaluation metrics for the benchmark.

Measures:
- Response Relevance (1-5, LLM-judged)
- Context Utilization (%)
- Token Efficiency (tokens per useful info unit)
- Memory Hit Rate (per backend)
- Token Budget Breakdown (system / memory / response splits)
"""

from __future__ import annotations
import json
from collections import defaultdict
from langchain_openai import ChatOpenAI
from src.config import OPENAI_API_KEY, OPENAI_MODEL

# ── Prompts ───────────────────────────────────────────────────────────────────

RELEVANCE_PROMPT = """You are an evaluation judge. Rate the following response on a scale of 1-5
for RELEVANCE to the conversation context and query.

Scoring:
1 = Completely irrelevant, ignores context
2 = Partially relevant but misses key context
3 = Relevant but doesn't fully utilize available context
4 = Highly relevant, uses most available context
5 = Perfectly relevant, fully leverages all available context

Conversation history:
{history}

Current query: {query}

Agent response: {response}

Respond with ONLY a JSON object: {{"score": <1-5>, "explanation": "<brief reason>"}}"""

CONTEXT_UTIL_PROMPT = """Evaluate how well this response utilizes the conversation context.
Rate from 0.0 to 1.0 where:
- 0.0 = No context used at all
- 0.5 = Some context referenced
- 1.0 = All relevant context fully utilized

Conversation history:
{history}

Query: {query}
Response: {response}

Respond with ONLY a JSON object: {{"score": <0.0-1.0>, "explanation": "<brief reason>"}}"""


# ── Main class ────────────────────────────────────────────────────────────────

class BenchmarkMetrics:
    """Collects and computes all benchmark metrics."""

    # Memory backend labels shown in tables
    MEMORY_TYPES = ["short_term", "long_term", "episodic", "semantic"]

    def __init__(self):
        self.llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)

    # ── LLM-judged metrics ────────────────────────────────────────────────────

    async def evaluate_relevance(self, history: str, query: str, response: str) -> dict:
        try:
            result = await self.llm.ainvoke(
                RELEVANCE_PROMPT.format(history=history, query=query, response=response)
            )
            content = result.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content)
            return {"score": int(parsed.get("score", 3)), "explanation": parsed.get("explanation", "")}
        except Exception as e:
            return {"score": 3, "explanation": f"Evaluation error: {e}"}

    async def evaluate_context_utilization(self, history: str, query: str, response: str) -> dict:
        try:
            result = await self.llm.ainvoke(
                CONTEXT_UTIL_PROMPT.format(history=history, query=query, response=response)
            )
            content = result.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content)
            return {"score": float(parsed.get("score", 0.5)), "explanation": parsed.get("explanation", "")}
        except Exception as e:
            return {"score": 0.5, "explanation": f"Evaluation error: {e}"}

    # ── Token metrics ─────────────────────────────────────────────────────────

    def calculate_token_efficiency(self, response: str, context_tokens: int) -> dict:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(OPENAI_MODEL)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        response_tokens = len(enc.encode(response))
        words = len(response.split())
        sentences = max(1, response.count('.') + response.count('!') + response.count('?'))
        info_density = words / response_tokens if response_tokens > 0 else 0
        return {
            "response_tokens": response_tokens,
            "context_tokens_used": context_tokens,
            "total_tokens": response_tokens + context_tokens,
            "info_density": round(info_density, 3),
            "words": words,
            "sentences": sentences,
        }

    def calculate_token_budget(
        self,
        response: str,
        context_util: dict,
    ) -> dict:
        """Break down token usage into: system, memory, history, response.

        Uses the context_utilization dict returned by ContextWindowManager.
        """
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(OPENAI_MODEL)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        by_priority = context_util.get("by_priority", {})
        response_tokens = len(enc.encode(response))
        system_tokens = by_priority.get("critical", 0)
        high_tokens = by_priority.get("high", 0)      # recent turns
        medium_tokens = by_priority.get("medium", 0)  # retrieved memories
        low_tokens = by_priority.get("low", 0)        # older history / episodic

        total = system_tokens + high_tokens + medium_tokens + low_tokens + response_tokens or 1
        return {
            "system_tokens": system_tokens,
            "history_tokens": high_tokens + low_tokens,
            "memory_tokens": medium_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total,
            # percentage shares
            "pct_system": round(system_tokens / total * 100, 1),
            "pct_history": round((high_tokens + low_tokens) / total * 100, 1),
            "pct_memory": round(medium_tokens / total * 100, 1),
            "pct_response": round(response_tokens / total * 100, 1),
            "trimmed_count": context_util.get("trimmed_count", 0),
            "utilization_pct": context_util.get("utilization_pct", 0.0),
        }

    # ── Memory hit rate ───────────────────────────────────────────────────────

    @staticmethod
    def compute_memory_hit_rates(conv_results: list[dict]) -> dict:
        """Compute how often each memory backend was routed to across all turns.

        Returns:
            {
              "per_backend": {"short_term": N, "long_term": N, "episodic": N, "semantic": N},
              "total_routed_turns": N,
              "hit_rate_pct": {"short_term": X.x, ...},
              "per_conversation": [{name, counts, dominant_type}, ...],
            }
        """
        total_counts: dict[str, int] = defaultdict(int)
        per_conv: list[dict] = []

        for conv in conv_results:
            conv_counts: dict[str, int] = defaultdict(int)
            for turn in conv.get("turns", []):
                mtype = turn.get("routed_memory_type", "")
                if mtype and mtype != "none":
                    total_counts[mtype] += 1
                    conv_counts[mtype] += 1
            dominant = max(conv_counts, key=conv_counts.get) if conv_counts else "—"
            per_conv.append({
                "name": conv["name"],
                "counts": dict(conv_counts),
                "dominant_type": dominant,
            })

        total = sum(total_counts.values()) or 1
        hit_rates = {k: round(v / total * 100, 1) for k, v in total_counts.items()}
        return {
            "per_backend": dict(total_counts),
            "total_routed_turns": total,
            "hit_rate_pct": hit_rates,
            "per_conversation": per_conv,
        }

    @staticmethod
    def compute_token_budget_summary(conv_results: list[dict]) -> dict:
        """Aggregate token budget breakdown across all conversations."""
        totals = defaultdict(float)
        count = 0
        for conv in conv_results:
            for turn in conv.get("turns", []):
                tb = turn.get("token_budget")
                if tb:
                    for key in ("system_tokens", "history_tokens", "memory_tokens", "response_tokens", "total_tokens"):
                        totals[key] += tb.get(key, 0)
                    count += 1
        if count == 0:
            return {}
        avgs = {k: round(v / count, 1) for k, v in totals.items()}
        grand_total = avgs.get("total_tokens", 1) or 1
        avgs["pct_system"] = round(avgs["system_tokens"] / grand_total * 100, 1)
        avgs["pct_history"] = round(avgs["history_tokens"] / grand_total * 100, 1)
        avgs["pct_memory"] = round(avgs["memory_tokens"] / grand_total * 100, 1)
        avgs["pct_response"] = round(avgs["response_tokens"] / grand_total * 100, 1)
        return avgs
