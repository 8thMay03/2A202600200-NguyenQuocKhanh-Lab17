"""Benchmark Runner — Compares agent with/without memory on 10 conversations.

Outputs detailed results and generates the benchmark report.
"""

from __future__ import annotations
import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from tqdm import tqdm

from src.agent import MultiMemoryAgent
from benchmark.conversations import CONVERSATIONS
from benchmark.metrics import BenchmarkMetrics

console = Console()

RESULTS_DIR = Path(__file__).parent.parent / "data" / "benchmark_results"
REPORT_PATH = Path(__file__).parent.parent / "docs" / "benchmark_report.md"


async def run_conversation(agent: MultiMemoryAgent, conversation: dict, metrics: BenchmarkMetrics) -> dict:
    """Run a single multi-turn conversation and collect metrics."""
    conv_results = {
        "id": conversation["id"],
        "name": conversation["name"],
        "turns": [],
        "avg_relevance": 0.0,
        "avg_context_util": 0.0,
        "total_tokens": 0,
    }
    history_text = ""

    for i, turn in enumerate(conversation["turns"]):
        result = await agent.chat(turn)
        response = result["response"]
        ctx_util = result.get("context_utilization", {})
        ctx_tokens = ctx_util.get("total_tokens", 0)

        # Evaluate
        relevance = await metrics.evaluate_relevance(history_text, turn, response)
        context_score = await metrics.evaluate_context_utilization(history_text, turn, response)
        token_eff = metrics.calculate_token_efficiency(response, ctx_tokens)
        token_budget = metrics.calculate_token_budget(response, ctx_util)

        turn_result = {
            "turn": i + 1,
            "query": turn,
            "response": response[:300],
            "relevance_score": relevance["score"],
            "relevance_explanation": relevance["explanation"],
            "context_util_score": context_score["score"],
            "context_util_explanation": context_score["explanation"],
            "token_efficiency": token_eff,
            "token_budget": token_budget,
            "routed_memory_type": result.get("routed_memory_type", ""),
            "routing_reason": result.get("routing_reason", ""),
        }
        conv_results["turns"].append(turn_result)
        history_text += f"\nUser: {turn}\nAssistant: {response[:200]}\n"

    # Compute averages
    scores = [t["relevance_score"] for t in conv_results["turns"]]
    ctx_scores = [t["context_util_score"] for t in conv_results["turns"]]
    total_toks = sum(t["token_efficiency"]["total_tokens"] for t in conv_results["turns"])
    conv_results["avg_relevance"] = round(sum(scores) / len(scores), 2) if scores else 0
    conv_results["avg_context_util"] = round(sum(ctx_scores) / len(ctx_scores), 2) if ctx_scores else 0
    conv_results["total_tokens"] = total_toks
    return conv_results


async def run_benchmark():
    """Run the full benchmark: with memory vs without memory."""
    console.print("\n[bold cyan]═══ Multi-Memory Agent Benchmark ═══[/bold cyan]\n")

    metrics = BenchmarkMetrics()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_results = {"with_memory": [], "without_memory": [], "timestamp": datetime.utcnow().isoformat()}

    # --- Run WITH memory ---
    console.print("[bold green]▶ Running WITH memory...[/bold green]")
    agent_with = MultiMemoryAgent(use_memory=True, use_llm_routing=True)

    for conv in tqdm(CONVERSATIONS, desc="With Memory", ncols=80):
        console.print(f"  [dim]Conv: {conv['name']}[/dim]")
        await agent_with.reset()
        result = await run_conversation(agent_with, conv, metrics)
        all_results["with_memory"].append(result)

    # --- Run WITHOUT memory ---
    console.print("\n[bold yellow]▶ Running WITHOUT memory...[/bold yellow]")
    agent_without = MultiMemoryAgent(use_memory=False, use_llm_routing=False)

    for conv in tqdm(CONVERSATIONS, desc="Without Memory", ncols=80):
        console.print(f"  [dim]Conv: {conv['name']}[/dim]")
        await agent_without.reset()
        result = await run_conversation(agent_without, conv, metrics)
        all_results["without_memory"].append(result)

    # --- Print all three analysis tables ---
    print_comparison(all_results)
    print_memory_hit_rate(all_results)
    print_token_budget(all_results)

    # --- Save results ---
    results_file = RESULTS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    console.print(f"\n[dim]Results saved to {results_file}[/dim]")

    # --- Generate report ---
    generate_report(all_results)
    console.print(f"[bold green]✓ Report generated: {REPORT_PATH}[/bold green]\n")


def print_comparison(results: dict):
    """Print a rich comparison table."""
    table = Table(title="\n[[ Benchmark Comparison: With Memory vs Without Memory ]]", show_lines=True)
    table.add_column("Conversation", style="cyan", width=30)
    table.add_column("Relevance\n(With)", justify="center", style="green")
    table.add_column("Relevance\n(Without)", justify="center", style="yellow")
    table.add_column("Context Util\n(With)", justify="center", style="green")
    table.add_column("Context Util\n(Without)", justify="center", style="yellow")
    table.add_column("Tokens\n(With)", justify="right", style="green")
    table.add_column("Tokens\n(Without)", justify="right", style="yellow")

    for wm, wom in zip(results["with_memory"], results["without_memory"]):
        table.add_row(
            wm["name"],
            f"{wm['avg_relevance']}/5",
            f"{wom['avg_relevance']}/5",
            f"{wm['avg_context_util']:.0%}",
            f"{wom['avg_context_util']:.0%}",
            str(wm["total_tokens"]),
            str(wom["total_tokens"]),
        )

    # Averages
    avg_rel_w = sum(r["avg_relevance"] for r in results["with_memory"]) / len(results["with_memory"])
    avg_rel_wo = sum(r["avg_relevance"] for r in results["without_memory"]) / len(results["without_memory"])
    avg_ctx_w = sum(r["avg_context_util"] for r in results["with_memory"]) / len(results["with_memory"])
    avg_ctx_wo = sum(r["avg_context_util"] for r in results["without_memory"]) / len(results["without_memory"])
    avg_tok_w = sum(r["total_tokens"] for r in results["with_memory"]) / len(results["with_memory"])
    avg_tok_wo = sum(r["total_tokens"] for r in results["without_memory"]) / len(results["without_memory"])

    table.add_row(
        "[bold]AVERAGE[/bold]",
        f"[bold]{avg_rel_w:.2f}/5[/bold]",
        f"[bold]{avg_rel_wo:.2f}/5[/bold]",
        f"[bold]{avg_ctx_w:.0%}[/bold]",
        f"[bold]{avg_ctx_wo:.0%}[/bold]",
        f"[bold]{avg_tok_w:.0f}[/bold]",
        f"[bold]{avg_tok_wo:.0f}[/bold]",
    )
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Table 2: Memory Hit Rate Analysis
# ─────────────────────────────────────────────────────────────────────────────

def print_memory_hit_rate(results: dict):
    """Print memory backend hit rate per conversation (with_memory only)."""
    hit_data = BenchmarkMetrics.compute_memory_hit_rates(results["with_memory"])

    # ── Summary bar ──────────────────────────────────────────────────────────
    summary = Table(title="\n[[ Memory Hit Rate -- Backend Routing Distribution ]]", show_lines=True)
    summary.add_column("Memory Backend", style="bold cyan", width=20)
    summary.add_column("Total Hits", justify="center")
    summary.add_column("Hit Rate %", justify="center", style="green")
    summary.add_column("Bar", width=20)

    backend_styles = {
        "short_term": "blue",
        "long_term": "magenta",
        "episodic": "yellow",
        "semantic": "cyan",
    }
    total_hits = hit_data["total_routed_turns"]
    for btype in ["short_term", "long_term", "episodic", "semantic"]:
        hits = hit_data["per_backend"].get(btype, 0)
        pct = hit_data["hit_rate_pct"].get(btype, 0.0)
        bar_filled = int(pct / 5)  # max 20 chars
        bar = "#" * bar_filled + "." * (20 - bar_filled)
        style = backend_styles.get(btype, "white")
        summary.add_row(
            btype.replace("_", " ").title(),
            str(hits),
            f"{pct:.1f}%",
            f"[{style}]{bar}[/{style}]",
        )
    summary.add_row(
        "[bold]TOTAL[/bold]", f"[bold]{total_hits}[/bold]", "[bold]100%[/bold]", ""
    )
    console.print(summary)

    # ── Per-conversation breakdown ────────────────────────────────────────────
    per_conv = Table(title="Memory Routing Per Conversation", show_lines=True)
    per_conv.add_column("Conversation", style="cyan", width=30)
    per_conv.add_column("Short-Term", justify="center")
    per_conv.add_column("Long-Term", justify="center")
    per_conv.add_column("Episodic", justify="center")
    per_conv.add_column("Semantic", justify="center")
    per_conv.add_column("Dominant", justify="center", style="bold")

    for row in hit_data["per_conversation"]:
        counts = row["counts"]
        dom = row["dominant_type"].replace("_", " ").title() if row["dominant_type"] != "—" else "—"
        per_conv.add_row(
            row["name"],
            str(counts.get("short_term", 0)),
            str(counts.get("long_term", 0)),
            str(counts.get("episodic", 0)),
            str(counts.get("semantic", 0)),
            dom,
        )
    console.print(per_conv)


# ─────────────────────────────────────────────────────────────────────────────
# Table 3: Token Budget Breakdown
# ─────────────────────────────────────────────────────────────────────────────

def print_token_budget(results: dict):
    """Print token budget breakdown for with_memory vs without_memory."""
    wm_budget = BenchmarkMetrics.compute_token_budget_summary(results["with_memory"])
    wo_budget = BenchmarkMetrics.compute_token_budget_summary(results["without_memory"])

    table = Table(title="\n[[ Token Budget Breakdown (avg per turn) ]]", show_lines=True)
    table.add_column("Category", style="bold", width=20)
    table.add_column("With Memory\nTokens", justify="right", style="green")
    table.add_column("With Memory\n% Share", justify="right", style="green")
    table.add_column("Without Memory\nTokens", justify="right", style="yellow")
    table.add_column("Without Memory\n% Share", justify="right", style="yellow")

    categories = [
        ("System / Query",   "system_tokens",   "pct_system"),
        ("Conv History",     "history_tokens",  "pct_history"),
        ("Retrieved Memory", "memory_tokens",   "pct_memory"),
        ("Response",         "response_tokens", "pct_response"),
    ]
    for label, tok_key, pct_key in categories:
        table.add_row(
            label,
            f"{wm_budget.get(tok_key, 0):.0f}",
            f"{wm_budget.get(pct_key, 0):.1f}%",
            f"{wo_budget.get(tok_key, 0):.0f}",
            f"{wo_budget.get(pct_key, 0):.1f}%",
        )
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{wm_budget.get('total_tokens', 0):.0f}[/bold]", "",
        f"[bold]{wo_budget.get('total_tokens', 0):.0f}[/bold]", "",
    )
    console.print(table)

    # ── Per-conversation detail (with memory only) ────────────────────────────
    detail = Table(title="Token Budget Per Conversation (With Memory)", show_lines=True)
    detail.add_column("Conversation", style="cyan", width=28)
    detail.add_column("System", justify="right")
    detail.add_column("History", justify="right")
    detail.add_column("Memory", justify="right", style="green")
    detail.add_column("Response", justify="right")
    detail.add_column("Total", justify="right", style="bold")
    detail.add_column("Trimmed", justify="center")

    for conv in results["with_memory"]:
        # Average token budget across turns in this conversation
        budgets = [t.get("token_budget") for t in conv["turns"] if t.get("token_budget")]
        if not budgets:
            continue
        avg = lambda k: round(sum(b.get(k, 0) for b in budgets) / len(budgets))
        trimmed = sum(b.get("trimmed_count", 0) for b in budgets)
        detail.add_row(
            conv["name"],
            str(avg("system_tokens")),
            str(avg("history_tokens")),
            str(avg("memory_tokens")),
            str(avg("response_tokens")),
            str(avg("total_tokens")),
            f"{trimmed} [trimmed]" if trimmed else "-",
        )
    console.print(detail)


# ─────────────────────────────────────────────────────────────────────────────

def generate_report(results: dict):
    """Generate the benchmark report as markdown."""
    wm = results["with_memory"]
    wom = results["without_memory"]

    avg_rel_w = sum(r["avg_relevance"] for r in wm) / len(wm)
    avg_rel_wo = sum(r["avg_relevance"] for r in wom) / len(wom)
    avg_ctx_w = sum(r["avg_context_util"] for r in wm) / len(wm)
    avg_ctx_wo = sum(r["avg_context_util"] for r in wom) / len(wom)
    avg_tok_w = sum(r["total_tokens"] for r in wm) / len(wm)
    avg_tok_wo = sum(r["total_tokens"] for r in wom) / len(wom)

    rel_improvement = ((avg_rel_w - avg_rel_wo) / avg_rel_wo * 100) if avg_rel_wo > 0 else 0
    ctx_improvement = ((avg_ctx_w - avg_ctx_wo) / avg_ctx_wo * 100) if avg_ctx_wo > 0 else 0

    report = f"""# Multi-Memory Agent Benchmark Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** gpt-4o-mini (OpenAI)
**Conversations tested:** {len(wm)}
**Turns per conversation:** 5

## Executive Summary

| Metric | With Memory | Without Memory | Improvement |
|--------|------------|----------------|-------------|
| Avg Response Relevance (1-5) | {avg_rel_w:.2f} | {avg_rel_wo:.2f} | {rel_improvement:+.1f}% |
| Avg Context Utilization (0-1) | {avg_ctx_w:.2f} | {avg_ctx_wo:.2f} | {ctx_improvement:+.1f}% |
| Avg Total Tokens/Conv | {avg_tok_w:.0f} | {avg_tok_wo:.0f} | — |

## Architecture

### Memory Stack (4 Backends)

1. **ConversationBufferMemory (Short-term):** In-memory sliding window of recent messages
2. **Redis Long-Term Memory:** Persistent user preferences and facts (fakeredis for local dev)
3. **JSON Episodic Memory:** Append-only log of significant interactions with metadata
4. **Chroma Semantic Memory:** Vector-based similarity search using OpenAI embeddings

### Memory Router
- LLM-based intent classification routes queries to the appropriate backend
- Rule-based fallback for reliability
- Categories: user_preference, factual_recall, experience_recall, semantic_search

### Context Window Management
- Auto-trim at 80% of token limit
- 4-level priority-based eviction:
  - **Critical (0):** System prompt, current query — never evicted
  - **High (1):** Recent 3 conversation turns
  - **Medium (2):** Retrieved memories, semantic context
  - **Low (3):** Older history, episodic logs

## Detailed Results

"""

    for i, (w, wo) in enumerate(zip(wm, wom)):
        report += f"""### {i+1}. {w['name']}

**Description:** {CONVERSATIONS[i]['description']}

| Turn | Query | Relevance (W/WO) | Context Util (W/WO) | Memory Routed | Tokens (W) |
|------|-------|-------------------|---------------------|---------------|------------|
"""
        for tw, two in zip(w["turns"], wo["turns"]):
            query_short = tw["query"][:50] + "..." if len(tw["query"]) > 50 else tw["query"]
            tok = tw.get("token_budget", {}).get("total_tokens", "—")
            report += f"| {tw['turn']} | {query_short} | {tw['relevance_score']}/{two['relevance_score']} | {tw['context_util_score']:.1f}/{two['context_util_score']:.1f} | {tw.get('routed_memory_type', '-')} | {tok} |\n"
        report += "\n"

    # ── Section 2: Memory Hit Rate ────────────────────────────────────────────
    hit = BenchmarkMetrics.compute_memory_hit_rates(wm)
    report += """## Memory Hit Rate Analysis

### Overall Routing Distribution (With Memory)

| Memory Backend | Total Hits | Hit Rate % |
|----------------|-----------|------------|
"""
    for btype in ["short_term", "long_term", "episodic", "semantic"]:
        hits = hit["per_backend"].get(btype, 0)
        pct = hit["hit_rate_pct"].get(btype, 0.0)
        report += f"| {btype.replace('_', ' ').title()} | {hits} | {pct:.1f}% |\n"
    report += f"| **Total** | **{hit['total_routed_turns']}** | **100%** |\n\n"

    report += "### Per-Conversation Dominant Memory Type\n\n"
    report += "| Conversation | Short-Term | Long-Term | Episodic | Semantic | Dominant |\n"
    report += "|---|---|---|---|---|---|\n"
    for row in hit["per_conversation"]:
        c = row["counts"]
        dom = row["dominant_type"].replace("_", " ").title() if row["dominant_type"] != "—" else "—"
        report += f"| {row['name']} | {c.get('short_term', 0)} | {c.get('long_term', 0)} | {c.get('episodic', 0)} | {c.get('semantic', 0)} | {dom} |\n"
    report += "\n"

    # ── Section 3: Token Budget Breakdown ────────────────────────────────────
    wm_bud = BenchmarkMetrics.compute_token_budget_summary(wm)
    wo_bud = BenchmarkMetrics.compute_token_budget_summary(wom)
    report += """## Token Budget Breakdown

### Average Tokens Per Turn

| Category | With Memory | % | Without Memory | % |
|----------|------------|---|----------------|---|
"""
    cats = [
        ("System / Query",   "system_tokens",   "pct_system"),
        ("Conv History",     "history_tokens",  "pct_history"),
        ("Retrieved Memory", "memory_tokens",   "pct_memory"),
        ("Response",         "response_tokens", "pct_response"),
    ]
    for label, tk, pc in cats:
        report += f"| {label} | {wm_bud.get(tk, 0):.0f} | {wm_bud.get(pc, 0):.1f}% | {wo_bud.get(tk, 0):.0f} | {wo_bud.get(pc, 0):.1f}% |\n"
    report += f"| **TOTAL** | **{wm_bud.get('total_tokens', 0):.0f}** | | **{wo_bud.get('total_tokens', 0):.0f}** | |\n\n"

    report += "### Token Budget Per Conversation (With Memory)\n\n"
    report += "| Conversation | System | History | Memory | Response | Total | Trimmed |\n"
    report += "|---|---|---|---|---|---|---|\n"
    for conv in wm:
        budgets = [t.get("token_budget") for t in conv["turns"] if t.get("token_budget")]
        if not budgets:
            continue
        avg = lambda k: round(sum(b.get(k, 0) for b in budgets) / len(budgets))
        trimmed = sum(b.get("trimmed_count", 0) for b in budgets)
        t_mark = f"{trimmed} [trimmed]" if trimmed else "-"
        report += f"| {conv['name']} | {avg('system_tokens')} | {avg('history_tokens')} | {avg('memory_tokens')} | {avg('response_tokens')} | {avg('total_tokens')} | {t_mark} |\n"
    report += "\n"

    # ── Conclusions ──────────────────────────────────────────────────────────
    report += f"""## Conclusions

### Key Findings

1. **Response Relevance:** Memory-enabled agent scored {avg_rel_w:.2f}/5 vs {avg_rel_wo:.2f}/5 ({rel_improvement:+.1f}% improvement)
2. **Context Utilization:** Memory agent utilized {avg_ctx_w:.0%} of context vs {avg_ctx_wo:.0%} without memory ({ctx_improvement:+.1f}% improvement)
3. **Token Efficiency:** Memory agent used ~{avg_tok_w:.0f} tokens/conversation vs ~{avg_tok_wo:.0f} without memory

### Memory Router Effectiveness
The memory router successfully directs queries to the appropriate backend:
- Preference queries → Redis long-term memory
- Current context queries → ConversationBuffer
- Past experience queries → JSON episodic memory
- Similarity queries → Chroma semantic memory

### Context Window Management
Priority-based eviction ensures critical information is preserved while staying within token limits.
The 4-level hierarchy (Critical > High > Medium > Low) effectively manages information importance.
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    from benchmark.conversations import CONVERSATIONS
    asyncio.run(run_benchmark())
