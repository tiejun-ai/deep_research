# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An open-source implementation of the core architecture described in Anthropic's "How we built our multi-agent research system" (https://www.anthropic.com/engineering/multi-agent-research-system). The goal is to showcase the canonical multi-agent (orchestrator-worker) pattern: given a research query, a lead agent coordinates several subagents that research in parallel, and the system produces a synthesized report with citations.

## Architecture (Orchestrator-Worker Multi-Agent)

The system follows the orchestrator-worker pattern from the article:

1. **Lead agent (orchestrator)** — plans its approach, decomposes the query into focused subtasks, and spawns multiple subagents in parallel (the article uses 3–5). It persists its plan to memory so context survives a long run, and decides when enough research has been gathered.
2. **Subagents (workers)** — each is given an objective, an output format, guidance on tools/sources to use, and clear task boundaries. Each runs its own iterative search loop, evaluating tool results and refining its next query, then returns condensed findings to the lead.
3. **Citation pass** — a dedicated step (CitationAgent) maps the final report's claims back to their sources so all claims are properly attributed.
4. **Synthesis** — the lead composes the final report from the subagents' findings, with citations attached.

Design principles to preserve:
- **Prompt-driven coordination** — division of labor, search strategy, and effort budgets live in the prompts, not in hardcoded rules.
- **Parallelism** — subagents run concurrently, and each subagent may issue multiple tool calls in parallel.
- **Tunable** — the number of subagents and per-agent iteration/effort budgets are configurable parameters.
- **Observable** — log the lead agent's plan, each subagent's searches (queries and sources), and the agents' thought process as the run progresses, so the multi-agent behavior can be followed and debugged.
- **Shared logic** — factor common behavior (e.g., the agent loop, LLM calls, search/tool handling) into shared classes and functions reused across agents, to reduce code size and duplication.

Output is markdown rendered to a PDF file. The report carries inline citations from the citation pass; to keep things simple, it does not append a full list of web search result URLs.

## Tech Stack

- Language: Python
- Web search: Tavily Python SDK (`TavilyClient`)
- LLM: LiteLLM Python SDK (model is a selectable parameter; provider-agnostic). The lead agent and subagents can use different models, but keep model choice model-agnostic via LiteLLM.

## Code Structure

- A single Jupyter notebook file.

## Code Guidelines

- Keep code as simple, concise, and clear as possible; add key comments to explain non-obvious logic.
- Keep text cells in notebooks to explain reasoning between code cells.
- Keep subagent count and iteration/effort budgets configurable so the multi-agent behavior can be tuned.
