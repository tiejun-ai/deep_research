# deep_research

  * Multi-Agent Deep Research: the orchestrator-worker pattern from Anthropic's [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
    1. **Lead agent (orchestrator)** decomposes the query into focused, non-overlapping subtasks
    2. **Subagents (workers)** research those subtasks **in parallel**, each running its own web-search loop (Tavily as a **tool** for the LLM)
    3. **Citation pass** attributes the report's claims back to their sources with inline citations
    4. **Synthesis**: the lead composes a final markdown report from the subagents' findings
    5. Prompt-driven coordination, with tunable knobs — `MAX_NUM_AGENTS` (default 3) and `MAX_AGENT_LOOP_TIMES` (default 3)

## Files

| File | Description |
|------|-------------|
| [deep_research.ipynb](deep_research.ipynb) | Jupyter notebook — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiejun-ai/deep_research/blob/main/deep_research.ipynb) |
| [deep_research.py](deep_research.py) | Python script version |
| [deep_research_mini.py](deep_research_mini.py) | Minimal single Python file version |
| [CLAUDE.md](CLAUDE.md) | Claude Code instructions — project context, tech stack, architecture, and code guidelines |
