# deep_research

  * Multi-Agent Deep Research: An open-source implementation of the core of Anthropic's Deep Research, as described in Anthropic's Engineering Blog: [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
    1. **Lead agent (orchestrator)** decomposes the query into focused, non-overlapping subtasks
    2. **Subagents (workers)** research those subtasks **in parallel**, each running its own web-search loop (Tavily as a **tool** for the LLM)
    3. **Citation pass** attributes the report's claims back to their sources with inline citations
    4. **Synthesis**: the lead composes a final markdown report from the subagents' findings
    5. Prompt-driven coordination, with tunable knobs — `MAX_NUM_AGENTS` (default 3) and `MAX_AGENT_LOOP_TIMES` (default 3)

## Files

| File | Description |
|------|-------------|
| [deep_research.ipynb](deep_research.ipynb) | Jupyter notebook — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiejun-ai/deep_research/blob/main/deep_research.ipynb) |
| [deep_research.demo1.ipynb](deep_research.demo1.ipynb) | Demo run with saved output — English query ("Paris travel planning: suggest number of days") — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiejun-ai/deep_research/blob/main/deep_research.demo1.ipynb) |
| [deep_research.demo2.ipynb](deep_research.demo2.ipynb) | Demo run with saved output — Chinese query ("巴黎旅行攻略 几天合适") — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiejun-ai/deep_research/blob/main/deep_research.demo2.ipynb) |
| [deep_research.py](deep_research.py) | Single Python file version |
| [deep_research_mini.py](deep_research_mini.py) | Minimal single Python file version |
| [CLAUDE.md](CLAUDE.md) | Claude Code instructions — project context, tech stack, architecture, and code guidelines |
