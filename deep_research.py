"""Multi-Agent Deep Research.

An open-source implementation of the orchestrator-worker pattern from Anthropic's
"How we built our multi-agent research system". Given a research query, a lead agent
decomposes it into focused subtasks, subagents research them in parallel (each running its
own web-search loop), a citation pass attributes claims to sources, and the lead synthesizes
a final markdown report (printed to stdout).

Dependencies: litellm tavily-python
Run: OPENAI_API_KEY=... TAVILY_API_KEY=... python deep_research.py
"""

import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import litellm
from tavily import TavilyClient

# --- API keys: set these in your environment before running ---
# os.environ["OPENAI_API_KEY"] = "sk-..."
# os.environ["TAVILY_API_KEY"] = "tvly-..."
assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY (used by LiteLLM)"
assert os.environ.get("TAVILY_API_KEY"), "Set TAVILY_API_KEY"

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


class Config:
    # Models are selected via LiteLLM strings, so the system is provider-agnostic.
    # Defaults are cheap OpenAI models; the lead is slightly stronger than the workers.
    LEAD_MODEL = "gpt-4.1-mini"
    SUBAGENT_MODEL = "gpt-4o-mini"

    # Tunable multi-agent budgets (the knobs from CLAUDE.md).
    MAX_NUM_AGENTS = 3        # max subagents the lead spawns in parallel
    MAX_AGENT_LOOP_TIMES = 3  # max search-loop iterations per agent

    MAX_TOKENS = 2000         # default per-call output budget
    TAVILY_RESULTS = 4        # results per web search


# --- Observability & shared utilities -------------------------------------------------
# Every agent reuses the same three helpers, so common logic lives in one place.

_log_lock = threading.Lock()  # subagents log from parallel threads


def log(event, agent="system", **fields):
    # One structured logger used everywhere so the multi-agent run is followable.
    detail = "  ".join(f"{k}={v}" for k, v in fields.items())
    with _log_lock:
        print(f"[{agent:>11}] {event:<16} {detail}")


def web_search(query, agent="system"):
    # Tavily web search, exposed to agents as the `web_search` tool.
    log("search", agent=agent, query=repr(query))
    results = [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in tavily.search(query=query, max_results=Config.TAVILY_RESULTS)["results"]
    ]
    log("search_results", agent=agent, urls=[r["url"] for r in results])
    return results


def llm(model, messages, tools=None, response_format=None, max_tokens=None, agent="system"):
    # Thin LiteLLM wrapper. Returns the assistant message object.
    resp = litellm.completion(
        model=model,
        messages=messages,
        tools=tools,
        response_format=response_format,
        max_tokens=max_tokens or Config.MAX_TOKENS,
        temperature=0.2,
    )
    return resp.choices[0].message


# Tool schema advertised to agents (OpenAI/LiteLLM function-calling format).
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information. Returns titles, URLs, and snippets.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
}


# --- Shared agent loop ----------------------------------------------------------------
# `Agent` holds the one iterative tool-use loop that both the lead and the subagents reuse.

def _assistant_to_dict(msg):
    # Normalize a LiteLLM assistant message back into a plain dict for the next turn.
    d = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


class Agent:
    def __init__(self, model, system_prompt, name="agent"):
        self.model = model
        self.system_prompt = system_prompt
        self.name = name
        self.sources = []  # URLs gathered during this run

    def run(self, task):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        for _ in range(Config.MAX_AGENT_LOOP_TIMES):
            msg = llm(self.model, messages, tools=[WEB_SEARCH_TOOL], agent=self.name)
            messages.append(_assistant_to_dict(msg))

            if not msg.tool_calls:
                return msg.content  # agent decided it has enough -> done

            if msg.content:
                log("thought", agent=self.name, text=repr(msg.content[:160]))

            # Execute every requested search and feed results back in.
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                results = web_search(args.get("query", ""), agent=self.name)
                self.sources.extend(r["url"] for r in results)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(results),
                })

        # Budget exhausted: ask for a final answer without further tool use.
        messages.append({"role": "user", "content": "Stop searching and give your final answer now."})
        return llm(self.model, messages, agent=self.name).content


# --- Subagent (worker) ----------------------------------------------------------------
# Each subagent is given an objective and an output format, runs the shared loop, and
# returns condensed findings together with the source URLs it actually used.

SUBAGENT_SYSTEM = (
    "You are a research subagent in a multi-agent system. You are given one focused objective.\n"
    "Use the web_search tool to gather evidence: issue searches, refine your queries based on "
    "what you find, and stop once you have enough to answer well.\n"
    "Then write condensed findings (not raw search dumps). For every claim, note the source URL "
    "inline, e.g. (source: https://example.com/page).\n"
    "Output format requested by the lead: {output_format}"
)


class Subagent(Agent):
    def __init__(self, objective, output_format, idx):
        super().__init__(
            model=Config.SUBAGENT_MODEL,
            system_prompt=SUBAGENT_SYSTEM.format(output_format=output_format),
            name=f"subagent-{idx}",
        )
        self.objective = objective

    def research(self):
        log("start", agent=self.name, objective=repr(self.objective))
        findings = self.run(self.objective)
        log("done", agent=self.name)
        return {
            "objective": self.objective,
            "findings": findings,
            "sources": list(dict.fromkeys(self.sources)),  # dedupe, keep order
        }


# --- Lead agent, citation pass, and synthesis -----------------------------------------
# The lead plans the work, fans out to subagents in parallel, synthesizes their findings
# into one report, then runs a citation pass. Citations are inline only.

PLANNER_SYSTEM = (
    "You are the lead agent (orchestrator) of a multi-agent research system.\n"
    "Decompose the user's research query into focused, non-overlapping subtasks, each suitable "
    "for one subagent researching in parallel.\n"
    "Respond ONLY with JSON of the form: "
    '{"subtasks": [{"objective": "...", "output_format": "..."}]}.\n'
    'Each "objective" is a clear research task; each "output_format" tells the subagent how to '
    "structure its findings."
)

SYNTH_SYSTEM = (
    "You are the lead agent synthesizing a final research report from your subagents' findings.\n"
    "Write a well-structured markdown report that directly answers the original query. Integrate "
    "the findings into a coherent narrative rather than concatenating them.\n"
    "Format the report properly: in a list, all items should be of the same category.\n"
    "Preserve the inline source attributions the subagents placed next to their claims. Do NOT add "
    "a separate bibliography or a full list of source URLs at the end."
)

CITATION_SYSTEM = (
    "You are the citation agent. Ensure every substantive claim in the report carries an inline "
    "citation to one of the available source URLs, formatted as a markdown link like "
    "([example.com](https://example.com/page)).\n"
    "Keep citations inline and minimal. Do NOT append a bibliography or a list of all URLs at the "
    "end. Return the full report markdown with citations applied."
)


class CitationAgent:
    def add_citations(self, report, sources):
        sources = sorted(set(sources))
        log("citation_pass", agent="citation", num_sources=len(sources))
        msg = llm(
            Config.LEAD_MODEL,
            [
                {"role": "system", "content": CITATION_SYSTEM},
                {"role": "user", "content": f"Report:\n{report}\n\nAvailable source URLs:\n" + "\n".join(sources)},
            ],
            max_tokens=3000,
            agent="citation",
        )
        return msg.content


class LeadAgent:
    def __init__(self):
        self.model = Config.LEAD_MODEL

    def plan(self, query):
        msg = llm(
            self.model,
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": f"Research query: {query}\nReturn at most {Config.MAX_NUM_AGENTS} subtasks."},
            ],
            response_format={"type": "json_object"},
            agent="lead",
        )
        subtasks = json.loads(msg.content)["subtasks"][: Config.MAX_NUM_AGENTS]
        log("plan", agent="lead", subtasks=[s["objective"] for s in subtasks])
        return subtasks

    def synthesize(self, query, results):
        blocks = "\n\n".join(
            f"## Subtask: {r['objective']}\n{r['findings']}" for r in results
        )
        msg = llm(
            self.model,
            [
                {"role": "system", "content": SYNTH_SYSTEM},
                {"role": "user", "content": f"Original query: {query}\n\nSubagent findings:\n{blocks}"},
            ],
            max_tokens=3000,
            agent="lead",
        )
        log("synthesized", agent="lead")
        return msg.content

    def research(self, query):
        log("query", agent="lead", text=repr(query))
        subtasks = self.plan(query)
        subagents = [
            Subagent(st["objective"], st.get("output_format", "concise bullet points"), i)
            for i, st in enumerate(subtasks)
        ]

        # Fan out: subagents research concurrently (litellm.completion is sync -> threads).
        with ThreadPoolExecutor(max_workers=Config.MAX_NUM_AGENTS) as ex:
            results = list(ex.map(lambda a: a.research(), subagents))

        report = self.synthesize(query, results)
        all_sources = [u for r in results for u in r["sources"]]
        return CitationAgent().add_citations(report, all_sources)


if __name__ == "__main__":
    query = "Best practices for prompt engineering?"

    report = LeadAgent().research(query)

    print("\n===== FINAL REPORT (markdown) =====\n")
    print(report)
