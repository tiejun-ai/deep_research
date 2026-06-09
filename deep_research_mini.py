"""Multi-Agent Deep Research (minimal). Deps: litellm tavily-python."""

import os
import json
from concurrent.futures import ThreadPoolExecutor

import litellm
from tavily import TavilyClient

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

LEAD_MODEL = "gpt-4.1-mini"
SUBAGENT_MODEL = "gpt-4o-mini"
MAX_NUM_AGENTS = 3
MAX_AGENT_LOOP_TIMES = 3

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web. Returns titles, URLs, and snippets.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

def web_search(query, agent="system"):
    print(f"[{agent}] search: {query!r}")
    return [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in tavily.search(query=query, max_results=4)["results"]
    ]

def llm(model, messages, tools=None, response_format=None, max_tokens=2000):
    return litellm.completion(
        model=model, messages=messages, tools=tools,
        response_format=response_format, max_tokens=max_tokens, temperature=0.2,
    ).choices[0].message

def _assistant_to_dict(msg):
    d = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return d

class Agent:
    def __init__(self, model, system_prompt, name="agent"):
        self.model = model
        self.system_prompt = system_prompt
        self.name = name
        self.sources = []

    def run(self, task):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        for _ in range(MAX_AGENT_LOOP_TIMES):
            msg = llm(self.model, messages, tools=[WEB_SEARCH_TOOL])
            messages.append(_assistant_to_dict(msg))
            if not msg.tool_calls:
                return msg.content
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                results = web_search(args.get("query", ""), agent=self.name)
                self.sources.extend(r["url"] for r in results)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(results)})
        messages.append({"role": "user", "content": "Stop searching and give your final answer now."})
        return llm(self.model, messages).content

SUBAGENT_SYSTEM = (
    "You are a research subagent. You are given one focused objective.\n"
    "Use the web_search tool to gather evidence, refining queries until you can answer well.\n"
    "Then write condensed findings. For every claim, note the source URL inline, e.g. "
    "(source: https://example.com/page).\n"
    "Output format requested by the lead: {output_format}"
)

PLANNER_SYSTEM = (
    "You are the lead agent of a multi-agent research system.\n"
    "Decompose the query into focused, non-overlapping subtasks for parallel subagents.\n"
    'Respond ONLY with JSON: {"subtasks": [{"objective": "...", "output_format": "..."}]}.'
)

SYNTH_SYSTEM = (
    "You are the lead agent synthesizing a final research report from subagents' findings.\n"
    "Write a well-structured markdown report that answers the query as a coherent narrative.\n"
    "Format the report properly: in a list, all items should be of the same category.\n"
    "Preserve the subagents' inline source attributions. Do NOT add a bibliography or URL list."
)

CITATION_SYSTEM = (
    "You are the citation agent. Ensure every substantive claim carries an inline citation to one "
    "of the available source URLs, formatted as a markdown link like ([example.com](https://example.com/page)).\n"
    "Keep citations inline and minimal. Do NOT append a bibliography. Return the full report markdown."
)

def research(query):
    print(f"[lead] query: {query!r}")
    plan = llm(LEAD_MODEL,
               [{"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": f"Research query: {query}\nReturn at most {MAX_NUM_AGENTS} subtasks."}],
               response_format={"type": "json_object"})
    subtasks = json.loads(plan.content)["subtasks"][:MAX_NUM_AGENTS]
    print(f"[lead] plan: {[s['objective'] for s in subtasks]}")

    def run_subagent(i, st):
        agent = Agent(SUBAGENT_MODEL,
                      SUBAGENT_SYSTEM.format(output_format=st.get("output_format", "concise bullet points")),
                      name=f"subagent-{i}")
        findings = agent.run(st["objective"])
        return {"objective": st["objective"], "findings": findings, "sources": agent.sources}

    with ThreadPoolExecutor(max_workers=MAX_NUM_AGENTS) as ex:
        results = list(ex.map(lambda p: run_subagent(*p), enumerate(subtasks)))

    blocks = "\n\n".join(f"## Subtask: {r['objective']}\n{r['findings']}" for r in results)
    report = llm(LEAD_MODEL,
                 [{"role": "system", "content": SYNTH_SYSTEM},
                  {"role": "user", "content": f"Original query: {query}\n\nSubagent findings:\n{blocks}"}],
                 max_tokens=3000).content

    sources = sorted({u for r in results for u in r["sources"]})
    return llm(LEAD_MODEL,
               [{"role": "system", "content": CITATION_SYSTEM},
                {"role": "user", "content": f"Report:\n{report}\n\nAvailable source URLs:\n" + "\n".join(sources)}],
               max_tokens=3000).content

if __name__ == "__main__":
    print(research("Best practices for prompt engineering?"))
