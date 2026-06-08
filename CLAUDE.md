# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A deep research demonstration. Given a user query, the system produces a long-form research report: it plans the research, runs many web searches across sub-questions, reads and synthesizes the findings, and writes a structured report with inline citations to the sources used.

This is the deeper-research successor to the single-shot `ai_websearch` project: instead of one answer from a few searches, it decomposes the query, gathers evidence over multiple rounds, and composes a multi-section report.

## Architecture (Plan, Agent Loop, Tool Use, Report Synthesis)

The pipeline runs in stages:

1. **Plan** — prompt the LLM to decompose the query into a set of focused sub-questions / search topics.
2. **Research (agent loop with tool use)** — for each sub-question, iteratively prompt the LLM with a web-search tool. The LLM issues searches, the system executes them and feeds results back into context, repeating until the LLM has enough to answer. A configurable parameter caps the number of LLM calls / iterations per sub-question.
3. **Synthesize** — collect findings across all sub-questions and prompt the LLM to write the final report, with inline citations referencing the collected sources.

Output is markdown (converted to HTML for Jupyter display), containing the report and a list of cited sources with URLs.

## Tech Stack

- Language: Python
- Web search: Tavily Python SDK (`TavilyClient`)
- LLM: LiteLLM Python SDK (model is a selectable parameter, default to a cheap OpenAI model)

## Code Structure

- A single Jupyter notebook file.

## Code Guidelines

- Simple and concise; add key comments to explain non-obvious logic.
- Keep text cells in notebooks to explain reasoning between code cells.
- Keep iteration/LLM-call counts configurable so deep research depth can be tuned.
