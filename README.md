---
title: Learn Deep Agents with a Tax Briefing Agent
description: A citation-aware LangChain Deep Agents example with authoritative local sources
ms.date: 2026-08-31
ms.topic: tutorial
---

## What you will build

This project builds an educational tax briefing agent with
[LangChain Deep Agents](https://github.com/langchain-ai/deepagents). It is small
enough to trace from the user request to the final answer.

The example has six moving parts:

1. A coordinator receives the question and plans the work.
2. The built-in `task` tool delegates substantive research to `tax-researcher`.
3. The specialist searches a versioned local corpus of official tax guidance.
4. A `jurisdiction-researcher` compares date-filtered evidence across jurisdictions.
5. The coordinator writes a citation-backed briefing and identifies evidence gaps.
6. An optional filesystem backend persists the briefing in an isolated artifact directory.

Deep Agents installs filesystem, context-management, and general-purpose subagent
capabilities by default. The `--workspace` option connects those filesystem tools
to a confined host directory so generated briefings survive after the run.

## Prerequisites

Install these tools before starting:

* Python 3.11 or later
* [uv](https://docs.astral.sh/uv/getting-started/installation/)
* [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) for the
  live Azure OpenAI example

The committed lock file uses the Tsinghua PyPI mirror because the standard PyPI
file host was unavailable in the environment where this example was built.

## Step 1: Inspect the graph

Inspection mode needs no API key and makes no model call:

```bash
uv sync --no-editable
uv run --offline --no-sync python -m deep_agent_learning --inspect
```

Expected flow:

```text
Coordinator
  -> task(subagent_type='tax-researcher')
Tax researcher
  -> search_tax_sources(query, jurisdictions, effective_on, limit)
  -> read_tax_source(excerpt_id)
Coordinator
  -> task(subagent_type='jurisdiction-researcher')
Jurisdiction researcher
  -> search_tax_sources(query, jurisdictions, effective_on, limit)
  -> read_tax_source(excerpt_id)
Coordinator
  -> final answer
```

Open [src/deep_agent_learning/agent.py](src/deep_agent_learning/agent.py) and find
`create_agent`. Its call to `create_deep_agent` constructs and returns a compiled
LangGraph graph rather than invoking the model immediately.

## Step 2: Understand the retrieval tools

`search_tax_sources` and `read_tax_source` are ordinary typed Python functions.
Deep Agents exposes them to the model as tools. Search applies lexical ranking,
jurisdiction filtering, and inclusive effective-date filtering over the packaged
[`tax_sources.json`](src/deep_agent_learning/knowledge/tax_sources.json) corpus.
Read resolves a stable excerpt ID to the exact text and citation metadata.

Try retrieval without an LLM:

```bash
uv run --offline --no-sync python -c \
  "from deep_agent_learning import search_tax_sources; print(search_tax_sources('pass-through entity election', ['New York', 'California'], '2026-08-31'))"
```

Each result includes its issuing authority, jurisdiction, source type, publication
date, effective range, URL, section, excerpt text, and stable identifiers. The
corpus currently contains selected official New York and California PTET guidance
for learning purposes. It is bounded and reproducible, not comprehensive.

## Step 3: Understand the subagent

The `tax_researcher` dictionary defines a specialist with four important fields:

* `name` becomes the value used by the coordinator's `task` call
* `description` tells the coordinator when to delegate
* `system_prompt` defines the specialist's behavior
* `tools` limits the specialist to the bounded retrieval tools, in addition to the
  Deep Agents middleware tools it receives

The subagent gets an isolated context window. Its detailed tool interactions do
not crowd the coordinator's conversation; only its result comes back through the
`task` tool.

## Step 4: Understand the coordinator

The outer `create_deep_agent` call supplies the model, coordinator prompt, and
subagent list. The coordinator can call the built-in `task` tool because a
subagent is registered.

The live request uses the standard LangGraph input shape:

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]}
)
```

The graph loops through model and tool calls until the coordinator produces a
final response. The complete state comes back from `invoke`, so the example prints
the content of the final message.

## Step 5: Run the live agent

Sign in with the Azure CLI. `DefaultAzureCredential` uses this session locally:

```bash
az login
az account set --subscription "<subscription-name-or-id>"
```

Your identity needs the `Cognitive Services OpenAI User` role on the Azure OpenAI
resource. Create your local configuration from the tracked example:

```bash
cp .env.example .env
```

Open `.env` and replace `<resource-name>` and `<deployment-name>` with the actual
values from your Azure OpenAI resource:

```dotenv
AZURE_OPENAI_ENDPOINT="https://<resource-name>.openai.azure.com/"
AZURE_OPENAI_CHAT_DEPLOYMENT="<deployment-name>"
AZURE_OPENAI_API_VERSION="2024-10-21"
```

These settings identify the Azure resource and model deployment; they are not
credentials. Authentication still comes from `az login`. The application loads
`.env` automatically, while values exported in your shell take precedence. The
real `.env` remains local and must not be committed.

Install dependencies and run the agent:

```bash
uv sync --no-editable
uv run --offline --no-sync deep-agent-learning
```

No `AZURE_OPENAI_API_KEY` is needed. In Azure-hosted environments, the same code
can use a managed identity through `DefaultAzureCredential`. The default question
compares New York and California pass-through entity tax guidance as of August 31,
2026. You can pass another question as a positional argument:

```bash
uv run --offline --no-sync deep-agent-learning \
  "Compare PTET eligibility in New York and California as of 2026-08-31"
```

To use the public OpenAI service instead, set `OPENAI_API_KEY` and pass a
`provider:model` identifier with `--model`.

To persist the final answer as `artifacts/briefing.md`, provide an artifact
workspace:

```bash
uv run --offline --no-sync deep-agent-learning \
  --workspace artifacts \
  "Create a cited PTET comparison for New York and California as of 2026-08-31."
```

The agent sees the virtual path `/briefing.md`, while `FilesystemBackend` maps it
under the selected host directory. The generated `artifacts/` directory is
ignored by Git.

To resume a conversation across separate commands, provide a SQLite checkpoint
database and reuse the same thread ID:

```bash
uv run --offline --no-sync deep-agent-learning \
  --checkpoint-db .deep-agent/checkpoints.sqlite \
  --thread-id tax-session \
  "Research California's PTE election as of 2026-08-31."

uv run --offline --no-sync deep-agent-learning \
  --checkpoint-db .deep-agent/checkpoints.sqlite \
  --thread-id tax-session \
  "Which source and effective date did we use?"
```

Checkpoint databases contain conversation and graph state. The local
`.deep-agent/` directory is ignored by Git and should not contain credentials.

To upload a run tree to LangSmith, add `LANGSMITH_API_KEY` to your local `.env`
and enable tracing explicitly:

```bash
uv run --offline --no-sync deep-agent-learning \
  --trace \
  --trace-project deep-agent-learning \
  "Compare New York and California PTET guidance as of 2026-08-31."
```

Tracing can include prompts, responses, and tool results. Use synthetic or
approved data and apply your organization's retention and access policies.

## Step 6: Observe the reasoning loop

Read the returned message history or connect LangSmith tracing to see this sequence:

1. The coordinator decides to delegate.
2. `task` starts `tax-researcher` in an isolated context.
3. The specialist searches by issue, jurisdiction, and effective date.
4. The specialist reads cited excerpts and returns supported claims.
5. The coordinator synthesizes the final response.

The important distinction is that a deep agent is not a single longer prompt. It
is an agent harness for long-running work, with planning, tools, context management,
filesystem access, and subagent delegation assembled around a LangGraph loop.

## Verify the project

Run the offline checks before making paid model calls:

```bash
uv run --offline --no-sync pytest
uv run --offline --no-sync ruff check .
```

## Complete tutorial

Continue with the consolidated blog-style guide,
[Deep Agents from Scratch](docs/README.md). It builds the example one boundary
at a time: bounded retrieval, provenance-aware specialist routing, filesystem
artifacts, SQLite checkpoints, LangSmith tracing, evaluation, and migration to
another domain.
