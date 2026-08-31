---
title: Learn Deep Agents with a Tax Briefing Agent
description: A small LangChain Deep Agents example with a custom tool and specialist subagent
ms.date: 2026-08-31
ms.topic: tutorial
---

## What you will build

This project builds an educational tax briefing agent with
[LangChain Deep Agents](https://github.com/langchain-ai/deepagents). It is small
enough to trace from the user request to the final answer.

The example has four moving parts:

1. A coordinator receives the question and plans the work.
2. The built-in `task` tool delegates research to `tax-researcher`.
3. The subagent calls `lookup_tax_topic` in its isolated context.
4. The result returns to the coordinator, which writes the final briefing.

Deep Agents also installs filesystem, context-management, and general-purpose
subagent capabilities by default. This lesson focuses on custom tools and explicit
delegation before introducing those features.

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
  -> lookup_tax_topic(topic)
Coordinator
  -> final answer
```

Open [src/deep_agent_learning/agent.py](src/deep_agent_learning/agent.py) and find
`create_agent`. Its call to `create_deep_agent` constructs and returns a compiled
LangGraph graph rather than invoking the model immediately.

## Step 2: Understand the tool

`lookup_tax_topic` is a normal typed Python function with a descriptive docstring.
Deep Agents exposes it to the model as a tool. The model uses the function name,
signature, and docstring to decide when and how to call it.

Try the tool without an LLM:

```bash
uv run --offline --no-sync python -c "from deep_agent_learning import lookup_tax_topic; print(lookup_tax_topic('sales tax'))"
```

Keeping the first tool deterministic makes the agent trace easier to understand
and the behavior easy to test.

## Step 3: Understand the subagent

The `tax_researcher` dictionary defines a specialist with four important fields:

* `name` becomes the value used by the coordinator's `task` call
* `description` tells the coordinator when to delegate
* `system_prompt` defines the specialist's behavior
* `tools` limits the specialist to the local catalog tool, in addition to the
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
compares sales tax and income tax. You can pass another question as a positional
argument:

```bash
uv run --offline --no-sync deep-agent-learning \
  "Explain income tax and identify when it is collected"
```

To use the public OpenAI service instead, set `OPENAI_API_KEY` and pass a
`provider:model` identifier with `--model`.

## Step 6: Observe the reasoning loop

Read the returned message history or connect LangSmith tracing to see this sequence:

1. The coordinator decides to delegate.
2. `task` starts `tax-researcher` in an isolated context.
3. The specialist calls `lookup_tax_topic` for each requested concept.
4. The specialist summarizes its findings.
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

## Next experiments

After the first live run, make one change at a time:

1. [Add `property tax` to `TAX_CATALOG` and test the extension](docs/series/02-extend-the-tax-tool/README.md).
2. Add a second specialist for jurisdiction research.
3. Add a filesystem backend and ask the agent to write a briefing file.
4. Add a checkpointer so a conversation can resume across turns.
5. Enable LangSmith tracing and inspect the parent and subagent runs.
