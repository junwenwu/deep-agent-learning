---
name: migrate-deep-agent-domain
description: "Adapt this tax Deep Agents example to another educational domain - Brought to you by junwenwu/deep-agent-learning"
---

# Migrate Deep Agent Domain

## Overview

Replace the repository's tax briefing domain with a user-specified educational
domain. Preserve the proven Deep Agents runtime and change only domain-owned
contracts unless the user explicitly requests an infrastructure change.

Use this skill when the user asks to adapt, repurpose, or migrate the example to
support, engineering, product research, compliance education, or another
informational domain.

The workflow replaces the domain in the current checkout. To preserve the tax
version or host multiple domains, fork or copy the repository first and invoke
the skill in the destination checkout. Coexistence inside one Python package is
outside this skill's scope.

## Prerequisites

* A working checkout of this repository
* Python 3.11 or later and `uv`
* A target domain and at least one concrete user request
* Approved knowledge sources for deterministic tools
* Safety, privacy, and output constraints for the target domain

Azure credentials are needed only for live model validation. LangSmith
credentials are needed only for hosted trace validation. Never request secrets
through chat or add them to source control.

## Quick Start

Invoke `/migrate-deep-agent-domain` with a request such as:

```text
Migrate this example to IT support. Use one specialist for product troubleshooting
and one for account access policy. Produce a support-summary.md artifact. Ground
answers only in the local catalogs and keep Azure keyless authentication.
```

## Required Inputs

Collect or infer these values before editing:

| Input | Purpose | Example |
| --- | --- | --- |
| Domain | Names the subject area | IT support |
| User jobs | Defines requests the agent must handle | Diagnose a product issue |
| Specialist boundaries | Separates responsibilities | Troubleshooting and access policy |
| Knowledge sources | Grounds each tool | Approved local product catalog |
| Output artifact | Defines the durable deliverable | `support-summary.md` |
| Safety constraints | Limits unsupported or sensitive behavior | No credential collection |
| Domain caveat | Qualifies the final answer | Escalate unresolved access issues |

Ask concise clarifying questions only when a missing value changes architecture,
data access, or safety. Do not block on cosmetic naming choices that can be
derived from the domain.

This skill creates educational and informational agents. For legal, healthcare,
finance, employment, security, or another high-impact subject, limit the example
to synthetic data and non-consequential education. Before implementation,
establish:

* Allowed informational tasks and prohibited actions
* Refusal and human-escalation rules
* Sensitive-data collection, storage, and deletion limits
* Source ownership, provenance, freshness, and citation requirements
* Human review required before consequential use
* Whether prompts and results are approved for external telemetry

Do not present the migrated example as production-ready for regulated or
high-impact decisions. A production system requires a separate domain, security,
privacy, and compliance review.

## Migration Workflow

### Step 1: Establish the baseline

1. Inspect the current worktree and do not overwrite unrelated user changes.
2. Run the existing offline tests and Ruff before editing.
3. Run `uv run --offline --no-sync deep-agent-learning --inspect` to capture the
   current team shape.
4. Confirm `.env`, `artifacts/`, and `.deep-agent/` remain ignored.
5. Inventory the public `__all__`, function signatures, CLI flags and exit codes,
   environment variables, entry point, artifact behavior, and checkpoint behavior.

Stop and report pre-existing failures. Do not disguise them as migration
regressions.

### Step 2: Design responsibilities before names

Map each target user job to a specialist only when it has a distinct
responsibility, source, or action set. Add facts to an existing tool when the
responsibility does not change.

For each specialist, define:

* One stable kebab-case name
* A description that tells the coordinator when to delegate
* A system prompt that requires its approved tools
* A minimal custom tool list
* A result contract the coordinator can synthesize

For a local corpus, preserve the existing evidence contract: stable excerpt and
source identifiers, issuing authority, jurisdiction, source type, publication and
effective dates, URL, section, and exact approved text. Preserve explicit empty
results when no approved evidence matches. Do not fall back to model memory.

Avoid specialists with overlapping descriptions and identical tools. Keep
cross-domain planning and final synthesis in the coordinator.

### Step 3: Replace deterministic domain retrieval

Update `src/deep_agent_learning/research.py` and the packaged knowledge corpus:

1. Replace `knowledge/tax_sources.json` with approved domain evidence records.
2. Rename `search_tax_sources` and `read_tax_source` to clear domain verbs.
3. Preserve typed parameters, descriptive docstrings, input validation, source
   confinement, stable IDs, and deterministic empty results.
4. Preserve relevant metadata filters, such as jurisdiction and effective date,
   or replace them with explicit target-domain filters.
5. Return only approved source content. Do not use the model as an untracked
   fallback when retrieval misses.
6. Export renamed public symbols from `src/deep_agent_learning/__init__.py`.
7. Remove replaced tax-only exports and preserve generic public APIs such as
   `create_agent`, model configuration, CLI flags, and exit codes.

If the target source is an API, database, or retrieval index, keep network and
authentication handling inside the tool boundary. Add timeouts, explicit errors,
and tests with fakes. Never hardcode credentials.

### Step 4: Rewrite the agent contracts

Update `src/deep_agent_learning/agent.py`:

1. Import the new tools.
2. Rename specialist variables and their `name` fields.
3. Rewrite each description around responsibility and routing signals.
4. Rewrite system prompts around approved sources, forbidden invention, and the
   specialist result contract.
5. Assign each specialist only its required tools.
6. Register every specialist in `subagents`.
7. Rewrite the coordinator prompt with single-route, mixed-route, synthesis, and
   domain-caveat rules.

Keep these runtime extension points unchanged by default:

* `resolve_model(model_name)`
* `FilesystemBackend(root_dir=workspace, virtual_mode=True)`
* The optional `BaseCheckpointSaver`
* `create_deep_agent` graph construction

Prompt grounding is guidance, not enforcement. Add deterministic output
validation when the target domain requires strict provenance or regulated
language.

### Step 5: Adapt the user-facing contract

Update `src/deep_agent_learning/cli.py`:

1. Replace `DEFAULT_QUESTION` with a representative domain request.
2. Rename `ARTIFACT_NAME` when the target deliverable differs.
3. Update artifact instructions and inspection labels.
4. Replace tax-specific trace tags with stable domain tags.
5. Keep checkpoint option pairing and trace credential gating intact.
6. Keep `virtual_mode=True` and use the repository's dedicated ignored
   `artifacts/` directory for validation. Do not broaden filesystem access or use
   the repository root, home directory, secret-bearing directories, or symlinks
   that resolve outside the intended artifact root.

Update package metadata or the console-script name only when the user requests a
full product rename. A domain migration does not require changing the Python
package name.

### Step 6: Rewrite tests as behavioral contracts

Update `tests/test_agent.py` before live calls. Cover:

* A successful lookup for every catalog or source
* Case and whitespace normalization where applicable
* Deterministic behavior for unknown values and source errors
* Exact specialist registration and names
* Distinct, least-capability tool assignments
* Coordinator support for mixed requests
* Updated inspection output
* Updated artifact request and host-side verification
* Checkpoint thread propagation
* Trace project, tags, metadata, and flush behavior
* Missing credential failure paths without network calls
* Generic public imports, CLI flags, and exit codes from the baseline inventory

Assert meaningful domain terms and contracts rather than copying whole prose
responses. Fake external systems at their boundary.

### Step 7: Update documentation and configuration

Update `README.md`, `docs/README.md`, and `.env.example` when variable names or
external sources change. Explain:

* The new coordinator and specialist responsibilities
* Trusted knowledge sources and known limits
* Keyless Azure authentication
* Artifact and checkpoint data boundaries
* Tracing data exposure
* One concept-only, one second-specialist, and one mixed example

Keep real `.env` values, generated artifacts, checkpoints, credentials, personal
data, and proprietary source exports out of Git.

### Step 8: Validate from narrow to broad

After the first source edit, run the narrowest relevant test. Then rebuild the
non-editable wheel and run the full checks:

```bash
uv sync --no-editable --reinstall-package deep-agent-learning --offline
uv run --offline --no-sync pytest -q
uv run --offline --no-sync ruff check .
git diff --check
```

Run inspection without credentials:

```bash
uv run --offline --no-sync deep-agent-learning --inspect
```

Verify ignored runtime data:

```bash
git check-ignore -v .env artifacts/briefing.md .deep-agent/checkpoints.sqlite
```

Adapt the artifact path in that command when `ARTIFACT_NAME` changes.

### Step 9: Validate live behavior deliberately

Run live Azure requests only after offline checks pass and the user has approved
the domain data boundary. High-impact educational examples also require approval
of refusal, escalation, provenance, sensitive-data, and human-review behavior.
Test at least:

1. A request routed to the first specialist
2. A request routed to the second specialist
3. A mixed request that needs both
4. An artifact request, when enabled
5. A resumed thread, when checkpointing is in scope

Enable LangSmith only when a local key is already configured and the prompts are
approved for external telemetry. Evaluate routing, tool grounding, completion,
and unnecessary calls. Do not claim hosted validation when no trace was uploaded.

## Invariants

Preserve these properties unless the user explicitly changes them:

* Azure OpenAI uses `DefaultAzureCredential`, not an API key
* Shell environment values override `.env`
* Real `.env` files are ignored
* Specialists use isolated, least-capability tool sets
* Unknown domain values do not trigger invented tool results
* The filesystem uses a dedicated root and `virtual_mode=True`
* Artifacts are verified outside the model
* Checkpoint database and thread ID are supplied together
* Thread IDs are not treated as authorization
* Tracing is opt-in and excludes secrets from tags and metadata
* Offline tests run before paid or externally traced requests

## Troubleshooting

### Source changes do not appear

The project uses a non-editable wheel. Force a local package reinstall:

```bash
uv sync --no-editable --reinstall-package deep-agent-learning --offline
```

### The coordinator chooses the wrong specialist

Make descriptions mutually distinct, align coordinator routing rules with those
descriptions, and confirm tool lists do not overlap accidentally. Add an offline
construction assertion and inspect a live trace with synthetic data.

### The model adds unsupported content

Tighten specialist and coordinator contracts, capture tool results, and add
deterministic output validation. Do not rely on prompt wording alone for strict
provenance.

### The artifact is missing

Confirm the CLI appends the write instruction, the backend receives the intended
workspace, the model writes the configured virtual path, and host-side code
checks the matching file name.

### A resumed thread has no history

Confirm both commands use the same database path and thread ID. Keep the
`SqliteSaver` open throughout graph invocation.

### Tracing fails before invocation

Set `LANGSMITH_API_KEY` only in the local ignored environment and use `--trace`.
Azure CLI credentials cannot authenticate to LangSmith.

> Brought to you by junwenwu/deep-agent-learning
