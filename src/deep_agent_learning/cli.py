"""Command-line interface for the Deep Agents example."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import Client, tracing_context

from deep_agent_learning.agent import create_agent
from deep_agent_learning.models import (
    AZURE_DEPLOYMENT_MODEL,
    AZURE_DEPLOYMENT_MODEL_VERSION,
    AZURE_MODEL,
    DEFAULT_MODEL,
    configure_azure_environment,
)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2
ARTIFACT_NAME = "briefing.md"
DEFAULT_TRACE_PROJECT = "deep-agent-learning"
DEFAULT_QUESTION = (
    "Compare sales tax and income tax. Explain who pays each and when it is collected."
)


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description="Run the Deep Agents tax briefing example.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--model",
        default=os.environ.get("DEEP_AGENT_MODEL", DEFAULT_MODEL),
        help="LangChain model identifier (default: %(default)s)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Show the agent structure without calling a model.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Persist a Markdown briefing in this local artifact directory.",
    )
    parser.add_argument(
        "--checkpoint-db",
        type=Path,
        help="Persist conversation checkpoints in this SQLite database.",
    )
    parser.add_argument(
        "--thread-id",
        help="Resume the conversation associated with this checkpoint thread.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Upload the run tree to LangSmith using LANGSMITH_API_KEY.",
    )
    parser.add_argument(
        "--trace-project",
        default=os.environ.get("LANGSMITH_PROJECT", DEFAULT_TRACE_PROJECT),
        help="LangSmith project for traced runs (default: %(default)s).",
    )
    return parser


def describe_agent(
    model: str,
    workspace: Path | None = None,
    checkpoint_db: Path | None = None,
    thread_id: str | None = None,
    trace: bool = False,
    trace_project: str = DEFAULT_TRACE_PROJECT,
) -> str:
    """Return a no-credentials view of the example's control flow."""
    description = [f"Model: {model}"]
    if model == AZURE_MODEL:
        configure_azure_environment()
        deployment = os.environ.get(
            "AZURE_OPENAI_CHAT_DEPLOYMENT", "<not configured>"
        )
        description.extend(
            [
                f"Azure deployment: {deployment}",
                f"Underlying model: {AZURE_DEPLOYMENT_MODEL} ({AZURE_DEPLOYMENT_MODEL_VERSION})",
            ]
        )
    description.extend(
        [
            "Coordinator: plans and synthesizes the briefing",
            "  -> task(subagent_type='tax-researcher')",
            "Tax researcher: researches concepts in an isolated context",
            "  -> lookup_tax_topic(topic)",
            "Coordinator: routes jurisdiction research",
            "  -> task(subagent_type='jurisdiction-researcher')",
            "Jurisdiction researcher: researches where rules apply",
            "  -> lookup_tax_jurisdiction(jurisdiction)",
            "Result: returns to the coordinator for the final answer",
        ]
    )
    if workspace is not None:
        description.extend(
            [
                f"Artifact workspace: {workspace.resolve()}",
                f"  -> write_file('/{ARTIFACT_NAME}')",
            ]
        )
    if checkpoint_db is not None and thread_id is not None:
        description.extend(
            [
                f"Checkpoint database: {checkpoint_db.resolve()}",
                f"Thread ID: {thread_id}",
            ]
        )
    if trace:
        description.extend(
            [
                "LangSmith tracing: enabled",
                f"Trace project: {trace_project}",
            ]
        )
    return "\n".join(description)


def invoke_agent(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    """Build and invoke the agent with optional checkpointing."""
    if args.checkpoint_db is not None:
        args.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(args.checkpoint_db)) as checkpointer:
            agent = create_agent(
                args.model,
                workspace=args.workspace,
                checkpointer=checkpointer,
            )
            return agent.invoke(
                payload,
                config={"configurable": {"thread_id": args.thread_id}},
            )

    agent = create_agent(args.model, workspace=args.workspace)
    return agent.invoke(payload)


def main() -> int:
    """Run inspection mode or invoke the live agent."""
    args = create_parser().parse_args()
    if (args.checkpoint_db is None) != (args.thread_id is None):
        print(
            "--checkpoint-db and --thread-id must be provided together.",
            file=sys.stderr,
        )
        return EXIT_ERROR
    configure_azure_environment()
    if args.inspect:
        print(
            describe_agent(
                args.model,
                args.workspace,
                args.checkpoint_db,
                args.thread_id,
                args.trace,
                args.trace_project,
            )
        )
        return EXIT_SUCCESS
    if args.trace and not os.environ.get("LANGSMITH_API_KEY"):
        print(
            "LANGSMITH_API_KEY is required when --trace is enabled.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    if not os.environ.get("OPENAI_API_KEY") and args.model.startswith("openai:"):
        print(
            "OPENAI_API_KEY is not set. Set it in your shell or run with --inspect first.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    question = args.question
    if args.workspace is not None:
        question += (
            f"\n\nAfter synthesizing the answer, use write_file to save the same briefing "
            f"as Markdown at /{ARTIFACT_NAME}."
        )
    payload = {"messages": [{"role": "user", "content": question}]}
    trace_client = Client() if args.trace else None
    trace_scope = (
        tracing_context(
            enabled=True,
            project_name=args.trace_project,
            tags=["deep-agent-learning", "tax-briefing"],
            metadata={
                "model": args.model,
                "thread_id": args.thread_id or "not-configured",
                "artifact_enabled": args.workspace is not None,
            },
            client=trace_client,
        )
        if args.trace
        else nullcontext()
    )
    try:
        with trace_scope:
            result = invoke_agent(args, payload)
    except ValueError as error:
        print(error, file=sys.stderr)
        return EXIT_ERROR
    finally:
        if trace_client is not None:
            trace_client.flush()
    print(result["messages"][-1].content)
    if args.workspace is not None:
        artifact_path = args.workspace / ARTIFACT_NAME
        if not artifact_path.is_file():
            print(f"Expected artifact was not created: {artifact_path}", file=sys.stderr)
            return EXIT_FAILURE
        print(f"Artifact: {artifact_path.resolve()}")
    if args.trace:
        print(f"LangSmith project: {args.trace_project}")
    return EXIT_SUCCESS
