"""Command-line interface for the Deep Agents example."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
    return parser


def describe_agent(model: str, workspace: Path | None = None) -> str:
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
    return "\n".join(description)


def main() -> int:
    """Run inspection mode or invoke the live agent."""
    args = create_parser().parse_args()
    if args.inspect:
        print(describe_agent(args.model, args.workspace))
        return EXIT_SUCCESS

    if not os.environ.get("OPENAI_API_KEY") and args.model.startswith("openai:"):
        print(
            "OPENAI_API_KEY is not set. Set it in your shell or run with --inspect first.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    try:
        agent = create_agent(args.model, workspace=args.workspace)
    except ValueError as error:
        print(error, file=sys.stderr)
        return EXIT_ERROR
    question = args.question
    if args.workspace is not None:
        question += (
            f"\n\nAfter synthesizing the answer, use write_file to save the same briefing "
            f"as Markdown at /{ARTIFACT_NAME}."
        )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(result["messages"][-1].content)
    if args.workspace is not None:
        artifact_path = args.workspace / ARTIFACT_NAME
        if not artifact_path.is_file():
            print(f"Expected artifact was not created: {artifact_path}", file=sys.stderr)
            return EXIT_FAILURE
        print(f"Artifact: {artifact_path.resolve()}")
    return EXIT_SUCCESS