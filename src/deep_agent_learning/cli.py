"""Command-line interface for the Deep Agents example."""

from __future__ import annotations

import argparse
import os
import sys

from deep_agent_learning.agent import create_agent
from deep_agent_learning.models import (
    AZURE_DEPLOYMENT_MODEL,
    AZURE_DEPLOYMENT_MODEL_VERSION,
    AZURE_MODEL,
    DEFAULT_MODEL,
    configure_azure_environment,
)

EXIT_SUCCESS = 0
EXIT_ERROR = 2
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
    return parser


def describe_agent(model: str) -> str:
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
    return "\n".join(description)


def main() -> int:
    """Run inspection mode or invoke the live agent."""
    args = create_parser().parse_args()
    if args.inspect:
        print(describe_agent(args.model))
        return EXIT_SUCCESS

    if not os.environ.get("OPENAI_API_KEY") and args.model.startswith("openai:"):
        print(
            "OPENAI_API_KEY is not set. Set it in your shell or run with --inspect first.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    try:
        agent = create_agent(args.model)
    except ValueError as error:
        print(error, file=sys.stderr)
        return EXIT_ERROR
    result = agent.invoke({"messages": [{"role": "user", "content": args.question}]})
    print(result["messages"][-1].content)
    return EXIT_SUCCESS