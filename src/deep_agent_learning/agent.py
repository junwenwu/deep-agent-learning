"""Coordinator and subagent construction."""

from __future__ import annotations

from typing import Any

from deep_agent_learning.models import resolve_model
from deep_agent_learning.tools import lookup_tax_topic


def create_agent(model_name: str) -> Any:
    """Build the coordinator and its tax research subagent.

    Args:
        model_name: ``azure-openai`` or a LangChain ``provider:model`` identifier.

    Returns:
        A compiled LangGraph agent.
    """
    from deepagents import create_deep_agent

    model = resolve_model(model_name)
    tax_researcher = {
        "name": "tax-researcher",
        "description": (
            "Look up and compare tax concepts using the local catalog. "
            "Delegate here when a request mentions one or more tax topics."
        ),
        "system_prompt": (
            "You are an educational tax researcher. Use lookup_tax_topic for every topic. "
            "Report the retrieved facts, distinguish general concepts from jurisdiction-specific "
            "rules, and do not give personal tax advice."
        ),
        "tools": [lookup_tax_topic],
        "model": model,
    }

    return create_deep_agent(
        model=model,
        system_prompt=(
            "You coordinate educational tax briefings. Plan the request, delegate tax research "
            "to tax-researcher with the task tool, then synthesize a concise answer. State that "
            "actual rules vary by jurisdiction."
        ),
        subagents=[tax_researcher],
    )