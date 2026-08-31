"""Coordinator and subagent construction."""

from __future__ import annotations

from typing import Any

from deep_agent_learning.models import resolve_model
from deep_agent_learning.tools import lookup_tax_jurisdiction, lookup_tax_topic


def create_agent(model_name: str) -> Any:
    """Build the coordinator and its specialist research team.

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
            "Return only facts present in the tool output. Do not add facts, calculations, "
            "examples, or advice from your own knowledge."
        ),
        "tools": [lookup_tax_topic],
        "model": model,
    }
    jurisdiction_researcher = {
        "name": "jurisdiction-researcher",
        "description": (
            "Explain federal, state, and local tax jurisdiction levels. Delegate here when a "
            "request asks where tax rules apply or how jurisdiction levels differ."
        ),
        "system_prompt": (
            "You are an educational tax jurisdiction researcher. Use lookup_tax_jurisdiction "
            "for every requested jurisdiction level. Return only facts present in the tool "
            "output. Do not add facts, examples, or advice from your own knowledge."
        ),
        "tools": [lookup_tax_jurisdiction],
        "model": model,
    }

    return create_deep_agent(
        model=model,
        system_prompt=(
            "You coordinate educational tax briefings. Plan the request and delegate tax-concept "
            "questions to tax-researcher. Delegate questions about federal, state, or local "
            "jurisdiction levels to jurisdiction-researcher. Use both specialists when a request "
            "needs both kinds of research, then synthesize a concise answer using only facts "
            "returned by the specialists. Do not add rates, formulas, examples, procedures, or "
            "jurisdiction-specific claims from your own knowledge. State that actual rules vary "
            "by jurisdiction."
        ),
        subagents=[tax_researcher, jurisdiction_researcher],
    )