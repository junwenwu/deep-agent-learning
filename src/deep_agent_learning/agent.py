"""Coordinator and subagent construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from deep_agent_learning.models import resolve_model
from deep_agent_learning.research import read_tax_source, search_tax_sources


def create_agent(
    model_name: str,
    workspace: Path | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Build the coordinator and its specialist research team.

    Args:
        model_name: ``azure-openai`` or a LangChain ``provider:model`` identifier.
        workspace: Optional local directory exposed as the agent's virtual filesystem root.
        checkpointer: Optional LangGraph checkpoint saver for thread state.

    Returns:
        A compiled LangGraph agent.
    """
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    model = resolve_model(model_name)
    backend = (
        FilesystemBackend(root_dir=workspace, virtual_mode=True)
        if workspace is not None
        else None
    )
    tax_researcher = {
        "name": "tax-researcher",
        "description": (
            "Find authoritative tax guidance and return evidence with citations. "
            "Delegate here for the substantive tax issue in a research request."
        ),
        "system_prompt": (
            "You are an educational tax researcher. Use search_tax_sources with the requested "
            "effective date and jurisdictions, then use read_tax_source for the excerpts you "
            "rely on. Return a list of claims, and attach the excerpt_id, URL, section, and a "
            "supporting quote to every claim. If the corpus has insufficient evidence, say so. "
            "Never rely on your own tax knowledge."
        ),
        "tools": [search_tax_sources, read_tax_source],
        "model": model,
    }
    jurisdiction_researcher = {
        "name": "jurisdiction-researcher",
        "description": (
            "Compare authoritative guidance across requested jurisdictions and effective dates. "
            "Delegate here when a request asks how jurisdiction-specific rules differ."
        ),
        "system_prompt": (
            "You are an educational jurisdiction researcher. Search each requested jurisdiction "
            "separately with search_tax_sources and the requested effective date. Read every "
            "excerpt used with read_tax_source. Compare only supported claims and attach the "
            "excerpt_id, URL, section, and supporting quote. Explicitly identify missing or "
            "non-comparable evidence. Never rely on your own tax knowledge."
        ),
        "tools": [search_tax_sources, read_tax_source],
        "model": model,
    }

    return create_deep_agent(
        model=model,
        system_prompt=(
            "You coordinate educational, citation-backed tax research. Confirm the tax issue, "
            "jurisdictions, and effective date; state missing scope as an assumption. Delegate "
            "substantive research to tax-researcher and cross-jurisdiction comparison to "
            "jurisdiction-researcher. Synthesize only claims supported by retrieved excerpts. "
            "For every material claim, include its source title, section, URL, excerpt_id, and a "
            "short supporting quote. Separate conclusions, unresolved questions, and sources. "
            "Say 'insufficient evidence' when support is absent. This is educational research "
            "that requires qualified human review, not tax advice."
        ),
        subagents=[tax_researcher, jurisdiction_researcher],
        backend=backend,
        checkpointer=checkpointer,
    )
