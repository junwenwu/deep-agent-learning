"""Public API for the Deep Agents learning example."""

from deep_agent_learning.agent import create_agent
from deep_agent_learning.cli import (
    ARTIFACT_NAME,
    DEFAULT_QUESTION,
    DEFAULT_TRACE_PROJECT,
    EXIT_ERROR,
    EXIT_FAILURE,
    EXIT_SUCCESS,
    create_parser,
    describe_agent,
    main,
)
from deep_agent_learning.models import (
    AZURE_DEPLOYMENT_MODEL,
    AZURE_DEPLOYMENT_MODEL_VERSION,
    AZURE_MODEL,
    DEFAULT_MODEL,
    REQUIRED_AZURE_ENVIRONMENT,
    configure_azure_environment,
    resolve_model,
)
from deep_agent_learning.research import read_tax_source, search_tax_sources

__all__ = [
    "ARTIFACT_NAME",
    "AZURE_DEPLOYMENT_MODEL",
    "AZURE_DEPLOYMENT_MODEL_VERSION",
    "AZURE_MODEL",
    "DEFAULT_MODEL",
    "DEFAULT_QUESTION",
    "DEFAULT_TRACE_PROJECT",
    "EXIT_ERROR",
    "EXIT_FAILURE",
    "EXIT_SUCCESS",
    "REQUIRED_AZURE_ENVIRONMENT",
    "configure_azure_environment",
    "create_agent",
    "create_parser",
    "describe_agent",
    "main",
    "read_tax_source",
    "resolve_model",
    "search_tax_sources",
]
