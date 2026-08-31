"""Public API for the Deep Agents learning example."""

from deep_agent_learning.agent import create_agent
from deep_agent_learning.cli import (
    DEFAULT_QUESTION,
    EXIT_ERROR,
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
from deep_agent_learning.tools import TAX_CATALOG, lookup_tax_topic

__all__ = [
    "AZURE_DEPLOYMENT_MODEL",
    "AZURE_DEPLOYMENT_MODEL_VERSION",
    "AZURE_MODEL",
    "DEFAULT_MODEL",
    "DEFAULT_QUESTION",
    "EXIT_ERROR",
    "EXIT_SUCCESS",
    "REQUIRED_AZURE_ENVIRONMENT",
    "TAX_CATALOG",
    "configure_azure_environment",
    "create_agent",
    "create_parser",
    "describe_agent",
    "lookup_tax_topic",
    "main",
    "resolve_model",
]
