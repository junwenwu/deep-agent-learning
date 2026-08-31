"""Model configuration for the Deep Agents example."""

from __future__ import annotations

import os
from typing import Any

AZURE_MODEL = "azure-openai"
DEFAULT_MODEL = AZURE_MODEL
AZURE_DEPLOYMENT_MODEL = "gpt-5-mini"
AZURE_DEPLOYMENT_MODEL_VERSION = "2025-08-07"
DEFAULT_AZURE_ENVIRONMENT = {
    "AZURE_OPENAI_ENDPOINT": "https://deepagent-learning.cognitiveservices.azure.com/",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "DeepAgent_Learning",
    "AZURE_OPENAI_API_VERSION": "2024-10-21",
}


def configure_azure_environment() -> None:
    """Set non-secret Azure defaults without replacing shell configuration."""
    for name, value in DEFAULT_AZURE_ENVIRONMENT.items():
        os.environ.setdefault(name, value)


def resolve_model(model: str) -> Any:
    """Resolve a model name to a LangChain model configuration.

    Args:
        model: ``azure-openai`` or a LangChain ``provider:model`` identifier.

    Returns:
        An Azure chat model instance or the unchanged LangChain model identifier.

    Raises:
        ValueError: If an Azure OpenAI environment variable is missing.
    """
    if model != AZURE_MODEL:
        return model

    configure_azure_environment()
    required_variables = (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
    )
    missing_variables = [name for name in required_variables if not os.environ.get(name)]
    if missing_variables:
        missing = ", ".join(missing_variables)
        raise ValueError(f"Missing Azure OpenAI environment variables: {missing}")

    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from langchain_openai import AzureChatOpenAI

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_ad_token_provider=token_provider,
    )