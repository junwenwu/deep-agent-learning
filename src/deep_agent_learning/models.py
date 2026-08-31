"""Model configuration for the Deep Agents example."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

AZURE_MODEL = "azure-openai"
DEFAULT_MODEL = AZURE_MODEL
AZURE_DEPLOYMENT_MODEL = "gpt-5-mini"
AZURE_DEPLOYMENT_MODEL_VERSION = "2025-08-07"
REQUIRED_AZURE_ENVIRONMENT = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
)


def configure_azure_environment(dotenv_path: str | Path | None = None) -> None:
    """Load Azure configuration from a dotenv file without replacing shell values."""
    load_dotenv(dotenv_path=dotenv_path, override=False)


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
    missing_variables = [
        name for name in REQUIRED_AZURE_ENVIRONMENT if not os.environ.get(name)
    ]
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