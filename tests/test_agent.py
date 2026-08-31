"""Offline tests for the Deep Agents learning example."""

from __future__ import annotations

import os
from typing import Any

from deep_agent_learning import (
    EXIT_ERROR,
    REQUIRED_AZURE_ENVIRONMENT,
    configure_azure_environment,
    create_agent,
    describe_agent,
    lookup_tax_jurisdiction,
    lookup_tax_topic,
    main,
    resolve_model,
)


def test_lookup_tax_topic_is_case_insensitive() -> None:
    result = lookup_tax_topic("  Sales Tax ")

    assert "seller" in result
    assert "buyer" in result


def test_lookup_property_tax_is_case_insensitive() -> None:
    result = lookup_tax_topic("  PROPERTY Tax ")

    assert "owner" in result
    assert "local tax authority" in result


def test_lookup_tax_topic_lists_known_topics_for_unknown_value() -> None:
    result = lookup_tax_topic("estate tax")

    assert "No exact match" in result
    assert "income tax" in result
    assert "property tax" in result
    assert "sales tax" in result


def test_lookup_tax_jurisdiction_is_case_insensitive() -> None:
    result = lookup_tax_jurisdiction("  STATE ")

    assert "within one state" in result
    assert "local taxes" in result


def test_lookup_tax_jurisdiction_lists_known_levels() -> None:
    result = lookup_tax_jurisdiction("international")

    assert "No exact match" in result
    assert "federal, local, state" in result


def test_describe_agent_shows_delegation_path() -> None:
    result = describe_agent("openai:test-model")

    assert "task(subagent_type='tax-researcher')" in result
    assert "lookup_tax_topic(topic)" in result
    assert "task(subagent_type='jurisdiction-researcher')" in result
    assert "lookup_tax_jurisdiction(jurisdiction)" in result


def test_create_agent_registers_specialists_with_distinct_tools(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def capture_agent(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("deep_agent_learning.agent.resolve_model", lambda model: model)
    monkeypatch.setattr("deepagents.create_deep_agent", capture_agent)

    assert create_agent("openai:test-model") == "compiled-agent"
    subagents = captured["subagents"]
    assert [subagent["name"] for subagent in subagents] == [
        "tax-researcher",
        "jurisdiction-researcher",
    ]
    assert subagents[0]["tools"] == [lookup_tax_topic]
    assert subagents[1]["tools"] == [lookup_tax_jurisdiction]
    assert "Return only facts present in the tool output" in subagents[0]["system_prompt"]
    assert "Return only facts present in the tool output" in subagents[1]["system_prompt"]
    assert "Use both specialists" in captured["system_prompt"]
    assert "using only facts returned by the specialists" in captured["system_prompt"]


def test_describe_agent_shows_underlying_azure_model() -> None:
    result = describe_agent("azure-openai")

    assert "DeepAgent_Learning" in result
    assert "gpt-5-mini (2025-08-07)" in result


def test_main_explains_missing_openai_key(monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "sys.argv", ["deep-agent-learning", "--model", "openai:test-model"]
    )

    assert main() == EXIT_ERROR
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().err


def test_configure_azure_environment_loads_dotenv(monkeypatch, tmp_path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "AZURE_OPENAI_ENDPOINT=https://example.openai.azure.com/\n"
        "AZURE_OPENAI_CHAT_DEPLOYMENT=example-deployment\n"
        "AZURE_OPENAI_API_VERSION=2024-10-21\n",
        encoding="utf-8",
    )
    for variable in REQUIRED_AZURE_ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)

    configure_azure_environment(dotenv_path)

    assert os.environ["AZURE_OPENAI_ENDPOINT"] == "https://example.openai.azure.com/"
    assert os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"] == "example-deployment"
    assert os.environ["AZURE_OPENAI_API_VERSION"] == "2024-10-21"


def test_configure_azure_environment_preserves_overrides(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")

    configure_azure_environment("missing.env")

    assert os.environ["AZURE_OPENAI_ENDPOINT"] == "https://example.openai.azure.com/"


def test_resolve_model_preserves_provider_model_name() -> None:
    assert resolve_model("openai:test-model") == "openai:test-model"