"""Offline tests for the Deep Agents learning example."""

from __future__ import annotations

import os

from deep_agent_learning import (
    DEFAULT_AZURE_ENVIRONMENT,
    EXIT_ERROR,
    configure_azure_environment,
    describe_agent,
    lookup_tax_topic,
    main,
    resolve_model,
)


def test_lookup_tax_topic_is_case_insensitive() -> None:
    result = lookup_tax_topic("  Sales Tax ")

    assert "seller" in result
    assert "buyer" in result


def test_lookup_tax_topic_lists_known_topics_for_unknown_value() -> None:
    result = lookup_tax_topic("property tax")

    assert "No exact match" in result
    assert "income tax" in result
    assert "sales tax" in result


def test_describe_agent_shows_delegation_path() -> None:
    result = describe_agent("openai:test-model")

    assert "task(subagent_type='tax-researcher')" in result
    assert "lookup_tax_topic(topic)" in result


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


def test_configure_azure_environment_sets_defaults(monkeypatch) -> None:
    for variable in DEFAULT_AZURE_ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)

    configure_azure_environment()

    for variable, value in DEFAULT_AZURE_ENVIRONMENT.items():
        assert os.environ[variable] == value


def test_configure_azure_environment_preserves_overrides(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")

    configure_azure_environment()

    assert os.environ["AZURE_OPENAI_ENDPOINT"] == "https://example.openai.azure.com/"


def test_resolve_model_preserves_provider_model_name() -> None:
    assert resolve_model("openai:test-model") == "openai:test-model"