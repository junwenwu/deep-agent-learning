"""Offline tests for the Deep Agents learning example."""

from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from deep_agent_learning import (
    ARTIFACT_NAME,
    EXIT_ERROR,
    EXIT_FAILURE,
    EXIT_SUCCESS,
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


def test_create_agent_registers_specialists_with_distinct_tools(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, Any] = {}

    def capture_agent(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("deep_agent_learning.agent.resolve_model", lambda model: model)
    monkeypatch.setattr("deepagents.create_deep_agent", capture_agent)

    assert create_agent("openai:test-model", workspace=tmp_path) == "compiled-agent"
    assert captured["checkpointer"] is None
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
    backend = captured["backend"]
    assert type(backend).__name__ == "FilesystemBackend"
    assert backend.cwd == tmp_path.resolve()
    assert backend.virtual_mode is True


def test_describe_agent_shows_artifact_workspace(tmp_path) -> None:
    result = describe_agent("openai:test-model", tmp_path)

    assert f"Artifact workspace: {tmp_path.resolve()}" in result
    assert f"write_file('/{ARTIFACT_NAME}')" in result


def test_describe_agent_shows_checkpoint_thread(tmp_path) -> None:
    checkpoint_db = tmp_path / "state" / "checkpoints.sqlite"

    result = describe_agent(
        "openai:test-model",
        checkpoint_db=checkpoint_db,
        thread_id="tax-session",
    )

    assert f"Checkpoint database: {checkpoint_db.resolve()}" in result
    assert "Thread ID: tax-session" in result


def test_describe_agent_shows_langsmith_project() -> None:
    result = describe_agent(
        "openai:test-model",
        trace=True,
        trace_project="tax-learning",
    )

    assert "LangSmith tracing: enabled" in result
    assert "Trace project: tax-learning" in result


def test_main_requests_and_reports_artifact(monkeypatch, tmp_path, capsys) -> None:
    captured: dict[str, Any] = {}

    class ArtifactAgent:
        def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
            captured["payload"] = payload
            (tmp_path / ARTIFACT_NAME).write_text("# Briefing\n", encoding="utf-8")
            return {"messages": [SimpleNamespace(content="Briefing complete.")]}

    def create_artifact_agent(model_name: str, workspace=None) -> ArtifactAgent:
        captured["model_name"] = model_name
        captured["workspace"] = workspace
        return ArtifactAgent()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("deep_agent_learning.cli.create_agent", create_artifact_agent)
    monkeypatch.setattr(
        "sys.argv",
        [
            "deep-agent-learning",
            "Create a tax briefing.",
            "--model",
            "openai:test-model",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert main() == EXIT_SUCCESS
    assert captured["workspace"] == tmp_path
    assert f"write_file to save the same briefing as Markdown at /{ARTIFACT_NAME}" in (
        captured["payload"]["messages"][0]["content"]
    )
    assert f"Artifact: {(tmp_path / ARTIFACT_NAME).resolve()}" in capsys.readouterr().out


def test_main_fails_when_requested_artifact_is_missing(
    monkeypatch, tmp_path, capsys
) -> None:
    class NoArtifactAgent:
        def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {"messages": [SimpleNamespace(content="No artifact created.")]}

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "deep_agent_learning.cli.create_agent",
        lambda model_name, workspace=None: NoArtifactAgent(),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "deep-agent-learning",
            "Create a tax briefing.",
            "--model",
            "openai:test-model",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert main() == EXIT_FAILURE
    assert "Expected artifact was not created" in capsys.readouterr().err


def test_main_passes_sqlite_checkpointer_and_thread_config(
    monkeypatch, tmp_path, capsys
) -> None:
    captured: dict[str, Any] = {}
    checkpoint_db = tmp_path / "state" / "checkpoints.sqlite"

    class CheckpointAgent:
        def invoke(
            self, payload: dict[str, Any], config: dict[str, Any]
        ) -> dict[str, Any]:
            captured["payload"] = payload
            captured["config"] = config
            return {"messages": [SimpleNamespace(content="Conversation saved.")]}

    def create_checkpoint_agent(
        model_name: str, workspace=None, checkpointer=None
    ) -> CheckpointAgent:
        captured["model_name"] = model_name
        captured["checkpointer"] = checkpointer
        return CheckpointAgent()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "deep_agent_learning.cli.create_agent", create_checkpoint_agent
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "deep-agent-learning",
            "Explain property tax.",
            "--model",
            "openai:test-model",
            "--checkpoint-db",
            str(checkpoint_db),
            "--thread-id",
            "tax-session",
        ],
    )

    assert main() == EXIT_SUCCESS
    assert type(captured["checkpointer"]).__name__ == "SqliteSaver"
    assert captured["config"] == {"configurable": {"thread_id": "tax-session"}}
    assert checkpoint_db.parent.is_dir()
    assert "Conversation saved." in capsys.readouterr().out


def test_main_rejects_incomplete_checkpoint_options(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "deep-agent-learning",
            "--checkpoint-db",
            str(tmp_path / "checkpoints.sqlite"),
        ],
    )

    assert main() == EXIT_ERROR
    assert "must be provided together" in capsys.readouterr().err


def test_main_rejects_trace_without_langsmith_key(monkeypatch, capsys) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setattr("deep_agent_learning.cli.configure_azure_environment", lambda: None)
    monkeypatch.setattr("sys.argv", ["deep-agent-learning", "--trace"])

    assert main() == EXIT_ERROR
    assert "LANGSMITH_API_KEY is required" in capsys.readouterr().err


def test_main_inspects_trace_without_langsmith_key(monkeypatch, capsys) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setattr("deep_agent_learning.cli.configure_azure_environment", lambda: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "deep-agent-learning",
            "--inspect",
            "--trace",
            "--trace-project",
            "tax-learning",
        ],
    )

    assert main() == EXIT_SUCCESS
    output = capsys.readouterr().out
    assert "LangSmith tracing: enabled" in output
    assert "Trace project: tax-learning" in output


def test_main_traces_run_with_project_and_metadata(monkeypatch, capsys) -> None:
    captured: dict[str, Any] = {}

    class TracedAgent:
        def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
            captured["payload"] = payload
            return {"messages": [SimpleNamespace(content="Traced briefing.")]}

    class FakeClient:
        def flush(self) -> None:
            captured["flushed"] = True

    @contextmanager
    def fake_tracing_context(**kwargs: Any):
        captured["trace_context"] = kwargs
        yield

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.setattr("deep_agent_learning.cli.Client", FakeClient)
    monkeypatch.setattr("deep_agent_learning.cli.tracing_context", fake_tracing_context)
    monkeypatch.setattr(
        "deep_agent_learning.cli.create_agent",
        lambda model_name, workspace=None: TracedAgent(),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "deep-agent-learning",
            "Explain property tax.",
            "--model",
            "openai:test-model",
            "--trace",
            "--trace-project",
            "tax-learning",
        ],
    )

    assert main() == EXIT_SUCCESS
    trace_context = captured["trace_context"]
    assert trace_context["enabled"] is True
    assert trace_context["project_name"] == "tax-learning"
    assert trace_context["tags"] == ["deep-agent-learning", "tax-briefing"]
    assert trace_context["metadata"] == {
        "model": "openai:test-model",
        "thread_id": "not-configured",
        "artifact_enabled": False,
    }
    assert captured["flushed"] is True
    assert "LangSmith project: tax-learning" in capsys.readouterr().out


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