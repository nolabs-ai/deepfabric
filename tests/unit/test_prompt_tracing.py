"""File-only tracing at the shared LLM request boundary."""
# ruff: noqa: PLR2004

import asyncio
import json

from unittest.mock import AsyncMock, Mock, patch

import pytest

from pydantic import create_model

from deepfabric.builders_agent import AgentResponse, UserQuestion
from deepfabric.llm.client import LLMClient, make_dynamic_model
from deepfabric.llm.prompt_trace import prompt_tracing
from deepfabric.schemas import AgentStep, Conversation, GraphSubtopics, TopicList

CustomRecord = create_model("CustomRecord", answer=(str, ...))


@pytest.fixture(autouse=True)
def tracing_session(request, tmp_path):
    enabled = not request.node.name.startswith("test_disabled_tracing")
    with prompt_tracing(
        enabled=enabled, path=str(tmp_path / "prompts.jsonl"), verbose=True
    ) as session:
        yield session


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("DEEPFABRIC_PROMPT_TRACE_FILE", raising=False)
    with (
        patch("deepfabric.llm.client.make_outlines_model", return_value=Mock()),
        patch("deepfabric.llm.client.make_async_outlines_model", return_value=AsyncMock()),
    ):
        yield LLMClient("openai", "test-model", rate_limit_config={"max_retries": 0})


def read_traces(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    "schema",
    [TopicList, GraphSubtopics, Conversation, UserQuestion, AgentStep, AgentResponse, CustomRecord],
    ids=["tree", "graph", "single-shot", "user-question", "agent-step", "agent-response", "custom"],
)
async def test_composed_prompt_and_stage_are_traced(client, schema, tmp_path, monkeypatch, capsys):
    trace_path = tmp_path / "prompts.jsonl"
    monkeypatch.setenv("DEEPFABRIC_PROMPT_TRACE_FILE", str(trace_path))
    prompt = 'System instructions\nRoot → child\nUser: "generate a sample"'
    expected = object()
    with patch.object(schema, "model_validate_json", return_value=expected):
        result = await client.generate_async(prompt, schema, max_tokens=123)

    assert result is expected
    assert read_traces(trace_path) == [
        {
            "provider": "openai",
            "model": "test-model",
            "schema_name": schema.__name__,
            "schema": schema.model_json_schema(),
            "dry_run": False,
            "prompt": prompt,
        }
    ]
    assert client.async_model.await_args.args[0] == prompt
    assert client.async_model.await_args.kwargs == {"max_tokens": 123}
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("setting", [None, "", "./should-not-be-created.jsonl"])
async def test_disabled_tracing_does_not_open_files(client, monkeypatch, setting):
    if setting is not None:
        monkeypatch.setenv("DEEPFABRIC_PROMPT_TRACE_FILE", setting)
    client.async_model.return_value = '{"answer": "ok"}'
    with patch("pathlib.Path.open") as open_file:
        result = await client.generate_async("a prompt", CustomRecord)
    assert result.answer == "ok"
    open_file.assert_not_called()


async def test_message_roles_and_content_are_preserved(client, tmp_path, monkeypatch):
    trace_path = tmp_path / "prompts.jsonl"
    monkeypatch.setenv("DEEPFABRIC_PROMPT_TRACE_FILE", str(trace_path))
    messages = [
        {"role": "system", "content": "Be precise.\nUse the supplied context."},
        {"role": "user", "content": [{"type": "text", "text": "Explain café"}]},
    ]
    client.async_model.return_value = '{"answer": "ok"}'
    await client.generate_async(messages, CustomRecord)
    assert read_traces(trace_path)[0]["prompt"] == messages
    assert client.async_model.await_args.args[0] is messages


async def test_sync_fallback_appends_one_record_per_request(client, tmp_path, monkeypatch, capsys):
    trace_path = tmp_path / "prompts.jsonl"
    monkeypatch.setenv("DEEPFABRIC_PROMPT_TRACE_FILE", str(trace_path))
    client._async_model = None
    client._async_model_initialized = True
    client.model.return_value = '{"answer": "ok"}'
    prompts = [f"prompt {index}" for index in range(20)]
    await asyncio.gather(*(client.generate_async(prompt, CustomRecord) for prompt in prompts))
    assert sorted(record["prompt"] for record in read_traces(trace_path)) == sorted(prompts)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


async def test_failed_request_is_traced_before_provider_call(client, tmp_path, monkeypatch):
    trace_path = tmp_path / "prompts.jsonl"
    monkeypatch.setenv("DEEPFABRIC_PROMPT_TRACE_FILE", str(trace_path))
    client.async_model.side_effect = ValueError("provider failure")
    with pytest.raises(ValueError, match="provider failure"):
        await client.generate_async("failed prompt", CustomRecord)
    assert read_traces(trace_path)[0]["prompt"] == "failed prompt"


async def test_retries_append_each_attempt(client, tmp_path, monkeypatch):
    trace_path = tmp_path / "prompts.jsonl"
    monkeypatch.setenv("DEEPFABRIC_PROMPT_TRACE_FILE", str(trace_path))
    client.retry_handler.config.max_retries = 1
    client.async_model.side_effect = [TimeoutError("request timeout"), '{"answer": "ok"}']
    with (
        patch("deepfabric.llm.retry_handler.asyncio.sleep", new_callable=AsyncMock),
        patch.object(client.retry_handler, "on_backoff_handler"),
    ):
        result = await client.generate_async("retry prompt", CustomRecord)
    assert result.answer == "ok"
    assert [record["prompt"] for record in read_traces(trace_path)] == [
        "retry prompt",
        "retry prompt",
    ]


@pytest.mark.parametrize("provider", ["anthropic", "gemini", "ollama", "openrouter"])
async def test_provider_paths_are_traced(client, provider, tmp_path, monkeypatch):
    trace_path = tmp_path / "prompts.jsonl"
    monkeypatch.setenv("DEEPFABRIC_PROMPT_TRACE_FILE", str(trace_path))
    client.provider = provider
    client.async_model.return_value = '{"answer": "ok"}'
    await client.generate_async("provider prompt", CustomRecord)
    record = read_traces(trace_path)[0]
    assert record["provider"] == provider
    assert record["prompt"] == "provider prompt"


async def test_unwritable_trace_fails_before_provider_request(client, tmp_path, tracing_session):
    tracing_session.path = tmp_path
    with pytest.raises(IsADirectoryError):
        await client.generate_async("a prompt", CustomRecord)
    client.async_model.assert_not_awaited()


@pytest.mark.parametrize("verbose, expected_count", [(False, 1), (True, 3)])
async def test_first_per_stage_or_all_calls(client, tmp_path, verbose, expected_count):
    client.async_model.return_value = '{"answer": "ok"}'
    path = tmp_path / "first.jsonl"
    with prompt_tracing(enabled=True, path=str(path), verbose=verbose):
        for i in range(3):
            await client.generate_async(f"topic {i}", CustomRecord)
    assert len(read_traces(path)) == expected_count
    assert read_traces(path)[0]["prompt"] == "topic 0"


async def test_new_run_resets_first_stage_and_does_not_leak(client, tmp_path):
    client.async_model.return_value = '{"answer": "ok"}'
    path = tmp_path / "runs.jsonl"
    for _ in range(2):
        with prompt_tracing(enabled=True, path=str(path)):
            await client.generate_async("prompt", CustomRecord)
            await client.generate_async("suppressed", CustomRecord)
    assert len(read_traces(path)) == 2


@pytest.mark.parametrize(
    "explicit, environment, expected",
    [
        ("flag.jsonl", "env.jsonl", "flag.jsonl"),
        (None, "env.jsonl", "env.jsonl"),
        (None, None, "prompt-trace.jsonl"),
    ],
)
def test_destination_precedence(monkeypatch, explicit, environment, expected):
    if environment is None:
        monkeypatch.delenv("DEEPFABRIC_PROMPT_TRACE_FILE", raising=False)
    else:
        monkeypatch.setenv("DEEPFABRIC_PROMPT_TRACE_FILE", environment)
    with prompt_tracing(enabled=True, path=explicit) as session:
        assert str(session.path) == expected


async def test_dynamic_custom_schema_is_traced_without_pydantic_base(client, tmp_path):
    schema = make_dynamic_model({"type": "object", "properties": {"answer": {"type": "string"}}})
    client.async_model.return_value = '{"answer": "ok"}'
    path = tmp_path / "dynamic.jsonl"
    with prompt_tracing(enabled=True, path=str(path)):
        result = await client.generate_async("custom prompt", schema)
    assert result.model_dump() == {"answer": "ok"}
    assert read_traces(path)[0]["schema"] == schema.model_json_schema()
