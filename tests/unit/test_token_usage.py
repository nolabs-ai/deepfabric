"""Provider usage survives structured-output parsing and failed attempts."""
# ruff: noqa: PLR2004

import asyncio

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest

from google.genai import types as genai_types
from pydantic import BaseModel

from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.llm.client import AnthropicModel, GeminiModel, LLMClient
from deepfabric.llm.token_usage import (
    TokenUsageTracker,
    normalize_usage,
    record_usage,
    track_token_usage,
)


class Answer(BaseModel):
    answer: str


def gemini_response(text='{"answer":"ok"}', usage=None):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]))],
        usage_metadata=usage
        if usage is not None
        else {
            "prompt_token_count": 10,
            "candidates_token_count": 4,
            "thoughts_token_count": 2,
            "total_token_count": 16,
            "cached_content_token_count": 3,
        },
    )


def native_client(native, provider):
    with (
        patch("deepfabric.llm.client.make_outlines_model", return_value=native),
        patch("deepfabric.llm.client.make_async_outlines_model", return_value=native),
    ):
        client = LLMClient(provider, "test", rate_limit_config={"max_retries": 0})
        assert client.async_model is native
        return client


@pytest.mark.parametrize(
    "provider,raw,expected",
    [
        (
            "gemini",
            {
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "thoughts_token_count": 3,
                "cached_content_token_count": 2,
                "tool_use_prompt_token_count": 4,
                "total_token_count": 22,
            },
            {
                "input_tokens": 14,
                "output_tokens": 8,
                "total_tokens": 22,
                "cached_input_tokens": 2,
                "reasoning_tokens": 3,
            },
        ),
        (
            "anthropic",
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
            },
            {
                "input_tokens": 15,
                "output_tokens": 5,
                "total_tokens": 20,
                "cached_input_tokens": 3,
                "cache_creation_input_tokens": 2,
            },
        ),
        (
            "openai",
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cached_input_tokens": 3,
                "reasoning_tokens": 2,
            },
        ),
    ],
)
def test_normalization_preserves_provider_fields(provider, raw, expected):
    normalized = normalize_usage(provider, raw)
    assert normalized["raw_usage"] == raw
    for key, value in expected.items():
        assert normalized["tokens"][key] == value


@pytest.mark.parametrize("raw", [None, {}, {"input_tokens": -1, "output_tokens": True}])
def test_unknown_or_invalid_counts_are_not_zero(raw):
    assert normalize_usage("anthropic", raw)["tokens"]["total_tokens"] is None


def test_reported_zero_tokens_are_valid():
    tracker = TokenUsageTracker()
    with track_token_usage(tracker):
        record_usage(
            "openai", "test", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
    report = tracker.snapshot()
    assert report["complete"]
    assert report["tokens"]["total_tokens"] == 0


async def test_gemini_bad_json_and_success_both_count_without_console_output(capsys):
    transport = AsyncMock(side_effect=[gemini_response("not json"), gemini_response()])
    native = GeminiModel(
        SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=transport))),
        "test",
    )
    client = native_client(native, "gemini")
    run_usage = TokenUsageTracker()
    with track_token_usage(run_usage):
        with pytest.raises(DataSetGeneratorError, match="validation failed"):
            await client.generate_async("first", Answer)
        assert (await client.generate_async("second", Answer)).answer == "ok"
    for report in (run_usage.snapshot(), client.token_usage.snapshot()):
        assert report["requests"] == 2
        assert report["tokens"]["input_tokens"] == 20
        assert report["tokens"]["output_tokens"] == 12
        assert report["tokens"]["total_tokens"] == 32
        assert report["tokens"]["reasoning_tokens"] == 4
        assert report["complete"]
        assert [r["stage"] for r in report["records"]] == ["Answer", "Answer"]
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


async def test_blocked_gemini_response_still_records_usage():
    response = gemini_response()
    response.candidates = []
    transport = AsyncMock(return_value=response)
    native = GeminiModel(
        SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=transport))),
        "test",
    )
    client = native_client(native, "gemini")
    with pytest.raises(DataSetGeneratorError, match="empty response"):
        await client.generate_async("blocked", Answer)
    assert client.token_usage.snapshot()["tokens"]["total_tokens"] == 16


async def test_transport_retry_has_unknown_first_attempt_and_reported_second():
    transport = AsyncMock(side_effect=[TimeoutError("request timeout"), gemini_response()])
    native = GeminiModel(
        SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=transport))),
        "test",
    )
    client = native_client(native, "gemini")
    client.retry_handler.config.max_retries = 1
    with (
        patch("deepfabric.llm.retry_handler.asyncio.sleep", new_callable=AsyncMock),
        patch.object(client.retry_handler, "on_backoff_handler"),
    ):
        await client.generate_async("retry", Answer)
    report = client.token_usage.snapshot()
    assert report["requests"] == 2
    assert report["requests_without_usage"] == 1
    assert report["tokens"]["total_tokens"] == 16
    assert not report["complete"]
    assert [r["response_received"] for r in report["records"]] == [False, True]


async def test_anthropic_truncation_records_cache_and_output():
    response = SimpleNamespace(
        stop_reason="max_tokens",
        content=[],
        usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
        },
    )
    transport = AsyncMock(return_value=response)
    native = AnthropicModel(
        SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=transport))), "test"
    )
    client = native_client(native, "anthropic")
    with pytest.raises(DataSetGeneratorError, match="truncated"):
        await client.generate_async("truncated", Answer)
    assert client.token_usage.snapshot()["tokens"]["total_tokens"] == 20
    assert client.token_usage.snapshot()["requests"] == 1


@pytest.mark.parametrize("provider", ["openai", "openrouter", "ollama"])
@pytest.mark.parametrize("path", ["sync", "async", "fallback"])
async def test_outlines_preserves_usage_from_sdk_response(provider, path):
    def transport(_request):
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"answer":"ok"}'},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            },
        )

    sync_sdk = openai.OpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    async_sdk = openai.AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    try:
        with (
            patch("deepfabric.llm.client._create_openai_compatible_client", return_value=sync_sdk),
            patch(
                "deepfabric.llm.client._create_async_openai_compatible_client",
                return_value=async_sdk,
            ),
            (
                patch("deepfabric.llm.client.make_async_outlines_model", return_value=None)
                if path == "fallback"
                else nullcontext()
            ),
        ):
            client = LLMClient(provider, "test", rate_limit_config={"max_retries": 0})
            result = (
                client.generate("request", Answer)
                if path == "sync"
                else await client.generate_async("request", Answer)
            )
        assert result.answer == "ok"
        report = client.token_usage.snapshot()
        assert report["tokens"]["total_tokens"] == 14
        assert report["requests"] == 1
        assert report["complete"]
    finally:
        sync_sdk.close()
        await async_sdk.close()


async def test_concurrent_calls_and_multiple_clients_accumulate_once():
    async def transport(**_kwargs):
        await asyncio.sleep(0)
        return gemini_response()

    native = GeminiModel(
        SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=transport))),
        "test",
    )
    clients = [native_client(native, "gemini"), native_client(native, "gemini")]
    run_usage = TokenUsageTracker()
    with track_token_usage(run_usage):
        await asyncio.gather(*(clients[i % 2].generate_async(str(i), Answer) for i in range(10)))
    assert run_usage.snapshot()["requests"] == 10
    assert run_usage.snapshot()["tokens"]["total_tokens"] == 160
    assert [c.token_usage.snapshot()["requests"] for c in clients] == [5, 5]
    # A fresh session does not inherit the previous run's totals.
    next_run = TokenUsageTracker()
    with track_token_usage(next_run):
        await clients[0].generate_async("new run", Answer)
    assert next_run.snapshot()["requests"] == 1
    assert run_usage.snapshot()["requests"] == 10


def test_native_sdk_metadata_keeps_provider_counts():
    usage = genai_types.GenerateContentResponseUsageMetadata(
        prompt_token_count=10,
        candidates_token_count=4,
        thoughts_token_count=2,
        total_token_count=16,
        cached_content_token_count=3,
    )
    result = normalize_usage("gemini", usage)
    assert result["tokens"]["output_tokens"] == 6
    assert result["tokens"]["total_tokens"] == 16
    assert result["raw_usage"]["thoughts_token_count"] == 2
