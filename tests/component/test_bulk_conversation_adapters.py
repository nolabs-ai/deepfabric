"""Collection schemas through real client/adapter code, with provider I/O mocked."""
# ruff: noqa: PLR2004, S106

import json

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest

from jsonschema.validators import validator_for

from deepfabric.generator import DataSetGenerator
from deepfabric.llm.client import LLMClient
from deepfabric.schemas import Conversation


def record(question, cot):
    result = {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Answer"},
        ]
    }
    if cot:
        result["reasoning"] = {"style": "freetext", "content": "Explanation"}
    return result


def assert_collection_schema(schema):
    validator_for(schema).check_schema(schema)
    assert schema["type"] == "object"
    assert schema["required"] == ["items"]
    item = schema["properties"]["items"]["items"]
    assert item["type"] == "object"
    assert "messages" in item["required"]
    assert "ChatMessage" in item["$defs"] and "ReasoningTrace" in item["$defs"]
    references = []

    def visit(value):
        if isinstance(value, dict):
            ref = value.get("$ref")
            if ref is not None:
                references.append(ref)
                assert ref.startswith("#/properties/items/items/$defs/")
                target = schema
                for part in ref[2:].split("/"):
                    target = target[part.replace("~1", "/").replace("~0", "~")]
                assert isinstance(target, dict)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    assert references


@pytest.mark.parametrize(
    "provider,path",
    [
        ("openai", "sync"),
        ("openai", "async"),
        ("openrouter", "sync"),
        ("openrouter", "async"),
        ("ollama", "sync"),
        ("ollama", "async"),
        ("anthropic", "native"),
        ("gemini", "native"),
    ],
)
@pytest.mark.parametrize("cot", [False, True])
@pytest.mark.parametrize("partial", [False, True], ids=["complete", "malformed-then-partial"])
async def test_nested_collection_adapter_round_trip(
    tmp_path, monkeypatch, provider, path, cot, partial
):
    original_schema = Conversation.model_json_schema()
    payloads = iter(
        [
            '{"items": [',
            json.dumps({"items": [record("A", cot), {"messages": []}]}),
            json.dumps({"items": [record("B", cot)]}),
        ]
        if partial
        else [json.dumps({"items": [record("A", cot), record("B", cot)]})]
    )
    sent = []

    def transport(request):
        body = json.loads(request.content)
        sent.append(body)
        schema = body["response_format"]["json_schema"]["schema"]
        assert_collection_schema(schema)
        return httpx.Response(
            200,
            json={
                "id": "completion-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": next(payloads)},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    sync_client = openai.OpenAI(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(transport))
    )
    async_client = openai.AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport))
    )

    async def anthropic_create(**body):
        sent.append(body)
        assert_collection_schema(body["output_format"]["schema"])
        return SimpleNamespace(
            stop_reason="end_turn", content=[SimpleNamespace(text=next(payloads))]
        )

    async def gemini_generate(**body):
        sent.append(body)
        assert_collection_schema(body["config"].response_json_schema)
        return SimpleNamespace(
            text=next(payloads),
            candidates=[
                SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="response")]))
            ],
        )

    native_anthropic = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(create=AsyncMock(side_effect=anthropic_create))
        )
    )
    native_gemini = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=AsyncMock(side_effect=gemini_generate))
        )
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    try:
        with (
            patch(
                "deepfabric.llm.client._create_openai_compatible_client", return_value=sync_client
            ),
            patch(
                "deepfabric.llm.client._create_async_openai_compatible_client",
                return_value=async_client,
            ),
            patch("deepfabric.llm.client.anthropic.AsyncAnthropic", return_value=native_anthropic),
            patch("deepfabric.llm.client.genai.Client", return_value=native_gemini),
        ):
            client = LLMClient(
                provider=provider, model_name="test", rate_limit_config={"max_retries": 0}
            )
            if path == "sync":
                client._async_model_initialized = True
                client._async_model = None
            with patch("deepfabric.generator.LLMClient", return_value=client):
                engine = DataSetGenerator(
                    generation_system_prompt="Persona",
                    provider=provider,
                    model_name="test",
                    conversation_type="cot" if cot else "basic",
                    reasoning_style="freetext" if cot else None,
                    checkpoint_interval=1,
                    checkpoint_path=str(tmp_path),
                    output_save_as="records.jsonl",
                    max_tokens=6000,
                )
            engine.config = engine.config.model_copy(
                update={"topic_sample_mode": "bulk", "shared_scope": "full"}
            )
            topic_model = SimpleNamespace(
                get_unique_topics=lambda: [SimpleNamespace(uuid="topic", topic="Child")],
                get_path_by_id=lambda _uuid: ["Root", "Child"],
            )
            values = [
                value
                async for value in engine.create_data_with_events_async(
                    num_samples=2, topic_model=topic_model
                )
            ]
    finally:
        sync_client.close()
        await async_client.close()
    dataset = values[-1]
    assert len(dataset) == 2
    assert [row["messages"][1]["content"] for row in dataset] == ["A", "B"]
    assert engine._sample_slots == [("topic", 0), ("topic", 1)]
    assert engine.get_all_failures() == []
    requests = [
        value
        for value in values
        if isinstance(value, dict) and value.get("event") == "seed_request"
    ]
    assert [value["requested"] for value in requests] == ([2, 2, 1] if partial else [2])
    assert [value["attempt"] for value in requests] == ([1, 2, 3] if partial else [1])
    assert len(sent) == (3 if partial else 1)
    for body in sent:
        if provider == "gemini":
            assert body["config"].max_output_tokens == 6000
            assert body["contents"].startswith(engine.generation_prompt)
        else:
            assert body["max_tokens"] == 6000
            assert body["messages"][0]["role"] == "user"
    assert Conversation.model_json_schema() == original_schema
