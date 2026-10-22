"""Bulk conversation seeds with real builders and mocked provider/Spin I/O."""
# ruff: noqa: PLR2004, ARG001

import asyncio
import json

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from deepfabric.builders import SingleShotBuilder
from deepfabric.builders_agent import SingleTurnAgentBuilder
from deepfabric.generator import DataSetGenerator
from deepfabric.schemas import ChatMessage


@pytest.fixture
def make_engine(tmp_path):
    def make(**options):
        params = {
            "generation_system_prompt": "GENERATION_PERSONA",
            "dataset_system_prompt": "OUTPUT_PERSONA",
            "provider": "openai",
            "model_name": "test",
            "topic_sample_mode": "bulk",
            "instructions": "SPECIAL_INSTRUCTIONS",
            "sys_msg": False,
            "checkpoint_interval": 1,
            "checkpoint_path": str(tmp_path),
            "output_save_as": "conversations.jsonl",
        }
        params.update(options)
        with patch("deepfabric.generator.LLMClient"):
            engine = DataSetGenerator(**params)
        engine.build_examples_text = lambda _count: "DEMONSTRATION_SENTINEL"
        return engine

    return make


def script(engine, *values):
    iterator = iter(values)

    async def generate(prompt, schema, **kwargs):
        value = next(iterator)
        if isinstance(value, BaseException):
            raise value
        return schema.model_validate_json(json.dumps(value))

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)


async def run(engine, count=1, **kwargs):
    topics = kwargs.pop("topic_model", None) or SimpleNamespace(
        get_unique_topics=lambda: [SimpleNamespace(uuid="topic-0", topic="Topic")],
        get_path_by_id=lambda uuid: ["PRIVATE_ROOT", uuid],
    )
    events = [
        event
        async for event in engine.create_data_with_events_async(
            num_samples=count, topic_model=topics, **kwargs
        )
    ]
    return events[:-1], events[-1]


def unpack(call):
    return (
        (call.args[0], call.args[1])
        if call.args
        else (call.kwargs["prompt"], call.kwargs["schema"])
    )


@pytest.mark.parametrize("kind", ["basic", "cot"])
@pytest.mark.parametrize(
    "persona,path,shipping",
    [
        ("OUTPUT_PERSONA", False, True),
        (None, False, True),
        ("OUTPUT_PERSONA", True, True),
        (None, False, False),
    ],
)
async def test_prompt_matrix_and_exact_pin(make_engine, kind, persona, path, shipping):
    engine = make_engine(
        conversation_type=kind, dataset_system_prompt=persona, completion_includes_topic_path=path
    )
    seed = "  Explain café\nΩ?  "
    response = {"content": "An assistant answer"}
    if kind == "cot":
        response["reasoning"] = {"style": "freetext", "content": "Consider the request"}
    script(engine, {"items": [seed]}, response)
    _, dataset = await run(engine, sys_msg=shipping)
    row = dataset[0]
    assert row["messages"][int(shipping)]["content"] == seed
    assert row["messages"][-1]["content"] == response["content"]
    resolved = persona or "GENERATION_PERSONA"
    if shipping:
        assert row["messages"][0] == {"role": "system", "content": resolved}
    seed_prompt, _ = unpack(engine.llm_client.generate_async.call_args_list[0])
    completion_prompt, model = unpack(engine.llm_client.generate_async.call_args_list[1])
    assert "PRIVATE_ROOT" in seed_prompt
    assert "GENERATION_PERSONA" not in seed_prompt and "OUTPUT_PERSONA" not in seed_prompt
    assert ("PRIVATE_ROOT" in completion_prompt) is path
    assert resolved in completion_prompt
    assert set(model.model_json_schema()["properties"]) == (
        {"content", "reasoning"} if kind == "cot" else {"content"}
    )
    for prompt in (seed_prompt, completion_prompt):
        assert "SPECIAL_INSTRUCTIONS" in prompt and "DEMONSTRATION_SENTINEL" in prompt


async def test_partial_invalid_seeds_and_slot_retry_isolation(make_engine):
    engine = make_engine(sample_retries=1)
    script(
        engine,
        {"items": ["A", "", 12]},
        {"items": ["B"]},
        {"messages": [{"role": "user", "content": "drift"}]},
        {"content": ""},
        {"content": "B answer"},
    )
    events, dataset = await run(engine, 2)
    assert len(dataset) == 1
    assert dataset[0]["messages"][0]["content"] == "B"
    assert engine.get_all_failures()[0]["sample_ordinal"] == 0
    assert len(engine.llm_client.generate_async.call_args_list) == 5
    assert len([event for event in events if event.get("event") == "batch_complete"]) == 2


@pytest.mark.parametrize("provider", ["openai", "openrouter", "ollama", "anthropic", "gemini"])
@pytest.mark.parametrize("kind", ["basic", "cot"])
async def test_assistant_schema_through_real_adapters(make_engine, provider, kind):
    from jsonschema import validate  # noqa: PLC0415
    from outlines.models.openai import AsyncOpenAI  # noqa: PLC0415

    from deepfabric.llm.client import AnthropicModel, GeminiModel, LLMClient  # noqa: PLC0415

    sent = []

    def response(schema):
        sent.append(schema)
        if "items" in schema["properties"]:
            value = {"items": ["exact seed"]}
        else:
            assert "messages" not in schema["properties"]
            value = {"content": "An answer"}
            if kind == "cot":
                value["reasoning"] = {"style": "freetext", "content": "A reason"}
        validate(value, schema)
        return json.dumps(value)

    async def openai_create(**body):
        raw = response(body["response_format"]["json_schema"]["schema"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=raw, refusal=None))]
        )

    async def anthropic_create(**body):
        raw = response(body["output_format"]["schema"])
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(text=raw)])

    async def gemini_generate(**body):
        raw = response(body["config"].response_json_schema)
        return SimpleNamespace(
            text=raw,
            candidates=[
                SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=raw)]))
            ],
        )

    if provider == "anthropic":
        adapter = AnthropicModel(
            SimpleNamespace(
                beta=SimpleNamespace(
                    messages=SimpleNamespace(create=AsyncMock(side_effect=anthropic_create))
                )
            ),
            "test",
        )
    elif provider == "gemini":
        adapter = GeminiModel(
            SimpleNamespace(
                aio=SimpleNamespace(
                    models=SimpleNamespace(generate_content=AsyncMock(side_effect=gemini_generate))
                )
            ),
            "test",
        )
    else:
        adapter = AsyncOpenAI(
            SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=AsyncMock(side_effect=openai_create))
                )
            ),
            "test",
        )
    engine = make_engine(provider=provider, conversation_type=kind)
    with (
        patch("deepfabric.llm.client.make_outlines_model", return_value=adapter),
        patch("deepfabric.llm.client.make_async_outlines_model", return_value=adapter),
    ):
        engine.llm_client = LLMClient(
            provider=provider, model_name="test", rate_limit_config={"max_retries": 0}
        )
        _, dataset = await run(engine)
    assert len(dataset) == 1, engine.get_all_failures()
    assert dataset[0]["messages"][0]["content"] == "exact seed"
    assert len(sent) == 2


async def test_cot_wrong_style_retries_without_changing_user(make_engine):
    engine = make_engine(conversation_type="cot", sample_retries=1)
    script(
        engine,
        {"items": ["A"]},
        {"content": "reply", "reasoning": {"style": "agent", "content": []}},
        {"content": "reply", "reasoning": {"style": "freetext", "content": "thinking"}},
    )
    _, dataset = await run(engine)
    assert dataset[0]["messages"][0]["content"] == "A"
    assert dataset[0]["reasoning"]["style"] == "freetext"
    assert len(engine.llm_client.generate_async.call_args_list) == 3


async def test_failed_slot_resume_reuses_seed_and_successful_sibling(make_engine):
    engine = make_engine(sample_retries=0)
    script(engine, {"items": ["A", "B"]}, {"content": ""}, {"content": "B reply"})
    await run(engine, 2, sys_msg=True)
    restored = make_engine(sample_retries=0)
    assert restored.load_checkpoint(retry_failed=True)
    script(restored, {"content": "A reply"})
    _, dataset = await run(restored, 2, sys_msg=True)
    assert [row["messages"][1]["content"] for row in dataset] == ["B", "A"]
    assert len(restored.llm_client.generate_async.call_args_list) == 1
    assert restored.get_all_failures() == []


async def test_seed_only_resume_does_not_recollect(make_engine):
    engine = make_engine()

    async def seeds(prompt, schema, **kwargs):
        engine.stop_requested = True
        return schema.model_validate({"items": ["unchanged"]})

    engine.llm_client.generate_async = AsyncMock(side_effect=seeds)
    _, dataset = await run(engine)
    assert len(dataset) == 0
    restored = make_engine()
    assert restored.load_checkpoint()
    script(restored, {"content": "reply"})
    _, dataset = await run(restored)
    assert dataset[0]["messages"][0]["content"] == "unchanged"
    assert len(restored.llm_client.generate_async.call_args_list) == 1


async def test_builder_copies_pinned_message(make_engine):
    engine = make_engine()
    script(engine, {"content": "reply"})
    builder = SingleShotBuilder(engine.llm_client, engine.config)
    user = ChatMessage(role="user", content="exact")
    conversation = await builder.generate("context", pinned_user_message=user)
    user.content = "changed later"
    assert conversation.messages[0].content == "exact"


TOOL = {"name": "lookup", "description": "CATALOG_SENTINEL", "parameters": [], "returns": "result"}


@pytest.fixture
def spin_sessions():
    sessions = []

    def create(*args, **kwargs):
        session = SimpleNamespace(
            execute_tool=AsyncMock(
                return_value=SimpleNamespace(
                    success=True, result="observed result", error_type=None
                )
            ),
            cleanup=AsyncMock(),
            seed_files=AsyncMock(),
        )
        sessions.append(session)
        return session

    with (
        patch("deepfabric.builders_agent.SpinClient"),
        patch("deepfabric.builders_agent.SpinSession", side_effect=create),
    ):
        yield sessions


TOOL_STEP = {
    "thought": "Look it up",
    "tool_calls": [{"function_name": "lookup", "arguments": "{}", "reasoning": "Need evidence"}],
}
FINAL_STEP = {"thought": "Enough evidence", "tool_calls": [], "is_final": True}


@pytest.mark.parametrize("kind,style", [("basic", None), ("cot", "agent"), ("cot", "freetext")])
@pytest.mark.parametrize("include_path", [False, True])
async def test_agent_pins_request_and_carries_context_to_both_stages(
    make_engine, spin_sessions, kind, style, include_path
):
    engine = make_engine(
        custom_tools=[TOOL],
        spin_endpoint="http://spin.invalid",
        conversation_type=kind,
        reasoning_style=style,
        completion_includes_topic_path=include_path,
    )
    script(
        engine,
        {"items": ["  exact A  ", "exact B"]},
        TOOL_STEP,
        FINAL_STEP,
        {"content": "Answer based on evidence"},
        TOOL_STEP,
        FINAL_STEP,
        {"content": "Another evidence based answer"},
    )
    with patch.object(
        SingleTurnAgentBuilder, "_generate_user_question", new_callable=AsyncMock
    ) as question:
        _, dataset = await run(engine, 2, sys_msg=True)
    question.assert_not_awaited()
    assert len(dataset) == 2 and len(spin_sessions) == 2
    for row, seed in zip(dataset, ["  exact A  ", "exact B"], strict=True):
        assert row["messages"][1]["content"] == seed
        call_id = row["messages"][2]["tool_calls"][0]["id"]
        assert row["messages"][3]["tool_call_id"] == call_id
        assert row["tool_context"]["executions"][0]["result"] == "observed result"
        if style == "freetext":
            assert isinstance(row["reasoning"]["content"], str)
    for session in spin_sessions:
        session.execute_tool.assert_awaited_once()
        session.cleanup.assert_awaited_once()
    calls = engine.llm_client.generate_async.call_args_list
    assert "CATALOG_SENTINEL" in unpack(calls[0])[0]
    for call in calls[1:]:
        prompt, _ = unpack(call)
        for text in (
            "OUTPUT_PERSONA",
            "SPECIAL_INSTRUCTIONS",
            "DEMONSTRATION_SENTINEL",
            "CATALOG_SENTINEL",
        ):
            assert text in prompt
        assert "GENERATION_PERSONA" not in prompt
        assert ("PRIVATE_ROOT" in prompt) is include_path


async def test_agent_retry_budget_is_not_multiplied_and_sessions_are_fresh(
    make_engine, spin_sessions
):
    engine = make_engine(custom_tools=[TOOL], spin_endpoint="http://spin.invalid", sample_retries=2)
    script(
        engine,
        {"items": ["A", "B"]},
        {"thought": ""},
        {"thought": ""},
        {"thought": ""},
        TOOL_STEP,
        FINAL_STEP,
        {"content": "Sibling still succeeds"},
    )
    _, dataset = await run(engine, 2)
    assert len(dataset) == 1 and dataset[0]["messages"][0]["content"] == "B"
    assert len(spin_sessions) == 4
    assert len(engine.llm_client.generate_async.call_args_list) == 7
    for session in spin_sessions:
        session.cleanup.assert_awaited_once()
    for session in spin_sessions[:3]:
        session.execute_tool.assert_not_awaited()
    assert engine.get_all_failures()[0]["sample_ordinal"] == 0


@pytest.mark.parametrize(
    "failure", [asyncio.CancelledError(), RuntimeError("provider unavailable")]
)
async def test_agent_cleanup_on_interrupted_completion(make_engine, spin_sessions, failure):
    engine = make_engine(custom_tools=[TOOL], spin_endpoint="http://spin.invalid")
    builder = SingleTurnAgentBuilder(engine.llm_client, engine.config, engine.tool_registry)
    script(engine, TOOL_STEP, FINAL_STEP, failure)
    with pytest.raises(type(failure)):
        await builder.generate(
            "completion context", pinned_user_message=ChatMessage(role="user", content="original")
        )
    assert len(spin_sessions) == 1
    spin_sessions[0].execute_tool.assert_awaited_once()
    spin_sessions[0].cleanup.assert_awaited_once()
    assert builder._completion_context is None


async def test_independent_agent_still_generates_its_question(make_engine, spin_sessions):
    engine = make_engine(
        topic_sample_mode="independent", custom_tools=[TOOL], spin_endpoint="http://spin.invalid"
    )
    script(
        engine,
        {"content": "A generated user question"},
        TOOL_STEP,
        FINAL_STEP,
        {"content": "A complete final answer"},
    )
    _, dataset = await run(engine)
    assert dataset[0]["messages"][0]["content"] == "A generated user question"
    assert unpack(engine.llm_client.generate_async.call_args_list[0])[1].__name__ == "UserQuestion"
    spin_sessions[0].cleanup.assert_awaited_once()


async def test_agent_without_tool_execution_exhausts_only_its_own_slot(make_engine, spin_sessions):
    engine = make_engine(custom_tools=[TOOL], spin_endpoint="http://spin.invalid", sample_retries=1)
    script(
        engine,
        {"items": ["A", "B"]},
        FINAL_STEP,
        {"content": "A reply without execution"},
        FINAL_STEP,
        {"content": "Another reply without execution"},
        TOOL_STEP,
        FINAL_STEP,
        {"content": "B executes the tool successfully"},
    )
    _, dataset = await run(engine, 2)
    assert len(dataset) == 1 and dataset[0]["messages"][0]["content"] == "B"
    assert len(spin_sessions) == 3
    for session in spin_sessions:
        session.cleanup.assert_awaited_once()
    assert "at least one tool execution" in engine.get_all_failures()[0]["error"]


async def test_pre_seed_checkpoint_rebuilds_context_after_shipping_system(make_engine):
    topics = SimpleNamespace(
        get_unique_topics=lambda: [SimpleNamespace(uuid="topic-0", topic="Topic")],
        get_path_by_id=lambda uuid: ["PRIVATE_ROOT", uuid],
    )
    writer = make_engine(topic_sample_mode="independent")
    writer._prepare_generation_run(writer._prepare_topic_allocations(2, topics))
    writer._save_checkpoint(
        [
            {
                "messages": [
                    {"role": "system", "content": "SHIPPED_SYSTEM"},
                    {"role": "user", "content": "Existing seed"},
                    {"role": "assistant", "content": "Existing reply"},
                ]
            }
        ],
        [],
        [("topic-0", 1)],
        sample_slots=[("topic-0", 1)],
    )
    metadata = json.loads(writer._checkpoint_metadata_path.read_text())
    for key in ("topic_sample_mode", "shared_scope", "seed_field", "seeds", "collection_attempts"):
        metadata.pop(key, None)
    writer._checkpoint_metadata_path.write_text(json.dumps(metadata))
    restored = make_engine()
    assert restored.load_checkpoint()
    script(restored, {"items": ["New seed"]}, {"content": "New reply"})
    _, dataset = await run(restored, 2)
    prompt, _ = unpack(restored.llm_client.generate_async.call_args_list[0])
    assert '["Existing seed"]' in prompt and "SHIPPED_SYSTEM" not in prompt
    assert len(dataset) == 2
    assert restored._sample_slots == [("topic-0", 1), ("topic-0", 0)]


async def test_topics_complete_concurrently_but_siblings_are_serial(make_engine):
    engine = make_engine(checkpoint_interval=None)
    topics = SimpleNamespace(
        get_unique_topics=lambda: [
            SimpleNamespace(uuid=f"topic-{i}", topic=f"Topic {i}") for i in range(2)
        ],
        get_path_by_id=lambda uuid: ["PRIVATE_ROOT", uuid],
    )
    active = set()
    overlapped = asyncio.Event()
    order = {"0": [], "1": []}

    async def generate(prompt, schema, **kwargs):
        if "items" in schema.model_json_schema()["properties"]:
            topic = "0" if "topic-0" in prompt else "1"
            return schema.model_validate({"items": [f"seed-{topic}-A", f"seed-{topic}-B"]})
        topic = "0" if "seed-0" in prompt else "1"
        assert topic not in active
        active.add(topic)
        if len(active) == 2:
            overlapped.set()
        try:
            await asyncio.wait_for(overlapped.wait(), timeout=2)
            await asyncio.sleep(0)
            order[topic].append("A" if f"seed-{topic}-A" in prompt else "B")
            return schema.model_validate({"content": "reply"})
        finally:
            active.remove(topic)

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    _, dataset = await run(engine, 4, topic_model=topics, batch_size=2)
    assert len(dataset) == 4 and overlapped.is_set()
    assert order == {"0": ["A", "B"], "1": ["A", "B"]}
