"""Component tests for bulk conversation seeds."""
# ruff: noqa: PLR2004, ARG001

import asyncio
import json

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from deepfabric.builders import SingleShotBuilder
from deepfabric.builders_agent import SingleTurnAgentBuilder
from tests.support import bulk_conversation_seeds as _support
from tests.support.bulk_conversation_seeds import FINAL_STEP, TOOL, TOOL_STEP, script

make_engine = _support.make_engine
spin_sessions = _support.spin_sessions


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
    assert "GENERATION_PERSONA" in seed_prompt and "OUTPUT_PERSONA" not in seed_prompt
    assert seed_prompt.count("GENERATION_PERSONA") == (1 if kind == "cot" else 2)
    assert seed_prompt.count("SPECIAL_INSTRUCTIONS") == 2
    assert "Topic/Scenario:" in seed_prompt
    assert "Generate 1 user request for this topic." in seed_prompt
    assert 'an "items" list containing only the user-message text strings' in seed_prompt
    assert "Do NOT add assistant messages" in seed_prompt
    if kind == "cot":
        assert "Generate reasoning problems" in seed_prompt
        assert "REASONING FORMAT:" not in seed_prompt
        assert completion_prompt.startswith(resolved + "\n\nTopic/Scenario:")
    assert ("PRIVATE_ROOT" in completion_prompt) is path
    assert resolved in completion_prompt
    pinned_index = completion_prompt.index("<pinned_user_request>")
    assert completion_prompt.index("Here is the user request you should answer:") < pinned_index
    assert completion_prompt.index("SPECIAL_INSTRUCTIONS") < pinned_index
    assert completion_prompt.index("DEMONSTRATION_SENTINEL") < pinned_index
    if path:
        assert completion_prompt.index("PRIVATE_ROOT") < pinned_index
    if kind == "cot":
        assert pinned_index < completion_prompt.index("REASONING FORMAT:")
        assert completion_prompt.index("REASONING FORMAT:") < completion_prompt.index("IMPORTANT:")
        assert "Generate and return only the 'assistant' message" in completion_prompt
        assert completion_prompt.rstrip().endswith(
            "Do NOT include any 'system' role messages - those are added separately."
        )
    else:
        assert pinned_index < completion_prompt.index("Return only")
        assert completion_prompt.rstrip().endswith(
            "Do not reproduce a user turn or a message array."
        )
    assert "The pinned seed supplies the task context" not in completion_prompt
    assert "Use the supplied topic context and pinned seed" not in completion_prompt
    assert set(model.model_json_schema()["properties"]) == (
        {"content", "reasoning"} if kind == "cot" else {"content"}
    )
    for prompt in (seed_prompt, completion_prompt):
        assert "SPECIAL_INSTRUCTIONS" in prompt and "DEMONSTRATION_SENTINEL" in prompt


@pytest.mark.parametrize("requested", [1, 3])
def test_seed_framing_omits_unconfigured_instruction_sections(make_engine, requested):
    engine = make_engine(instructions="")
    prompt = engine._build_bulk_collection_prompt({"path": ["ROOT", "TOPIC"]}, requested, [], 0)
    assert prompt.count("GENERATION_PERSONA") == 2
    expected = (
        "Generate 1 user request for this topic."
        if requested == 1
        else f"Generate {requested} diverse user requests for this topic."
    )
    assert expected in prompt
    assert "Additional Instructions:" not in prompt
    assert "<instructions>" not in prompt
    assert "<accepted_siblings>" not in prompt


@pytest.mark.parametrize("style", ["freetext", "agent"])
@pytest.mark.parametrize("requested", [1, 3])
def test_cot_seed_history_order_and_completion_reasoning_format(make_engine, style, requested):
    engine = make_engine(conversation_type="cot", reasoning_style=style)
    allocation = {"path": ["ROOT", "TOPIC"]}
    seed = engine._build_bulk_collection_prompt(allocation, requested, ["Earlier question"], 0)
    assert seed.startswith("GENERATION_PERSONA\n\nTopic/Scenario: Generate reasoning problems")
    assert seed.index("Additional Instructions:") < seed.index("<accepted_siblings>")
    assert seed.index("</accepted_siblings>") < seed.index(f"Generate {requested} ")
    assert "REASONING FORMAT:" not in seed
    assert 'an "items" list containing only the user-message text strings' in seed
    if style == "agent":
        assert "Create realistic dialogues" not in seed
        assert (
            "Create realistic reasoning problems that require methodical reasoning to solve."
            in seed
        )
        baseline = engine.build_prompt(engine._get_prompt_template(), 0, allocation["path"])
        assert "Create realistic dialogues where complex problems are solved" in baseline
    completion = engine._conversation_completion_prompt(allocation, "Pinned question", 0)
    assert completion.startswith("OUTPUT_PERSONA\n\nTopic/Scenario:")
    assert completion.index("Pinned question") < completion.index("REASONING FORMAT:")
    assert (
        "plain string (not a list)." if style == "freetext" else "ReasoningStep objects."
    ) in completion
    independent = SingleShotBuilder(engine.llm_client, engine.config)._build_prompt("SCENARIO")
    reasoning_format = next(
        line for line in independent.splitlines() if line.startswith("REASONING FORMAT:")
    )
    assert reasoning_format in completion


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


@pytest.mark.parametrize("kind,style", [("basic", None), ("cot", "agent"), ("cot", "freetext")])
@pytest.mark.parametrize("include_path", [False, True])
async def test_agent_pins_request_and_keeps_generation_context_out_of_completion(
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
    assert "Should require using tools to answer" in unpack(calls[0])[0]
    for call in calls[1:]:
        prompt, _ = unpack(call)
        assert "CATALOG_SENTINEL" in prompt
        if unpack(call)[1].__name__ == "AgentResponse":
            assert "OUTPUT_PERSONA" in prompt
        else:
            assert "OUTPUT_PERSONA" not in prompt
        assert "SPECIAL_INSTRUCTIONS" not in prompt
        assert "DEMONSTRATION_SENTINEL" not in prompt
        assert "GENERATION_PERSONA" not in prompt
        assert "PRIVATE_ROOT" not in prompt


@pytest.mark.parametrize(
    "mode,options",
    [
        ("sequential", {"shared_scope": "seed"}),
        ("sequential", {"shared_scope": "full"}),
        ("bulk", {"completion_includes_topic_path": False}),
        ("bulk", {"completion_includes_topic_path": True}),
    ],
)
@pytest.mark.parametrize("kind", ["basic", "cot"])
async def test_agent_prompts_match_independent_except_question_collection_shape(
    make_engine, spin_sessions, mode, options, kind
):
    prompts = []
    question_prompts = []
    for sample_mode, settings in [("independent", {}), (mode, options)]:
        engine = make_engine(
            topic_sample_mode=sample_mode,
            conversation_type=kind,
            reasoning_style="agent" if kind == "cot" else None,
            output_save_as=f"{sample_mode}.jsonl",
            custom_tools=[TOOL],
            spin_endpoint="http://spin.invalid",
            **settings,
        )
        question = "Look up the current value"
        script(
            engine,
            {"items": [question]} if sample_mode == "bulk" else {"content": question},
            TOOL_STEP,
            FINAL_STEP,
            {"content": "The observed value"},
        )
        _, dataset = await run(engine)
        assert len(dataset) == 1, engine.get_all_failures()
        question_prompt, _ = unpack(engine.llm_client.generate_async.call_args_list[0])
        if sample_mode == "bulk":
            question_prompt = question_prompt.replace(
                "Generate 1 user request for this topic.",
                "Generate a short, natural user question for this scenario:",
            ).replace(
                'Generate only the user questions as an object with an "items" list of strings:',
                "Generate only the user's question:",
            )
        question_prompts.append(question_prompt)
        prompts.append(
            [
                (schema.__name__, prompt)
                for prompt, schema in map(unpack, engine.llm_client.generate_async.call_args_list)
                if schema.__name__ in ("AgentStep", "AgentResponse")
            ]
        )
    assert [name for name, _ in prompts[0]] == ["AgentStep", "AgentStep", "AgentResponse"]
    assert prompts[1] == prompts[0]
    assert question_prompts[1] == question_prompts[0]
    assert question_prompts[0].count("GENERATION_PERSONA") == (0 if kind == "cot" else 1)
    assert ("ARGUMENT REQUIREMENTS:" in question_prompts[0]) is (kind == "cot")
    assert ("You may use 1 to 3 tools" in question_prompts[0]) is (kind == "cot")
    assert "Should require using tools to answer" in question_prompts[0]


@pytest.mark.parametrize("kind", ["basic", "cot"])
@pytest.mark.parametrize("requested", [1, 4])
def test_agent_bulk_question_prompt_differs_only_in_collection_request_and_output_shape(
    make_engine, kind, requested
):
    engine = make_engine(custom_tools=[TOOL], max_tools_per_query=2, conversation_type=kind)
    path = ["ROOT", "TOPIC"]
    independent = SingleTurnAgentBuilder.build_user_question_prompt(
        engine.build_prompt(engine._get_prompt_template(), 0, path)
    )
    bulk = engine._build_bulk_collection_prompt({"path": path}, requested, [], 0)
    collection_request = (
        "Generate 1 user request for this topic."
        if requested == 1
        else "Generate 4 diverse user requests for this topic."
    )
    expected = independent.replace(
        "Generate a short, natural user question for this scenario:",
        collection_request,
    ).replace(
        "Generate only the user's question:",
        'Generate only the user questions as an object with an "items" list of strings:',
    )
    assert bulk == expected
    assert "Generate 1 short, natural user questions" not in bulk
    assert ("diverse user requests" in bulk) is (requested > 1)
    assert ("You may use 1 to 2 tools" in bulk) is (kind == "cot")
    retry = engine._build_bulk_collection_prompt({"path": path}, 1, ["Already asked"], 0)
    assert "<accepted_siblings>" in retry and "Already asked" in retry
    assert "Generate 1 user request for this topic." in retry


async def test_independent_agent_question_prompt_preserves_legacy_wrapper(make_engine):
    engine = make_engine(custom_tools=[TOOL], topic_sample_mode="independent")
    script(engine, {"content": "A complete user question"})
    builder = SingleTurnAgentBuilder(engine.llm_client, engine.config, engine.tool_registry)
    await builder._generate_user_question("SCENARIO")
    prompt, _ = unpack(engine.llm_client.generate_async.call_args)
    assert (
        prompt
        == """Generate a short, natural user question for this scenario:
SCENARIO

Requirements:
- Just the user's question - no reasoning, no explanations, no examples
- Should require using tools to answer
- 1-2 sentences maximum
- Natural, conversational tone

Example format: "Can you tell me the weather in Paris tomorrow and suggest what to wear?"

Generate only the user's question:"""
    )


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
