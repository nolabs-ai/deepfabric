"""Sequential whole-sample generation; provider and tool I/O stay mocked."""
# ruff: noqa: PLR2004, ARG001

import asyncio
import json
import re

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.generator import DataSetGenerator
from deepfabric.progress import ProgressReporter

SCHEMA = {
    "type": "object",
    "properties": {
        "seed": {"type": ["string", "null"]},
        "value": {"type": "integer", "minimum": 0},
    },
    "required": ["seed", "value"],
    "additionalProperties": False,
}


def topics(count=1):
    return SimpleNamespace(
        get_unique_topics=lambda: [
            SimpleNamespace(uuid=f"topic-{i}", topic=f"Topic {i}") for i in range(count)
        ],
        get_path_by_id=lambda uuid: ["ROOT_PATH", uuid],
    )


@pytest.fixture
def make_engine(tmp_path):
    def make(**options):
        params = {
            "generation_system_prompt": "GENERATION_PERSONA",
            "dataset_system_prompt": "OUTPUT_PERSONA",
            "instructions": "MY_INSTRUCTIONS",
            "provider": "openai",
            "model_name": "test",
            "topic_sample_mode": "sequential",
            "checkpoint_interval": 2,
            "checkpoint_path": str(tmp_path),
            "output_save_as": "sequential.jsonl",
            "sys_msg": False,
        }
        params.update(options)
        with patch("deepfabric.generator.LLMClient"):
            engine = DataSetGenerator(**params)
        engine.build_examples_text = lambda _count: "MY_EXAMPLE"
        return engine

    return make


def record(seed, cot=False):
    row = {
        "messages": [
            {"role": "user", "content": seed},
            {"role": "assistant", "content": f"ANSWER:{seed}"},
        ]
    }
    if cot:
        row["reasoning"] = {"style": "freetext", "content": "Explanation"}
    return row


def script(engine, *values):
    iterator = iter(values)

    async def generate(prompt, schema, **kwargs):
        value = next(iterator)
        if isinstance(value, BaseException):
            raise value
        return schema.model_validate_json(json.dumps(value))

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)


async def run(engine, count, **kwargs):
    events = [
        event
        async for event in engine.create_data_with_events_async(
            num_samples=count, topic_model=kwargs.pop("topic_model", topics()), **kwargs
        )
    ]
    return events[:-1], events[-1]


def prompt_of(call):
    return call.args[0] if call.args else call.kwargs["prompt"]


def history(prompt):
    match = re.search(r'<prior_successes scope="\w+">(.*?)</prior_successes>', prompt, flags=re.S)
    return json.loads(match.group(1)) if match else []


@pytest.mark.parametrize("kind", ["basic", "cot", "custom"])
@pytest.mark.parametrize("scope", ["seed", "full"])
@pytest.mark.parametrize("shipping", [False, True])
async def test_public_modes_generate_whole_samples_with_successful_history(
    make_engine, kind, scope, shipping
):
    options = {"shared_scope": scope}
    if kind == "custom":
        options.update(output_schema=SCHEMA, seed_field="seed")
        values = [{"seed": "A", "value": 1}, {"seed": None, "value": 2}, {"seed": "C", "value": 3}]
    else:
        options["conversation_type"] = kind
        values = [record(seed, kind == "cot") for seed in ["A", "B", "C"]]
    engine = make_engine(**options)
    script(engine, *values)
    events, dataset = await run(engine, 3, sys_msg=shipping)
    assert len(dataset) == 3, engine.get_all_failures()
    calls = engine.llm_client.generate_async.call_args_list
    assert len(calls) == 3
    prompts = [prompt_of(call) for call in calls]
    assert history(prompts[0]) == []
    if scope == "seed":
        assert history(prompts[1]) == ["A"]
        assert history(prompts[2]) == ["A", None if kind == "custom" else "B"]
        assert "ANSWER:A" not in prompts[1]
    else:
        prior = history(prompts[2])
        if kind == "custom":
            assert prior == [values[1], values[0]]
        else:
            assert [row["messages"][int(shipping)]["content"] for row in prior] == ["B", "A"]
    for prompt in prompts:
        for text in ("ROOT_PATH", "MY_INSTRUCTIONS", "MY_EXAMPLE", "GENERATION_PERSONA"):
            assert text in prompt
    assert "OUTPUT_PERSONA" not in prompts[0]
    if kind != "custom":
        for row, seed in zip(dataset, ["A", "B", "C"], strict=True):
            assert row["messages"][int(shipping)]["content"] == seed
            assert row["metadata"]["topic_id"] == "topic-0"
            if shipping:
                assert row["messages"][0]["content"] == "OUTPUT_PERSONA"
    metadata = json.loads(engine._checkpoint_metadata_path.read_text())
    assert "seeds" not in metadata and metadata["collection_attempts"] == {}
    assert not any(event.get("event") in ("seed_request", "seed_response") for event in events)
    assert len([event for event in events if event.get("event") == "batch_complete"]) == 3
    assert (
        next(event for event in events if event.get("event") == "generation_complete")[
            "unaccounted"
        ]
        == 0
    )


async def test_full_history_includes_all_earlier_successes_most_recent_first(make_engine):
    engine = make_engine(shared_scope="full")
    count = 12
    script(engine, *(record(f"seed-{i}") for i in range(count)))
    _, dataset = await run(engine, count)
    assert len(dataset) == count
    last = history(prompt_of(engine.llm_client.generate_async.call_args))
    assert [row["messages"][0]["content"] for row in last] == [
        f"seed-{i}" for i in range(count - 2, -1, -1)
    ]


@pytest.mark.parametrize("scope", ["seed", "full"])
async def test_failures_never_enter_history_and_budgets_are_slot_owned(make_engine, scope):
    engine = make_engine(
        output_schema=SCHEMA, seed_field="seed", shared_scope=scope, sample_retries=1, max_retries=8
    )
    bad = {"seed": "REJECTED", "value": -1}
    script(engine, {"seed": "A", "value": 1}, bad, bad, {"seed": "C", "value": 3})
    _, dataset = await run(engine, 3)
    assert len(dataset) == 2
    assert len(engine.llm_client.generate_async.call_args_list) == 4
    prompts = [prompt_of(call) for call in engine.llm_client.generate_async.call_args_list]
    assert history(prompts[1]) == history(prompts[2]) == history(prompts[3])
    assert "REJECTED" not in prompts[3]
    assert engine.get_all_failures()[0]["sample_ordinal"] == 1


@pytest.mark.parametrize("scope", ["seed", "full"])
async def test_retry_failed_excludes_later_successes_and_rebuilds_ordinal_order(make_engine, scope):
    options = {
        "output_schema": SCHEMA,
        "seed_field": "seed",
        "shared_scope": scope,
        "sample_retries": 0,
    }
    writer = make_engine(**options)
    script(
        writer,
        {"seed": "A", "value": 1},
        {"seed": "bad", "value": -1},
        {"seed": "C", "value": 3},
        {"seed": "bad", "value": -1},
    )
    await run(writer, 4)
    restored = make_engine(**options)
    assert restored.load_checkpoint(retry_failed=True)
    script(restored, {"seed": "B", "value": 2}, {"seed": "D", "value": 4})
    _, dataset = await run(restored, 4)
    assert len(dataset) == 4 and restored.get_all_failures() == []
    calls = restored.llm_client.generate_async.call_args_list
    first, second = [history(prompt_of(call)) for call in calls]
    if scope == "seed":
        assert first == ["A"] and second == ["A", "B", "C"]
    else:
        assert [row["seed"] for row in first] == ["A"]
        assert [row["seed"] for row in second] == ["C", "B", "A"]
    assert restored._sample_slots == [("topic-0", i) for i in [0, 2, 1, 3]]


@pytest.mark.parametrize("scope", ["seed", "full"])
async def test_ordinary_resume_reads_only_persisted_successes(make_engine, scope):
    engine = make_engine(shared_scope=scope)
    script(engine, record("A"), record("B"), record("C"))
    stream = engine.create_data_with_events_async(num_samples=3, topic_model=topics(), sys_msg=True)
    async for event in stream:
        if isinstance(event, dict) and event.get("event") == "batch_complete":
            break
    await stream.aclose()
    restored = make_engine(shared_scope=scope)
    assert restored.load_checkpoint()
    completed = len(restored._completed)
    assert 1 <= completed < 3
    script(restored, *(record(f"resumed-{i}") for i in range(3 - completed)))
    _, dataset = await run(restored, 3, sys_msg=True)
    assert len(dataset) == 3
    saved = history(prompt_of(restored.llm_client.generate_async.call_args_list[0]))
    assert len(saved) == completed
    if scope == "seed":
        assert saved[0] == "A" and "OUTPUT_PERSONA" not in saved
    else:
        assert saved[-1]["messages"][1]["content"] == "A"


async def test_persona_falls_back_and_full_custom_schema_needs_no_seed_field(make_engine):
    engine = make_engine(output_schema=SCHEMA, shared_scope="full", dataset_system_prompt=None)
    script(engine, {"seed": "A", "value": 1})
    _, dataset = await run(engine, 1)
    assert len(dataset) == 1
    assert "GENERATION_PERSONA" in prompt_of(engine.llm_client.generate_async.call_args)


async def test_non_validation_failure_does_not_retry_or_poison_sibling(make_engine):
    engine = make_engine(sample_retries=4)
    script(engine, RuntimeError("authentication error"), record("B"))
    _, dataset = await run(engine, 2)
    assert len(dataset) == 1
    assert len(engine.llm_client.generate_async.call_args_list) == 2
    assert history(prompt_of(engine.llm_client.generate_async.call_args)) == []


async def test_fixed_groups_and_serial_siblings_overlap_across_topics(make_engine):
    engine = make_engine(checkpoint_interval=None)
    active = set()
    barrier = asyncio.Event()
    finished = set()
    seen = {f"topic-{i}": [] for i in range(3)}

    async def generate(prompt, schema, **kwargs):
        topic = re.search(r"ROOT_PATH -> (topic-\d)", prompt).group(1)
        if topic == "topic-2":
            assert finished == {"topic-0", "topic-1"}
        assert topic not in active
        active.add(topic)
        if len(active) == 2:
            barrier.set()
        try:
            if topic != "topic-2":
                await asyncio.wait_for(barrier.wait(), timeout=2)
            await asyncio.sleep(0)
            prior = history(prompt)
            assert prior == seen[topic]
            seed = f"{topic}-seed-{len(prior)}"
            seen[topic].append(seed)
            if len(seen[topic]) == 2:
                finished.add(topic)
            return schema.model_validate(record(seed))
        finally:
            active.remove(topic)

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    events, dataset = await run(engine, 6, batch_size=2, topic_model=topics(3))
    assert len(dataset) == 6 and barrier.is_set(), engine.get_all_failures()
    assert all(len(seeds) == 2 for seeds in seen.values())
    assert len([e for e in events if e.get("event") == "topic_group_start"]) == 2


@pytest.fixture
def sessions():
    created = []

    def new(*args, **kwargs):
        session = SimpleNamespace(
            execute_tool=AsyncMock(
                return_value=SimpleNamespace(success=True, result="Observed value", error_type=None)
            ),
            cleanup=AsyncMock(),
            seed_files=AsyncMock(),
        )
        created.append(session)
        return session

    with (
        patch("deepfabric.builders_agent.SpinClient"),
        patch("deepfabric.builders_agent.SpinSession", side_effect=new),
    ):
        yield created


TOOL = {"name": "lookup", "description": "LOOKUP_CATALOG", "parameters": [], "returns": "value"}
STEP = {
    "thought": "Look it up",
    "tool_calls": [{"function_name": "lookup", "arguments": "{}", "reasoning": "Need evidence"}],
}
DONE = {"thought": "Enough evidence", "tool_calls": [], "is_final": True}


@pytest.mark.parametrize("scope", ["seed", "full"])
@pytest.mark.parametrize("style", ["agent", "freetext"])
async def test_agent_generates_new_questions_and_receives_history_in_every_stage(
    make_engine, sessions, scope, style
):
    engine = make_engine(
        shared_scope=scope,
        conversation_type="cot",
        reasoning_style=style,
        custom_tools=[TOOL],
        spin_endpoint="http://spin.invalid",
    )
    script(
        engine,
        {"content": "First generated user request"},
        STEP,
        DONE,
        {"content": "First complete assistant reply"},
        {"content": "Second generated user request"},
        STEP,
        DONE,
        {"content": "Second complete assistant reply"},
    )
    _, dataset = await run(engine, 2, sys_msg=True)
    assert len(dataset) == 2, engine.get_all_failures()
    calls = engine.llm_client.generate_async.call_args_list
    assert calls[0].kwargs["schema"].__name__ == "UserQuestion"
    assert calls[4].kwargs["schema"].__name__ == "UserQuestion"
    for call in calls[4:]:
        prompt = prompt_of(call)
        assert "First generated user request" in prompt
        for text in (
            "GENERATION_PERSONA",
            "MY_INSTRUCTIONS",
            "MY_EXAMPLE",
            "LOOKUP_CATALOG",
            "ROOT_PATH",
        ):
            assert text in prompt
        if scope == "seed":
            assert "First complete assistant reply" not in prompt
        else:
            assert "First complete assistant reply" in prompt
    assert len(sessions) == 2
    for session in sessions:
        session.execute_tool.assert_awaited_once()
        session.cleanup.assert_awaited_once()
    for row in dataset:
        assert row["messages"][0]["content"] == "OUTPUT_PERSONA"
        assert row["messages"][2]["tool_calls"][0]["id"] == row["messages"][3]["tool_call_id"]
        assert isinstance(row["reasoning"]["content"], str if style == "freetext" else list)


async def test_agent_retry_is_whole_sample_with_new_question_and_fresh_session(
    make_engine, sessions
):
    engine = make_engine(custom_tools=[TOOL], spin_endpoint="http://spin.invalid", sample_retries=1)
    script(
        engine,
        {"content": "Rejected attempt user request"},
        {"thought": ""},
        {"content": "Successful attempt user request"},
        STEP,
        DONE,
        {"content": "Successful complete assistant reply"},
        {"content": "Next sibling user request"},
        STEP,
        DONE,
        {"content": "Next sibling assistant reply"},
    )
    _, dataset = await run(engine, 2)
    assert len(dataset) == 2, engine.get_all_failures()
    calls = engine.llm_client.generate_async.call_args_list
    assert len(calls) == 10 and len(sessions) == 3
    assert calls[2].kwargs["schema"].__name__ == "UserQuestion"
    assert history(prompt_of(calls[2])) == []
    assert history(prompt_of(calls[6])) == ["Successful attempt user request"]
    assert "Rejected attempt user request" not in prompt_of(calls[6])
    for session in sessions:
        session.cleanup.assert_awaited_once()


@pytest.mark.parametrize("provider", ["openai", "openrouter", "ollama", "anthropic", "gemini"])
@pytest.mark.parametrize("custom", [False, True])
async def test_whole_sample_schemas_through_real_provider_adapters(make_engine, provider, custom):
    from jsonschema import validate  # noqa: PLC0415
    from outlines.models.openai import AsyncOpenAI  # noqa: PLC0415

    from deepfabric.llm.client import AnthropicModel, GeminiModel, LLMClient  # noqa: PLC0415
    from deepfabric.schemas import Conversation  # noqa: PLC0415

    sent = []

    def response(schema):
        sent.append(schema)
        assert "items" not in schema["properties"]
        value = (
            {"seed": "A", "value": 1}
            if custom
            else Conversation.model_validate(record("A", cot=True)).model_dump(exclude_none=False)
        )
        value = {key: item for key, item in value.items() if key in schema["properties"]}
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
    options = (
        {"output_schema": SCHEMA, "seed_field": "seed"} if custom else {"conversation_type": "cot"}
    )
    engine = make_engine(provider=provider, **options)
    with (
        patch("deepfabric.llm.client.make_outlines_model", return_value=adapter),
        patch("deepfabric.llm.client.make_async_outlines_model", return_value=adapter),
    ):
        engine.llm_client = LLMClient(
            provider=provider, model_name="test", rate_limit_config={"max_retries": 0}
        )
        _, dataset = await run(engine, 2)
    assert len(dataset) == 2, engine.get_all_failures()
    assert len(sent) == 2


async def test_cancelled_agent_attempt_cleans_session_and_resumes_from_rows(make_engine, sessions):
    engine = make_engine(
        custom_tools=[TOOL], spin_endpoint="http://spin.invalid", checkpoint_interval=1
    )
    started = asyncio.Event()
    values = iter(
        [
            {"content": "First successful user request"},
            STEP,
            DONE,
            {"content": "First complete assistant reply"},
            {"content": "Interrupted generated user request"},
        ]
    )

    async def generate(prompt, schema, **kwargs):
        value = next(values, None)
        if value is not None:
            return schema.model_validate(value)
        started.set()
        return await asyncio.Future()

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    stream = engine.create_data_with_events_async(num_samples=2, topic_model=topics())

    async def consume():
        async for _ in stream:
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(sessions) == 2
    for session in sessions:
        session.cleanup.assert_awaited_once()
    restored = make_engine(
        custom_tools=[TOOL], spin_endpoint="http://spin.invalid", checkpoint_interval=1
    )
    assert restored.load_checkpoint()
    script(
        restored,
        {"content": "New request after interruption"},
        STEP,
        DONE,
        {"content": "A successful complete assistant reply"},
    )
    _, dataset = await run(restored, 2)
    assert len(dataset) == 2
    assert dataset[1]["messages"][0]["content"] == "New request after interruption"
    assert history(prompt_of(restored.llm_client.generate_async.call_args_list[0])) == [
        "First successful user request"
    ]


async def test_checkpoint_write_failure_recovery_keeps_unique_slots_and_history(make_engine):
    engine = make_engine(checkpoint_interval=1)
    engine._save_samples_to_file = Mock()
    save = engine._save_checkpoint_metadata
    injected = False

    def fail_once():
        nonlocal injected
        if not injected and engine._flushed_samples_count:
            injected = True
            raise OSError("injected sequential checkpoint failure")
        return save()

    engine._save_checkpoint_metadata = fail_once
    script(engine, record("A"), record("B"), record("C"))
    with pytest.raises(DataSetGeneratorError):
        await run(engine, 3)
    assert injected
    restored = make_engine(checkpoint_interval=1)
    assert restored.load_checkpoint()
    completed = len(restored._completed)
    assert 1 <= completed <= 3
    script(restored, *(record(f"resumed-{i}") for i in range(3 - completed)))
    _, dataset = await run(restored, 3)
    assert len(dataset) == 3
    assert len(restored._sample_slots) == len(set(restored._sample_slots)) == 3
    if completed < 3:
        assert (
            len(history(prompt_of(restored.llm_client.generate_async.call_args_list[0])))
            == completed
        )


@pytest.mark.parametrize(
    "invalid",
    [
        {"messages": [{"role": "user", "content": "invalid"}]},
        {
            "messages": [
                {"role": "user", "content": "invalid"},
                {"role": "assistant", "content": "answer"},
            ],
            "reasoning": {"style": "agent", "content": []},
        },
    ],
)
async def test_conversation_validation_retry_has_balanced_progress_and_success_only_context(
    make_engine, invalid
):
    engine = make_engine(conversation_type="cot", sample_retries=1)
    engine.progress_reporter = Mock(spec=ProgressReporter)
    script(engine, invalid, record("A", cot=True), record("B", cot=True))
    _, dataset = await run(engine, 2)
    assert len(dataset) == 2
    calls = engine.llm_client.generate_async.call_args_list
    assert history(prompt_of(calls[0])) == history(prompt_of(calls[1])) == []
    assert history(prompt_of(calls[2])) == ["A"]
    assert "invalid" not in prompt_of(calls[2])
    reporter = engine.progress_reporter
    assert reporter.emit_step_start.call_count == reporter.emit_step_complete.call_count == 3
    reporter.emit_retry.assert_called_once()
    assert reporter.emit_retry.call_args.kwargs["sample_idx"] == 1
    assert [call.kwargs["sample_idx"] for call in reporter.emit_step_start.call_args_list] == [
        1,
        1,
        2,
    ]
