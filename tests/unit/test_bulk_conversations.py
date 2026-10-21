"""Whole-conversation bulk execution and shared-call progress, behind the full-scope gate."""
# ruff: noqa: PLR2004, ARG001

import asyncio
import json

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from datasets import Dataset as HFDataset

from deepfabric.bulk import estimate_record_tokens
from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.generator import DataSetGenerator
from deepfabric.progress import ProgressReporter
from deepfabric.schemas import Conversation


def conversation(question="Question", answer="Answer", *, cot=False):
    result = {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }
    if cot:
        result["reasoning"] = {"style": "freetext", "content": "Reasoning with concrete steps"}
    return result


def topics():
    return SimpleNamespace(
        get_unique_topics=lambda: [SimpleNamespace(uuid="topic", topic="Child")],
        get_path_by_id=lambda _uuid: ["Root requirements", "Child"],
    )


@pytest.fixture
def make_engine(tmp_path):
    def make(**options):
        params = {
            "generation_system_prompt": "Generation persona",
            "provider": "openai",
            "dataset_system_prompt": "Dataset persona",
            "model_name": "test",
            "checkpoint_interval": 10,
            "checkpoint_path": str(tmp_path),
            "output_save_as": "conversations.jsonl",
        }
        params.update(options)
        with patch("deepfabric.generator.LLMClient"):
            engine = DataSetGenerator(**params)
        # No public bypass: the separate truncation fix still gates full-scope config.
        engine.config = engine.config.model_copy(
            update={"topic_sample_mode": "bulk", "shared_scope": "full"}
        )
        return engine

    return make


def responses(engine, *values):
    values = iter(values)

    async def generate(prompt, schema, **kwargs):
        value = next(values)
        if isinstance(value, BaseException):
            raise value
        return schema.model_validate_json(
            value if isinstance(value, str) else json.dumps({"items": value})
        )

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)


async def run(engine, target, **kwargs):
    values = [
        value
        async for value in engine.create_data_with_events_async(
            num_samples=target, topic_model=kwargs.pop("topic_model", topics()), **kwargs
        )
    ]
    return [value for value in values if isinstance(value, dict)], values[-1]


def events_of(events, name):
    return [event for event in events if event.get("event") == name]


@pytest.mark.parametrize("cot", [False, True])
async def test_full_conversations_keep_reasoning_persona_and_slot_identities(make_engine, cot):
    engine = make_engine(
        conversation_type="cot" if cot else "basic", reasoning_style="freetext" if cot else None
    )
    responses(
        engine, [conversation("A", cot=cot), conversation("B", cot=cot), conversation("C", cot=cot)]
    )
    events, dataset = await run(engine, 3)
    assert len(dataset) == 3
    assert engine._sample_slots == [("topic", i) for i in range(3)]
    for record in dataset:
        sample = Conversation.model_validate(record)
        assert [message.role for message in sample.messages] == ["system", "user", "assistant"]
        assert sample.messages[0].content == "Dataset persona"
        assert sample.metadata["topic_id"] == "topic"
        if cot:
            assert sample.reasoning.style == "freetext"
    settled = events_of(events, "batch_complete")
    assert [event["sample_ordinal"] for event in settled] == [0, 1, 2]
    assert all(event["samples_generated"] == 1 for event in settled)
    assert len(events_of(events, "seed_request")) == len(events_of(events, "seed_response")) == 1
    assert events[-1]["unaccounted"] == 0
    prompt = engine.llm_client.generate_async.call_args.args[0]
    assert "Root requirements -> Child" in prompt
    assert "Dataset persona" in prompt and "Generation persona" not in prompt
    assert "Do not generate system messages" in prompt
    if cot:
        assert "reasoning trace" in prompt and "freetext" in prompt


@pytest.mark.parametrize(
    "config_value,override,expected",
    [(False, None, False), (True, False, False), (False, True, True)],
)
async def test_system_message_inclusion_respects_per_run_override(
    make_engine, config_value, override, expected
):
    engine = make_engine(sys_msg=config_value, dataset_system_prompt=None)
    responses(engine, [conversation()])
    _, dataset = await run(engine, 1, sys_msg=override)
    roles = [message["role"] for message in dataset[0]["messages"]]
    assert roles == (["system", "user", "assistant"] if expected else ["user", "assistant"])
    prompt = engine.llm_client.generate_async.call_args.args[0]
    assert "Generation persona" in prompt
    if expected:
        assert dataset[0]["messages"][0]["content"] == "Generation persona"


@pytest.mark.parametrize(
    "invalid",
    [
        {"messages": []},
        {"messages": [{"role": "user", "content": "Incomplete"}]},
        {
            "messages": [
                {"role": "system", "content": "Wrong"},
                {"role": "assistant", "content": "Reply"},
            ]
        },
        {
            "messages": [
                {"role": "assistant", "content": "Wrong start"},
                {"role": "user", "content": "Wrong end"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Request"},
                {"role": "tool", "content": "Unexpected"},
                {"role": "assistant", "content": "Reply"},
            ]
        },
        {**conversation(), "reasoning": {"style": "freetext", "content": []}},
    ],
)
async def test_invalid_conversation_does_not_discard_valid_sibling(make_engine, invalid):
    engine = make_engine(sample_retries=0)
    responses(engine, [invalid, conversation("Valid")])
    events, dataset = await run(engine, 2)
    assert len(dataset) == 1
    assert dataset[0]["messages"][1]["content"] == "Valid"
    assert [failure["sample_ordinal"] for failure in engine.get_all_failures()] == [1]
    assert [event["sample_ordinal"] for event in events_of(events, "batch_complete")] == [0, 1]
    assert events[-1]["unaccounted"] == 0


async def test_cot_requires_a_trace_of_the_configured_style(make_engine):
    engine = make_engine(conversation_type="cot", reasoning_style="freetext", sample_retries=0)
    wrong_style = {**conversation(), "reasoning": {"style": "agent", "content": []}}
    responses(engine, [conversation(), wrong_style, conversation(cot=True)])
    _, dataset = await run(engine, 2)
    assert len(dataset) == 1
    assert len(engine.get_all_failures()) == 1


async def test_generated_topic_identity_is_overridden_and_input_is_not_mutated(make_engine):
    engine = make_engine()
    generated = {**conversation(), "metadata": {"topic_id": "invented", "domain": "test"}}
    original = deepcopy(generated)
    responses(engine, [generated])
    _, dataset = await run(engine, 1)
    assert dataset[0]["metadata"] == {"topic_id": "topic", "domain": "test"}
    assert generated == original


async def test_prompt_examples_and_instructions_reach_full_collection(make_engine):
    engine = make_engine(
        instructions="Use concrete values",
        example_data=HFDataset.from_list([conversation("Example question")]),
    )
    responses(engine, [conversation()])
    await run(engine, 1, num_example_demonstrations=1)
    prompt = engine.llm_client.generate_async.call_args.args[0]
    assert "Use concrete values" in prompt and "Example question" in prompt


async def test_partial_conversations_resume_with_successful_sibling_context(make_engine):
    engine = make_engine()

    async def initial(prompt, schema, **kwargs):
        engine.stop_requested = True
        return schema.model_validate_json(json.dumps({"items": [conversation("Saved question")]}))

    engine.llm_client.generate_async = AsyncMock(side_effect=initial)
    await run(engine, 3)
    restored = make_engine()
    assert restored.load_checkpoint()
    responses(restored, [conversation("Second"), conversation("Third")])
    events, dataset = await run(restored, 3)
    assert len(dataset) == 3
    request = events_of(events, "seed_request")[0]
    assert request["requested"] == 2 and request["attempt"] == 2
    assert "Saved question" in restored.llm_client.generate_async.call_args.args[0]
    assert [event["sample_ordinal"] for event in events_of(events, "batch_complete")] == [1, 2]
    assert all(record["metadata"]["topic_id"] == "topic" for record in dataset)


async def test_progress_labels_preview_and_slot_counts_are_separate(make_engine):
    engine = make_engine()
    observer = Mock()
    reporter = ProgressReporter()
    reporter.attach(observer)
    engine.progress_reporter = reporter
    order = []

    async def generate(prompt, schema, **kwargs):
        observer.on_step_start.assert_called_once()
        assert not engine._completed
        order.append("provider")
        return schema.model_validate_json(
            json.dumps({"items": [conversation("A"), conversation("B")]})
        )

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    with patch("deepfabric.generator.simulate_stream") as preview:
        events, dataset = await run(engine, 2)
    observer.on_step_complete.assert_called_once()
    label, metadata = observer.on_step_start.call_args.args
    assert label == "Generating 2 samples for topic topic (attempt 1/3)"
    assert metadata["topic_path"] == ["Root requirements", "Child"]
    assert "sample_idx" not in metadata
    complete_label, complete_metadata = observer.on_step_complete.call_args.args
    assert complete_label == label
    assert complete_metadata["accepted"] == 2 and complete_metadata["missing"] == 0
    assert complete_metadata["success"] is True
    assert preview.call_args.kwargs["source"] == "bulk_collection"
    assert json.loads(preview.call_args.args[1])["items"][0]["messages"][0]["content"] == "A"
    assert len(dataset) == len(events_of(events, "batch_complete")) == 2
    assert order == ["provider"]
    for event in events_of(events, "seed_request") + events_of(events, "seed_response"):
        assert "samples_generated" not in event and "samples_failed" not in event


async def test_each_parse_failure_and_success_gets_one_progress_pair(make_engine):
    engine = make_engine()
    reporter = Mock()
    engine.progress_reporter = reporter
    responses(engine, '{"items": [', [conversation()])
    with patch("deepfabric.generator.simulate_stream") as preview:
        events, _ = await run(engine, 1)
    assert reporter.emit_step_start.call_count == reporter.emit_step_complete.call_count == 2
    assert reporter.emit_step_complete.call_args_list[0].kwargs["success"] is False
    assert reporter.emit_step_complete.call_args_list[1].kwargs["success"] is True
    assert preview.call_count == 1
    assert len(events_of(events, "seed_response")) == 2


async def test_cancelled_shared_request_completes_progress_without_settling_slots(make_engine):
    engine = make_engine()
    reporter = Mock()
    engine.progress_reporter = reporter
    started = asyncio.Event()

    async def generate(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    stream = engine.create_data_with_events_async(num_samples=2, topic_model=topics())
    async for event in stream:
        if event.get("event") == "seed_request":
            await started.wait()
            break
    await stream.aclose()
    assert reporter.emit_step_start.call_count == reporter.emit_step_complete.call_count == 1
    assert reporter.emit_step_complete.call_args.kwargs["success"] is False
    assert not engine._completed


@pytest.mark.parametrize(
    "custom_schema,target,budget,expected",
    [
        (None, 3, 2000, True),
        (None, 2, 2000, False),
        ({"type": "object", "properties": {}}, 20, 2000, True),
        ({"type": "object", "properties": {}}, 2, 2000, False),
    ],
)
async def test_quota_aware_token_warning_never_scales_budget(
    make_engine, caplog, custom_schema, target, budget, expected
):
    engine = make_engine(output_schema=custom_schema, max_tokens=budget)
    records = [conversation(str(i)) if custom_schema is None else {} for i in range(target)]
    responses(engine, records)
    await run(engine, target)
    assert ("may exceed max_tokens" in caplog.text) is expected
    assert engine.config.max_tokens == budget
    assert engine.llm_client.generate_async.call_args.kwargs["max_tokens"] == budget
    if expected:
        assert f"up to {target} outstanding records" in caplog.text
        assert "shared_scope: seed" in caplog.text


def test_token_estimate_accounts_for_wide_custom_schema():
    wide = {"type": "object", "properties": {f"field_{i}": {"type": "string"} for i in range(100)}}
    assert estimate_record_tokens(wide) > estimate_record_tokens({"type": "object"})


@pytest.mark.parametrize("output_schema", [None, {"type": "object"}])
def test_public_full_scope_is_still_gated(output_schema):
    with pytest.raises(DataSetGeneratorError, match="requires the truncation fix"):
        DataSetGenerator(
            generation_system_prompt="Persona",
            provider="openai",
            model_name="test",
            output_schema=output_schema,
            topic_sample_mode="bulk",
            shared_scope="full",
        )


async def test_token_warning_uses_remaining_quota_on_resume(make_engine, caplog):
    engine = make_engine()

    async def initial(prompt, schema, **kwargs):
        engine.stop_requested = True
        return schema.model_validate_json(json.dumps({"items": [conversation("Saved")]}))

    engine.llm_client.generate_async = AsyncMock(side_effect=initial)
    await run(engine, 3)
    assert "may exceed max_tokens" in caplog.text
    restored = make_engine()
    assert restored.load_checkpoint()
    caplog.clear()
    responses(restored, [conversation("Next"), conversation("Last")])
    await run(restored, 3)
    assert "may exceed max_tokens" not in caplog.text


async def test_preview_stream_reaches_observer_without_changing_slot_counts(make_engine):
    from deepfabric.stream_simulator import StreamSimulatorConfig, simulate_stream  # noqa: PLC0415

    engine = make_engine()
    observer = Mock()
    reporter = ProgressReporter()
    reporter.attach(observer)
    engine.progress_reporter = reporter
    stream_tasks = []

    def preview(reporter, content, **metadata):
        task = simulate_stream(
            reporter,
            content,
            config=StreamSimulatorConfig(chunk_size=100, chunk_delay_ms=0),
            **metadata,
        )
        stream_tasks.append(task)
        return task

    responses(engine, [conversation("Preview content")])
    with patch("deepfabric.generator.simulate_stream", side_effect=preview):
        events, dataset = await run(engine, 1)
        await asyncio.gather(*stream_tasks)
    chunks = "".join(call.args[1] for call in observer.on_stream_chunk.call_args_list)
    assert "Preview content" in chunks
    assert all(
        call.args[0] == "bulk_collection" for call in observer.on_stream_chunk.call_args_list
    )
    assert len(dataset) == len(events_of(events, "batch_complete")) == 1
    assert events[-1]["unaccounted"] == 0


def test_embedded_conversation_schema_preserves_nested_references():
    from jsonschema import validate  # noqa: PLC0415

    from deepfabric.bulk import make_collection_model  # noqa: PLC0415

    model = make_collection_model(Conversation.model_json_schema())
    validate({"items": [conversation(cot=True)]}, model.model_json_schema())


def concurrent_topics():
    return SimpleNamespace(
        get_unique_topics=lambda: [SimpleNamespace(uuid=name, topic=name) for name in ("A", "B")],
        get_path_by_id=lambda uuid: ["Root requirements", uuid],
    )


async def test_concurrent_mixed_conversations_keep_independent_budgets_and_slots(make_engine):
    from deepfabric.checkpoint import validate_checkpoint  # noqa: PLC0415

    engine = make_engine(max_tokens=6000)
    both_started = asyncio.Event()
    calls = {"A": 0, "B": 0}
    active = maximum = 0

    async def generate(prompt, schema, **kwargs):
        nonlocal active, maximum
        topic_id = prompt.split("Root requirements -> ")[1].split("\n")[0]
        calls[topic_id] += 1
        attempt = calls[topic_id]
        active += 1
        maximum = max(maximum, active)
        try:
            if attempt == 1:
                if all(calls.values()):
                    both_started.set()
                await asyncio.wait_for(both_started.wait(), 2)
                items = [conversation(f"{topic_id}-first"), {"messages": []}]
            elif topic_id == "A" and attempt == 2:
                return schema.model_validate_json('{"items": [')
            elif topic_id == "A":
                items = [conversation("A-second"), {"messages": []}]
            else:
                items = [conversation("B-second"), conversation("B-third")]
            return schema.model_validate_json(json.dumps({"items": items}))
        finally:
            active -= 1

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    events, dataset = await run(engine, 6, batch_size=2, topic_model=concurrent_topics())
    assert maximum == 2 and active == 0
    assert calls == {"A": 3, "B": 2}
    requests = events_of(events, "seed_request")
    assert [event["requested"] for event in requests if event["topic_id"] == "A"] == [3, 2, 2]
    assert [event["requested"] for event in requests if event["topic_id"] == "B"] == [3, 2]
    assert len(dataset) == 5
    assert {
        (failure["topic_id"], failure["sample_ordinal"]) for failure in engine.get_all_failures()
    } == {("A", 2)}
    assert len(events_of(events, "batch_complete")) == 6
    rows_by_slot = dict(zip(engine._sample_slots, dataset, strict=True))
    assert rows_by_slot[("A", 0)]["messages"][1]["content"] == "A-first"
    assert rows_by_slot[("A", 1)]["messages"][1]["content"] == "A-second"
    assert rows_by_slot[("B", 2)]["messages"][1]["content"] == "B-third"
    assert events[-1]["unaccounted"] == 0
    metadata = json.loads(engine._checkpoint_metadata_path.read_text())
    validate_checkpoint(metadata, engine._checkpoint_samples_path, engine._checkpoint_failures_path)


async def test_concurrent_checkpoint_metadata_failure_never_acknowledges_unwritten_state(
    make_engine,
):
    from deepfabric.checkpoint import validate_checkpoint  # noqa: PLC0415

    engine = make_engine(max_tokens=6000)
    engine._save_samples_to_file = Mock()
    both_started = asyncio.Event()
    calls = {"A": 0, "B": 0}
    acknowledgements = []
    original_state = engine._checkpoint_topic_state
    original_metadata = engine._save_checkpoint_metadata
    failed = False
    state_calls = {"A": 0, "B": 0}
    both_delivered = asyncio.Event()

    async def checkpoint_state(queue, topic_id, **state):
        state_calls[topic_id] += 1
        if state_calls[topic_id] == 2:
            if all(count == 2 for count in state_calls.values()):
                both_delivered.set()
            await asyncio.wait_for(both_delivered.wait(), 2)
        await original_state(queue, topic_id, **state)
        metadata = json.loads(engine._checkpoint_metadata_path.read_text())
        validate_checkpoint(
            metadata, engine._checkpoint_samples_path, engine._checkpoint_failures_path
        )
        acknowledgements.append((topic_id, state["attempts"]))

    def save_metadata():
        nonlocal failed
        if engine._flushed_samples_count > 0 and not failed:
            failed = True
            raise OSError("injected metadata write failure")
        return original_metadata()

    async def generate(prompt, schema, **kwargs):
        topic_id = prompt.split("Root requirements -> ")[1].split("\n")[0]
        calls[topic_id] += 1
        if all(calls.values()):
            both_started.set()
        await asyncio.wait_for(both_started.wait(), 2)
        return schema.model_validate_json(
            json.dumps({"items": [conversation(f"{topic_id}-saved"), {"messages": []}]})
        )

    engine._checkpoint_topic_state = checkpoint_state
    engine._save_checkpoint_metadata = save_metadata
    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    with pytest.raises(DataSetGeneratorError) as error:
        await run(engine, 6, batch_size=2, topic_model=concurrent_topics())
    assert isinstance(error.value.__cause__, OSError)
    assert failed
    assert calls == {"A": 1, "B": 1}
    # Only the initial, zero-row attempt reservations were acknowledged.
    assert sorted(acknowledgements) == [("A", 1), ("B", 1)]
    metadata = json.loads(engine._checkpoint_metadata_path.read_text())
    validate_checkpoint(metadata, engine._checkpoint_samples_path, engine._checkpoint_failures_path)
    saved_rows = engine._load_all_samples_from_checkpoint()
    assert len(saved_rows) == len(engine._sample_slots) == 2
    assert len(set(engine._sample_slots)) == 2
    restored = make_engine(max_tokens=6000)
    assert restored.load_checkpoint()
    assert restored._completed == {("A", 0), ("B", 0)}
    assert restored._collection_attempts == {"A": 1, "B": 1}

    async def complete_remaining(prompt, schema, **_kwargs):
        topic_id = prompt.split("Root requirements -> ")[1].split("\n")[0]
        assert f"{topic_id}-saved" in prompt
        return schema.model_validate_json(
            json.dumps(
                {
                    "items": [
                        conversation(f"{topic_id}-next"),
                        conversation(f"{topic_id}-last"),
                    ]
                }
            )
        )

    restored.llm_client.generate_async = AsyncMock(side_effect=complete_remaining)
    events, dataset = await run(restored, 6, batch_size=2, topic_model=concurrent_topics())
    assert len(dataset) == len(set(restored._sample_slots)) == 6
    assert restored.llm_client.generate_async.await_count == 2
    assert all(
        event["requested"] == 2 and event["attempt"] == 2
        for event in events_of(events, "seed_request")
    )
    assert events[-1]["unaccounted"] == 0


@pytest.mark.parametrize("fault", ["sample_append", "metadata_json", "metadata_replace"])
def test_checkpoint_write_failure_preserves_previous_files_and_pending_rows(
    make_engine, tmp_path, fault
):
    from contextlib import ExitStack  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from deepfabric.checkpoint import validate_checkpoint  # noqa: PLC0415

    engine = make_engine()
    engine._prepare_generation_run(engine._prepare_topic_allocations(3, topics()))
    first, second = conversation("Saved"), conversation("Pending")
    failure = {"topic_id": "topic", "sample_ordinal": 2, "error": "Terminal sample failure"}
    engine._save_checkpoint([first], [], [("topic", 0)], sample_slots=[("topic", 0)])
    original_metadata = engine._checkpoint_metadata_path.read_bytes()
    original_samples = engine._checkpoint_samples_path.read_bytes()
    engine._samples.append(second)
    engine.failed_samples.append(failure)
    original_open = Path.open

    def partial_json(_metadata, stream, **_kwargs):
        stream.write('{"partial":')
        raise OSError("injected partial metadata write")

    class PartialAppend:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, text):
            self.stream.write(text[:8])
            self.stream.flush()
            raise OSError("injected partial sample append")

    def open_path(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return (
            PartialAppend(stream)
            if path == engine._checkpoint_samples_path and mode == "a"
            else stream
        )

    with ExitStack() as patches:
        if fault == "sample_append":
            patches.enter_context(patch.object(Path, "open", open_path))
        elif fault == "metadata_json":
            patches.enter_context(patch("deepfabric.generator.json.dump", side_effect=partial_json))
        else:
            patches.enter_context(
                patch(
                    "deepfabric.generator.os.replace",
                    side_effect=OSError("injected metadata replace failure"),
                )
            )
        with pytest.raises(OSError, match="injected"):
            engine._save_checkpoint(
                [second], [failure], [("topic", 1), ("topic", 2)], sample_slots=[("topic", 1)]
            )
    assert engine._checkpoint_metadata_path.read_bytes() == original_metadata
    assert engine._checkpoint_samples_path.read_bytes() == original_samples
    assert (
        not engine._checkpoint_failures_path.exists()
        or engine._checkpoint_failures_path.read_bytes() == b""
    )
    assert engine._sample_slots == [("topic", 0)]
    assert engine._completed == {("topic", 0)}
    assert engine._flushed_samples_count == 1 and engine._flushed_failures_count == 0
    assert engine._samples == [second] and engine.failed_samples == [failure]
    assert not list(tmp_path.glob(".*.tmp"))
    validate_checkpoint(
        json.loads(original_metadata),
        engine._checkpoint_samples_path,
        engine._checkpoint_failures_path,
    )
    # Retrying the same coordinator buffers writes each row/failure exactly once.
    engine._save_checkpoint(
        [second], [failure], [("topic", 1), ("topic", 2)], sample_slots=[("topic", 1)]
    )
    metadata = json.loads(engine._checkpoint_metadata_path.read_text())
    validate_checkpoint(metadata, engine._checkpoint_samples_path, engine._checkpoint_failures_path)
    assert engine._load_all_samples_from_checkpoint() == [first, second]
    assert engine.get_all_failures() == [failure]
