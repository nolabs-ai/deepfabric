"""Bulk full-record custom schemas, exercised internally while public config is gated."""
# ruff: noqa: PLR2004, ARG001

import asyncio
import json

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from jsonschema import validate

from deepfabric.bulk import make_collection_model
from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.generator import DataSetGenerator
from deepfabric.llm.client import LLMClient
from deepfabric.sampling import topic_plan

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "price": {"type": "number", "minimum": 0},
    },
    "required": ["name", "price"],
    "additionalProperties": False,
}


def row(name):
    return {"name": name, "price": 1}


def topics(count=1):
    nodes = [SimpleNamespace(uuid=f"topic-{i}", topic=f"Topic {i}") for i in range(count)]
    return SimpleNamespace(
        get_unique_topics=lambda: nodes,
        get_path_by_id=lambda uuid: ["Root specification", uuid],
    )


@pytest.fixture
def make_engine(tmp_path):
    def make(**options):
        params = {
            "generation_system_prompt": "Generation persona",
            "dataset_system_prompt": "Dataset persona",
            "provider": "openai",
            "model_name": "test",
            "output_schema": deepcopy(SCHEMA),
            "checkpoint_interval": 10,
            "checkpoint_path": str(tmp_path),
            "output_save_as": "records.jsonl",
        }
        params.update(options)
        with patch("deepfabric.generator.LLMClient"):
            engine = DataSetGenerator(**params)
        # The truncation gate remains intact. Exercise the internal capability only.
        engine.config = engine.config.model_copy(
            update={"topic_sample_mode": "bulk", "shared_scope": "full"}
        )
        return engine

    return make


def responses(engine, *values):
    pending = iter(values)

    async def generate(prompt, schema, **kwargs):
        value = next(pending)
        if isinstance(value, Exception):
            raise value
        raw = value if isinstance(value, str) else json.dumps({"items": value})
        return schema.model_validate_json(raw)

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)


async def run(engine, target, **kwargs):
    events = [
        event
        async for event in engine.create_data_with_events_async(
            num_samples=target, topic_model=kwargs.pop("topic_model", topics()), **kwargs
        )
    ]
    return [event for event in events if isinstance(event, dict)], events[-1]


def requests(events):
    return [event for event in events if event.get("event") == "seed_request"]


def settled(events):
    return [event for event in events if event.get("event") == "batch_complete"]


async def test_full_quota_records_flatten_with_stable_slots(make_engine):
    engine = make_engine(instructions="Use concrete values", max_retries=10, max_tokens=1234)
    responses(engine, [row("A"), row("B"), row("C")])
    events, dataset = await run(engine, 3)
    assert list(dataset) == [row("A"), row("B"), row("C")]
    assert engine._sample_slots == [("topic-0", ordinal) for ordinal in range(3)]
    assert [event["sample_ordinal"] for event in settled(events)] == [0, 1, 2]
    assert requests(events)[0]["requested"] == 3
    call = engine.llm_client.generate_async.call_args
    prompt, model = call.args
    assert "Generate 3" in prompt
    assert "Dataset persona" in prompt and "Generation persona" not in prompt
    assert "Root specification -> topic-0" in prompt and "Use concrete values" in prompt
    assert model.model_json_schema()["properties"]["items"]["items"] == SCHEMA
    assert call.kwargs["max_tokens"] == 1234
    assert "max_retries" not in call.kwargs
    assert engine._collection_attempts == {"topic-0": 1}
    assert events[-1]["unaccounted"] == 0


async def test_partial_requests_shrink_and_include_accepted_siblings(make_engine):
    engine = make_engine()
    responses(engine, [row("A"), row("B")], [row("C")], [row("D"), row("E")])
    events, dataset = await run(engine, 5)
    assert [event["requested"] for event in requests(events)] == [5, 3, 2]
    replies = [event for event in events if event.get("event") == "seed_response"]
    assert [event["accepted"] for event in replies] == [2, 1, 2]
    assert [event["missing"] for event in replies] == [3, 2, 0]
    prompts = [call.args[0] for call in engine.llm_client.generate_async.call_args_list]
    assert json.dumps([row("A"), row("B")]) in prompts[1]
    assert json.dumps([row("A"), row("B"), row("C")]) in prompts[2]
    assert [record["name"] for record in dataset] == list("ABCDE")
    assert len(set(engine._sample_slots)) == 5


async def test_partial_progress_does_not_reset_budget_and_logs_have_no_content(make_engine, caplog):
    engine = make_engine(sample_retries=2, max_retries=10)
    sample_content = "PRIVATE_SAMPLE_CONTENT"
    responses(engine, [row(sample_content)], [row("B")], [row("C")])
    events, dataset = await run(engine, 5)
    assert engine.llm_client.generate_async.await_count == 3
    assert len(dataset) == 3
    assert [failure["sample_ordinal"] for failure in engine.get_all_failures()] == [3, 4]
    assert len(settled(events)) == 5
    assert sum(event["samples_generated"] for event in settled(events)) == 3
    assert sum(event["samples_failed"] for event in settled(events)) == 2
    assert "attempt 3/3" in caplog.text and "No retries remain" in caplog.text
    assert "accepted 3/5" in caplog.text and sample_content not in caplog.text
    assert events[-1]["unaccounted"] == 0
    assert all(sample_content not in json.dumps(event) for event in events)


async def test_parse_errors_and_shortfall_share_budget(make_engine):
    engine = make_engine(sample_retries=2)
    responses(engine, '{"items": [', [row("A")], '{"items": [')
    events, dataset = await run(engine, 3)
    assert [event["requested"] for event in requests(events)] == [3, 3, 2]
    assert len(dataset) == 1
    assert len(engine.get_all_failures()) == 2
    assert engine._collection_attempts == {"topic-0": 3}


@pytest.mark.parametrize("bad", ['{"items": "bad"}', "[]", "{}"])
async def test_invalid_envelopes_accept_nothing(make_engine, bad):
    engine = make_engine(sample_retries=0)
    responses(engine, bad)
    events, dataset = await run(engine, 2)
    assert len(dataset) == 0
    assert len(engine.get_all_failures()) == 2
    assert requests(events)[0]["requested"] == 2


async def test_valid_items_survive_invalid_siblings(make_engine):
    engine = make_engine()
    responses(
        engine, [row("A"), {"name": "bad"}, {"name": "negative", "price": -1}, "bad", row("B")]
    )
    events, dataset = await run(engine, 2)
    assert list(dataset) == [row("A"), row("B")]
    assert engine.llm_client.generate_async.await_count == 1
    assert len(settled(events)) == 2


async def test_excess_records_do_not_exceed_quota(make_engine):
    engine = make_engine()
    responses(engine, [row("A"), row("B"), row("C")])
    events, dataset = await run(engine, 2)
    assert list(dataset) == [row("A"), row("B")]
    assert len(settled(events)) == 2
    assert engine._sample_slots == [("topic-0", 0), ("topic-0", 1)]


async def test_non_retryable_error_fails_only_outstanding_slots(make_engine):
    engine = make_engine()
    responses(engine, [row("A")], RuntimeError("Authentication denied"))
    events, dataset = await run(engine, 3)
    assert list(dataset) == [row("A")]
    assert engine.llm_client.generate_async.await_count == 2
    assert len(engine.get_all_failures()) == 2
    assert events[-1]["unaccounted"] == 0


async def test_ordinary_resume_preserves_budget_rows_and_retry_context(make_engine):
    engine = make_engine(sample_retries=2)

    async def first_response(prompt, schema, **kwargs):
        engine.stop_requested = True
        return schema.model_validate_json(json.dumps({"items": [row("A")]}))

    engine.llm_client.generate_async = AsyncMock(side_effect=first_response)
    events, dataset = await run(engine, 4)
    assert len(dataset) == 1
    assert events[-1]["event"] == "generation_stopped"
    restored = make_engine(sample_retries=2)
    assert restored.load_checkpoint()
    responses(restored, [row("B")], [row("C")])
    events, dataset = await run(restored, 4)
    assert [event["attempt"] for event in requests(events)] == [2, 3]
    assert [event["requested"] for event in requests(events)] == [3, 2]
    assert json.dumps([row("A")]) in restored.llm_client.generate_async.call_args_list[0].args[0]
    assert [record["name"] for record in dataset] == list("ABC")
    assert [failure["sample_ordinal"] for failure in restored.get_all_failures()] == [3]
    assert restored._sample_slots == [("topic-0", ordinal) for ordinal in range(3)]


async def test_retry_failed_reopens_only_failed_slots_with_fresh_budget(make_engine):
    engine = make_engine(sample_retries=0)
    responses(engine, [row("A")])
    await run(engine, 3)
    restored = make_engine(sample_retries=0)
    assert restored.load_checkpoint(retry_failed=True)
    responses(restored, [row("B"), row("C")])
    events, dataset = await run(restored, 3)
    assert [record["name"] for record in dataset] == list("ABC")
    assert [event["sample_ordinal"] for event in settled(events)] == [1, 2]
    assert requests(events)[0]["attempt"] == 1
    assert requests(events)[0]["requested"] == 2
    assert json.dumps([row("A")]) in restored.llm_client.generate_async.call_args.args[0]
    assert restored.get_all_failures() == []


async def test_resume_exhausted_unsettled_budget_makes_no_request(make_engine):
    engine = make_engine(sample_retries=0)
    allocation_plan = topic_plan(
        2, [{"uuid": "topic-0", "topic": "Topic 0", "path": ["Root specification", "topic-0"]}]
    )
    engine._prepare_generation_run(allocation_plan)
    engine._collection_attempts = {"topic-0": 1}
    engine._save_checkpoint_metadata()
    restored = make_engine(sample_retries=0)
    assert restored.load_checkpoint()
    restored.llm_client.generate_async = AsyncMock()
    events, dataset = await run(restored, 2)
    restored.llm_client.generate_async.assert_not_called()
    assert len(dataset) == 0
    assert len(settled(events)) == 2
    assert events[-1]["unaccounted"] == 0


async def test_interrupted_inflight_request_consumes_one_attempt(make_engine):
    engine = make_engine()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def generate(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    stream = engine.create_data_with_events_async(num_samples=2, topic_model=topics())
    async for event in stream:
        if event.get("event") == "seed_request":
            await started.wait()
            break
    await stream.aclose()
    assert cancelled.is_set()
    restored = make_engine()
    assert restored.load_checkpoint()
    assert restored._collection_attempts == {"topic-0": 1}
    assert restored._completed == set()


async def test_topics_share_global_limit_and_fixed_group_barrier(make_engine):
    engine = make_engine(checkpoint_interval=None)
    active = maximum = 0
    order = []

    async def generate(prompt, schema, **kwargs):
        nonlocal active, maximum
        topic_id = prompt.split("Root specification -> ")[1].split("\n")[0]
        active += 1
        maximum = max(maximum, active)
        order.append(("start", topic_id))
        await asyncio.sleep(0.01)
        active -= 1
        order.append(("end", topic_id))
        return schema.model_validate_json(
            json.dumps({"items": [row(f"{topic_id}-{i}") for i in range(2)]})
        )

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    events, dataset = await run(engine, 6, batch_size=2, topic_model=topics(3))
    assert maximum == 2
    assert order.index(("start", "topic-2")) > order.index(("end", "topic-0"))
    assert order.index(("start", "topic-2")) > order.index(("end", "topic-1"))
    assert len(dataset) == 6
    assert len(settled(events)) == 6
    assert len(engine._completed) == 6


def test_collection_embedding_preserves_local_refs_and_original_schema():
    item_schema = {
        "type": "object",
        "$defs": {"Name": {"type": "string", "minLength": 2}},
        "properties": {"name": {"$ref": "#/$defs/Name"}},
        "required": ["name"],
    }
    original = deepcopy(item_schema)
    collection = make_collection_model(item_schema)
    validate({"items": [{"name": "Valid"}]}, collection.model_json_schema())
    assert item_schema == original
    with pytest.raises(ValueError, match="malformed JSON"):
        collection.model_validate_json('{"items": [{"name": "Valid"}')


async def test_public_full_scope_remains_gated():
    with pytest.raises(DataSetGeneratorError, match="requires the truncation fix"):
        DataSetGenerator(
            generation_system_prompt="Persona",
            provider="openai",
            model_name="test",
            output_schema=SCHEMA,
            topic_sample_mode="bulk",
            shared_scope="full",
        )


async def test_empty_responses_consume_the_same_bounded_budget(make_engine):
    engine = make_engine(sample_retries=1)
    responses(engine, [], [])
    events, dataset = await run(engine, 3)
    assert [event["requested"] for event in requests(events)] == [3, 3]
    assert engine.llm_client.generate_async.await_count == 2
    assert len(dataset) == 0
    assert len(engine.get_all_failures()) == 3


async def test_one_topic_exhausting_budget_does_not_consume_sibling_budget(make_engine):
    engine = make_engine(sample_retries=1)
    calls = {"topic-0": 0, "topic-1": 0}

    async def generate(prompt, schema, **kwargs):
        topic_id = prompt.split("Root specification -> ")[1].split("\n")[0]
        calls[topic_id] += 1
        items = [] if topic_id == "topic-0" else [row("A"), row("B"), row("C")]
        return schema.model_validate_json(json.dumps({"items": items}))

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    events, dataset = await run(engine, 6, batch_size=2, topic_model=topics(2))
    assert calls == {"topic-0": 2, "topic-1": 1}
    assert len(dataset) == 3
    assert len(engine.get_all_failures()) == 3
    assert events[-1]["unaccounted"] == 0


def test_collection_embedding_preserves_literal_reference_data():
    literal = {"$ref": "#/$defs/Example"}
    schema = {
        "type": "object",
        "properties": {"literal": {"const": literal}},
        "required": ["literal"],
    }
    collection = make_collection_model(schema)
    validate({"items": [{"literal": literal}]}, collection.model_json_schema())
    assert (
        collection.model_json_schema()["properties"]["items"]["items"]["properties"]["literal"][
            "const"
        ]
        == literal
    )


async def test_collection_round_trip_through_llm_client_and_outlines_schema(make_engine):
    from outlines.types import JsonSchema  # noqa: PLC0415

    engine = make_engine()

    def model(prompt, schema, **kwargs):
        # Exercise the actual Outlines schema conversion, with no provider network call.
        JsonSchema(schema)
        return json.dumps({"items": [row("A"), {"name": "invalid"}, row("B")]})

    with (
        patch("deepfabric.llm.client.make_outlines_model", return_value=model),
        patch("deepfabric.llm.client.make_async_outlines_model", return_value=None),
    ):
        engine.llm_client = LLMClient(provider="openai", model_name="test")
        events, dataset = await run(engine, 2)
    assert list(dataset) == [row("A"), row("B")]
    assert len(settled(events)) == 2


async def test_resume_assigns_records_to_outstanding_holes_in_ordinal_order(make_engine):
    engine = make_engine()
    allocation_plan = engine._prepare_topic_allocations(3, topics())
    engine._prepare_generation_run(allocation_plan)
    engine._save_checkpoint([row("B")], [], [("topic-0", 1)], sample_slots=[("topic-0", 1)])
    restored = make_engine()
    assert restored.load_checkpoint()
    responses(restored, [row("A"), row("C")])
    events, dataset = await run(restored, 3)
    assert [event["sample_ordinal"] for event in settled(events)] == [0, 2]
    assert restored._sample_slots == [("topic-0", 1), ("topic-0", 0), ("topic-0", 2)]
    assert [record["name"] for record in dataset] == list("BAC")
    assert requests(events)[0]["requested"] == 2


async def test_complete_checkpoint_never_regenerates_records(make_engine):
    engine = make_engine()
    responses(engine, [row("A"), row("B")])
    await run(engine, 2)
    restored = make_engine()
    assert restored.load_checkpoint()
    restored.llm_client.generate_async = AsyncMock()
    events, dataset = await run(restored, 2)
    restored.llm_client.generate_async.assert_not_called()
    assert list(dataset) == [row("A"), row("B")]
    assert settled(events) == []
    assert events[-1]["unaccounted"] == 0


async def test_schema_permitted_null_values_are_preserved(make_engine):
    schema = deepcopy(SCHEMA)
    schema["properties"]["name"] = {"type": ["string", "null"]}
    engine = make_engine(output_schema=schema)
    responses(engine, [{"name": None, "price": 1}])
    _, dataset = await run(engine, 1)
    assert list(dataset) == [{"name": None, "price": 1}]
