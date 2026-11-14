"""Component tests for bulk custom seeds."""
# ruff: noqa: PLR2004, ARG001

import asyncio
import json

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from datasets import Dataset as HFDataset
from jsonschema import validate

from deepfabric.config import DeepFabricConfig
from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.generator import DataSetGenerator
from deepfabric.progress import ProgressReporter
from tests.support.bulk_custom_seeds import SCHEMA


def topics(count=1):
    return SimpleNamespace(
        get_unique_topics=lambda: [
            SimpleNamespace(uuid=f"topic-{i}", topic=f"Topic {i}") for i in range(count)
        ],
        get_path_by_id=lambda uuid: ["PRIVATE_ROOT_REQUIREMENTS", uuid],
    )


@pytest.fixture
def make_engine(tmp_path):
    def make(**options):
        params = {
            "generation_system_prompt": "GENERATION_PERSONA",
            "dataset_system_prompt": "DATASET_PERSONA",
            "provider": "openai",
            "model_name": "test",
            "topic_sample_mode": "bulk",
            "output_schema": deepcopy(SCHEMA),
            "seed_field": "name",
            "checkpoint_interval": 10,
            "checkpoint_path": str(tmp_path),
            "output_save_as": "seeded.jsonl",
        }
        params.update(options)
        with patch("deepfabric.generator.LLMClient"):
            return DataSetGenerator(**params)

    return make


def script(engine, *values):
    values = iter(values)

    async def generate(prompt, schema, **kwargs):
        value = next(values)
        if isinstance(value, BaseException):
            raise value
        return schema.model_validate_json(value if isinstance(value, str) else json.dumps(value))

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


def seed_calls(engine):
    return [
        call
        for call in engine.llm_client.generate_async.call_args_list
        if "items" in call.args[1].model_json_schema().get("properties", {})
    ]


async def test_public_seed_mode_pins_values_with_narrower_completion_schema(make_engine):
    engine = make_engine(instructions="Use concrete values")
    original = deepcopy(engine.config.output_schema)
    script(engine, {"items": ["A", "B"]}, {"price": 1}, {"price": 2})
    events, dataset = await run(engine, 2)
    assert list(dataset) == [{"name": "A", "price": 1}, {"name": "B", "price": 2}]
    assert engine._sample_slots == [("topic-0", 0), ("topic-0", 1)]
    assert len(seed_calls(engine)) == 1
    calls = engine.llm_client.generate_async.call_args_list
    seed_prompt = calls[0].args[0]
    assert "PRIVATE_ROOT_REQUIREMENTS -> topic-0" in seed_prompt
    assert "GENERATION_PERSONA" in seed_prompt and "DATASET_PERSONA" not in seed_prompt
    assert "Use concrete values" in seed_prompt
    for call in calls[1:]:
        prompt, model = call.args
        schema = model.model_json_schema()
        assert "name" not in schema["properties"] and "name" not in schema["required"]
        assert "DATASET_PERSONA" in prompt and "GENERATION_PERSONA" not in prompt
        assert "PRIVATE_ROOT_REQUIREMENTS" not in prompt and "topic-0" not in prompt
        assert "Use concrete values" in prompt
    assert engine.config.output_schema == original
    assert engine._topic_seeds == {"topic-0": {"0": "A", "1": "B"}}
    assert len(events_of(events, "batch_complete")) == 2
    assert events[-1]["unaccounted"] == 0


@pytest.mark.parametrize("include_path", [False, True])
async def test_completion_topic_path_is_opt_in_and_persona_falls_back(make_engine, include_path):
    engine = make_engine(dataset_system_prompt=None, completion_includes_topic_path=include_path)
    script(engine, {"items": ["A"]}, {"price": 1})
    await run(engine, 1)
    prompt = engine.llm_client.generate_async.call_args.args[0]
    assert "GENERATION_PERSONA" in prompt
    assert ("PRIVATE_ROOT_REQUIREMENTS -> topic-0" in prompt) is include_path


@pytest.mark.parametrize(
    "value,value_schema",
    [
        ("A", {"type": "string"}),
        (7, {"type": "integer"}),
        (1.5, {"type": "number"}),
        (False, {"type": "boolean"}),
        (None, {"type": "null"}),
        (
            {"sku": "A", "variant": 2},
            {
                "type": "object",
                "properties": {"sku": {"type": "string"}, "variant": {"type": "integer"}},
                "required": ["sku", "variant"],
                "additionalProperties": False,
            },
        ),
        (["A", "B"], {"type": "array", "items": {"type": "string"}}),
    ],
)
async def test_seed_types_and_null_values_are_preserved(make_engine, value, value_schema):
    schema = deepcopy(SCHEMA)
    schema["properties"]["name"] = value_schema
    engine = make_engine(output_schema=schema)
    script(engine, {"items": [value]}, {"price": 1})
    _, dataset = await run(engine, 1)
    assert dataset[0] == {"name": value, "price": 1}
    assert engine._topic_seeds["topic-0"]["0"] == value
    assert type(engine._topic_seeds["topic-0"]["0"]) is type(value)


async def test_partial_seed_collection_finishes_before_any_completion(make_engine):
    engine = make_engine()
    script(
        engine,
        {"items": ["A", 123]},
        {"items": ["B", "C"]},
        {"price": 1},
        {"price": 2},
        {"price": 3},
    )
    events, dataset = await run(engine, 3)
    assert len(dataset) == 3
    requests = events_of(events, "seed_request")
    assert [event["requested"] for event in requests] == [3, 2]
    calls = engine.llm_client.generate_async.call_args_list
    assert all("items" in call.args[1].model_json_schema()["properties"] for call in calls[:2])
    assert all("items" not in call.args[1].model_json_schema()["properties"] for call in calls[2:])
    assert json.dumps(["A"]) in calls[1].args[0]
    assert events_of(events, "seed_response")[-1]["accepted"] == 2


async def test_collection_shortfall_preserves_seeds_and_marks_only_seedless_slots_failed(
    make_engine,
):
    engine = make_engine(max_retries=10)
    script(
        engine,
        {"items": ["A"]},
        {"items": ["B"]},
        {"items": ["C"]},
        {"price": 1},
        {"price": 2},
        {"price": 3},
    )
    events, dataset = await run(engine, 5)
    assert len(seed_calls(engine)) == 3
    assert len(dataset) == 3
    assert [
        (failure["topic_id"], failure["sample_ordinal"]) for failure in engine.get_all_failures()
    ] == [("topic-0", 3), ("topic-0", 4)]
    assert engine._collection_attempts == {"topic-0": 3}
    assert len(events_of(events, "batch_complete")) == 5
    assert events[-1]["unaccounted"] == 0


async def test_parse_failures_and_partial_seeds_share_one_budget(make_engine):
    engine = make_engine()
    script(engine, '{"items": [', {"items": ["A"]}, {"items": ["B"]}, {"price": 1}, {"price": 2})
    events, dataset = await run(engine, 3)
    assert [event["requested"] for event in events_of(events, "seed_request")] == [3, 3, 2]
    assert len(dataset) == 2
    assert len(engine.get_all_failures()) == 1


async def test_slot_retry_exhaustion_does_not_consume_sibling_or_collection_budgets(make_engine):
    engine = make_engine()
    script(
        engine,
        {"items": ["A", "B"]},
        {"price": -1},
        {"price": -1},
        {"price": -1},
        "{",
        {"price": 2},
    )
    _, dataset = await run(engine, 2)
    assert list(dataset) == [{"name": "B", "price": 2}]
    assert len(seed_calls(engine)) == 1
    assert engine.llm_client.generate_async.await_count == 6
    assert engine.get_all_failures()[0]["sample_ordinal"] == 0
    assert engine._collection_attempts == {"topic-0": 1}


async def test_completion_cannot_repeat_or_change_pinned_field(make_engine):
    engine = make_engine(sample_retries=1)
    script(
        engine, {"items": ["Pinned"]}, {"name": "Drift", "price": 1}, {"name": "Pinned", "price": 1}
    )
    _, dataset = await run(engine, 1)
    assert len(dataset) == 0
    assert len(engine.get_all_failures()) == 1
    assert engine._topic_seeds["topic-0"]["0"] == "Pinned"


async def test_assembled_record_is_validated_against_original_schema(make_engine):
    schema = deepcopy(SCHEMA)
    schema["allOf"] = [
        {
            "if": {"required": ["name"], "properties": {"name": {"const": "A"}}},
            "then": {"properties": {"price": {"minimum": 5}}},
        }
    ]
    engine = make_engine(output_schema=schema)
    script(engine, {"items": ["A"]}, {"price": 1}, {"price": 5})
    _, dataset = await run(engine, 1)
    assert list(dataset) == [{"name": "A", "price": 5}]
    assert engine.llm_client.generate_async.await_count == 3


async def test_seed_only_schema_needs_no_completion_call(make_engine):
    schema = deepcopy(SCHEMA)
    schema["properties"].pop("price")
    schema["required"] = ["name"]
    engine = make_engine(output_schema=schema)
    script(engine, {"items": ["A", "B"]})
    events, dataset = await run(engine, 2)
    assert list(dataset) == [{"name": "A"}, {"name": "B"}]
    assert engine.llm_client.generate_async.await_count == 1
    assert len(events_of(events, "batch_complete")) == 2


async def test_seed_only_checkpoint_resumes_without_recollecting(make_engine):
    engine = make_engine()

    async def seeds(prompt, schema, **kwargs):
        engine.stop_requested = True
        return schema.model_validate_json(json.dumps({"items": ["A", "B"]}))

    engine.llm_client.generate_async = AsyncMock(side_effect=seeds)
    events, dataset = await run(engine, 2)
    assert len(dataset) == 0 and not events_of(events, "batch_complete")
    metadata = json.loads(engine._checkpoint_metadata_path.read_text())
    assert metadata["seeds"] == {"topic-0": {"0": "A", "1": "B"}}
    assert metadata["total_samples"] == 0 and metadata["completed"] == []
    restored = make_engine()
    assert restored.load_checkpoint()
    script(restored, {"price": 1}, {"price": 2})
    events, dataset = await run(restored, 2)
    assert len(dataset) == 2 and not seed_calls(restored)
    assert events_of(events, "seed_request") == []
    assert restored._collection_attempts == {"topic-0": 1}


async def test_partial_seed_resume_keeps_original_collection_budget(make_engine):
    engine = make_engine()

    async def seeds(prompt, schema, **kwargs):
        engine.stop_requested = True
        return schema.model_validate_json('{"items": ["A"]}')

    engine.llm_client.generate_async = AsyncMock(side_effect=seeds)
    await run(engine, 4)
    restored = make_engine()
    assert restored.load_checkpoint()
    script(restored, {"items": ["B"]}, {"items": ["C"]}, {"price": 1}, {"price": 2}, {"price": 3})
    events, dataset = await run(restored, 4)
    assert [event["attempt"] for event in events_of(events, "seed_request")] == [2, 3]
    assert [event["requested"] for event in events_of(events, "seed_request")] == [3, 2]
    assert len(dataset) == 3
    assert engine._topic_seeds["topic-0"]["0"] == restored._topic_seeds["topic-0"]["0"] == "A"
    assert restored.get_all_failures()[0]["sample_ordinal"] == 3


async def test_retry_failed_completion_reuses_its_seed(make_engine):
    engine = make_engine(sample_retries=0)
    script(engine, {"items": ["A", "B"]}, {"price": -1}, {"price": 2})
    await run(engine, 2)
    restored = make_engine(sample_retries=0)
    assert restored.load_checkpoint(retry_failed=True)
    script(restored, {"price": 1})
    events, dataset = await run(restored, 2)
    assert not seed_calls(restored)
    assert [event["sample_ordinal"] for event in events_of(events, "batch_complete")] == [0]
    assert list(dataset) == [{"name": "B", "price": 2}, {"name": "A", "price": 1}]
    assert restored.get_all_failures() == []


async def test_retry_failed_seedless_slot_collects_only_missing_seed(make_engine):
    engine = make_engine(sample_retries=0)
    script(engine, {"items": ["A"]}, {"price": 1})
    await run(engine, 2)
    restored = make_engine(sample_retries=0)
    assert restored.load_checkpoint(retry_failed=True)
    script(restored, {"items": ["B"]}, {"price": 2})
    events, dataset = await run(restored, 2)
    assert events_of(events, "seed_request")[0]["requested"] == 1
    assert events_of(events, "seed_request")[0]["attempt"] == 1
    assert json.dumps(["A"]) in seed_calls(restored)[0].args[0]
    assert len(dataset) == 2


async def test_progress_and_persistence_keep_seeds_out_of_sample_counts(make_engine):
    engine = make_engine()
    observer = Mock()
    engine.progress_reporter = ProgressReporter()
    engine.progress_reporter.attach(observer)
    phases = []

    async def generate(prompt, schema, **kwargs):
        if "items" in schema.model_json_schema()["properties"]:
            phases.append("seeds")
            return schema.model_validate_json('{"items": ["A", "B"]}')
        metadata = json.loads(engine._checkpoint_metadata_path.read_text())
        assert metadata["seeds"] == {"topic-0": {"0": "A", "1": "B"}}
        if phases == ["seeds"]:
            assert metadata["total_samples"] == 0 and not engine._completed
        phases.append("completion")
        return schema.model_validate_json('{"price": 1}')

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    with patch("deepfabric.generator.simulate_stream"):
        events, dataset = await run(engine, 2)
    starts = observer.on_step_start.call_args_list
    assert starts[0].args[0].startswith("Generating 2 seeds for topic topic-0 (attempt 1/3;")
    assert "sample_idx" not in starts[0].args[1]
    assert [call.args[1]["sample_idx"] for call in starts[1:]] == [1, 2]
    assert phases == ["seeds", "completion", "completion"]
    assert len(dataset) == len(events_of(events, "batch_complete")) == 2


async def test_seed_mode_works_without_checkpointing(make_engine):
    engine = make_engine(checkpoint_interval=None)
    script(engine, {"items": ["A", "B"]}, {"price": 1}, {"price": 2})
    events, dataset = await run(engine, 2)
    assert len(dataset) == 2
    assert engine._topic_seeds == {"topic-0": {"0": "A", "1": "B"}}
    assert not events_of(events, "checkpoint_saved")


async def test_inflight_completion_cancellation_keeps_seed_for_resume(make_engine):
    engine = make_engine()
    started = asyncio.Event()

    async def generate(prompt, schema, **kwargs):
        if "items" in schema.model_json_schema()["properties"]:
            return schema.model_validate_json('{"items": ["A", "B"]}')
        started.set()
        await asyncio.Event().wait()
        return None

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    stream = engine.create_data_with_events_async(num_samples=2, topic_model=topics())
    async for event in stream:
        if event.get("event") == "seed_response":
            await asyncio.wait_for(started.wait(), 2)
            break
    await stream.aclose()
    restored = make_engine()
    assert restored.load_checkpoint()
    assert not restored._completed
    script(restored, {"price": 1}, {"price": 2})
    _, dataset = await run(restored, 2)
    assert len(dataset) == 2 and not seed_calls(restored)


async def test_seed_checkpoint_failure_does_not_start_completions(make_engine):
    engine = make_engine()
    engine._save_samples_to_file = Mock()
    original = engine._save_checkpoint_metadata
    failed = False

    def save():
        nonlocal failed
        if engine._topic_seeds and not failed:
            failed = True
            raise OSError("injected seed checkpoint failure")
        return original()

    engine._save_checkpoint_metadata = save
    script(engine, {"items": ["A", "B"]})
    with pytest.raises(DataSetGeneratorError):
        await run(engine, 2)
    assert failed and engine.llm_client.generate_async.await_count == 1
    restored = make_engine()
    assert restored.load_checkpoint()
    script(restored, {"price": 1}, {"price": 2})
    _, dataset = await run(restored, 2)
    assert len(dataset) == 2 and not seed_calls(restored)


async def test_seed_mode_keeps_topic_groups_and_serial_completions(make_engine):
    engine = make_engine(checkpoint_interval=None)
    active = maximum = 0
    topic_active = {f"topic-{i}": 0 for i in range(3)}
    done = set()

    async def generate(prompt, schema, **kwargs):
        nonlocal active, maximum
        if "items" in schema.model_json_schema()["properties"]:
            topic_id = prompt.split("PRIVATE_ROOT_REQUIREMENTS -> ")[1].split("\n")[0]
            if topic_id == "topic-2":
                assert done == {"topic-0", "topic-1"}
            value = {"items": [f"{topic_id}-A", f"{topic_id}-B"]}
        else:
            seed = json.loads(prompt.split("<pinned_seed>\n")[1].split("\n</pinned_seed>")[0])
            topic_id = seed.rsplit("-", 1)[0]
            value = {"price": 1}
        topic_active[topic_id] += 1
        assert topic_active[topic_id] == 1
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        topic_active[topic_id] -= 1
        if "items" not in value and seed.endswith("-B"):
            done.add(topic_id)
        return schema.model_validate_json(json.dumps(value))

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    _, dataset = await run(engine, 6, batch_size=2, topic_model=topics(3))
    assert len(dataset) == 6 and maximum == 2
    assert engine.llm_client.generate_async.await_count == 9


async def test_yaml_generation_parameters_enable_custom_seed_mode(tmp_path):
    config = DeepFabricConfig(
        topics={"prompt": "Topics"},
        generation={
            "system_prompt": "Persona",
            "topic_sample_mode": "bulk",
            "output_schema": SCHEMA,
            "seed_field": "name",
        },
        output={"save_as": "records.jsonl"},
    )
    with patch("deepfabric.generator.LLMClient"):
        engine = DataSetGenerator(**config.get_generation_params())
    script(engine, {"items": ["A"]}, {"price": 1})
    _, dataset = await run(engine, 1)
    assert list(dataset) == [{"name": "A", "price": 1}]


async def test_examples_and_configured_tool_catalog_reach_both_phases(make_engine):
    registry = SimpleNamespace(
        tools=[
            SimpleNamespace(
                name="catalog_lookup",
                description="Find concrete catalog entries",
                parameters=[
                    SimpleNamespace(
                        name="query", type="str", description="Search text", required=True
                    ),
                ],
            )
        ]
    )
    with patch.object(
        DataSetGenerator,
        "_initialize_tool_registry",
        autospec=True,
        side_effect=lambda engine: setattr(engine, "tool_registry", registry),
    ):
        engine = make_engine(
            custom_tools=[{"name": "catalog_lookup"}],
            example_data=HFDataset.from_list(
                [
                    {"name": "Example product", "price": 4},
                ]
            ),
        )
    script(engine, {"items": ["A"]}, {"price": 1})
    with patch("deepfabric.generator.ConversationBuilderFactory.create") as builder:
        await run(engine, 1, num_example_demonstrations=1)
    builder.assert_not_called()
    for call in engine.llm_client.generate_async.call_args_list:
        assert "Example product" in call.args[0]
        assert "catalog_lookup" in call.args[0]
        assert "Search text" in call.args[0]


async def test_non_retryable_seed_error_does_not_start_completions(make_engine):
    engine = make_engine()
    script(engine, RuntimeError("Provider authentication error"))
    events, dataset = await run(engine, 2)
    assert len(dataset) == 0
    assert engine.llm_client.generate_async.await_count == 1
    assert len(engine.get_all_failures()) == 2
    assert events[-1]["unaccounted"] == 0


async def test_non_retryable_completion_error_does_not_hold_sibling_budget(make_engine):
    engine = make_engine()
    script(
        engine, {"items": ["A", "B"]}, RuntimeError("Provider authentication error"), {"price": 2}
    )
    _, dataset = await run(engine, 2)
    assert list(dataset) == [{"name": "B", "price": 2}]
    assert engine.llm_client.generate_async.await_count == 3
    assert engine.get_all_failures()[0]["sample_ordinal"] == 0


async def test_excess_seeds_are_ignored_after_quota_fills(make_engine):
    engine = make_engine()
    script(engine, {"items": ["A", "B", "Excess"]}, {"price": 1}, {"price": 2})
    _, dataset = await run(engine, 2)
    assert len(dataset) == 2
    assert engine._topic_seeds == {"topic-0": {"0": "A", "1": "B"}}


@pytest.mark.parametrize("provider", ["openai", "openrouter", "ollama", "anthropic", "gemini"])
@pytest.mark.parametrize("seed_kind", ["reference-object", "nullable-string"])
async def test_seed_and_completion_schemas_through_real_provider_adapters(
    make_engine, provider, seed_kind
):
    from outlines.models.openai import AsyncOpenAI  # noqa: PLC0415

    from deepfabric.llm.client import AnthropicModel, GeminiModel, LLMClient  # noqa: PLC0415

    schema = deepcopy(SCHEMA)
    if seed_kind == "reference-object":
        value = {"sku": "A", "variant": 2}
        schema["$defs"] = {
            "Seed": {
                "type": "object",
                "properties": {"sku": {"type": "string"}, "variant": {"type": "integer"}},
                "required": ["sku", "variant"],
                "additionalProperties": False,
            }
        }
        schema["properties"]["name"] = {"$ref": "#/$defs/Seed"}
    else:
        value = None
        schema["properties"]["name"] = {"type": ["string", "null"]}
    engine = make_engine(provider=provider, output_schema=schema)
    sent = []

    def response(prepared_schema, body):
        sent.append(body)
        if "items" in prepared_schema["properties"]:
            # These are the actual, transformed schemas delivered to the adapter.
            validate({"items": [value]}, prepared_schema)
            return json.dumps({"items": [value]})
        assert "name" not in prepared_schema["properties"]
        assert "name" not in prepared_schema["required"]
        validate({"price": 1}, prepared_schema)
        return '{"price": 1}'

    async def openai_create(**body):
        raw = response(body["response_format"]["json_schema"]["schema"], body)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=raw, refusal=None))]
        )

    async def anthropic_create(**body):
        raw = response(body["output_format"]["schema"], body)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(text=raw)])

    async def gemini_generate(**body):
        raw = response(body["config"].response_json_schema, body)
        return SimpleNamespace(
            text=raw,
            candidates=[
                SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=raw)]))
            ],
        )

    if provider == "anthropic":
        sdk = SimpleNamespace(
            beta=SimpleNamespace(
                messages=SimpleNamespace(create=AsyncMock(side_effect=anthropic_create))
            )
        )
        adapter = AnthropicModel(sdk, "test")
    elif provider == "gemini":
        sdk = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=AsyncMock(side_effect=gemini_generate))
            )
        )
        adapter = GeminiModel(sdk, "test")
    else:
        sdk = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(side_effect=openai_create))
            )
        )
        adapter = AsyncOpenAI(sdk, "test")
    with (
        patch("deepfabric.llm.client.make_outlines_model", return_value=adapter),
        patch("deepfabric.llm.client.make_async_outlines_model", return_value=adapter),
    ):
        engine.llm_client = LLMClient(
            provider=provider, model_name="test", rate_limit_config={"max_retries": 0}
        )
        events, dataset = await run(engine, 1)
    assert list(dataset) == [{"name": value, "price": 1}], engine.get_all_failures()
    assert len(sent) == 2
    assert len(events_of(events, "batch_complete")) == 1
    assert engine.config.output_schema == schema


async def test_pre_seed_v5_checkpoint_resumes_with_only_outstanding_seeds(make_engine):
    writer = make_engine(topic_sample_mode="independent", seed_field=None)
    writer._prepare_generation_run(writer._prepare_topic_allocations(3, topics()))
    writer._save_checkpoint(
        [{"name": "Existing", "price": 7}], [], [("topic-0", 1)], sample_slots=[("topic-0", 1)]
    )
    metadata = json.loads(writer._checkpoint_metadata_path.read_text())
    for key in ("topic_sample_mode", "shared_scope", "seed_field", "seeds", "collection_attempts"):
        metadata.pop(key, None)
    writer._checkpoint_metadata_path.write_text(json.dumps(metadata))
    restored = make_engine()
    assert restored.load_checkpoint()
    script(restored, {"items": ["A", "C"]}, {"price": 1}, {"price": 3})
    events, dataset = await run(restored, 3)
    assert events_of(events, "seed_request")[0]["requested"] == 2
    assert json.dumps(["Existing"]) in seed_calls(restored)[0].args[0]
    assert restored._sample_slots == [("topic-0", 1), ("topic-0", 0), ("topic-0", 2)]
    assert list(dataset) == [
        {"name": "Existing", "price": 7},
        {"name": "A", "price": 1},
        {"name": "C", "price": 3},
    ]


@pytest.mark.parametrize("override,expected", [(None, [600, 400]), (900, [900, 900])])
async def test_seed_collection_and_completion_have_separate_token_budgets(
    make_engine, override, expected
):
    engine = make_engine(max_tokens=200, bulk={"max_tokens_per_call": override})
    reporter = Mock(spec=ProgressReporter)
    engine.progress_reporter = reporter
    script(
        engine, {"items": ["A"]}, {"items": ["B", "C"]}, {"price": 1}, {"price": 2}, {"price": 3}
    )
    events, dataset = await run(engine, 3)
    assert len(dataset) == 3
    calls = engine.llm_client.generate_async.call_args_list
    assert [call.kwargs["max_tokens"] for call in calls] == [*expected, 200, 200, 200]
    resolved = events_of(events, "seed_request")
    assert [event["max_tokens_per_call"] for event in resolved] == expected
    assert {event["completion_max_tokens"] for event in resolved} == {200}
    collection_steps = [
        call
        for call in reporter.emit_step_start.call_args_list
        if "max_tokens_per_call" in call.kwargs
    ]
    assert len(collection_steps) == 2
    assert "completions separately capped at 200 per call" in collection_steps[0].args[0]
    assert ("calculated as 200 x 3" in collection_steps[0].args[0]) is (override is None)
