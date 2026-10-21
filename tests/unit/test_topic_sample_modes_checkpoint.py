"""Additive v5 topic state and coordinator ownership of metadata-only progress."""

import asyncio
import json

from unittest.mock import AsyncMock, patch

import pytest

from deepfabric.checkpoint import validate_checkpoint
from deepfabric.constants import CHECKPOINT_VERSION
from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.generator import DataSetGenerator
from deepfabric.sampling import topic_plan


@pytest.fixture
def make_engine(tmp_path):
    def make(**options):
        with patch("deepfabric.generator.LLMClient"):
            return DataSetGenerator(
                generation_system_prompt="Persona",
                provider="openai",
                model_name="test",
                checkpoint_interval=10,
                checkpoint_path=str(tmp_path),
                output_save_as="samples.jsonl",
                **options,
            )

    return make


def plan():
    return topic_plan(3, [{"uuid": "topic", "topic": "A", "path": ["Root", "A"]}])


def save_state(engine, *, failure=False):
    engine._generation_plan = plan()
    engine._initialize_checkpoint_paths()
    engine._topic_seeds = {"topic": {"0": "accepted", "1": "pinned", "2": "pending"}}
    engine._collection_attempts = {"topic": 2}
    engine._save_checkpoint(
        [{"name": "accepted"}],
        [{"topic_id": "topic", "sample_ordinal": 1, "error": "failed"}] if failure else [],
        [("topic", 0), ("topic", 1)] if failure else [("topic", 0)],
        sample_slots=[("topic", 0)],
    )


def test_additive_metadata_round_trip(make_engine):
    writer = make_engine(topic_sample_mode="bulk")
    save_state(writer)
    metadata = json.loads(writer._checkpoint_metadata_path.read_text())
    assert metadata["version"] == CHECKPOINT_VERSION == 5  # noqa: PLR2004 - v5 remains compatible
    assert metadata["plan"] == plan()
    assert "seeds" not in metadata["plan"]
    restored = make_engine(topic_sample_mode="bulk")
    assert restored.load_checkpoint()
    assert restored._topic_seeds == writer._topic_seeds
    assert restored._collection_attempts == {"topic": 2}
    assert restored._completed == {("topic", 0)}


def test_pre_seed_v5_loads_under_bulk(make_engine, caplog):
    writer = make_engine()
    save_state(writer)
    metadata = json.loads(writer._checkpoint_metadata_path.read_text())
    for key in ("topic_sample_mode", "shared_scope", "seed_field", "seeds", "collection_attempts"):
        metadata.pop(key)
    writer._checkpoint_metadata_path.write_text(json.dumps(metadata))
    restored = make_engine(topic_sample_mode="bulk")
    assert restored.load_checkpoint()
    assert restored._completed == {("topic", 0)}
    assert restored._topic_seeds == {}
    assert restored._collection_attempts == {}
    assert "topic_sample_mode" in caplog.text


@pytest.mark.parametrize(
    "before,after,field",
    [
        ({"topic_sample_mode": "bulk"}, {}, "topic_sample_mode"),
        ({}, {"topic_sample_mode": "bulk"}, "topic_sample_mode"),
        (
            {"topic_sample_mode": "sequential"},
            {"topic_sample_mode": "sequential", "shared_scope": "full"},
            "shared_scope",
        ),
        (
            {"topic_sample_mode": "bulk", "seed_field": "name"},
            {"topic_sample_mode": "bulk", "seed_field": "other"},
            "seed_field",
        ),
    ],
)
def test_setting_changes_warn_and_discard_only_unsettled_state(
    make_engine, caplog, before, after, field
):
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "other": {"type": "string"}},
        "required": ["name", "other"],
    }
    if "seed_field" in before:
        before = {**before, "output_schema": schema}
        after = {**after, "output_schema": schema}
    writer = make_engine(**before)
    save_state(writer)
    restored = make_engine(**after)
    assert restored.load_checkpoint()
    assert restored._topic_seeds == {"topic": {"0": "accepted"}}
    assert restored._collection_attempts == {}
    assert restored._sample_slots == [("topic", 0)]
    assert restored._flushed_samples_count == 1
    assert field in caplog.text
    assert "Discarded unsettled" in caplog.text


@pytest.mark.parametrize("retry_at_load", [False, True])
def test_retry_failed_keeps_seed_but_resets_collection_budget(make_engine, retry_at_load):
    writer = make_engine(topic_sample_mode="bulk")
    save_state(writer, failure=True)
    restored = make_engine(topic_sample_mode="bulk")
    assert restored.load_checkpoint(retry_failed=retry_at_load)
    if not retry_at_load:
        assert restored._collection_attempts == {"topic": 2}
        restored.retry_checkpoint_failures()
    assert restored._topic_seeds["topic"]["1"] == "pinned"
    assert restored._collection_attempts == {}
    assert restored._completed == {("topic", 0)}
    again = make_engine(topic_sample_mode="bulk")
    assert again.load_checkpoint()
    assert again._topic_seeds == restored._topic_seeds
    assert again._collection_attempts == {}


def test_plan_changes_still_reject(make_engine):
    writer = make_engine()
    save_state(writer)
    restored = make_engine(topic_sample_mode="bulk")
    assert restored.load_checkpoint()
    with pytest.raises(DataSetGeneratorError, match="plan differs"):
        restored._prepare_generation_run(topic_plan(4, plan()["allocations"]))


async def test_metadata_only_progress_is_saved_before_worker_continues(make_engine):
    engine = make_engine(topic_sample_mode="bulk")
    allocation_plan = plan()
    engine._prepare_generation_run(allocation_plan)
    observed = []

    async def worker(_allocation, _offset, queue, *_args):
        await engine._checkpoint_topic_state(queue, "topic", attempts=1)
        metadata = json.loads(engine._checkpoint_metadata_path.read_text())
        observed.append(metadata)
        await engine._checkpoint_topic_state(queue, "topic", seeds={"0": "pinned"}, attempts=2)
        observed.append(json.loads(engine._checkpoint_metadata_path.read_text()))
        engine.stop_requested = True
        await queue.put(("done", None))

    engine._generate_topic_samples_async = worker
    events = [
        event
        async for event in engine._run_topic_based_generation_async(
            plan=allocation_plan,
            batch_size=1,
            data_creation_prompt="test",
            num_example_demonstrations=0,
            include_sys_msg=True,
        )
    ]
    assert observed[0]["collection_attempts"] == {"topic": 1}
    assert observed[0]["seeds"] == {}
    assert observed[1]["seeds"] == {"topic": {"0": "pinned"}}
    assert all(item["total_samples"] == 0 and item["completed"] == [] for item in observed)
    assert not any(
        event.get("event") == "batch_complete" for event in events if isinstance(event, dict)
    )
    stopped = next(
        event
        for event in events
        if isinstance(event, dict) and event.get("event") == "generation_stopped"
    )
    assert stopped["unaccounted"] == allocation_plan["target"]
    restored = make_engine(topic_sample_mode="bulk")
    assert restored.load_checkpoint()
    assert restored._collection_attempts == {"topic": 2}
    assert restored._topic_seeds == {"topic": {"0": "pinned"}}


async def test_state_flush_includes_pending_rows(make_engine):
    engine = make_engine()
    allocation_plan = plan()
    engine._prepare_generation_run(allocation_plan)

    async def worker(_allocation, _offset, queue, *_args):
        await queue.put(("outcome", (("topic", 0), {"name": "row"}, None)))
        await engine._checkpoint_topic_state(queue, "topic", attempts=1)
        metadata = json.loads(engine._checkpoint_metadata_path.read_text())
        validate_checkpoint(
            metadata, engine._checkpoint_samples_path, engine._checkpoint_failures_path
        )
        assert metadata["total_samples"] == 1
        engine.stop_requested = True
        await queue.put(("done", None))

    engine._generate_topic_samples_async = worker
    async for _ in engine._run_topic_based_generation_async(
        plan=allocation_plan,
        batch_size=1,
        data_creation_prompt="test",
        num_example_demonstrations=0,
        include_sys_msg=True,
    ):
        pass


@pytest.mark.parametrize("mode", ["bulk", "sequential"])
async def test_unimplemented_modes_fail_before_generation(make_engine, mode):
    engine = make_engine(topic_sample_mode=mode)
    engine._generate_sample_outcome = AsyncMock()
    with pytest.raises(DataSetGeneratorError, match="require a topic model"):
        await engine.create_data_async(num_samples=1)
    with pytest.raises(DataSetGeneratorError, match="not implemented yet"):
        await engine.create_data_async(num_samples=1, topic_model=object())
    engine._generate_sample_outcome.assert_not_called()


def test_clear_checkpoint_clears_topic_state(make_engine):
    engine = make_engine()
    save_state(engine)
    engine.clear_checkpoint()
    assert engine._topic_seeds == {}
    assert engine._collection_attempts == {}


@pytest.mark.parametrize(
    "state",
    [
        {"seeds": []},
        {"seeds": {"unknown": {"0": "seed"}}},
        {"seeds": {"topic": {"3": "seed"}}},
        {"seeds": {"topic": {"01": "seed"}}},
        {"seeds": {"topic": {"invalid": "seed"}}},
        {"collection_attempts": {"topic": -1}},
        {"collection_attempts": {"topic": True}},
        {"collection_attempts": {"unknown": 1}},
    ],
)
def test_invalid_optional_state_rejected(make_engine, state):
    engine = make_engine()
    save_state(engine)
    metadata = json.loads(engine._checkpoint_metadata_path.read_text())
    metadata.update(state)
    engine._checkpoint_metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(DataSetGeneratorError, match="Invalid checkpoint"):
        make_engine().load_checkpoint()


async def test_metadata_state_without_checkpointing(make_engine):
    engine = make_engine()
    engine.config.checkpoint_interval = None
    engine._prepare_generation_run(plan())

    async def worker(_allocation, _offset, queue, *_args):
        await engine._checkpoint_topic_state(queue, "topic", seeds={"0": "seed"}, attempts=1)
        engine.stop_requested = True
        await queue.put(("done", None))

    engine._generate_topic_samples_async = worker
    events = [
        event
        async for event in engine._run_topic_based_generation_async(
            plan=plan(),
            batch_size=1,
            data_creation_prompt="test",
            num_example_demonstrations=0,
            include_sys_msg=True,
        )
    ]
    assert engine._topic_seeds == {"topic": {"0": "seed"}}
    assert engine._collection_attempts == {"topic": 1}
    assert not any(
        event.get("event") == "checkpoint_saved" for event in events if isinstance(event, dict)
    )
    assert engine._checkpoint_metadata_path is None
    engine._prepare_generation_run(plan())
    assert engine._topic_seeds == {}
    assert engine._collection_attempts == {}


async def test_closing_event_stream_preserves_state_and_cancels_worker(make_engine):
    engine = make_engine(topic_sample_mode="bulk")
    engine._prepare_generation_run(plan())
    worker_stopped = asyncio.Event()

    async def worker(_allocation, _offset, queue, *_args):
        try:
            await engine._checkpoint_topic_state(queue, "topic", seeds={"0": "seed"}, attempts=1)
            await asyncio.Event().wait()
        finally:
            worker_stopped.set()

    engine._generate_topic_samples_async = worker
    stream = engine._run_topic_based_generation_async(
        plan=plan(),
        batch_size=1,
        data_creation_prompt="test",
        num_example_demonstrations=0,
        include_sys_msg=True,
    )
    async for event in stream:
        if event.get("event") == "checkpoint_saved":
            break
    await stream.aclose()
    assert worker_stopped.is_set()
    restored = make_engine(topic_sample_mode="bulk")
    assert restored.load_checkpoint()
    assert restored._collection_attempts == {"topic": 1}
    assert restored._topic_seeds == {"topic": {"0": "seed"}}
    assert restored._completed == set()


@pytest.mark.parametrize("mode", ["independent", "sequential"])
def test_modes_without_seed_calls_omit_seed_metadata(make_engine, mode):
    engine = make_engine(topic_sample_mode=mode)
    engine._prepare_generation_run(plan())
    engine._save_checkpoint_metadata()
    metadata = json.loads(engine._checkpoint_metadata_path.read_text())
    assert "seeds" not in metadata
    assert metadata["plan"] == plan()
