"""Regressions for independent invocation state and partial error/interruption results."""
# ruff: noqa: PLR2004, ARG001

import asyncio
import json

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from datasets import Dataset

from deepfabric.constants import ERROR_DATASET_FILENAME, INTERRUPTED_DATASET_FILENAME
from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.generator import DataSetGenerator
from deepfabric.tree import Tree


@pytest.fixture
def tree():
    with patch("deepfabric.tree.LLMClient"):
        model = Tree(topic_prompt="Root")
    model.tree_paths = [["Root", "A"], ["Root", "B"]]
    return model


@pytest.fixture
def engine(tmp_path):
    with patch("deepfabric.generator.LLMClient"):
        result = DataSetGenerator(
            generation_system_prompt="Generate",
            provider="openai",
            model_name="test",
            checkpoint_path=str(tmp_path / "checkpoints"),
            output_save_as="samples.jsonl",
        )

    async def outcome(prompt, include_sys_msg, sample_idx, topic_path, identity):
        await asyncio.sleep(0)
        return {"topic_id": identity[0], "ordinal": identity[1], "index": sample_idx}, None

    result._generate_sample_outcome = AsyncMock(side_effect=outcome)
    return result


@pytest.mark.parametrize("topic_based", [False, True])
@pytest.mark.parametrize("with_events", [False, True])
async def test_reused_async_generator_starts_independent_runs(
    engine, tree, topic_based, with_events
):
    model = tree if topic_based else None
    for target in (3, 3, 5):
        before = engine._generate_sample_outcome.await_count
        if with_events:
            outputs = [
                e
                async for e in engine.create_data_with_events_async(
                    num_samples=target, batch_size=2, topic_model=model
                )
            ]
            result = outputs[-1]
            start = outputs[0]
            complete = next(
                e
                for e in outputs
                if isinstance(e, dict) and e.get("event") == "generation_complete"
            )
            assert start["resumed_samples"] == start["resumed_failures"] == 0
            assert start["outstanding_slots"] == target
            assert not start["resumed_from_checkpoint"]
            assert complete["new_samples"] == complete["total_samples"] == target
            assert complete["unaccounted"] == complete["skipped_slots"] == 0
        else:
            result = await engine.create_data_async(
                num_samples=target, batch_size=2, topic_model=model
            )
        assert len(result) == target
        assert engine._generate_sample_outcome.await_count - before == target
        assert len(engine._completed) == target


@pytest.mark.parametrize("with_events", [False, True])
def test_reused_sync_generator_starts_independent_runs(engine, tree, with_events):
    for target in (2, 2, 3):
        before = engine._generate_sample_outcome.await_count
        if with_events:
            result = list(engine.create_data_with_events(num_samples=target, topic_model=tree))[-1]
        else:
            result = engine.create_data(num_samples=target, topic_model=tree)
        assert len(result) == target
        assert engine._generate_sample_outcome.await_count - before == target


async def test_new_run_resets_failures_analysis_and_stop_state(engine, tree):
    success = engine._generate_sample_outcome.side_effect
    first_attempt = True

    async def outcome(*args):
        nonlocal first_attempt
        if first_attempt:
            first_attempt = False
            identity = args[-1]
            engine.failure_analysis["api_errors"].append("old failure")
            return None, {
                "topic_id": identity[0],
                "sample_ordinal": identity[1],
                "error": "old failure",
            }
        return await success(*args)

    engine._generate_sample_outcome.side_effect = outcome
    assert len(await engine.create_data_async(num_samples=3, topic_model=tree)) == 2
    assert len(engine.failed_samples) == 1
    engine.stop_requested = (
        True  # A stop request from the previous run must not poison a fresh invocation.
    )
    outputs = [
        e async for e in engine.create_data_with_events_async(num_samples=2, topic_model=tree)
    ]
    assert len(outputs[-1]) == 2
    assert not engine.stop_requested
    assert not engine.failed_samples
    assert not any(engine.failure_analysis.values())
    assert outputs[0]["resumed_failures"] == 0


async def test_new_checkpoint_location_starts_fresh_without_touching_old_files(
    engine, tree, tmp_path
):
    engine.config.checkpoint_interval = 1
    await engine.create_data_async(num_samples=3, topic_model=tree)
    old_files = {p: p.read_bytes() for p in engine._get_checkpoint_paths() if p.exists()}
    engine.config.checkpoint_path = str(tmp_path / "new-checkpoints")
    result = await engine.create_data_async(num_samples=2, topic_model=tree)
    assert len(result) == 2
    assert engine._flushed_samples_count == len(engine._sample_slots) == 2
    assert all(p.read_bytes() == content for p, content in old_files.items())
    assert json.loads(engine._checkpoint_metadata_path.read_text())["plan"]["target"] == 2


async def test_existing_checkpoint_requires_load_even_on_same_engine(engine, tree):
    engine.config.checkpoint_interval = 1
    await engine.create_data_async(num_samples=3, topic_model=tree)
    before = {p: p.read_bytes() for p in engine._get_checkpoint_paths() if p.exists()}
    calls = engine._generate_sample_outcome.await_count
    with pytest.raises(DataSetGeneratorError, match="Existing checkpoint"):
        await engine.create_data_async(num_samples=2, topic_model=tree)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert engine._generate_sample_outcome.await_count == calls
    assert engine.load_checkpoint()
    with pytest.raises(DataSetGeneratorError, match="plan differs"):
        await engine.create_data_async(num_samples=2, topic_model=tree)
    result = await engine.create_data_async(num_samples=3, topic_model=tree)
    assert len(result) == 3
    assert engine._generate_sample_outcome.await_count == calls
    # Explicit load selects resume for one invocation, not all future invocations.
    with pytest.raises(DataSetGeneratorError, match="Existing checkpoint"):
        await engine.create_data_async(num_samples=3, topic_model=tree)


@pytest.mark.parametrize("change", ["location", "disabled"])
async def test_loaded_checkpoint_cannot_change_storage_before_resume(
    engine, tree, tmp_path, change
):
    engine.config.checkpoint_interval = 1
    await engine.create_data_async(num_samples=3, topic_model=tree)
    assert engine.load_checkpoint()
    original_location = engine.config.checkpoint_path
    if change == "location":
        engine.config.checkpoint_path = str(tmp_path / "elsewhere")
    else:
        engine.config.checkpoint_interval = None
    with pytest.raises(DataSetGeneratorError, match="Checkpoint location|remain enabled"):
        await engine.create_data_async(num_samples=3, topic_model=tree)
    # Correcting a rejected configuration must leave the explicit resume usable.
    engine.config.checkpoint_path = original_location
    engine.config.checkpoint_interval = 1
    assert len(await engine.create_data_async(num_samples=3, topic_model=tree)) == 3


async def test_disabling_checkpointing_and_changing_strategy_starts_fresh(engine, tree):
    engine.config.checkpoint_interval = 1
    await engine.create_data_async(num_samples=3, topic_model=tree)
    old_files = {p: p.read_bytes() for p in engine._get_checkpoint_paths() if p.exists()}
    engine.config.checkpoint_interval = None
    result = await engine.create_data_async(num_samples=2, batch_size=2)
    assert len(result) == 2
    assert engine._generation_plan == {"generation_mode": "step_based", "target": 2}
    assert engine._flushed_samples_count == 0
    assert engine._checkpoint_samples_path is None
    assert all(p.read_bytes() == content for p, content in old_files.items())


@pytest.mark.parametrize("topic_based", [False, True])
@pytest.mark.parametrize("interval", [None, 1, 10])
@pytest.mark.parametrize("interrupted", [False, True])
async def test_partial_fallback_and_lifecycle(
    engine, tree, tmp_path, monkeypatch, topic_based, interval, interrupted
):
    monkeypatch.chdir(tmp_path)
    engine.config.checkpoint_interval = interval
    # One topic/batch task makes the position of the fatal error deterministic.
    tree.tree_paths = tree.tree_paths[:1]
    failure = (
        KeyboardInterrupt("interrupted") if interrupted else RuntimeError("fatal pipeline error")
    )
    success = engine._generate_sample_outcome.side_effect

    async def outcome(*args):
        if args[-1][1] == 2:
            raise failure
        return await success(*args)

    engine._generate_sample_outcome.side_effect = outcome
    events = []

    async def consume():
        async for event in engine.create_data_with_events_async(
            num_samples=6, batch_size=1, topic_model=tree if topic_based else None
        ):
            events.append(event)

    if interrupted:
        await asyncio.wait_for(consume(), 2)
        assert isinstance(events[-1], Dataset)
        assert len(events[-1]) == 2
    else:
        with pytest.raises(DataSetGeneratorError, match="failed") as error:
            await asyncio.wait_for(consume(), 2)
        assert error.value.__cause__ is failure
    lifecycle = [e for e in events if isinstance(e, dict)]
    assert lifecycle[-1]["event"] == (
        "generation_interrupted" if interrupted else "generation_error"
    )
    assert lifecycle[-1]["total_samples"] == 2
    assert lifecycle[-1]["failed_samples"] == 0
    assert lifecycle[-1]["unaccounted"] == 4
    assert not any(e["event"] == "generation_complete" for e in lifecycle)
    filename = INTERRUPTED_DATASET_FILENAME if interrupted else ERROR_DATASET_FILENAME
    rows = [json.loads(line) for line in Path(filename).read_text().splitlines()]
    assert [row["ordinal"] for row in rows] == [0, 1]
    assert engine._generate_sample_outcome.await_count == 3
    if interval is not None:
        metadata = json.loads(engine._checkpoint_metadata_path.read_text())
        assert (
            metadata["total_samples"]
            == len(metadata["sample_slots"])
            == len(metadata["completed"])
            == 2
        )
        assert engine.load_checkpoint()
        engine._generate_sample_outcome.side_effect = success
        result = await engine.create_data_async(
            num_samples=6, batch_size=1, topic_model=tree if topic_based else None
        )
        assert len(result) == 6
        assert (
            engine._generate_sample_outcome.await_count == 7
        )  # 3 original attempts, 4 outstanding slots.


async def test_export_failure_does_not_mask_generation_cause(
    engine, tree, tmp_path, monkeypatch, caplog
):
    monkeypatch.chdir(tmp_path)
    failure = RuntimeError("primary generation failure")
    engine._generate_sample_outcome.side_effect = failure
    with (
        patch.object(engine, "_save_samples_to_file", side_effect=OSError("export failed")),
        pytest.raises(DataSetGeneratorError) as error,
    ):
        await engine.create_data_async(num_samples=1, topic_model=tree)
    assert error.value.__cause__ is failure
    assert "Unable to save partial dataset" in caplog.text


async def test_queued_success_is_preserved_when_another_worker_fails(
    engine, tree, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    uuids = [topic.uuid for topic in tree.get_unique_topics()]
    started = asyncio.Event()
    cancelled = asyncio.Event()
    failure = RuntimeError("other topic failed")

    async def topics(allocation, offset, queue, *args):
        if allocation["uuid"] == uuids[0]:
            # The error is observed first; the settled success is still waiting in the queue.
            await queue.put(("error", failure))
            await queue.put(("outcome", ((uuids[0], 0), {"ordinal": 0}, None)))
            started.set()
        else:
            try:
                await started.wait()
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    engine._generate_topic_samples_async = topics
    with pytest.raises(DataSetGeneratorError) as error:
        await asyncio.wait_for(
            engine.create_data_async(num_samples=2, batch_size=2, topic_model=tree), 2
        )
    assert error.value.__cause__ is failure
    assert cancelled.is_set()
    assert [json.loads(line) for line in Path(ERROR_DATASET_FILENAME).read_text().splitlines()] == [
        {"ordinal": 0}
    ]
