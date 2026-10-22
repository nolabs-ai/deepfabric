"""Component tests for sequential prompt warnings."""
# ruff: noqa: PLR2004, ARG001

import asyncio
import json
import logging
import re

from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from datasets import Dataset
from rich.console import Console

from deepfabric.dataset_manager import handle_dataset_events_async
from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.tui import DatasetGenerationTUI, DeepFabricTUI
from tests.support import sequential_prompt_warnings as _support

make_engine = _support.make_engine


def topics(count=1):
    return SimpleNamespace(
        get_unique_topics=lambda: [
            SimpleNamespace(uuid=f"topic-{i}", topic=f"Topic {i}") for i in range(count)
        ],
        get_path_by_id=lambda uuid: ["ROOT", uuid],
    )


def response(schema, question):
    return schema.model_validate_json(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        )
    )


async def answer(prompt, schema, **kwargs):
    return response(schema, "Q")


def warning_logs(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "deepfabric.generator" and record.levelno == logging.WARNING
    ]


@pytest.mark.parametrize("scope", ["seed", "full"])
async def test_all_scopes_warn_once_per_run_count_topics_and_keep_history(
    make_engine, monkeypatch, caplog, scope
):
    monkeypatch.setattr("deepfabric.generator.SEQUENTIAL_PROMPT_WARNING_CHARS", 1)
    engine = make_engine(shared_scope=scope, sample_retries=1)
    attempts = {"topic-0": 0, "topic-1": 0}
    histories = {"topic-0": [], "topic-1": []}

    async def generate(prompt, schema, **kwargs):
        topic = re.search(r"ROOT -> (topic-\d)", prompt).group(1)
        attempts[topic] += 1
        if attempts[topic] == 1:
            raise ValueError("validation error: injected retry")
        histories[topic].append(
            json.loads(
                re.search(
                    r'<prior_successes scope="\w+">(.*?)</prior_successes>', prompt, re.S
                ).group(1)
            )
        )
        return response(schema, f"{topic}-question-{attempts[topic]}")

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    events = [
        event
        async for event in engine.create_data_with_events_async(
            num_samples=4, topic_model=topics(2), batch_size=2
        )
    ]
    assert len(events[-1]) == 4
    warnings = [
        event
        for event in events
        if isinstance(event, dict) and event.get("event") == "sequential_prompt_warning"
    ]
    assert [event["affected_topics"] for event in warnings] == [1, 2]
    assert warnings[0]["message"] and warnings[1]["message"] is None
    assert (
        warnings[-1]["summary"]
        == "Large sequential prompts detected for 2 topics (1 warning suppressed)."
    )
    assert attempts == {"topic-0": 3, "topic-1": 3}
    assert all(len(history[1]) == 1 for history in histories.values())
    assert engine._topic_seeds == {} and engine._collection_attempts == {}
    logs = warning_logs(caplog)
    assert len(logs) == 2
    assert logs[0].startswith("Sequential prompt for topic ")
    assert logs[1] == warnings[-1]["summary"]
    final = next(
        e for e in events if isinstance(e, dict) and e.get("event") == "generation_complete"
    )
    assert final["total_samples"] == 4 and final["unaccounted"] == 0


async def test_small_prompts_do_not_warn(make_engine, caplog):
    engine = make_engine()
    engine.llm_client.generate_async = AsyncMock(side_effect=answer)
    await engine.create_data_async(num_samples=2, topic_model=topics())
    assert warning_logs(caplog) == []
    assert engine._large_sequential_prompt_topics == set()


@pytest.mark.parametrize("scope", ["seed", "full"])
async def test_default_threshold_detects_growing_history_without_trimming(
    make_engine, caplog, scope
):
    engine = make_engine(shared_scope=scope)
    histories = []

    async def generate(prompt, schema, **kwargs):
        histories.append(
            json.loads(
                re.search(
                    r'<prior_successes scope="\w+">(.*?)</prior_successes>', prompt, re.S
                ).group(1)
            )
        )
        return response(schema, "PRIVATE_SAMPLE_" + "x" * 10_000)

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    events = [
        event
        async for event in engine.create_data_with_events_async(num_samples=8, topic_model=topics())
    ]
    warnings = [
        event
        for event in events
        if isinstance(event, dict) and event.get("event") == "sequential_prompt_warning"
    ]
    assert len(warnings) == 1
    assert warnings[0]["characters"] >= 50_000
    assert 0 < warnings[0]["sample_ordinal"] < 8
    assert len(histories[-1]) == 7 and len(events[-1]) == 8
    assert "PRIVATE_SAMPLE_" not in "\n".join(warning_logs(caplog))


async def test_each_new_run_resets_affected_topic_count(make_engine, monkeypatch, caplog):
    monkeypatch.setattr("deepfabric.generator.SEQUENTIAL_PROMPT_WARNING_CHARS", 1)
    engine = make_engine()
    engine.llm_client.generate_async = AsyncMock(side_effect=answer)
    for _ in range(2):
        await engine.create_data_async(num_samples=1, topic_model=topics())
        assert engine._large_sequential_prompt_topics == {"topic-0"}
    logs = warning_logs(caplog)
    assert len(logs) == 4
    assert (
        logs[1]
        == logs[3]
        == "Large sequential prompts detected for 1 topic (0 warnings suppressed)."
    )


@pytest.mark.parametrize("ending", ["close", "cancel", "stop", "error"])
async def test_summary_on_every_interruption_path(make_engine, monkeypatch, caplog, ending):
    monkeypatch.setattr("deepfabric.generator.SEQUENTIAL_PROMPT_WARNING_CHARS", 1)
    engine = make_engine()
    engine._save_samples_to_file = Mock()
    started = asyncio.Event()

    async def generate(prompt, schema, **kwargs):
        started.set()
        if ending == "error":
            raise KeyboardInterrupt()
        if ending == "stop":
            engine.stop_requested = True
            return response(schema, "Q")
        return await asyncio.Future()

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    stream = engine.create_data_with_events_async(num_samples=2, topic_model=topics(), batch_size=1)
    if ending == "close":
        async for event in stream:
            if isinstance(event, dict) and event.get("event") == "sequential_prompt_warning":
                break
        await stream.aclose()
    elif ending == "cancel":

        async def consume():
            async for _ in stream:
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        async for _ in stream:
            pass
    logs = warning_logs(caplog)
    assert len(logs) == 2
    assert logs[-1] == "Large sequential prompts detected for 1 topic (0 warnings suppressed)."


async def test_summary_survives_worker_error(make_engine, monkeypatch, caplog):
    monkeypatch.setattr("deepfabric.generator.SEQUENTIAL_PROMPT_WARNING_CHARS", 1)
    engine = make_engine()
    engine._save_samples_to_file = Mock()
    engine._generate_sequential_sample_outcome = AsyncMock(side_effect=RuntimeError("worker error"))
    with pytest.raises(DataSetGeneratorError):
        await engine.create_data_async(num_samples=1, topic_model=topics())
    assert (
        warning_logs(caplog)[-1]
        == "Large sequential prompts detected for 1 topic (0 warnings suppressed)."
    )


@pytest.mark.parametrize("terminal", [False, True])
async def test_event_consumer_updates_live_counter_without_printing_each_topic(terminal):
    output = StringIO()
    tui = DatasetGenerationTUI(
        DeepFabricTUI(Console(file=output, width=200, force_terminal=terminal))
    )
    result = Dataset.from_list([{"question": "Q"}])
    progress = Mock()
    progress.add_task.side_effect = [0, 1]

    async def events():
        yield {"event": "generation_start", "model_name": "test", "total_samples": 1}
        for affected in [1, 2, 6]:
            yield {
                "event": "sequential_prompt_warning",
                "affected_topics": affected,
                "warnings_suppressed": affected - 1,
                "message": "First detailed warning" if affected == 1 else None,
            }
        yield {"event": "generation_complete", "total_samples": 1, "failed_samples": 0}
        yield result

    with (
        patch("deepfabric.dataset_manager.get_dataset_tui", return_value=tui),
        patch(
            "deepfabric.dataset_manager._get_tui_settings",
            return_value=SimpleNamespace(mode="simple"),
        ),
        patch("deepfabric.dataset_manager.Progress", return_value=progress),
    ):
        assert await handle_dataset_events_async(events()) is result
    assert tui.large_prompt_topics == 6
    assert list(tui.events_log).count("First detailed warning") == 1
    if terminal:
        descriptions = [call.kwargs.get("description") for call in progress.update.call_args_list]
        assert descriptions[-1] == "Generating • Large prompts: 6 topics (5 warnings suppressed)"
    else:
        progress.update.assert_not_called()
        assert "warnings suppressed" not in output.getvalue()


@pytest.mark.parametrize("ending", ["generation_interrupted", "generation_error", "cancel"])
async def test_rich_display_stops_on_interruption_before_final_summary(ending):
    output = StringIO()
    tui = DatasetGenerationTUI(DeepFabricTUI(Console(file=output, width=140)))
    live = Mock()

    async def events():
        yield {"event": "generation_start", "model_name": "test", "total_samples": 1}
        yield {
            "event": "sequential_prompt_warning",
            "affected_topics": 6,
            "warnings_suppressed": 5,
            "message": "First warning",
        }
        if ending == "cancel":
            raise asyncio.CancelledError()
        yield {"event": ending}
        live.stop.assert_called()
        assert tui.large_prompt_topics == 6

    with (
        patch("deepfabric.dataset_manager.get_dataset_tui", return_value=tui),
        patch(
            "deepfabric.dataset_manager._get_tui_settings",
            return_value=SimpleNamespace(mode="rich"),
        ),
        patch("deepfabric.dataset_manager.Live", return_value=live),
    ):
        if ending == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await handle_dataset_events_async(events())
        else:
            await handle_dataset_events_async(events())
    live.stop.assert_called()
    assert tui.live_display is None and tui.live_layout is None
