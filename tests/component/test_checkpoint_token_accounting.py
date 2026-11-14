"""Observed spend survives resumes independently of durable sample progress."""
# ruff: noqa: PLR2004

import asyncio
import json

from unittest.mock import patch

import pytest

from deepfabric.checkpoint import validate_checkpoint
from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.llm.token_usage import TokenUsageTracker, record_usage, track_token_usage
from tests.component.test_run_report import make_engine, topics


async def run(engine, tracker=None, batch_size=2):
    tracker = tracker or TokenUsageTracker()
    with track_token_usage(tracker):
        data = await engine.create_data_async(
            num_samples=4, batch_size=batch_size, topic_model=topics()
        )
    return data, tracker.snapshot()


async def test_retry_failed_multiple_resumes_keep_paid_attempts_without_double_counting(tmp_path):
    engine, _ = make_engine(
        tmp_path, "independent", invalid_first=True, checkpoint_interval=1, sample_retries=0
    )
    data, first = await run(engine, batch_size=1)
    assert len(data) == 3
    assert first["cumulative"]["requests"] == 4
    identity = first["checkpoint_accounting"]["generation_id"]
    current_id = first["checkpoint_accounting"]["current_invocation_id"]
    assert first["checkpoint_accounting"]["invocations"][current_id]["usage"]["requests"] == 4
    metadata = json.loads(engine._checkpoint_metadata_path.read_text())
    assert (
        "records" not in next(iter(metadata["token_accounting"]["invocations"].values()))["usage"]
    )
    for expected in (5, 5):
        resumed, calls = make_engine(tmp_path, "independent", checkpoint_interval=1)
        assert resumed.load_checkpoint(retry_failed=True)
        data, report = await run(resumed)
        assert len(data) == 4
        assert report["cumulative"]["requests"] == expected
        assert report["cumulative"]["tokens"]["total_tokens"] == expected * 12
        assert report["checkpoint_accounting"]["generation_id"] == identity
        assert report["checkpoint_accounting"]["current_invocation_id"] != current_id
        assert len(calls) == (
            1 if expected == 5 and len(report["checkpoint_accounting"]["invocations"]) == 2 else 0
        )
        engine = resumed
    assert len(report["checkpoint_accounting"]["invocations"]) == 3
    assert len(report["records"]) == 0  # Historical request events are never reconstructed.


async def test_accounting_only_updates_leave_unflushed_samples_out_of_metadata(tmp_path):
    engine, _ = make_engine(tmp_path, "independent", checkpoint_interval=99)
    tracker = TokenUsageTracker()
    observed = []

    def inspect():
        metadata = json.loads(engine._checkpoint_metadata_path.read_text())
        observed.append(metadata)
        validate_checkpoint(
            metadata, engine._checkpoint_samples_path, engine._checkpoint_failures_path
        )

    # Subscribe after the persistence callback, before each actual record.
    original = engine._persist_accounting

    def persist(status=None):
        original(status)
        if status is None:
            inspect()

    with patch.object(engine, "_persist_accounting", side_effect=persist):
        await run(engine, tracker)
    assert len(observed) == 4
    assert all(m["total_samples"] == 0 and m["sample_slots"] == [] for m in observed)
    assert [
        next(iter(m["token_accounting"]["invocations"].values()))["usage"]["requests"]
        for m in observed
    ] == [1, 2, 3, 4]


async def test_cli_scoped_topic_usage_included_but_no_import_for_loaded_graph(tmp_path):
    tracker = TokenUsageTracker()
    with track_token_usage(tracker):
        record_usage(
            "gemini", "topic-model", {"prompt_token_count": 20, "candidates_token_count": 5}
        )
    engine, _ = make_engine(tmp_path, "independent", checkpoint_interval=1)
    _, report = await run(engine, tracker)
    assert report["cumulative"]["requests"] == 5
    assert {(g["provider"], g["model"]) for g in report["cumulative"]["by_model"]} == {
        ("gemini", "topic-model"),
        ("openai", "test"),
    }
    # A newly loaded graph is only a topic object; there is no historical usage import.
    resumed, _ = make_engine(tmp_path, "independent", checkpoint_interval=1)
    resumed.load_checkpoint()
    _, report = await run(resumed)
    assert report["requests"] == 0
    assert report["cumulative"]["requests"] == 5


async def test_legacy_checkpoint_history_unknown_not_zero(tmp_path):
    engine, _ = make_engine(tmp_path, "independent", checkpoint_interval=1)
    await run(engine)
    path = engine._checkpoint_metadata_path
    metadata = json.loads(path.read_text())
    del metadata["token_accounting"]
    path.write_text(json.dumps(metadata))
    resumed, _ = make_engine(tmp_path, "independent", checkpoint_interval=1)
    resumed.load_checkpoint()
    _, report = await run(resumed)
    assert report["cumulative"]["prior_usage_unknown"]
    assert report["cumulative"]["tokens"]["total_tokens"] is None
    assert not report["cumulative"]["complete"]


async def test_accounting_write_failure_does_not_change_sample_result(tmp_path):
    engine, _ = make_engine(tmp_path, "independent", checkpoint_interval=99)
    original = engine._write_checkpoint_metadata

    def write(metadata):
        # Fail only accounting snapshots, not initial or final sample publication.
        if (
            metadata["total_samples"] == 0
            and next(iter(metadata["token_accounting"]["invocations"].values()))["usage"][
                "requests"
            ]
        ):
            raise OSError("simulated accounting disk failure")
        return original(metadata)

    with patch.object(engine, "_write_checkpoint_metadata", side_effect=write):
        data, report = await run(engine)
    assert len(data) == 4
    assert engine._accounting_persistence_failed
    assert report["cumulative"]["tokens"]["total_tokens"] == 48


async def test_python_same_engine_without_cli_tracker_is_invocation_local(tmp_path):
    engine, calls = make_engine(tmp_path, "independent", checkpoint_interval=1)
    for _ in range(2):
        data = await engine.create_data_async(num_samples=4, batch_size=2, topic_model=topics())
        assert len(data) == 4
        assert engine._accounting.cumulative()["requests"] == 4
        engine.clear_checkpoint()
    assert len(calls) == 8


async def test_reused_outer_tracker_counts_only_new_session_records(tmp_path):
    tracker = TokenUsageTracker()
    engine, _ = make_engine(tmp_path, "independent", checkpoint_interval=1)
    await run(engine, tracker)
    resumed, _ = make_engine(tmp_path, "independent", checkpoint_interval=1)
    resumed.load_checkpoint()
    _, report = await run(resumed, tracker)
    sessions = list(report["checkpoint_accounting"]["invocations"].values())
    assert [s["usage"]["requests"] for s in sessions] == [4, 0]
    assert report["cumulative"]["requests"] == 4


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(invocations=[]),
        lambda s: s.update(version=99),
        lambda s: next(iter(s["invocations"].values())).update(sequence=-1),
        lambda s: next(iter(s["invocations"].values())).update(status="invented"),
        lambda s: next(iter(s["invocations"].values()))["usage"].update(requests=-1),
    ],
)
async def test_invalid_accounting_rejected_before_resume(tmp_path, mutation):
    engine, _ = make_engine(tmp_path, "independent", checkpoint_interval=1)
    await run(engine)
    path = engine._checkpoint_metadata_path
    metadata = json.loads(path.read_text())
    mutation(metadata["token_accounting"])
    path.write_text(json.dumps(metadata))
    resumed, _ = make_engine(tmp_path, "independent", checkpoint_interval=1)
    with pytest.raises(DataSetGeneratorError, match="Invalid checkpoint token_accounting"):
        resumed.load_checkpoint()


@pytest.mark.parametrize("mode", ["bulk", "sequential"])
async def test_other_modes_persist_all_observed_calls(tmp_path, mode):
    engine, calls = make_engine(tmp_path, mode, checkpoint_interval=2)
    data, report = await run(engine)
    assert len(data) == 4
    assert report["cumulative"]["requests"] == len(calls)
    assert report["cumulative"]["tokens"]["total_tokens"] == len(calls) * 12
    metadata = json.loads(engine._checkpoint_metadata_path.read_text())
    validate_checkpoint(metadata, engine._checkpoint_samples_path, engine._checkpoint_failures_path)


async def test_zero_success_usage_survives_failed_slot_retry(tmp_path):
    engine, _ = make_engine(tmp_path, "independent", checkpoint_interval=99, sample_retries=0)

    async def fail(*_args, **_kwargs):
        record_usage("openai", "test", {"prompt_tokens": 10, "completion_tokens": 2})
        raise ValueError("validation error: unusable output")

    with patch.object(engine.llm_client, "generate_async", side_effect=fail):
        data, report = await run(engine)
    assert len(data) == 0
    assert len(engine.get_all_failures()) == 4
    assert report["cumulative"]["requests"] == 4
    resumed, _ = make_engine(tmp_path, "independent", checkpoint_interval=1)
    resumed.load_checkpoint(retry_failed=True)
    data, report = await run(resumed)
    assert len(data) == 4
    assert report["cumulative"]["requests"] == 8
    assert report["cumulative"]["tokens"]["total_tokens"] == 96


async def test_cancellation_after_observed_response_retains_spend_before_outcome(tmp_path):
    engine, _ = make_engine(tmp_path, "independent", checkpoint_interval=99)
    observed = asyncio.Event()

    async def awaiting_settlement(*_args, **_kwargs):
        record_usage("openai", "test", {"prompt_tokens": 10, "completion_tokens": 2})
        observed.set()
        await asyncio.Event().wait()

    with patch.object(engine.llm_client, "generate_async", side_effect=awaiting_settlement):
        task = asyncio.create_task(run(engine, batch_size=1))
        await asyncio.wait_for(observed.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    metadata = json.loads(engine._checkpoint_metadata_path.read_text())
    assert metadata["total_samples"] == 0
    prior_id, prior = next(iter(metadata["token_accounting"]["invocations"].items()))
    assert prior["status"] == "unfinished"
    assert prior["usage"]["tokens"]["total_tokens"] == 12
    resumed, _ = make_engine(tmp_path, "independent", checkpoint_interval=1)
    resumed.load_checkpoint()
    data, report = await run(resumed)
    assert len(data) == 4
    assert report["cumulative"]["tokens"]["total_tokens"] == 60
    assert prior_id in report["cumulative"]["unfinished_invocations"]
    assert not report["cumulative"]["complete"]


async def test_missing_usage_and_changed_model_remain_distinct(tmp_path):
    engine, _ = make_engine(tmp_path, "independent", checkpoint_interval=1)
    await run(engine)
    tracker = TokenUsageTracker()
    with track_token_usage(tracker):
        record_usage("anthropic", "new-model", None, response_received=False)
    resumed, _ = make_engine(tmp_path, "independent", checkpoint_interval=1)
    resumed.load_checkpoint()
    _, report = await run(resumed, tracker)
    cumulative = report["cumulative"]
    assert cumulative["requests"] == 5
    assert cumulative["requests_without_usage"] == 1
    assert cumulative["tokens"]["total_tokens"] == 48
    assert cumulative["field_coverage"]["total_tokens"] == 4
    assert not cumulative["complete"]
    groups = {(g["provider"], g["model"]): g for g in cumulative["by_model"]}
    assert groups[("openai", "test")]["tokens"]["total_tokens"] == 48
    assert groups[("anthropic", "new-model")]["tokens"]["total_tokens"] is None
