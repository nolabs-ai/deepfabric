"""Run reports retain request and slot identities without changing generation."""
# ruff: noqa: PLR2004

import asyncio
import json

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from deepfabric.generator import DataSetGenerator
from deepfabric.llm.client import LLMClient
from deepfabric.llm.token_usage import TokenUsageTracker, record_usage, track_token_usage
from deepfabric.run_report import RunReport, record_event, reporting, response_metrics

SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string"}, "value": {"type": "integer"}},
    "required": ["command", "value"],
    "additionalProperties": False,
}


def topics():
    return SimpleNamespace(
        get_unique_topics=lambda: [
            SimpleNamespace(uuid=f"id-{i}", topic="identical topic") for i in range(2)
        ],
        get_path_by_id=lambda uuid: ["root", f"parent-{uuid}", "identical topic"],
    )


def make_engine(tmp_path, mode, *, invalid_first=False, short_collection=False, **options):
    calls = []

    class Model:
        async def __call__(self, prompt, schema, **kwargs):
            calls.append((prompt, kwargs))
            await asyncio.sleep(0)
            record_usage("openai", "test", {"prompt_tokens": 10, "completion_tokens": 2})
            response_metrics(
                "openai", "test", "length" if invalid_first and len(calls) == 1 else "stop"
            )
            if invalid_first and len(calls) == 1:
                raise ValueError("validation error: simulated malformed response")
            props = schema.model_json_schema()["properties"]
            if "items" in props:
                # A collection of pinned command seeds, followed by separate completions.
                return json.dumps(
                    {"items": ["one"] if short_collection and len(calls) == 1 else ["one", "two"]}
                )
            if "command" not in props:
                return '{"value": 1}'
            return json.dumps({"command": f"sample-{len(calls)}", "value": 1})

    model = Model()
    with (
        patch("deepfabric.llm.client.make_outlines_model", return_value=model),
        patch("deepfabric.llm.client.make_async_outlines_model", return_value=model),
    ):
        client = LLMClient("openai", "test", rate_limit_config={"max_retries": 0})
        assert client.async_model is model
    params = {
        "generation_system_prompt": "test",
        "provider": "openai",
        "model_name": "test",
        "output_schema": SCHEMA,
        "topic_sample_mode": mode,
        "output_save_as": "samples.jsonl",
        "checkpoint_path": str(tmp_path),
        "sample_retries": 1,
    }
    if mode != "independent":
        params.update(shared_scope="seed", seed_field="command")
    params.update(options)
    with patch("deepfabric.generator.LLMClient", return_value=client):
        engine = DataSetGenerator(**params)
    return engine, calls


@pytest.mark.parametrize("mode", ["independent", "bulk", "sequential"])
async def test_reports_all_modes_with_concurrent_identical_topics(tmp_path, mode):
    engine, _calls = make_engine(tmp_path, mode)
    report, tracker = RunReport(), TokenUsageTracker()
    with reporting(report), track_token_usage(tracker):
        dataset = await engine.create_data_async(num_samples=4, batch_size=2, topic_model=topics())
    result = report.snapshot(tracker.snapshot(), "completed")
    outcomes = [e for e in result["events"] if e["event"] == "sample_outcome"]
    requests = [e for e in result["events"] if e["event"] == "request"]
    assert result["summary"]["accepted_samples"] == len(dataset) == 4
    assert result["summary"]["requests"] == (6 if mode == "bulk" else 4)
    assert result["token_usage"]["tokens"]["total_tokens"] == len(requests) * 12
    assert [e["output_row"] for e in outcomes] == [1, 2, 3, 4]
    assert {(e["topic_id"], e["sample_ordinal"]) for e in outcomes} == {
        (uuid, ordinal) for uuid in ["id-0", "id-1"] for ordinal in [0, 1]
    }
    for outcome in outcomes:
        assert outcome["topic_path"][1] == f"parent-{outcome['topic_id']}"
    request_ids = {e["request_id"] for e in requests}
    assert len(request_ids) == len(requests)
    assert {r["request_id"] for r in result["token_usage"]["records"]} == request_ids
    for request in requests:
        assert request["topic_path"][1] == f"parent-{request['topic_id']}"
        assert request["duration_seconds"] >= 0
        assert request["finish_reason"] == "stop"
        assert request["parameters"]["temperature"] == engine.config.temperature
    if mode == "bulk":
        assert sum("sample_ordinal" not in r for r in requests) == 2
    else:
        assert all("sample_ordinal" in r for r in requests)


@pytest.mark.parametrize(
    "mode,category", [("independent", "sample_validation"), ("sequential", "sample_validation")]
)
async def test_validation_retries_and_truncation_are_separate(tmp_path, mode, category):
    engine, _calls = make_engine(tmp_path, mode, invalid_first=True)
    report = RunReport()
    with reporting(report):
        dataset = await engine.create_data_async(num_samples=1, batch_size=1, topic_model=topics())
    result = report.snapshot(engine.llm_client.token_usage.snapshot(), "completed")
    assert len(dataset) == 1
    assert result["summary"]["truncated_requests"] == 1
    assert result["summary"]["retries_by_category"] == {category: 1}
    assert result["summary"]["requests"] == 2


async def test_bulk_collection_shortfall_is_reported_separately(tmp_path):
    engine, _calls = make_engine(tmp_path, "bulk", short_collection=True)
    report = RunReport()
    with reporting(report):
        dataset = await engine.create_data_async(num_samples=4, batch_size=1, topic_model=topics())
    result = report.snapshot({}, "completed")
    assert len(dataset) == 4
    assert result["summary"]["retries_by_category"] == {"bulk_collection": 1}
    assert any(
        e["event"] == "generation_event" and e["details"]["event"] == "seed_response"
        for e in result["events"]
    )


async def test_report_retains_mapping_after_checkpoint_cleanup(tmp_path):
    engine, _calls = make_engine(tmp_path, "independent", checkpoint_interval=1)
    report = RunReport()
    with reporting(report):
        await engine.create_data_async(num_samples=2, topic_model=topics())
        engine.clear_checkpoint()
    path = tmp_path / "report.json"
    report.write(path, engine.llm_client.token_usage.snapshot(), "completed")
    result = json.loads(path.read_text())
    assert result["summary"]["accepted_samples"] == 2
    assert all(e["topic_path"] for e in result["events"] if e["event"] == "sample_outcome")
    assert not list(tmp_path.glob("*.checkpoint.json"))


async def test_no_topic_requests_have_slot_identity(tmp_path):
    engine, _calls = make_engine(tmp_path, "independent")
    report = RunReport()
    with reporting(report):
        await engine.create_data_async(num_samples=2, batch_size=2)
    requests = [e for e in report.snapshot({}, "completed")["events"] if e["event"] == "request"]
    assert {e["sample_ordinal"] for e in requests} == {0, 1}
    assert all(e["topic_path"] is None for e in requests)


def test_report_context_does_not_leak_between_runs():
    first, second = RunReport(), RunReport()
    with reporting(first):
        record_event("retry", category="api")
    record_event("retry", category="api")
    with reporting(second):
        record_event("generation_result", stopped=True)
    assert first.snapshot({}, "completed")["summary"]["retries"] == 1
    result = second.snapshot({}, "completed")
    assert result["summary"]["retries"] == 0
    assert result["status"] == "stopped"


async def test_api_retry_counts_backoff_and_unknown_usage(tmp_path):
    engine, _calls = make_engine(tmp_path, "independent")
    client = engine.llm_client
    original = client._async_model
    attempts = 0

    async def flaky(prompt, schema, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("request timeout")
        return await original(prompt, schema, **kwargs)

    client._async_model = flaky
    client.retry_handler.config.max_retries = 1
    report = RunReport()
    with reporting(report), patch.object(client.retry_handler, "calculate_delay", return_value=0):
        dataset = await engine.create_data_async(num_samples=1, topic_model=topics())
    result = report.snapshot(client.token_usage.snapshot(), "completed")
    assert len(dataset) == 1
    assert result["summary"]["retries_by_category"] == {"api": 1}
    assert result["summary"]["requests"] == 2
    assert result["token_usage"]["requests_without_usage"] == 1
    assert result["token_usage"]["complete"] is False
    retry = next(e for e in result["events"] if e["event"] == "retry")
    assert retry["backoff_seconds"] == 0
    assert retry["topic_id"] == "id-0"
    assert retry["sample_ordinal"] == 0


async def test_resumed_rows_remain_distinct_from_current_work(tmp_path):
    engine, _calls = make_engine(tmp_path, "independent", checkpoint_interval=1)
    original = engine._generate_sample_outcome

    async def stop_after_one(*args, **kwargs):
        result = await original(*args, **kwargs)
        engine.stop_requested = True
        return result

    engine._generate_sample_outcome = stop_after_one
    await engine.create_data_async(num_samples=4, batch_size=1, topic_model=topics())
    assert engine._flushed_samples_count == 1
    resumed, _calls = make_engine(tmp_path, "independent", checkpoint_interval=1)
    assert resumed.load_checkpoint()
    report = RunReport()
    with reporting(report):
        dataset = await resumed.create_data_async(num_samples=4, batch_size=1, topic_model=topics())
    result = report.snapshot(resumed.llm_client.token_usage.snapshot(), "completed")
    plan = next(e for e in result["events"] if e["event"] == "generation_plan")
    assert plan["resumed_samples"] == 1
    assert plan["resumed_output_slots"] == [["id-0", 0]]
    outcomes = [e for e in result["events"] if e["event"] == "sample_outcome"]
    assert [e["output_row"] for e in outcomes] == [2, 3, 4]
    assert result["summary"]["accepted_samples"] == 3
    assert len(dataset) == 4


async def test_sync_fallback_preserves_report_context_in_thread(tmp_path):
    engine, _calls = make_engine(tmp_path, "independent")
    client = engine.llm_client

    def sync_model(_prompt, _schema, **_kwargs):
        record_usage("openai", "test", {"prompt_tokens": 2, "completion_tokens": 1})
        response_metrics("openai", "test", "stop")
        return '{"command":"read", "value":1}'

    client._async_model = None
    client.model = sync_model
    report = RunReport()
    with reporting(report):
        dataset = await engine.create_data_async(num_samples=1, topic_model=topics())
    result = report.snapshot(client.token_usage.snapshot(), "completed")
    assert len(dataset) == 1
    assert result["summary"]["requests"] == 1
    request = next(e for e in result["events"] if e["event"] == "request")
    assert request["topic_id"] == "id-0"
    assert request["sample_ordinal"] == 0
    assert request["finish_reason"] == "stop"
