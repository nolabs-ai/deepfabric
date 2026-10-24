"""Large-quota and many-topic resource checks with provider I/O replaced."""
# ruff: noqa: PLR2004, ARG001

import asyncio
import json
import re
import time

from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from deepfabric.checkpoint import iter_jsonl
from deepfabric.generator import DataSetGenerator

MODES = ["independent", "bulk-seed", "sequential-seed", "sequential-full", "bulk-full-internal"]
SCHEMA = {
    "type": "object",
    "properties": {"seed": {"type": "string"}, "value": {"type": "integer"}},
    "required": ["seed", "value"],
    "additionalProperties": False,
}


def topics(count):
    return SimpleNamespace(
        get_unique_topics=lambda: [
            SimpleNamespace(uuid=f"topic-{i}", topic=f"Topic {i}") for i in range(count)
        ],
        get_path_by_id=lambda uuid: ["Generate distinct short records", uuid],
    )


class ResourceProbe:
    """Count actual worker/request lifetimes without imposing timing thresholds."""

    def __init__(self, engine, concurrency, stop_after=None):
        self.engine = engine
        self.concurrency = concurrency
        self.stop_after = stop_after
        self.calls = 0
        self.requests = set()
        self.workers = set()
        self.peak_requests = 0
        self.peak_workers = 0
        self.rows_read = 0
        self.scan_calls = 0
        self.issued = Counter()
        for topic, ordinal in engine._completed:
            self.issued[topic] = max(self.issued[topic], ordinal + 1)
        for topic, seeds in engine._topic_seeds.items():
            self.issued[topic] = max(self.issued[topic], len(seeds))

    async def generate(self, prompt, schema, **_kwargs):
        topic = re.search(r"topic-\d+", prompt).group()
        assert topic not in self.requests, "Sibling provider requests overlapped"
        self.requests.add(topic)
        self.calls += 1
        self.peak_requests = max(self.peak_requests, len(self.requests))
        try:
            if self.stop_after is not None and self.calls > self.stop_after:
                await asyncio.Future()
            await asyncio.sleep(0)
            properties = schema.model_json_schema()["properties"]
            if "items" in properties:
                count = int(re.search(r"Generate (\d+) ", prompt).group(1))
                start = self.issued[topic]
                self.issued[topic] += count
                items = [f"{topic} item-{i}" for i in range(start, start + count)]
                if self.engine.config.shared_scope == "full":
                    items = [
                        {"seed": seed, "value": i} for i, seed in enumerate(items, start=start)
                    ]
                value = {"items": items}
            elif "seed" in properties:
                ordinal = self.issued[topic]
                self.issued[topic] += 1
                value = {"seed": f"{topic} item-{ordinal}", "value": ordinal}
            else:
                ordinal = int(re.search(r"topic-\d+ item-(\d+)", prompt).group(1))
                value = {"value": ordinal}
            return schema.model_validate_json(json.dumps(value))
        finally:
            self.requests.remove(topic)

    async def worker(self, original, allocation, *args, **kwargs):
        index = int(allocation["uuid"].removeprefix("topic-"))
        assert all(i // self.concurrency == index // self.concurrency for i in self.workers)
        self.workers.add(index)
        self.peak_workers = max(self.peak_workers, len(self.workers))
        try:
            return await original(allocation, *args, **kwargs)
        finally:
            self.workers.remove(index)

    def read(self, *args, **kwargs):
        self.scan_calls += 1
        for row in iter_jsonl(*args, **kwargs):
            self.rows_read += 1
            yield row

    def attach(self):
        original = self.engine._generate_topic_samples_async

        async def worker(allocation, *args, **kwargs):
            return await self.worker(original, allocation, *args, **kwargs)

        self.engine._generate_topic_samples_async = worker
        self.engine.llm_client.generate_async = self.generate

    def report(self, record_property, seconds):
        for name, value in {
            "provider_requests": self.calls,
            "peak_requests": self.peak_requests,
            "peak_workers": self.peak_workers,
            "checkpoint_scans": self.scan_calls,
            "checkpoint_rows_read": self.rows_read,
            "elapsed_seconds": seconds,
        }.items():
            record_property(name, value)


def engine_for(tmp_path, mode, checkpoint):
    options = {
        "generation_system_prompt": "Generate compact, distinct records.",
        "provider": "openai",
        "model_name": "offline",
        "output_schema": SCHEMA,
        "seed_field": "seed",
        "sample_retries": 2,
        "checkpoint_interval": 25 if checkpoint else None,
        "checkpoint_path": str(tmp_path),
        "output_save_as": "scaling.jsonl",
    }
    if mode not in ("independent", "bulk-full-internal"):
        options.update(topic_sample_mode=mode.split("-")[0], shared_scope=mode.split("-")[1])
    with patch("deepfabric.generator.LLMClient"):
        engine = DataSetGenerator(**options)
    if mode == "bulk-full-internal":
        # Profile the existing internal collector; public bulk-full stays gated.
        engine.config = engine.config.model_copy(
            update={"topic_sample_mode": "bulk", "shared_scope": "full"}
        )
    return engine


def assert_records(engine, dataset, target):
    assert len(dataset) == target
    assert len(engine._completed) == target
    assert engine.get_all_failures() == []
    identities = []
    for row in dataset:
        match = re.fullmatch(r"(topic-\d+) item-(\d+)", row["seed"])
        assert match is not None
        topic, ordinal = match.groups()
        identities.append((topic, int(ordinal)))
        assert row["value"] == int(ordinal)
    assert len(set(identities)) == target
    assert set(identities) == engine._completed
    if engine.config.checkpoint_interval is not None:
        assert engine._sample_slots == identities
    else:
        # This list maps checkpoint rows to slots; memory-only runs do not fill it.
        assert engine._sample_slots == []


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("checkpoint", [False, True])
async def test_large_single_topic_quota_keeps_one_worker(
    tmp_path, mode, checkpoint, record_property
):
    target = 500
    engine = engine_for(tmp_path, mode, checkpoint)
    probe = ResourceProbe(engine, concurrency=8)
    probe.attach()
    started = time.perf_counter()
    with patch("deepfabric.generator.iter_jsonl", side_effect=probe.read):
        dataset = await engine.create_data_async(
            num_samples=target, topic_model=topics(1), batch_size=8
        )
    assert_records(engine, dataset, target)
    assert probe.peak_workers == probe.peak_requests == 1
    expected_calls = 1 if mode == "bulk-full-internal" else target + int(mode == "bulk-seed")
    assert probe.calls == expected_calls
    assert not probe.workers and not probe.requests
    probe.report(record_property, time.perf_counter() - started)


@pytest.mark.parametrize("mode", MODES)
async def test_many_topic_resume_preserves_rows_and_bounds_workers(tmp_path, mode, record_property):
    topic_count, quota, concurrency = 60, 4, 7
    target = topic_count * quota
    engine = engine_for(tmp_path, mode, checkpoint=True)
    # Keep the first run bounded even if workers outpace event consumption.
    first = ResourceProbe(engine, concurrency, stop_after=30)
    first.attach()
    stream = engine.create_data_with_events_async(
        num_samples=target, topic_model=topics(topic_count), batch_size=concurrency
    )
    settled = 0
    try:
        async for event in stream:
            if isinstance(event, dict) and event.get("event") == "batch_complete":
                settled += 1
                if settled == 12:
                    break
    finally:
        await stream.aclose()
    assert 0 < len(engine._completed) < target
    assert not first.workers and not first.requests
    saved = {row["seed"]: row for row in iter_jsonl(engine._checkpoint_samples_path)}
    assert saved

    restored = engine_for(tmp_path, mode, checkpoint=True)
    assert restored.load_checkpoint()
    remaining = target - len(restored._completed)
    probe = ResourceProbe(restored, concurrency)
    probe.attach()
    started = time.perf_counter()
    with patch("deepfabric.generator.iter_jsonl", side_effect=probe.read):
        dataset = await restored.create_data_async(
            num_samples=target, topic_model=topics(topic_count), batch_size=concurrency
        )
    assert_records(restored, dataset, target)
    for row in dataset:
        if row["seed"] in saved:
            assert row == saved[row["seed"]]
    assert 1 <= probe.peak_workers <= concurrency
    assert 1 <= probe.peak_requests <= concurrency
    if mode == "bulk-full-internal":
        assert 1 <= probe.calls <= remaining
    else:
        assert probe.calls >= remaining
    assert probe.scan_calls > 0 and probe.rows_read > 0
    assert not probe.workers and not probe.requests
    probe.report(record_property, time.perf_counter() - started)

    # Completed checkpoints must neither regenerate rows nor rewrite identities.
    complete = engine_for(tmp_path, mode, checkpoint=True)
    assert complete.load_checkpoint()
    with patch.object(
        complete.llm_client, "generate_async", side_effect=AssertionError("provider called")
    ):
        repeated = await complete.create_data_async(
            num_samples=target, topic_model=topics(topic_count), batch_size=concurrency
        )
    assert list(repeated) == list(dataset)
