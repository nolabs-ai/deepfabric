"""Offline component acceptance using real schedulers/builders and mocked external I/O.

Bulk full scope is exercised internally; its public truncation gate stays intact.
"""
# ruff: noqa: PLR2004, ARG001

import asyncio
import json
import re

from collections import Counter, deque
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from datasets import Dataset
from rich.console import Console

from deepfabric.builders import ConversationBuilderFactory
from deepfabric.dataset_manager import handle_dataset_events_async
from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.generator import DataSetGenerator
from deepfabric.tui import DatasetGenerationTUI, DeepFabricTUI

MODES = ["independent", "sequential-seed", "sequential-full", "bulk-seed", "bulk-full-internal"]
KINDS = ["basic", "cot", "custom", "agent"]
CASES = [
    pytest.param((mode, kind), id=f"{mode}-{kind}")
    for mode in MODES
    for kind in KINDS
    if not (mode == "bulk-full-internal" and kind == "agent")
]
SCHEMA = {
    "type": "object",
    "properties": {"seed": {"type": "string"}, "value": {"type": "integer", "minimum": 0}},
    "required": ["seed", "value"],
    "additionalProperties": False,
}
TOOL = {"name": "lookup", "description": "Look up a fact", "parameters": [], "returns": "fact"}


def topics(count=3):
    return SimpleNamespace(
        get_unique_topics=lambda: [
            SimpleNamespace(uuid=f"topic-{i}", topic=f"Topic {i}") for i in range(count)
        ],
        get_path_by_id=lambda uuid: ["Root", uuid],
    )


def question(topic, ordinal):
    return f"{topic} request {ordinal}"


def row_identity(row, kind):
    seed = (
        row["seed"]
        if kind == "custom"
        else next(message["content"] for message in row["messages"] if message["role"] == "user")
    )
    topic, ordinal = re.fullmatch(r"(topic-\d+) request (\d+)", seed).groups()
    return topic, int(ordinal)


class Provider:
    """Deterministic provider surrogate; all actual generation/assembly remains real."""

    def __init__(self, engine, case, target=7, topic_count=3, *, fail=False, resume=False):
        self.engine = engine
        self.mode, self.kind = case
        self.plan = engine._prepare_topic_allocations(target, topics(topic_count))
        completed = engine._completed if resume else set()
        self.pending = {
            allocation["uuid"]: deque(
                ordinal
                for ordinal in range(allocation["quota"])
                if (allocation["uuid"], ordinal) not in completed
            )
            for allocation in self.plan["allocations"]
        }
        self.fail = fail
        self.react_steps = Counter()
        self.calls = []
        self.active = set()
        self.max_active = 0
        self.overlap = None
        self.first_topics = set()

    def record(self, topic, ordinal):
        if self.kind == "custom":
            return {"seed": question(topic, ordinal), "value": ordinal}
        value = {
            "messages": [
                {"role": "user", "content": question(topic, ordinal)},
                {"role": "assistant", "content": "A complete assistant answer"},
            ],
            "metadata": {"topic_id": "provider-supplied-wrong-id"},
        }
        if self.kind == "cot":
            value["reasoning"] = {"style": "freetext", "content": "Consider the evidence"}
        return value

    def settle(self, topic):
        ordinal = self.pending[topic].popleft()
        self.react_steps[topic] = 0
        if self.fail and topic == "topic-0" and ordinal == 1:
            raise RuntimeError("Injected provider authentication error")
        return ordinal

    async def generate(self, prompt, schema, **_kwargs):
        topic = re.search(r"topic-\d+", prompt).group()
        assert topic not in self.active, "Sibling requests overlapped"
        self.active.add(topic)
        self.max_active = max(self.max_active, len(self.active))
        self.calls.append((topic, schema.__name__, prompt))
        try:
            if self.overlap is not None and topic in ("topic-0", "topic-1"):
                self.first_topics.add(topic)
                if len(self.first_topics) == 2:
                    self.overlap.set()
                await asyncio.wait_for(self.overlap.wait(), timeout=3)
            await asyncio.sleep(0)
            properties = schema.model_json_schema()["properties"]
            ordinal = self.pending[topic][0]
            if "items" in properties:
                if self.mode == "bulk-full-internal":
                    if self.fail and topic == "topic-0" and ordinal > 0:
                        self.pending[topic].clear()
                        raise RuntimeError("Injected collection authentication error")
                    count = 1 if self.fail and topic == "topic-0" else len(self.pending[topic])
                    payload = {
                        "items": [
                            self.record(topic, self.pending[topic].popleft()) for _ in range(count)
                        ]
                    }
                else:
                    saved = self.engine._topic_seeds.get(topic, {})
                    payload = {
                        "items": [
                            question(topic, index)
                            for index in self.pending[topic]
                            if str(index) not in saved
                        ]
                    }
            elif schema.__name__ == "UserQuestion":
                payload = {"content": question(topic, ordinal)}
            elif schema.__name__ == "AgentStep":
                self.react_steps[topic] += 1
                payload = (
                    {
                        "thought": "Look up evidence",
                        "tool_calls": [
                            {
                                "function_name": "lookup",
                                "arguments": "{}",
                                "reasoning": "Need facts",
                            }
                        ],
                    }
                    if self.react_steps[topic] == 1
                    else {"thought": "Enough evidence", "tool_calls": [], "is_final": True}
                )
            elif "messages" in properties or "seed" in properties:
                payload = self.record(topic, self.settle(topic))
            elif "value" in properties:
                payload = {"value": self.settle(topic)}
            else:
                self.settle(topic)
                payload = {"content": "A complete assistant answer"}
                if "reasoning" in properties:
                    payload["reasoning"] = {"style": "freetext", "content": "Consider the evidence"}
            return schema.model_validate_json(json.dumps(payload))
        finally:
            self.active.remove(topic)


@pytest.fixture
def sessions():
    created = []

    def create(*args, **kwargs):
        session = SimpleNamespace(
            execute_tool=AsyncMock(
                return_value=SimpleNamespace(success=True, result="Observed fact", error_type=None)
            ),
            cleanup=AsyncMock(),
            seed_files=AsyncMock(),
        )
        created.append(session)
        return session

    with (
        patch("deepfabric.builders_agent.SpinClient"),
        patch("deepfabric.builders_agent.SpinSession", side_effect=create),
    ):
        yield created


@pytest.fixture
def make_engine(tmp_path, sessions):
    def make(
        case, *, target=7, topic_count=3, fail=False, resume=False, retry_failed=False, **options
    ):
        mode, kind = case
        params = {
            "generation_system_prompt": "Generation persona",
            "dataset_system_prompt": "Shipping persona",
            "provider": "openai",
            "model_name": "test",
            "sample_retries": 0,
            "checkpoint_interval": None,
            "checkpoint_path": str(tmp_path),
            "output_save_as": "acceptance.jsonl",
            "sys_msg": True,
        }
        if mode != "bulk-full-internal":
            params["topic_sample_mode"] = mode.split("-")[0]
            if mode != "independent":
                params["shared_scope"] = mode.split("-")[1]
        if kind == "custom":
            params.update(output_schema=SCHEMA, seed_field="seed")
        elif kind == "agent":
            params.update(
                conversation_type="cot",
                reasoning_style="agent",
                custom_tools=[TOOL],
                spin_endpoint="http://spin.invalid",
            )
        else:
            params["conversation_type"] = kind
        params.update(options)
        with patch("deepfabric.generator.LLMClient"):
            engine = DataSetGenerator(**params)
        if mode == "bulk-full-internal":
            engine.config = engine.config.model_copy(
                update={"topic_sample_mode": "bulk", "shared_scope": "full"}
            )
        if resume:
            assert engine.load_checkpoint(retry_failed=retry_failed)
        provider = Provider(engine, case, target, topic_count, fail=fail, resume=resume)
        engine.llm_client.generate_async = AsyncMock(side_effect=provider.generate)
        return engine, provider

    return make


async def run(engine, target=7, topic_count=3, batch_size=2):
    values = [
        value
        async for value in engine.create_data_with_events_async(
            num_samples=target, topic_model=topics(topic_count), batch_size=batch_size
        )
    ]
    assert isinstance(values[-1], Dataset)
    return [value for value in values if isinstance(value, dict)], values[-1]


def events_of(events, name):
    return [event for event in events if event["event"] == name]


def inject_checkpoint_write_fault(engine):
    save = engine._save_checkpoint_metadata
    injected = []
    engine._save_samples_to_file = Mock()

    def fail_once():
        if engine._flushed_samples_count and not injected:
            injected.append(True)
            raise OSError("Injected cross-mode metadata write failure")
        return save()

    engine._save_checkpoint_metadata = fail_once
    return injected


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("checkpoint_interval", [None, 2], ids=["memory", "checkpoint"])
async def test_cross_mode_scheduler_events_and_slot_accounting(
    make_engine, sessions, case, checkpoint_interval
):
    engine, provider = make_engine(case, checkpoint_interval=checkpoint_interval)
    provider.overlap = asyncio.Event()
    active, finished = set(), set()
    peak = 0
    original = engine._generate_topic_samples_async

    async def worker(allocation, *args, **kwargs):
        nonlocal peak
        topic = allocation["uuid"]
        if topic == "topic-2":
            assert finished == {"topic-0", "topic-1"}, "Fixed topic group barrier was bypassed"
        active.add(topic)
        peak = max(peak, len(active))
        assert peak <= 2
        try:
            return await original(allocation, *args, **kwargs)
        finally:
            active.remove(topic)
            finished.add(topic)

    engine._generate_topic_samples_async = worker
    create = ConversationBuilderFactory.create
    builders = []

    def create_builder(**kwargs):
        builder = create(**kwargs)
        builders.append(builder)
        return builder

    with patch(
        "deepfabric.generator.ConversationBuilderFactory.create", side_effect=create_builder
    ):
        events, dataset = await asyncio.wait_for(run(engine), timeout=10)
    if case[1] == "custom" or case[0] == "bulk-full-internal":
        assert builders == []
    else:
        assert len(builders) == len({id(builder) for builder in builders}) == 7
    expected = {
        (allocation["uuid"], i)
        for allocation in provider.plan["allocations"]
        for i in range(allocation["quota"])
    }
    assert len(dataset) == len(engine._completed) == 7
    assert {row_identity(row, case[1]) for row in dataset} == engine._completed == expected
    assert peak == provider.max_active == 2 and not active and not provider.active
    assert finished == {"topic-0", "topic-1", "topic-2"}
    settlements = events_of(events, "batch_complete")
    assert [row_identity(row, case[1]) for row in dataset] == [
        (event["topic_id"], event["sample_ordinal"]) for event in settlements
    ]
    assert Counter((e["topic_id"], e["sample_ordinal"]) for e in settlements) == Counter(expected)
    assert all(e["samples_generated"] == 1 and e["samples_failed"] == 0 for e in settlements)
    complete = events_of(events, "generation_complete")[0]
    assert complete["new_samples"] == complete["total_samples"] == complete["expected_samples"] == 7
    assert complete["new_failures"] == complete["unaccounted"] == complete["skipped_slots"] == 0
    assert len(events_of(events, "topic_group_start")) == 2
    if case[0].startswith("bulk"):
        assert (
            len(events_of(events, "seed_request")) == len(events_of(events, "seed_response")) == 3
        )
    else:
        assert not events_of(events, "seed_request") and not events_of(events, "seed_response")
    if checkpoint_interval is not None:
        assert set(engine._sample_slots) == expected and len(engine._sample_slots) == 7
        assert engine._sample_slots == [row_identity(row, case[1]) for row in dataset]
        metadata = json.loads(engine._checkpoint_metadata_path.read_text())
        assert metadata["total_samples"] == 7 and metadata["total_failures"] == 0
        assert "seeds" not in metadata["plan"]
    if case[1] != "custom":
        for row in dataset:
            assert row["messages"][0]["role"] == "system"
            assert row["messages"][0]["content"] == "Shipping persona"
            assert row["metadata"]["topic_id"] == row_identity(row, case[1])[0]
    if case[1] == "agent":
        assert len(sessions) == 7
        for row in dataset:
            call_ids = [
                call["id"]
                for message in row["messages"]
                for call in message.get("tool_calls") or []
            ]
            response_ids = [
                message["tool_call_id"] for message in row["messages"] if message["role"] == "tool"
            ]
            assert call_ids == response_ids and len(call_ids) == 1
            assert len(row["tool_context"]["executions"]) == 1
        for session in sessions:
            session.execute_tool.assert_awaited_once()
            session.cleanup.assert_awaited_once()


@pytest.mark.parametrize("case", CASES)
async def test_cross_mode_zero_quota_topics_make_no_requests(make_engine, case):
    engine, provider = make_engine(case, target=2, topic_count=4)
    events, dataset = await run(engine, 2, 4)
    assert len(dataset) == 2
    assert {call[0] for call in provider.calls} == {"topic-0", "topic-1"}
    assert len(events_of(events, "topic_group_start")) == 1


@pytest.mark.parametrize("case", CASES)
async def test_cross_mode_completed_checkpoint_is_no_op(make_engine, case):
    engine, _ = make_engine(case, checkpoint_interval=1)
    _, expected = await run(engine)
    restored, provider = make_engine(case, checkpoint_interval=1, resume=True)
    events, dataset = await run(restored)
    assert list(dataset) == list(expected)
    assert provider.calls == [] and restored.llm_client.generate_async.await_count == 0
    assert not events_of(events, "batch_complete") and not events_of(events, "seed_request")
    complete = events_of(events, "generation_complete")[0]
    assert complete["skipped_slots"] == complete["total_samples"] == 7
    assert complete["new_samples"] == complete["new_failures"] == complete["unaccounted"] == 0


@pytest.mark.parametrize("case", CASES)
async def test_cross_mode_failed_slot_resume_preserves_rows_and_reopens_only_failures(
    make_engine, sessions, case
):
    engine, _ = make_engine(case, checkpoint_interval=1, fail=True)
    events, initial = await run(engine)
    failed = {
        (failure["topic_id"], failure["sample_ordinal"]) for failure in engine.get_all_failures()
    }
    assert failed == (
        {("topic-0", 1), ("topic-0", 2)} if case[0] == "bulk-full-internal" else {("topic-0", 1)}
    )
    assert len(initial) + len(failed) == 7
    first_complete = events_of(events, "generation_complete")[0]
    assert first_complete["failed_samples"] == len(failed) and first_complete["unaccounted"] == 0
    preserved = {row_identity(row, case[1]): row for row in initial}
    restored, provider = make_engine(case, checkpoint_interval=1, resume=True, retry_failed=True)
    events, dataset = await run(restored)
    assert len(dataset) == 7 and restored.get_all_failures() == []
    assert len(set(restored._sample_slots)) == len(restored._sample_slots) == 7
    assert Counter(
        (e["topic_id"], e["sample_ordinal"]) for e in events_of(events, "batch_complete")
    ) == Counter(failed)
    for row in dataset:
        identity = row_identity(row, case[1])
        if identity in preserved:
            assert row == preserved[identity]
    assert {call[0] for call in provider.calls} == {"topic-0"}
    if case[0] == "bulk-seed":
        assert not events_of(events, "seed_request"), (
            "A failed completion must reuse its pinned seed"
        )
    final = events_of(events, "generation_complete")[0]
    assert final["new_samples"] == len(failed) and final["skipped_slots"] == 7 - len(failed)
    assert final["total_samples"] == 7 and final["unaccounted"] == final["failed_samples"] == 0
    for session in sessions:
        session.cleanup.assert_awaited_once()


@pytest.mark.parametrize("case", CASES)
async def test_cross_mode_new_invocation_has_no_previous_sibling_context(make_engine, case):
    engine, _ = make_engine(case)
    await run(engine)
    provider = Provider(engine, case, target=3)
    engine.llm_client.generate_async = AsyncMock(side_effect=provider.generate)
    engine.stop_requested = True
    events, dataset = await run(engine, 3)
    assert len(dataset) == 3 and not engine.stop_requested
    assert len(engine._completed) == 3 and not engine.get_all_failures()
    assert events[0]["resumed_samples"] == events[0]["resumed_failures"] == 0
    assert events[0]["outstanding_slots"] == 3
    first_prompt = provider.calls[0][2]
    if case[0].startswith("sequential"):
        value = re.search(
            r'<prior_successes scope="\w+">(.*?)</prior_successes>', first_prompt, re.S
        )
        assert json.loads(value.group(1)) == []
    elif case[0].startswith("bulk"):
        value = re.search(r"<accepted_siblings>(.*?)</accepted_siblings>", first_prompt, re.S)
        assert json.loads(value.group(1)) == []
    else:
        assert "<prior_successes" not in first_prompt and "<accepted_siblings>" not in first_prompt


@pytest.mark.parametrize("case", CASES)
async def test_cross_mode_checkpoint_write_fault_recovers_unique_slot_identities(
    make_engine, sessions, case
):
    engine, _ = make_engine(case, checkpoint_interval=1, sample_retries=2)
    injected = inject_checkpoint_write_fault(engine)
    with pytest.raises(DataSetGeneratorError):
        await run(engine)
    assert injected
    restored, _ = make_engine(case, checkpoint_interval=1, sample_retries=2, resume=True)
    _, dataset = await run(restored)
    assert len(dataset) == 7
    assert len(set(restored._sample_slots)) == len(restored._sample_slots) == 7
    assert {row_identity(row, case[1]) for row in dataset} == set(restored._sample_slots)
    for session in sessions:
        session.cleanup.assert_awaited_once()


@pytest.mark.parametrize("kind", ["basic", "cot", "custom"])
async def test_exhausted_bulk_collection_after_write_fault_requires_explicit_retry(
    make_engine, kind
):
    case = ("bulk-full-internal", kind)
    engine, _ = make_engine(case, checkpoint_interval=1, sample_retries=0)
    injected = inject_checkpoint_write_fault(engine)
    with pytest.raises(DataSetGeneratorError):
        await run(engine)
    assert injected
    restored, provider = make_engine(case, checkpoint_interval=1, sample_retries=0, resume=True)
    exhausted = {
        topic for topic, attempts in restored._collection_attempts.items() if attempts >= 1
    }
    assert exhausted
    _, partial = await run(restored)
    assert not any(topic in exhausted for topic, _, _ in provider.calls)
    failures = restored.get_all_failures()
    assert failures and len(partial) + len(failures) == len(restored._completed) == 7
    previous = {row_identity(row, kind): row for row in partial}
    retried, _ = make_engine(
        case, checkpoint_interval=1, sample_retries=0, resume=True, retry_failed=True
    )
    events, dataset = await run(retried)
    assert len(dataset) == 7 and retried.get_all_failures() == []
    assert len(retried._sample_slots) == len(set(retried._sample_slots)) == 7
    assert len(events_of(events, "batch_complete")) == len(failures)
    for row in dataset:
        identity = row_identity(row, kind)
        if identity in previous:
            assert row == previous[identity]


@pytest.mark.parametrize("case", CASES)
async def test_cross_mode_cancel_inflight_sibling_then_resume(make_engine, sessions, case):
    engine, provider = make_engine(
        case, target=3, topic_count=1, checkpoint_interval=1, sample_retries=2
    )
    original = provider.generate
    blocked = asyncio.Event()

    async def generate(prompt, schema, **kwargs):
        if provider.pending["topic-0"][0] == 1:
            blocked.set()
            return await asyncio.Future()
        if case[0] == "bulk-full-internal":
            ordinal = provider.pending["topic-0"].popleft()
            return schema.model_validate_json(
                json.dumps({"items": [provider.record("topic-0", ordinal)]})
            )
        return await original(prompt, schema, **kwargs)

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)

    async def consume():
        return await run(engine, 3, 1, 1)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(blocked.wait(), timeout=3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    restored, _ = make_engine(
        case, target=3, topic_count=1, checkpoint_interval=1, sample_retries=2, resume=True
    )
    assert restored._completed == {("topic-0", 0)}
    events, dataset = await run(restored, 3, 1, 1)
    assert len(dataset) == 3 and len(set(restored._sample_slots)) == 3
    assert {(e["topic_id"], e["sample_ordinal"]) for e in events_of(events, "batch_complete")} == {
        ("topic-0", 1),
        ("topic-0", 2),
    }
    if case[0] == "bulk-seed":
        assert not events_of(events, "seed_request")
    for session in sessions:
        session.cleanup.assert_awaited_once()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("with_events", [False, True])
def test_cross_mode_sync_entrypoints(make_engine, mode, with_events):
    engine, _ = make_engine((mode, "basic"), target=3, topic_count=1)
    if with_events:
        values = list(engine.create_data_with_events(num_samples=3, topic_model=topics(1)))
        dataset = values[-1]
        assert (
            len(
                [
                    value
                    for value in values
                    if isinstance(value, dict) and value.get("event") == "batch_complete"
                ]
            )
            == 3
        )
    else:
        dataset = engine.create_data(num_samples=3, topic_model=topics(1))
    assert len(dataset) == 3


@pytest.mark.parametrize("kind", KINDS)
def test_public_bulk_full_gate_remains_in_place_before_client_initialization(kind):
    options = {"conversation_type": "cot" if kind in ("cot", "agent") else "basic"}
    if kind == "custom":
        options["output_schema"] = SCHEMA
    if kind == "agent":
        options["custom_tools"] = [TOOL]
    with (
        patch("deepfabric.generator.LLMClient") as client,
        pytest.raises(DataSetGeneratorError, match="truncation fix|agent builder"),
    ):
        DataSetGenerator(
            generation_system_prompt="Persona",
            provider="openai",
            model_name="test",
            topic_sample_mode="bulk",
            shared_scope="full",
            **options,
        )
    client.assert_not_called()


@pytest.mark.parametrize("case", CASES)
async def test_cross_mode_headless_progress_counts_settled_slots_and_resumed_history(
    make_engine, case
):
    output = StringIO()
    tui = DatasetGenerationTUI(DeepFabricTUI(Console(file=output, width=160, force_terminal=False)))
    engine, _ = make_engine(case, checkpoint_interval=1, fail=True)
    with (
        patch("deepfabric.dataset_manager.get_dataset_tui", return_value=tui),
        patch(
            "deepfabric.dataset_manager._get_tui_settings",
            return_value=SimpleNamespace(mode="simple"),
        ),
    ):
        partial = await handle_dataset_events_async(
            engine.create_data_with_events_async(num_samples=7, topic_model=topics(), batch_size=2)
        )
        assert len(partial) == (5 if case[0] == "bulk-full-internal" else 6)
        restored, _ = make_engine(case, checkpoint_interval=1, resume=True, retry_failed=True)
        complete = await handle_dataset_events_async(
            restored.create_data_with_events_async(
                num_samples=7, topic_model=topics(), batch_size=2
            )
        )
        assert len(complete) == 7
    counts = [int(count) for count in re.findall(r"Processed (\d+)/7 samples", output.getvalue())]
    assert counts.count(7) == 2 and max(counts) == 7
    assert "concurrency=2" in output.getvalue()
    assert tui.live_display is None and tui.simple_progress is None
