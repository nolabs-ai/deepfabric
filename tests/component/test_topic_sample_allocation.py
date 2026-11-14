"""Component tests for topic sample allocation."""
# ruff: noqa: PLR2004, ARG001

import asyncio
import json

from collections import Counter
from unittest.mock import AsyncMock, Mock, patch

import pytest

from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.generator import DataSetGenerator
from deepfabric.schemas import ChatMessage, Conversation
from deepfabric.tree import Tree


@pytest.fixture
def tree():
    with patch("deepfabric.tree.LLMClient"):
        model = Tree(topic_prompt="Root")
    model.tree_paths = [["Root", name] for name in "ABCD"]
    return model


@pytest.fixture
def make_engine(tmp_path):
    def make(**overrides):
        params = {
            "generation_system_prompt": "Generate varied questions",
            "provider": "openai",
            "model_name": "test",
            "output_save_as": "samples.jsonl",
            "checkpoint_path": str(tmp_path),
        }
        params.update(overrides)
        with patch("deepfabric.generator.LLMClient"):
            engine = DataSetGenerator(**params)

        async def outcome(prompt, include_sys_msg, sample_idx, topic_path, identity):
            await asyncio.sleep(0)
            return {"slot": list(map(str, identity)), "index": sample_idx, "prompt": prompt}, None

        engine._generate_sample_outcome = AsyncMock(side_effect=outcome)
        return engine

    return make


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "10", "auto", "50%"])
async def test_invalid_explicit_target(make_engine, invalid):
    with pytest.raises(DataSetGeneratorError, match="num_samples"):
        await make_engine().create_data_async(num_samples=invalid)


async def test_empty_topics(make_engine, tree):
    tree.tree_paths = []
    engine = make_engine()
    with pytest.raises(DataSetGeneratorError, match="no unique topics"):
        await engine.create_data_async(num_samples=1, topic_model=tree)
    engine._generate_sample_outcome.assert_not_called()


@pytest.mark.parametrize("target,batch", [(10, 4), (2, 10), (10, 1), (4, 20)])
async def test_exact_rows_and_topic_counts(make_engine, tree, target, batch):
    engine = make_engine()
    with patch.object(
        engine, "_generate_topic_samples_async", wraps=engine._generate_topic_samples_async
    ) as visits:
        dataset = await engine.create_data_async(
            num_samples=target, batch_size=batch, topic_model=tree
        )
    uuids = [t.uuid for t in tree.get_unique_topics()]
    base, extra = divmod(target, len(uuids))
    expected = {uuid: base + (i < extra) for i, uuid in enumerate(uuids) if base + (i < extra)}
    assert len(dataset) == target
    assert Counter(row["slot"][0] for row in dataset) == expected
    assert visits.call_count == len(expected)
    assert len({row["index"] for row in dataset}) == target


@pytest.mark.parametrize("events", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_sync_entry_points(make_engine, tree, events, explicit):
    engine = make_engine()
    kwargs = {"batch_size": 3, "topic_model": tree}
    kwargs.update(num_samples=5, num_steps=0) if explicit else kwargs.update(num_steps=2)
    if events:
        outputs = list(engine.create_data_with_events(**kwargs))
        result = outputs[-1]
        assert any(
            isinstance(e, dict) and e.get("generation_mode") == "topic_allocation" for e in outputs
        )
    else:
        result = engine.create_data(**kwargs)
    assert len(result) == (5 if explicit else 6)


async def test_async_entry_points_and_none_fallback(make_engine, tree):
    for events in (False, True):
        engine = make_engine()
        if events:
            output = [
                e
                async for e in engine.create_data_with_events_async(
                    num_samples=None, batch_size=2, topic_model=tree
                )
            ]
            result = output[-1]
        else:
            result = await engine.create_data_async(
                num_samples=None, batch_size=2, topic_model=tree
            )
        assert len(result) == 2  # one legacy step by default


async def test_topic_group_barrier_and_no_sibling_overlap(make_engine, tree):
    engine = make_engine()
    ids = [topic.uuid for topic in tree.get_unique_topics()]
    held = asyncio.Event()
    siblings_finished = asyncio.Event()
    active = Counter()
    started = []
    maximum = 0

    async def outcome(prompt, include_sys_msg, sample_idx, topic_path, identity):
        nonlocal maximum
        uuid, ordinal = identity
        active[uuid] += 1
        assert active[uuid] == 1
        maximum = max(maximum, sum(active.values()))
        started.append(identity)
        if uuid == ids[0] and ordinal == 0:
            await held.wait()
        await asyncio.sleep(0)
        active[uuid] -= 1
        if uuid == ids[1] and ordinal == 2:
            siblings_finished.set()
        return {"slot": list(map(str, identity))}, None

    engine._generate_sample_outcome.side_effect = outcome
    task = asyncio.create_task(
        engine.create_data_async(num_samples=12, batch_size=2, topic_model=tree)
    )
    try:
        await asyncio.wait_for(siblings_finished.wait(), 2)
        assert not any(uuid in ids[2:] for uuid, _ in started)
        held.set()
        result = await asyncio.wait_for(task, 2)
    finally:
        held.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert len(result) == 12
    assert maximum == 2
    for uuid in ids:
        assert [ordinal for item, ordinal in started if item == uuid] == [0, 1, 2]


async def test_single_large_topic_has_one_active_task(make_engine, tree):
    tree.tree_paths = tree.tree_paths[:1]
    engine = make_engine()
    with patch.object(
        engine, "_generate_topic_samples_async", wraps=engine._generate_topic_samples_async
    ) as visits:
        result = await engine.create_data_async(num_samples=20, batch_size=10, topic_model=tree)
    assert len(result) == 20
    assert visits.call_count == 1


async def test_step_short_batch_and_failure_slots(make_engine, caplog):
    engine = make_engine()
    original = engine._generate_sample_outcome.side_effect

    async def outcome(*args):
        identity = args[-1]
        if identity[1] == 1:
            return None, {"topic_id": identity[0], "sample_ordinal": identity[1], "error": "failed"}
        return await original(*args)

    engine._generate_sample_outcome.side_effect = outcome
    with caplog.at_level("INFO"):
        outputs = [
            e async for e in engine.create_data_with_events_async(num_samples=25, batch_size=10)
        ]
    assert len(outputs[-1]) == 24
    steps = [e for e in outputs if isinstance(e, dict) and e.get("event") == "step_start"]
    assert [e["batch_size"] for e in steps] == [10, 10, 5]
    assert caplog.text.count("Final step will generate 5 samples") == 1
    assert engine._generate_sample_outcome.await_count == 25
    complete = outputs[-2]
    assert complete["failed_samples"] == 1
    assert complete["unaccounted"] == 0


async def test_checkpoint_inside_topic_stop_resume_retry(make_engine, tree):
    tree.tree_paths = tree.tree_paths[:2]
    engine = make_engine(checkpoint_interval=1)
    original = engine._generate_sample_outcome.side_effect
    failed_slot = (tree.get_unique_topics()[0].uuid, 1)

    async def outcome(*args):
        if args[-1] == failed_slot:
            return None, {"topic_id": failed_slot[0], "sample_ordinal": 1, "error": "terminal"}
        return await original(*args)

    engine._generate_sample_outcome.side_effect = outcome
    events = []
    async for event in engine.create_data_with_events_async(
        num_samples=12, batch_size=2, topic_model=tree
    ):
        events.append(event)
        if (
            isinstance(event, dict)
            and event.get("event") == "checkpoint_saved"
            and len(engine._completed) >= 3
        ):
            engine.stop_requested = True
    assert engine.config.checkpoint_interval == 1
    assert any(isinstance(e, dict) and e.get("event") == "generation_stopped" for e in events)
    assert not any(isinstance(e, dict) and e.get("event") == "generation_complete" for e in events)
    assert len(engine._completed) < 12
    first_completed = set(engine._completed)
    metadata = json.loads(engine._checkpoint_metadata_path.read_text())
    assert metadata["plan"]["base_quota"] == 6
    restored = make_engine(checkpoint_interval=1)
    assert restored.load_checkpoint()
    result = await restored.create_data_async(num_samples=12, batch_size=2, topic_model=tree)
    assert len(result) == 11
    assert not first_completed & {
        call.args[-1] for call in restored._generate_sample_outcome.call_args_list
    }
    retry = make_engine(checkpoint_interval=1)
    assert retry.load_checkpoint(retry_failed=True)
    result = await retry.create_data_async(num_samples=12, batch_size=2, topic_model=tree)
    assert len(result) == 12
    assert [call.args[-1] for call in retry._generate_sample_outcome.call_args_list] == [
        failed_slot
    ]
    assert retry._flushed_failures_count == 0
    complete = make_engine(checkpoint_interval=1)
    assert complete.load_checkpoint()
    outputs = [
        e
        async for e in complete.create_data_with_events_async(
            num_samples=12, batch_size=2, topic_model=tree
        )
    ]
    complete._generate_sample_outcome.assert_not_called()
    assert len(outputs[-1]) == 12
    summary = next(
        e for e in outputs if isinstance(e, dict) and e.get("event") == "generation_complete"
    )
    assert summary["new_samples"] == 0
    assert summary["unaccounted"] == 0


async def test_changed_plan_rejected(make_engine, tree):
    engine = make_engine(checkpoint_interval=2)
    await engine.create_data_async(num_samples=5, topic_model=tree)
    restored = make_engine(checkpoint_interval=2)
    restored.load_checkpoint()
    with pytest.raises(DataSetGeneratorError, match="plan differs"):
        await restored.create_data_async(num_samples=6, topic_model=tree)
    tree.tree_paths[0].append("Changed context")
    with pytest.raises(DataSetGeneratorError, match="plan differs"):
        await restored.create_data_async(num_samples=5, topic_model=tree)
    tree.tree_paths[0].pop()
    tree.tree_paths.reverse()
    with pytest.raises(DataSetGeneratorError, match="plan differs"):
        await restored.create_data_async(num_samples=5, topic_model=tree)
    restored._generate_sample_outcome.assert_not_called()


async def test_no_topic_short_step_resume(make_engine):
    engine = make_engine(checkpoint_interval=1)
    async for event in engine.create_data_with_events_async(num_samples=5, batch_size=3):
        if isinstance(event, dict) and event.get("event") == "checkpoint_saved":
            engine.stop_requested = True
    restored = make_engine(checkpoint_interval=1)
    assert restored.load_checkpoint()
    result = await restored.create_data_async(num_samples=5, batch_size=3)
    assert len(result) == 5
    assert len({tuple(row["slot"]) for row in result}) == 5


async def test_fresh_builders_and_no_sibling_prompt_context(tree):
    with (
        patch("deepfabric.generator.LLMClient"),
        patch("deepfabric.generator.ConversationBuilderFactory.create") as factory,
    ):
        engine = DataSetGenerator(
            generation_system_prompt="Generate", provider="openai", model_name="test"
        )
        builders = []

        def builder(**kwargs):
            instance = Mock()
            instance.generate = AsyncMock(
                return_value=Conversation(
                    messages=[ChatMessage(role="user", content="UNIQUE GENERATED CONTENT")]
                )
            )
            builders.append(instance)
            return instance

        factory.side_effect = builder
        result = await engine.create_data_async(num_samples=10, batch_size=4, topic_model=tree)
    assert len(result) == len(builders) == 10
    assert len({id(b) for b in builders}) == 10
    assert all("UNIQUE GENERATED CONTENT" not in b.generate.call_args.args[0] for b in builders)


async def test_shared_graph_uuid_and_equal_text_are_distinct_allocations(make_engine):
    from deepfabric.graph import Graph  # noqa: PLC0415

    with patch("deepfabric.graph.LLMClient"):
        graph = Graph(topic_prompt="Root")
    first = graph.add_node("Same text", {"uuid": "shared"})
    duplicate = graph.add_node("Same text", {"uuid": "shared"})
    other = graph.add_node("Same text", {"uuid": "different"})
    graph.add_edge(graph.root.id, first.id)
    graph.add_edge(graph.root.id, duplicate.id)
    graph.add_edge(first.id, other.id)
    graph.add_edge(duplicate.id, other.id)
    engine = make_engine()
    with patch.object(
        graph, "get_all_paths", side_effect=AssertionError("must not enumerate routes")
    ):
        dataset = await engine.create_data_async(num_samples=5, batch_size=2, topic_model=graph)
    assert Counter(row["slot"][0] for row in dataset) == {"shared": 3, "different": 2}


async def test_custom_schema_keeps_rows_unmodified_and_resumes(tree, tmp_path):
    schema = {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    }
    with (
        patch("deepfabric.generator.LLMClient"),
        patch("deepfabric.generator.ConversationBuilderFactory.create") as factory,
    ):
        engine = DataSetGenerator(
            generation_system_prompt="Generate",
            provider="openai",
            model_name="test",
            output_schema=schema,
            output_format="custom",
            checkpoint_interval=1,
            checkpoint_path=str(tmp_path),
            output_save_as="custom.jsonl",
        )
        count = 0

        async def generate(prompt, schema_model, **kwargs):
            nonlocal count
            count += 1
            return schema_model({"question": f"Question {count}"})

        engine.llm_client.generate_async.side_effect = generate
        result = await engine.create_data_async(num_samples=5, batch_size=3, topic_model=tree)
        factory.assert_not_called()
    assert result.column_names == ["question"]
    assert len(result) == 5
    assert engine.load_checkpoint()
    outputs = [
        e
        async for e in engine.create_data_with_events_async(
            num_samples=5, batch_size=3, topic_model=tree
        )
    ]
    assert len(outputs[-1]) == 5
    assert count == 5


@pytest.mark.parametrize(
    "error,retries",
    [(ValueError("validation error"), 3), (RuntimeError("authentication failed"), 1)],
)
async def test_existing_sample_retry_policy(tree, error, retries):
    with (
        patch("deepfabric.generator.LLMClient"),
        patch("deepfabric.generator.ConversationBuilderFactory.create") as factory,
    ):
        engine = DataSetGenerator(
            generation_system_prompt="Generate",
            provider="openai",
            model_name="test",
            sample_retries=2,
        )
        builder = Mock()
        builder.generate = AsyncMock(side_effect=error)
        factory.return_value = builder
        events = [
            e async for e in engine.create_data_with_events_async(num_samples=1, topic_model=tree)
        ]
    assert builder.generate.await_count == retries
    assert len(events[-1]) == 0
    summary = events[-2]
    assert summary["failed_samples"] == 1
    assert summary["unaccounted"] == 0


async def test_early_close_cancels_bounded_workers_and_saves_seen_slots(make_engine, tree):
    engine = make_engine(checkpoint_interval=20)
    events = engine.create_data_with_events_async(num_samples=100, batch_size=2, topic_model=tree)
    seen = 0
    async for event in events:
        if isinstance(event, dict) and event.get("event") == "batch_complete":
            seen += 1
            break
    await asyncio.wait_for(events.aclose(), 2)
    assert seen == 1
    metadata = json.loads(engine._checkpoint_metadata_path.read_text())
    assert len(metadata["completed"]) == 1
    assert engine.load_checkpoint()


@pytest.mark.parametrize("version", [4, 5])
async def test_existing_checkpoint_never_overwritten_without_explicit_resume(
    make_engine, tree, version
):
    engine = make_engine(checkpoint_interval=1)
    await engine.create_data_async(num_samples=2, topic_model=tree)
    metadata = json.loads(engine._checkpoint_metadata_path.read_text())
    metadata["version"] = version
    engine._checkpoint_metadata_path.write_text(json.dumps(metadata))
    paths = engine._get_checkpoint_paths()
    before = {path: path.read_bytes() for path in paths if path.exists()}
    fresh = make_engine(checkpoint_interval=1)
    with pytest.raises(DataSetGeneratorError, match="version 4|Existing checkpoint"):
        await fresh.create_data_async(num_samples=2, topic_model=tree)
    fresh._generate_sample_outcome.assert_not_called()
    assert all(path.read_bytes() == content for path, content in before.items())
