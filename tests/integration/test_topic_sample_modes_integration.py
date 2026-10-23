"""Opt-in topic-mode checks against real providers and, separately, real Spin.

See deepfabric-topic-sample-modes-test-guide.md at the repository root for enable
flags, model settings, and artifacts.
These tests do not bypass the public bulk-full gate or use prompt-scenarios.csv.
"""
# ruff: noqa: PLR2004, ARG001

import json
import os
import time

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from jsonschema import validate

from deepfabric.checkpoint import iter_jsonl
from deepfabric.generator import DataSetGenerator
from deepfabric.llm.client import validate_provider_api_key
from deepfabric.schemas import Conversation
from deepfabric.spin.client import SpinSession

pytestmark = pytest.mark.skipif(
    os.getenv("DEEPFABRIC_TOPIC_MODES_LIVE") != "1",
    reason="Set DEEPFABRIC_TOPIC_MODES_LIVE=1 to enable paid/live topic-mode checks",
)

PROVIDERS = [
    pytest.param("openai", marks=pytest.mark.openai),
    pytest.param("gemini", marks=pytest.mark.gemini),
    "anthropic",
    "openrouter",
    "ollama",
]
MODES = ["independent", "bulk-seed", "sequential-seed", "sequential-full"]
SCHEMA = {
    "type": "object",
    "properties": {
        "request": {"type": "string", "minLength": 1},
        "answer": {"type": "string", "minLength": 1},
    },
    "required": ["request", "answer"],
    "additionalProperties": False,
}
PERSONA = "You are a concise, accurate assistant teaching basic Python programming."


@pytest.fixture(params=PROVIDERS)
def live_provider(request):
    provider = request.param
    selected = os.getenv("DEEPFABRIC_TOPIC_MODES_PROVIDERS", "openai").split(",")
    if provider not in {value.strip() for value in selected}:
        pytest.skip(f"{provider} not selected in DEEPFABRIC_TOPIC_MODES_PROVIDERS")
    model = os.getenv(f"{provider.upper()}_TEST_MODEL")
    if not model:
        pytest.skip(f"Set {provider.upper()}_TEST_MODEL to a structured-output-capable model")
    valid, reason = validate_provider_api_key(provider)
    if not valid:
        pytest.skip(reason)
    return {"provider": provider, "model_name": model}


def topics(count, subject="basic Python programming"):
    return SimpleNamespace(
        get_unique_topics=lambda: [
            SimpleNamespace(uuid=f"topic-{i}", topic=f"{subject}: learning objective {i + 1}")
            for i in range(count)
        ],
        get_path_by_id=lambda uuid: [
            f"Create varied, concrete beginner exercises about {subject}.",
            f"Learning objective {int(uuid.removeprefix('topic-')) + 1}",
        ],
    )


def make_engine(config, directory, mode, kind="custom", **overrides):
    options = {
        **config,
        "generation_system_prompt": PERSONA,
        "dataset_system_prompt": PERSONA,
        "instructions": "Use concrete, varied user requests and short correct answers.",
        "temperature": 0.4,
        "max_tokens": int(os.getenv("DEEPFABRIC_TOPIC_MODES_MAX_TOKENS", "2000")),
        "request_timeout": 90,
        "sample_retries": 2,
        "rate_limit": {"max_retries": 1},
        "checkpoint_interval": 1,
        "checkpoint_path": str(directory),
        "output_save_as": "samples.jsonl",
        "sys_msg": True,
    }
    if mode != "independent":
        options.update(topic_sample_mode=mode.split("-")[0], shared_scope=mode.split("-")[1])
    if kind == "custom":
        options.update(output_schema=SCHEMA, seed_field="request")
    else:
        options["conversation_type"] = kind
        if kind == "cot":
            options["reasoning_style"] = "freetext"
    options.update(overrides)
    return DataSetGenerator(**options)


def observe_requests(engine):
    """Measure the real adapter calls without saving prompt bodies or credentials."""
    calls = []
    original = engine.llm_client.generate_async

    async def generate(*args, **kwargs):
        prompt = args[0] if args else kwargs["prompt"]
        schema = args[1] if len(args) > 1 else kwargs["schema"]
        entry = {"prompt_characters": len(prompt), "schema": schema.__name__, "ok": False}
        calls.append(entry)
        started = time.perf_counter()
        try:
            result = await original(*args, **kwargs)
            entry["ok"] = True
            return result
        finally:
            entry["seconds"] = time.perf_counter() - started

    engine.llm_client.generate_async = generate
    return calls


def assert_result(engine, dataset, target, kind):
    assert len(dataset) == target, engine.get_all_failures()
    assert engine.get_all_failures() == []
    assert len(engine._completed) == len(set(engine._sample_slots)) == target
    for row in dataset:
        if kind == "custom":
            validate(row, SCHEMA)
        else:
            Conversation.model_validate(row)
            assert row["messages"][0]["role"] == "system"
            assert row["messages"][0]["content"] == PERSONA
            assert row["messages"][1]["role"] == "user"
            assert row["messages"][-1]["role"] == "assistant"
            if kind == "cot":
                assert row["reasoning"]["style"] == "freetext"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["basic", "cot", "custom"])
async def test_live_public_modes_and_completed_checkpoint(
    live_provider, tmp_path, mode, kind, record_property
):
    engine = make_engine(live_provider, tmp_path, mode, kind)
    calls = observe_requests(engine)
    dataset = await engine.create_data_async(num_samples=3, topic_model=topics(2), batch_size=2)
    assert_result(engine, dataset, 3, kind)
    assert Counter(topic for topic, _ in engine._sample_slots) == {"topic-0": 2, "topic-1": 1}
    assert engine.config.max_tokens == int(os.getenv("DEEPFABRIC_TOPIC_MODES_MAX_TOKENS", "2000"))
    record_property("requests", len(calls))

    restored = make_engine(live_provider, tmp_path, mode, kind)
    assert restored.load_checkpoint()

    async def unexpected_request(*args, **kwargs):
        pytest.fail("A completed checkpoint contacted the provider")

    restored.llm_client.generate_async = unexpected_request
    repeated = await restored.create_data_async(num_samples=3, topic_model=topics(2), batch_size=2)
    assert list(repeated) == list(dataset)
    assert restored._sample_slots == engine._sample_slots


@pytest.mark.parametrize("mode", MODES)
async def test_live_close_stream_then_resume(live_provider, tmp_path, mode):
    engine = make_engine(live_provider, tmp_path, mode)
    stream = engine.create_data_with_events_async(
        num_samples=3, topic_model=topics(1), batch_size=1
    )
    try:
        async for event in stream:
            if isinstance(event, dict) and event.get("event") == "batch_complete":
                break
    finally:
        await stream.aclose()
    saved = list(iter_jsonl(engine._checkpoint_samples_path))
    assert saved and len(saved) <= 3
    preserved = dict(zip(engine._sample_slots, saved, strict=True))
    restored = make_engine(live_provider, tmp_path, mode)
    assert restored.load_checkpoint()
    dataset = await restored.create_data_async(num_samples=3, topic_model=topics(1), batch_size=1)
    assert_result(restored, dataset, 3, "custom")
    rows = dict(zip(restored._sample_slots, dataset, strict=True))
    for identity, row in preserved.items():
        assert rows[identity] == row


@pytest.mark.spin
@pytest.mark.parametrize("mode", MODES)
async def test_live_agent_with_real_spin(live_provider, tmp_path, mode, monkeypatch):
    endpoint = os.getenv("SPIN_ENDPOINT")
    if os.getenv("DEEPFABRIC_TOPIC_MODES_SPIN") != "1" or not endpoint:
        pytest.skip("Set DEEPFABRIC_TOPIC_MODES_SPIN=1 and SPIN_ENDPOINT for live agent checks")
    created, executed, cleaned = [], [], []
    original_init = SpinSession.__init__
    original_execute, original_cleanup = SpinSession.execute_tool, SpinSession.cleanup

    def initialize(session, *args, **kwargs):
        original_init(session, *args, **kwargs)
        created.append(session.session_id)

    async def execute(session, *args, **kwargs):
        result = await original_execute(session, *args, **kwargs)
        executed.append((session.session_id, result.success))
        return result

    async def cleanup(session):
        result = await original_cleanup(session)
        cleaned.append((session.session_id, result))
        return result

    # Observers delegate to the actual Spin implementation and HTTP requests.
    monkeypatch.setattr(SpinSession, "__init__", initialize)
    monkeypatch.setattr(SpinSession, "execute_tool", execute)
    monkeypatch.setattr(SpinSession, "cleanup", cleanup)
    engine = make_engine(
        live_provider,
        tmp_path,
        mode,
        "basic",
        tool_components={"builtin": ["list_files"]},
        spin_endpoint=endpoint,
        max_agent_steps=3,
        instructions="The user must ask to list the session files. Call list_files before replying.",
    )
    dataset = await engine.create_data_async(
        num_samples=2, topic_model=topics(1, "listing files with list_files"), batch_size=1
    )
    assert_result(engine, dataset, 2, "basic")
    assert executed and all(success for _, success in executed)
    assert len({session for session, _ in executed}) >= 2
    assert len(created) == len(set(created))
    assert Counter(session for session, _ in cleaned) == Counter(created)
    assert all(success for _, success in cleaned)
    for row in dataset:
        calls = {
            call["id"] for message in row["messages"] for call in (message.get("tool_calls") or [])
        }
        results = {
            message["tool_call_id"] for message in row["messages"] if message["role"] == "tool"
        }
        assert calls and calls == results


@pytest.mark.parametrize("mode", ["bulk-seed", "sequential-seed", "sequential-full"])
async def test_live_diversity_comparison_artifact(live_provider, tmp_path, mode):
    root = os.getenv("DEEPFABRIC_TOPIC_MODES_ARTIFACT_DIR")
    if os.getenv("DEEPFABRIC_TOPIC_MODES_DIVERSITY") != "1" or not root:
        pytest.skip("Enable DEEPFABRIC_TOPIC_MODES_DIVERSITY=1 and set an artifact directory")
    target = int(os.getenv("DEEPFABRIC_TOPIC_MODES_SAMPLES", "12"))
    quota = int(os.getenv("DEEPFABRIC_TOPIC_MODES_QUOTA", "4"))
    assert quota > 1 and target >= quota and target % quota == 0
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **live_provider,
        "samples": target,
        "quota": quota,
        "scope": "custom request/answer records; one run per condition",
        "automatic_metrics": "Exact normalized request uniqueness only; not semantic diversity.",
        "review_required": [
            "Compare semantic request coverage against the quota=1 baseline.",
            "Check factual correctness, task usefulness, and answer quality.",
            "Repeat with representative schemas/topics and multiple runs before accepting the claim.",
        ],
        "runs": [],
    }
    for condition, selected_mode, count in [
        ("quota-1-workaround", "independent", target),
        ("coordinated", mode, target // quota),
    ]:
        engine = make_engine(live_provider, tmp_path / condition, selected_mode)
        calls = observe_requests(engine)
        started = time.perf_counter()
        dataset = await engine.create_data_async(
            num_samples=target, topic_model=topics(count), batch_size=1
        )
        assert_result(engine, dataset, target, "custom")
        normalized = {" ".join(row["request"].casefold().split()) for row in dataset}
        report["runs"].append(
            {
                "condition": condition,
                "mode": selected_mode,
                "topics": count,
                "topic_paths": [topics(count).get_path_by_id(f"topic-{i}") for i in range(count)],
                "configuration": engine.config.model_dump(mode="json"),
                "elapsed_seconds": time.perf_counter() - started,
                "requests": calls,
                "unique_normalized_requests": len(normalized),
                "sample_slots": engine._sample_slots,
                "rows": list(dataset),
            }
        )
    directory = Path(root) / live_provider["provider"] / mode
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (directory / f"comparison-{stamp}.json").write_text(json.dumps(report, indent=2) + "\n")
