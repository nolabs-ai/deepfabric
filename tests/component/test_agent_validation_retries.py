"""Shared agent acceptance and nested retry boundaries; external I/O is mocked."""
# ruff: noqa: PLR2004, ARG001

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from deepfabric.error_codes import DF_T02, DF_T03, classify_error
from deepfabric.exceptions import DataSetGeneratorError, ToolValidationError
from deepfabric.utils import is_validation_error
from tests.support import bulk_conversation_seeds as _support
from tests.support.bulk_conversation_seeds import FINAL_STEP, TOOL, TOOL_STEP, script

make_engine = _support.make_engine
spin_sessions = _support.spin_sessions


@pytest.fixture(
    params=[
        {"topic_sample_mode": "independent"},
        {"topic_sample_mode": "bulk", "shared_scope": "seed"},
        {"topic_sample_mode": "sequential", "shared_scope": "seed"},
        {"topic_sample_mode": "sequential", "shared_scope": "full"},
    ],
    ids=["independent", "bulk-seeded", "sequential-seeded", "sequential-full"],
)
def agent_options(request):
    return request.param


async def run(engine):
    topics = SimpleNamespace(
        get_unique_topics=lambda: [SimpleNamespace(uuid="topic-0", topic="Topic")],
        get_path_by_id=lambda uuid: ["ROOT", uuid],
    )
    events = [
        event
        async for event in engine.create_data_with_events_async(num_samples=1, topic_model=topics)
    ]
    return events[-1]


def question(engine, text="Pinned question"):
    return {"items": [text]} if engine.config.topic_sample_mode == "bulk" else {"content": text}


def prompt_of(call):
    return call.args[0] if call.args else call.kwargs["prompt"]


def schema_of(call):
    return call.args[1] if call.args else call.kwargs["schema"]


@pytest.mark.parametrize(
    "settings,expected_name",
    [
        ({"custom_tools": [TOOL]}, "lookup"),
        ({"tool_components": {"builtin": ["read_file"]}}, "read_file"),
    ],
)
def test_configured_tools_initialize_registry(make_engine, settings, expected_name):
    engine = make_engine(**settings)
    assert engine.tool_registry is not None
    assert expected_name in {tool.name for tool in engine.tool_registry.tools}


def test_registry_initialization_failure_raises(make_engine):
    with (
        patch("deepfabric.generator.load_tools_from_dict", side_effect=ValueError("bad tools")),
        pytest.raises(DataSetGeneratorError, match="Failed to initialize tool registry"),
    ):
        make_engine(custom_tools=[TOOL])


@pytest.mark.parametrize(
    "message,code",
    [
        ("Agent mode requires at least one tool execution", DF_T03),
        ("Sample has 2 tool executions, exceeds limit of 1", DF_T02),
    ],
)
def test_tool_validation_retryability_does_not_depend_on_message(message, code):
    assert is_validation_error(ToolValidationError(message))
    assert not is_validation_error(ValueError(message))
    assert classify_error(ToolValidationError(message)).error_code == code
    assert is_validation_error(ValueError("validation error: malformed response"))
    assert not is_validation_error(RuntimeError("provider unavailable"))


@pytest.mark.parametrize("missing_context", [False, True])
def test_tool_acceptance_uses_runtime_registry(make_engine, missing_context):
    conversation = SimpleNamespace(
        tool_context=None if missing_context else SimpleNamespace(executions=[])
    )
    engine = make_engine()
    engine._validate_conversation_tools(conversation)
    # A present registry activates validation even if configuration was changed later.
    configured = make_engine(custom_tools=[TOOL])
    engine.tool_registry = configured.tool_registry
    with pytest.raises(ToolValidationError, match="Agent mode requires at least one"):
        engine._validate_conversation_tools(conversation)


async def test_local_decision_retry_retains_session_and_observations(
    make_engine, spin_sessions, agent_options
):
    engine = make_engine(**agent_options, custom_tools=[TOOL], spin_endpoint="http://spin.invalid")
    script(
        engine,
        question(engine),
        TOOL_STEP,
        {"thought": ""},
        FINAL_STEP,
        {"content": "A complete assistant answer"},
    )
    dataset = await run(engine)
    assert len(dataset) == 1, engine.get_all_failures()
    assert len(spin_sessions) == 1
    spin_sessions[0].execute_tool.assert_awaited_once()
    spin_sessions[0].cleanup.assert_awaited_once()
    calls = engine.llm_client.generate_async.call_args_list
    assert len(calls) == 5
    first, retry = map(prompt_of, calls[2:4])
    for prompt in (first, retry):
        assert "Pinned question" in prompt and "observed result" in prompt
    assert "Previous Attempt Failed" not in first
    assert "Previous Attempt Failed" in retry and "validation error" in retry


async def test_whole_sample_retry_after_local_budget_preserves_bulk_seed(
    make_engine, spin_sessions, agent_options
):
    engine = make_engine(**agent_options, custom_tools=[TOOL], spin_endpoint="http://spin.invalid")
    bulk = engine.config.topic_sample_mode == "bulk"
    script(
        engine,
        question(engine),
        TOOL_STEP,
        *[{"thought": ""} for _ in range(3)],
        *([] if bulk else [{"content": "Replacement question"}]),
        TOOL_STEP,
        FINAL_STEP,
        {"content": "Recovered answer"},
    )
    dataset = await run(engine)
    assert len(dataset) == 1, engine.get_all_failures()
    assert dataset[0]["messages"][0]["content"] == (
        "Pinned question" if bulk else "Replacement question"
    )
    assert len(spin_sessions) == 2
    for session in spin_sessions:
        session.execute_tool.assert_awaited_once()
        session.cleanup.assert_awaited_once()
    calls = engine.llm_client.generate_async.call_args_list
    assert len(calls) == (8 if bulk else 9)
    if bulk:
        assert (
            sum("items" in schema_of(call).model_json_schema()["properties"] for call in calls) == 1
        )
        assert "Pinned question" in prompt_of(calls[5])


@pytest.mark.parametrize("sample_retries", [0, 2])
async def test_tool_acceptance_failure_retries_with_consistent_messages(
    make_engine, spin_sessions, agent_options, sample_retries
):
    engine = make_engine(
        **agent_options,
        custom_tools=[TOOL],
        spin_endpoint="http://spin.invalid",
        sample_retries=sample_retries,
    )
    responses = [question(engine)] if engine.config.topic_sample_mode == "bulk" else []
    for _ in range(sample_retries + 1):
        if engine.config.topic_sample_mode != "bulk":
            responses.append(question(engine))
        responses.extend([FINAL_STEP, {"content": "No tool used"}])
    script(engine, *responses)
    dataset = await run(engine)
    assert len(dataset) == 0
    assert len(spin_sessions) == sample_retries + 1
    assert (
        "Agent mode requires at least one tool execution" in engine.get_all_failures()[0]["error"]
    )
    for session in spin_sessions:
        session.cleanup.assert_awaited_once()


@pytest.mark.parametrize("strict", [False, True])
async def test_tool_execution_limit_is_shared(make_engine, spin_sessions, agent_options, strict):
    other_tool = {**TOOL, "name": "lookup_other"}
    engine = make_engine(
        **agent_options,
        custom_tools=[TOOL, other_tool],
        spin_endpoint="http://spin.invalid",
        max_tools_per_query=1,
        max_tools_strict=strict,
        sample_retries=0,
    )
    step = {
        **TOOL_STEP,
        "tool_calls": [
            TOOL_STEP["tool_calls"][0],
            {**TOOL_STEP["tool_calls"][0], "function_name": "lookup_other"},
        ],
    }
    script(engine, question(engine), step, FINAL_STEP, {"content": "A complete assistant answer"})
    dataset = await run(engine)
    if strict:
        assert len(dataset) == 0
        assert (
            "Sample has 2 tool executions, exceeds limit of 1"
            in engine.get_all_failures()[0]["error"]
        )
    else:
        assert len(dataset) == 1, engine.get_all_failures()
        assert len(dataset[0]["tool_context"]["executions"]) == 1
    assert spin_sessions[0].execute_tool.await_count == 2
    spin_sessions[0].cleanup.assert_awaited_once()


@pytest.mark.parametrize(
    "failure", [RuntimeError("provider unavailable"), ValueError("validation error: bad step")]
)
async def test_zero_retry_budget_and_non_validation_failures_are_bounded(
    make_engine, spin_sessions, agent_options, failure
):
    engine = make_engine(
        **agent_options,
        custom_tools=[TOOL],
        spin_endpoint="http://spin.invalid",
        sample_retries=0 if is_validation_error(failure) else 2,
    )
    script(engine, question(engine), failure)
    dataset = await run(engine)
    assert len(dataset) == 0
    assert len(engine.llm_client.generate_async.call_args_list) == 2
    assert len(spin_sessions) == 1
    spin_sessions[0].cleanup.assert_awaited_once()
