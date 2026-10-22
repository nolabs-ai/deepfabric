"""Unit tests for bulk conversation seeds."""
# ruff: noqa: PLR2004, ARG001

import asyncio

import pytest

from deepfabric.builders import SingleShotBuilder
from deepfabric.builders_agent import SingleTurnAgentBuilder
from deepfabric.schemas import ChatMessage
from tests.support import bulk_conversation_seeds as _support
from tests.support.bulk_conversation_seeds import FINAL_STEP, TOOL, TOOL_STEP, script

make_engine = _support.make_engine
spin_sessions = _support.spin_sessions


async def test_builder_copies_pinned_message(make_engine):
    engine = make_engine()
    script(engine, {"content": "reply"})
    builder = SingleShotBuilder(engine.llm_client, engine.config)
    user = ChatMessage(role="user", content="exact")
    conversation = await builder.generate("context", pinned_user_message=user)
    user.content = "changed later"
    assert conversation.messages[0].content == "exact"


@pytest.mark.parametrize(
    "failure", [asyncio.CancelledError(), RuntimeError("provider unavailable")]
)
async def test_agent_cleanup_on_interrupted_completion(make_engine, spin_sessions, failure):
    engine = make_engine(custom_tools=[TOOL], spin_endpoint="http://spin.invalid")
    builder = SingleTurnAgentBuilder(engine.llm_client, engine.config, engine.tool_registry)
    script(engine, TOOL_STEP, FINAL_STEP, failure)
    with pytest.raises(type(failure)):
        await builder.generate(
            "completion context", pinned_user_message=ChatMessage(role="user", content="original")
        )
    assert len(spin_sessions) == 1
    spin_sessions[0].execute_tool.assert_awaited_once()
    spin_sessions[0].cleanup.assert_awaited_once()
    assert builder._completion_context is None
