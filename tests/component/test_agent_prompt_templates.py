"""Agent-stage overrides reach actual provider calls without changing tool execution."""
# ruff: noqa: PLR2004, ARG001

import json

import pytest

from deepfabric.builders_agent import SingleTurnAgentBuilder
from deepfabric.prompt_templates import AGENT_FIELDS, GenerationPromptTemplates, template_manifest
from deepfabric.schemas import ChatMessage
from tests.component.test_bulk_conversation_seeds import run, unpack
from tests.support import bulk_conversation_seeds as support
from tests.support.bulk_conversation_seeds import FINAL_STEP, TOOL, TOOL_STEP, script

make_engine = support.make_engine
spin_sessions = support.spin_sessions


@pytest.mark.parametrize("mode", ["independent", "sequential", "bulk"])
async def test_agent_overrides_across_sampling_modes(make_engine, spin_sessions, mode):
    request = "Look up café {{ unknown }} exactly"
    engine = make_engine(
        custom_tools=[TOOL],
        spin_endpoint="http://spin.invalid",
        topic_sample_mode=mode,
        prompt_templates={
            "agent_step": "ACTION {{ user_request }}\n{{ steps }}\n{{ tool_results }}\n{{ raw_tools }}",
            "agent_response": "ANSWER {{ dataset_system_prompt }} {{ user_request }} {{ tool_results_text }}",
        },
    )
    question = {"items": [request]} if mode == "bulk" else {"content": request}
    script(engine, question, TOOL_STEP, FINAL_STEP, {"content": "The final assistant answer"})
    _, dataset = await run(engine)
    calls = [unpack(call) for call in engine.llm_client.generate_async.call_args_list]
    first_action, second_action, response = [prompt for prompt, _ in calls[1:]]
    assert first_action.startswith("ACTION " + request + "\n[]\n[]\n[")
    assert '"name": "lookup"' in first_action
    assert "Look it up" in second_action and "observed result" in second_action
    assert response == (
        "ANSWER OUTPUT_PERSONA " + request + " Tool: lookup\nResult: observed result"
    )
    assert dataset[0]["messages"][0]["content"] == request
    assert dataset[0]["messages"][-1]["content"] == "The final assistant answer"
    spin_sessions[0].execute_tool.assert_awaited_once()
    spin_sessions[0].cleanup.assert_awaited_once()


async def test_agent_context_types_retry_and_default_extension(make_engine):
    engine = make_engine(
        custom_tools=[TOOL],
        prompt_templates={
            "agent_step": "{{ default_prompt }}\nFEEDBACK={{ raw_retry_feedback }}",
            "agent_response": "{{ context }}\n{{ steps }}\n{{ tool_results }}",
        },
    )
    builder = SingleTurnAgentBuilder(engine.llm_client, engine.config, engine.tool_registry)
    user = ChatMessage(role="user", content="Please look this up")
    script(engine, FINAL_STEP, {"content": "The final assistant answer"})
    step = await builder._generate_next_step_impl(user, [], [], "invalid {{ literal }}")
    prompt, _ = unpack(engine.llm_client.generate_async.call_args)
    assert prompt.startswith("## User Request\nPlease look this up")
    assert prompt.endswith("FEEDBACK=invalid {{ literal }}")
    await builder._generate_agent_conclusion(user, [step], [], context="earlier conversation")
    prompt, _ = unpack(engine.llm_client.generate_async.call_args)
    assert prompt.startswith("earlier conversation\n[")
    assert json.loads(prompt.split("\n", 1)[1].rsplit("\n", 1)[0])[0]["is_final"] is True
    assert prompt.endswith("\n[]")


@pytest.mark.parametrize("stage", ["agent_step", "agent_response"])
def test_agent_sources_validate_snapshot_and_report(tmp_path, stage):
    path = tmp_path / "agent.txt"
    text = "\n".join("{{ " + field + " }}" for field in sorted(AGENT_FIELDS))
    path.write_text(text, encoding="utf-8")
    config = GenerationPromptTemplates.model_validate(
        {stage: {"file": "agent.txt"}}, context={"config_dir": tmp_path}
    )
    path.write_text("changed", encoding="utf-8")
    assert getattr(config, stage).content == text
    assert template_manifest(config)["overrides"][stage]["text"] == text
    with pytest.raises(ValueError, match="Unknown or unavailable"):
        GenerationPromptTemplates.model_validate({stage: "{{ typo }}"})
