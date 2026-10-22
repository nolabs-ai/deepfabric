"""Shared fixtures and helpers for bulk conversation seeds tests."""
# ruff: noqa: PLR2004, ARG001

import json

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from deepfabric.generator import DataSetGenerator


@pytest.fixture
def make_engine(tmp_path):
    def make(**options):
        params = {
            "generation_system_prompt": "GENERATION_PERSONA",
            "dataset_system_prompt": "OUTPUT_PERSONA",
            "provider": "openai",
            "model_name": "test",
            "topic_sample_mode": "bulk",
            "instructions": "SPECIAL_INSTRUCTIONS",
            "sys_msg": False,
            "checkpoint_interval": 1,
            "checkpoint_path": str(tmp_path),
            "output_save_as": "conversations.jsonl",
        }
        params.update(options)
        with patch("deepfabric.generator.LLMClient"):
            engine = DataSetGenerator(**params)
        engine.build_examples_text = lambda _count: "DEMONSTRATION_SENTINEL"
        return engine

    return make


def script(engine, *values):
    iterator = iter(values)

    async def generate(prompt, schema, **kwargs):
        value = next(iterator)
        if isinstance(value, BaseException):
            raise value
        return schema.model_validate_json(json.dumps(value))

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)


TOOL = {"name": "lookup", "description": "CATALOG_SENTINEL", "parameters": [], "returns": "result"}


@pytest.fixture
def spin_sessions():
    sessions = []

    def create(*args, **kwargs):
        session = SimpleNamespace(
            execute_tool=AsyncMock(
                return_value=SimpleNamespace(
                    success=True, result="observed result", error_type=None
                )
            ),
            cleanup=AsyncMock(),
            seed_files=AsyncMock(),
        )
        sessions.append(session)
        return session

    with (
        patch("deepfabric.builders_agent.SpinClient"),
        patch("deepfabric.builders_agent.SpinSession", side_effect=create),
    ):
        yield sessions


TOOL_STEP = {
    "thought": "Look it up",
    "tool_calls": [{"function_name": "lookup", "arguments": "{}", "reasoning": "Need evidence"}],
}

FINAL_STEP = {"thought": "Enough evidence", "tool_calls": [], "is_final": True}
