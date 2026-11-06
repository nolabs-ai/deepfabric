"""Shared fixtures and helpers for generator tests."""

from unittest.mock import patch

import pytest  # type: ignore

from deepfabric.generator import DataSetGenerator


@pytest.fixture
def engine_args():
    return {
        "instructions": "Test instructions",
        "generation_system_prompt": "Test system prompt",
        "dataset_system_prompt": "Test dataset system prompt",
        "provider": "openai",
        "model_name": "test-model",
        "prompt_template": None,
        "example_data": None,
        "temperature": 0.7,
        "max_retries": 3,
        "default_batch_size": 5,
        "default_num_examples": 3,
        "request_timeout": 30,
        "sys_msg": True,
    }


@pytest.fixture
def data_engine(engine_args):
    with patch("deepfabric.generator.LLMClient"):
        return DataSetGenerator(**engine_args)
