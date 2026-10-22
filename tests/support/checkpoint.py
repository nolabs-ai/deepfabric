"""Shared fixtures and helpers for checkpoint tests."""

import tempfile

from unittest.mock import patch

import pytest

from deepfabric.generator import DataSetGenerator


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoint files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def base_generator_params():
    """Base parameters for creating a DataSetGenerator."""
    return {
        "generation_system_prompt": "You are a helpful assistant.",
        "provider": "openai",
        "model_name": "gpt-4",
        "checkpoint_interval": 5,
        "checkpoint_path": ".checkpoints",
        "output_save_as": "test_dataset.jsonl",
    }


@pytest.fixture
def generator_with_checkpoint(temp_checkpoint_dir, base_generator_params):
    """Create a generator with checkpoint config and mocked LLM client."""
    params = {**base_generator_params, "checkpoint_path": temp_checkpoint_dir}
    with patch("deepfabric.generator.LLMClient"):
        return DataSetGenerator(**params)
