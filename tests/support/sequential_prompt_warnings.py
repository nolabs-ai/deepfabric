"""Shared fixtures and helpers for sequential prompt warnings tests."""
# ruff: noqa: PLR2004, ARG001

from unittest.mock import patch

import pytest

from deepfabric.generator import DataSetGenerator


@pytest.fixture
def make_engine():
    def make(**options):
        params = {
            "generation_system_prompt": "Persona",
            "provider": "openai",
            "model_name": "test",
            "topic_sample_mode": "sequential",
            "checkpoint_interval": None,
            "sys_msg": False,
        }
        params.update(options)
        with patch("deepfabric.generator.LLMClient"):
            return DataSetGenerator(**params)

    return make
