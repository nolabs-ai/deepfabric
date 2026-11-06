"""Unit tests for checkpoint."""

import os

from pathlib import Path
from unittest.mock import patch

import pytest

from deepfabric.constants import (
    CHECKPOINT_FAILURES_SUFFIX,
    CHECKPOINT_METADATA_SUFFIX,
    CHECKPOINT_SAMPLES_SUFFIX,
)
from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.generator import DataSetGenerator, DataSetGeneratorConfig
from tests.support import checkpoint as _support

base_generator_params = _support.base_generator_params
generator_with_checkpoint = _support.generator_with_checkpoint
temp_checkpoint_dir = _support.temp_checkpoint_dir


class TestCheckpointConfig:
    """Tests for checkpoint configuration in DataSetGeneratorConfig."""

    def test_checkpoint_config_defaults(self):
        """Test that checkpoint config has correct defaults."""
        config = DataSetGeneratorConfig(
            generation_system_prompt="Test",
            provider="openai",
            model_name="gpt-4",
        )
        assert config.checkpoint_interval is None
        assert config.checkpoint_path is None  # None means auto-resolve to XDG dir
        assert config.checkpoint_retry_failed is False
        assert config.output_save_as is None

    def test_checkpoint_config_with_values(self):
        """Test checkpoint config with custom values."""
        config = DataSetGeneratorConfig(
            generation_system_prompt="Test",
            provider="openai",
            model_name="gpt-4",
            checkpoint_interval=10,
            checkpoint_path="/custom/dir",
            checkpoint_retry_failed=True,
            output_save_as="output.jsonl",
        )
        assert config.checkpoint_interval == 10  # noqa: PLR2004
        assert config.checkpoint_path == "/custom/dir"
        assert config.checkpoint_retry_failed is True
        assert config.output_save_as == "output.jsonl"

    def test_checkpoint_interval_must_be_positive(self):
        """Test that checkpoint_interval must be >= 1."""
        with pytest.raises(ValueError):
            DataSetGeneratorConfig(
                generation_system_prompt="Test",
                provider="openai",
                model_name="gpt-4",
                checkpoint_interval=0,
            )


class TestCheckpointPaths:
    """Tests for checkpoint path generation."""

    def test_get_checkpoint_paths(self, temp_checkpoint_dir, base_generator_params):
        """Test that checkpoint paths are correctly derived from output_save_as."""
        params = {**base_generator_params, "checkpoint_path": temp_checkpoint_dir}
        with patch("deepfabric.generator.LLMClient"):
            generator = DataSetGenerator(**params)

        metadata_path, samples_path, failures_path = generator._get_checkpoint_paths()

        assert (
            metadata_path == Path(temp_checkpoint_dir) / f"test_dataset{CHECKPOINT_METADATA_SUFFIX}"
        )
        assert (
            samples_path == Path(temp_checkpoint_dir) / f"test_dataset{CHECKPOINT_SAMPLES_SUFFIX}"
        )
        assert (
            failures_path == Path(temp_checkpoint_dir) / f"test_dataset{CHECKPOINT_FAILURES_SUFFIX}"
        )

    def test_get_checkpoint_paths_creates_directory(
        self, temp_checkpoint_dir, base_generator_params
    ):
        """Test that checkpoint directory is created if it doesn't exist."""
        nested_dir = os.path.join(temp_checkpoint_dir, "nested", "checkpoints")
        params = {**base_generator_params, "checkpoint_path": nested_dir}
        with patch("deepfabric.generator.LLMClient"):
            generator = DataSetGenerator(**params)

        generator._get_checkpoint_paths()

        assert os.path.exists(nested_dir)

    def test_get_checkpoint_paths_requires_output_save_as(self, temp_checkpoint_dir):
        """Test that getting checkpoint paths fails without output_save_as."""
        with patch("deepfabric.generator.LLMClient"):
            generator = DataSetGenerator(
                generation_system_prompt="Test",
                provider="openai",
                model_name="gpt-4",
                checkpoint_interval=5,
                checkpoint_path=temp_checkpoint_dir,
                output_save_as=None,
            )

        with pytest.raises(DataSetGeneratorError, match="output_save_as not configured"):
            generator._get_checkpoint_paths()


class TestIsCompleted:
    def test_is_completed_returns_true_for_completed(self, generator_with_checkpoint):
        generator_with_checkpoint._completed.add(("uuid-1", 0))
        assert generator_with_checkpoint._is_completed("uuid-1", 0)

    def test_is_completed_returns_false_for_unprocessed(self, generator_with_checkpoint):
        assert not generator_with_checkpoint._is_completed("uuid-1", 0)

    def test_is_completed_distinguishes_sample_ordinals(self, generator_with_checkpoint):
        generator_with_checkpoint._completed.add(("uuid-1", 0))
        assert generator_with_checkpoint._is_completed("uuid-1", 0)
        assert not generator_with_checkpoint._is_completed("uuid-1", 1)
