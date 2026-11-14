"""Component tests for api key validation."""

import os

from unittest.mock import patch

import pytest

from click.testing import CliRunner

from deepfabric.cli import cli
from deepfabric.config import DeepFabricConfig


class TestCliApiKeyValidation:
    """Tests for CLI-level API key validation."""

    @pytest.fixture
    def sample_config_content(self):
        """Sample YAML content with openai provider (new format)."""
        return """
topics:
  prompt: "Test topic"
  mode: tree
  system_prompt: ""
  depth: 2
  degree: 2
  save_as: "test_tree.jsonl"
  llm:
    provider: "openai"
    model: "gpt-4o-mini"
    temperature: 0.7

generation:
  system_prompt: "Test generation"
  instructions: "Test instructions"
  conversation:
    type: basic
  llm:
    provider: "openai"
    model: "gpt-4o-mini"
    temperature: 0.7

output:
  system_prompt: "Test system prompt"
  include_system_message: true
  num_samples: 1
  batch_size: 1
  save_as: "test_dataset.jsonl"
"""

    @pytest.fixture
    def gemini_config_content(self):
        """Sample YAML content with gemini provider (new format)."""
        return """
topics:
  prompt: "Test topic"
  mode: tree
  system_prompt: ""
  depth: 2
  degree: 2
  save_as: "test_tree.jsonl"
  llm:
    provider: "gemini"
    model: "gemini-2.0-flash"
    temperature: 0.7

generation:
  system_prompt: "Test generation"
  instructions: "Test instructions"
  conversation:
    type: basic
  llm:
    provider: "gemini"
    model: "gemini-2.0-flash"
    temperature: 0.7

output:
  system_prompt: "Test system prompt"
  include_system_message: true
  num_samples: 1
  batch_size: 1
  save_as: "test_dataset.jsonl"
"""

    def test_config_get_configured_providers_tree_mode(self, tmp_path):
        """Test get_configured_providers returns correct providers for tree mode."""
        config_content = """
topics:
  prompt: "Test"
  mode: tree
  depth: 2
  degree: 2
  llm:
    provider: "openai"
    model: "gpt-4o-mini"

generation:
  system_prompt: "Test"
  conversation:
    type: basic
  llm:
    provider: "anthropic"
    model: "claude-3-haiku"

output:
  save_as: "test.jsonl"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        config = DeepFabricConfig.from_yaml(str(config_file))

        providers = config.get_configured_providers()
        assert "openai" in providers
        assert "anthropic" in providers

    def test_config_get_configured_providers_graph_mode(self, tmp_path):
        """Test get_configured_providers returns correct providers for graph mode."""
        config_content = """
topics:
  prompt: "Test"
  mode: graph
  prompt_style: anchored
  depth: 2
  degree: 2
  llm:
    provider: "gemini"
    model: "gemini-2.0-flash"

generation:
  system_prompt: "Test"
  conversation:
    type: basic
  llm:
    provider: "openai"
    model: "gpt-4o-mini"

output:
  save_as: "test.jsonl"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        config = DeepFabricConfig.from_yaml(str(config_file))

        providers = config.get_configured_providers()
        assert "gemini" in providers
        assert "openai" in providers

    @patch.dict(os.environ, {}, clear=True)
    def test_cli_validation_fails_without_openai_key(self, sample_config_content, tmp_path):
        """CLI should fail early when OPENAI_API_KEY is missing."""
        # Remove the key
        os.environ.pop("OPENAI_API_KEY", None)

        config_file = tmp_path / "config.yaml"
        config_file.write_text(sample_config_content)

        runner = CliRunner()
        result = runner.invoke(cli, ["generate", str(config_file)])

        assert result.exit_code == 1
        assert "API key verification failed" in result.output
        assert "OPENAI_API_KEY" in result.output

    @patch.dict(os.environ, {}, clear=True)
    def test_cli_validation_fails_without_gemini_key(self, gemini_config_content, tmp_path):
        """CLI should fail early when GEMINI_API_KEY/GOOGLE_API_KEY is missing."""
        # Remove both keys
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)

        config_file = tmp_path / "config.yaml"
        config_file.write_text(gemini_config_content)

        runner = CliRunner()
        result = runner.invoke(cli, ["generate", str(config_file)])

        assert result.exit_code == 1
        assert "API key verification failed" in result.output
        assert "GOOGLE_API_KEY" in result.output or "GEMINI_API_KEY" in result.output

    @patch.dict(os.environ, {}, clear=True)
    def test_cli_provider_override_validates_override_provider(
        self, sample_config_content, tmp_path
    ):
        """CLI should validate the override provider, not config provider."""
        # Remove both OpenAI and Anthropic keys
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)

        config_file = tmp_path / "config.yaml"
        config_file.write_text(sample_config_content)

        runner = CliRunner()
        # Override provider to anthropic
        result = runner.invoke(
            cli, ["generate", str(config_file), "--provider", "anthropic", "--model", "claude-3"]
        )

        assert result.exit_code == 1
        assert "API key verification failed" in result.output
        # Should check anthropic key, not openai
        assert "ANTHROPIC_API_KEY" in result.output
