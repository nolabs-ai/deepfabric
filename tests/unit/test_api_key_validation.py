"""Unit tests for api key validation."""

import os

from unittest.mock import patch

from deepfabric.llm import (
    PROVIDER_API_KEY_MAP,
    get_required_api_key_env_var,
    validate_provider_api_key,
)


class TestProviderApiKeyMap:
    """Tests for the PROVIDER_API_KEY_MAP constant."""

    def test_contains_all_major_providers(self):
        """Verify all major providers are in the map."""
        expected_providers = ["openai", "anthropic", "gemini", "openrouter", "ollama"]
        for provider in expected_providers:
            assert provider in PROVIDER_API_KEY_MAP

    def test_openai_requires_correct_key(self):
        """OpenAI should require OPENAI_API_KEY."""
        assert PROVIDER_API_KEY_MAP["openai"] == ["OPENAI_API_KEY"]

    def test_anthropic_requires_correct_key(self):
        """Anthropic should require ANTHROPIC_API_KEY."""
        assert PROVIDER_API_KEY_MAP["anthropic"] == ["ANTHROPIC_API_KEY"]

    def test_gemini_accepts_either_key(self):
        """Gemini should accept GOOGLE_API_KEY or GEMINI_API_KEY."""
        assert PROVIDER_API_KEY_MAP["gemini"] == ["GOOGLE_API_KEY", "GEMINI_API_KEY"]

    def test_openrouter_requires_correct_key(self):
        """OpenRouter should require OPENROUTER_API_KEY."""
        assert PROVIDER_API_KEY_MAP["openrouter"] == ["OPENROUTER_API_KEY"]

    def test_ollama_requires_no_key(self):
        """Ollama should not require an API key."""
        assert PROVIDER_API_KEY_MAP["ollama"] == []

    def test_test_providers_require_no_key(self):
        """Test providers should not require API keys."""
        assert PROVIDER_API_KEY_MAP["test"] == []
        assert PROVIDER_API_KEY_MAP["override"] == []


class TestValidateProviderApiKey:
    """Tests for the validate_provider_api_key function."""

    def test_unknown_provider_returns_error(self):
        """Unknown provider should return invalid with error message."""
        is_valid, error_msg = validate_provider_api_key("unknown_provider")
        assert is_valid is False
        assert error_msg == "Unknown provider: unknown_provider"

    def test_ollama_always_valid(self):
        """Ollama should always be valid (no key required)."""
        is_valid, error_msg = validate_provider_api_key("ollama")
        assert is_valid is True
        assert error_msg is None

    def test_test_provider_always_valid(self):
        """Test provider should always be valid (no key required)."""
        is_valid, error_msg = validate_provider_api_key("test")
        assert is_valid is True
        assert error_msg is None

    @patch.dict(os.environ, {}, clear=True)
    def test_openai_invalid_without_key(self):
        """OpenAI should be invalid without OPENAI_API_KEY."""
        # Ensure the key is not set
        os.environ.pop("OPENAI_API_KEY", None)

        is_valid, error_msg = validate_provider_api_key("openai")
        assert is_valid is False
        assert "OPENAI_API_KEY" in error_msg
        assert "not set" in error_msg

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    def test_openai_valid_with_key(self):
        """OpenAI should be valid with OPENAI_API_KEY set."""
        is_valid, error_msg = validate_provider_api_key("openai")
        assert is_valid is True
        assert error_msg is None

    @patch.dict(os.environ, {}, clear=True)
    def test_anthropic_invalid_without_key(self):
        """Anthropic should be invalid without ANTHROPIC_API_KEY."""
        os.environ.pop("ANTHROPIC_API_KEY", None)

        is_valid, error_msg = validate_provider_api_key("anthropic")
        assert is_valid is False
        assert "ANTHROPIC_API_KEY" in error_msg

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"})
    def test_anthropic_valid_with_key(self):
        """Anthropic should be valid with ANTHROPIC_API_KEY set."""
        is_valid, error_msg = validate_provider_api_key("anthropic")
        assert is_valid is True
        assert error_msg is None

    @patch.dict(os.environ, {}, clear=True)
    def test_gemini_invalid_without_any_key(self):
        """Gemini should be invalid without either key."""
        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)

        is_valid, error_msg = validate_provider_api_key("gemini")
        assert is_valid is False
        assert "GOOGLE_API_KEY" in error_msg
        assert "GEMINI_API_KEY" in error_msg

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    def test_gemini_valid_with_google_key(self):
        """Gemini should be valid with GOOGLE_API_KEY set."""
        is_valid, error_msg = validate_provider_api_key("gemini")
        assert is_valid is True
        assert error_msg is None

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_gemini_valid_with_gemini_key(self):
        """Gemini should be valid with GEMINI_API_KEY set."""
        is_valid, error_msg = validate_provider_api_key("gemini")
        assert is_valid is True
        assert error_msg is None

    @patch.dict(os.environ, {}, clear=True)
    def test_openrouter_invalid_without_key(self):
        """OpenRouter should be invalid without OPENROUTER_API_KEY."""
        os.environ.pop("OPENROUTER_API_KEY", None)

        is_valid, error_msg = validate_provider_api_key("openrouter")
        assert is_valid is False
        assert "OPENROUTER_API_KEY" in error_msg

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"})
    def test_openrouter_valid_with_key(self):
        """OpenRouter should be valid with OPENROUTER_API_KEY set."""
        is_valid, error_msg = validate_provider_api_key("openrouter")
        assert is_valid is True
        assert error_msg is None


class TestGetRequiredApiKeyEnvVar:
    """Tests for the get_required_api_key_env_var function."""

    def test_returns_none_for_unknown_provider(self):
        """Unknown provider should return None."""
        result = get_required_api_key_env_var("unknown")
        assert result is None

    def test_returns_none_for_ollama(self):
        """Ollama should return None (no key required)."""
        result = get_required_api_key_env_var("ollama")
        assert result is None

    def test_returns_single_key_for_openai(self):
        """OpenAI should return the single key name."""
        result = get_required_api_key_env_var("openai")
        assert result == "OPENAI_API_KEY"

    def test_returns_single_key_for_anthropic(self):
        """Anthropic should return the single key name."""
        result = get_required_api_key_env_var("anthropic")
        assert result == "ANTHROPIC_API_KEY"

    def test_returns_joined_keys_for_gemini(self):
        """Gemini should return both keys joined with 'or'."""
        result = get_required_api_key_env_var("gemini")
        assert result == "GOOGLE_API_KEY or GEMINI_API_KEY"

    def test_returns_single_key_for_openrouter(self):
        """OpenRouter should return the single key name."""
        result = get_required_api_key_env_var("openrouter")
        assert result == "OPENROUTER_API_KEY"
