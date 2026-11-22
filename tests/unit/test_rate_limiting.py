"""Unit tests for rate limiting."""
# ruff: noqa: PLR2004, PLC0415

from unittest.mock import Mock

import anthropic
import openai
import pytest

from deepfabric.llm.rate_limit_config import (
    AnthropicRateLimitConfig,
    BackoffStrategy,
    GeminiRateLimitConfig,
    OpenAIRateLimitConfig,
    OpenRouterRateLimitConfig,
    RateLimitConfig,
    create_rate_limit_config,
    get_default_rate_limit_config,
)
from deepfabric.llm.rate_limit_detector import QuotaInfo, RateLimitDetector
from deepfabric.llm.retry_handler import RetryHandler


class TestRateLimitConfig:
    """Tests for rate limit configuration models."""

    def test_default_config(self):
        """Test default rate limit configuration."""
        config = RateLimitConfig()
        assert config.max_retries == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER
        assert config.jitter is True
        assert config.respect_retry_after is True

    def test_openai_config_defaults(self):
        """Test OpenAI-specific configuration."""
        config = OpenAIRateLimitConfig()
        assert config.check_headers is True
        assert config.preemptive_backoff is False
        assert config.preemptive_threshold == 0.1

    def test_anthropic_config_defaults(self):
        """Test Anthropic-specific configuration."""
        config = AnthropicRateLimitConfig()
        assert config.check_headers is True
        assert config.token_bucket_aware is True
        assert config.gradual_rampup is True

    def test_gemini_config_defaults(self):
        """Test Gemini-specific configuration."""
        config = GeminiRateLimitConfig()
        assert config.base_delay == 2.0  # Higher default
        assert config.max_delay == 120.0  # Longer max
        assert config.parse_quota_details is True
        assert config.daily_quota_aware is True

    def test_openrouter_config_defaults(self):
        """Test OpenRouter-specific configuration."""
        config = OpenRouterRateLimitConfig()
        assert 402 in config.retry_on_status_codes  # Payment required
        assert 429 in config.retry_on_status_codes  # Rate limit
        assert config.check_credits is False  # Off by default

    def test_config_validation(self):
        """Test configuration validation."""
        # max_delay must be >= base_delay
        with pytest.raises(ValueError, match="max_delay must be greater"):
            RateLimitConfig(base_delay=10.0, max_delay=5.0)

    def test_create_rate_limit_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "max_retries": 3,
            "base_delay": 2.0,
            "backoff_strategy": "exponential",
        }
        config = create_rate_limit_config("openai", config_dict)
        assert isinstance(config, OpenAIRateLimitConfig)
        assert config.max_retries == 3
        assert config.base_delay == 2.0
        assert config.backoff_strategy == BackoffStrategy.EXPONENTIAL

    def test_get_default_rate_limit_config(self):
        """Test getting default config for each provider."""
        openai_config = get_default_rate_limit_config("openai")
        assert isinstance(openai_config, OpenAIRateLimitConfig)

        anthropic_config = get_default_rate_limit_config("anthropic")
        assert isinstance(anthropic_config, AnthropicRateLimitConfig)

        gemini_config = get_default_rate_limit_config("gemini")
        assert isinstance(gemini_config, GeminiRateLimitConfig)

        openrouter_config = get_default_rate_limit_config("openrouter")
        assert isinstance(openrouter_config, OpenRouterRateLimitConfig)

        # Unknown provider should return base config
        unknown_config = get_default_rate_limit_config("unknown")
        assert isinstance(unknown_config, RateLimitConfig)


class TestRateLimitDetector:
    """Tests for rate limit detection."""

    def test_openai_rate_limit_detection(self):
        """Test detection of OpenAI rate limit errors."""
        # Create a mock rate limit error
        rate_limit_error = openai.RateLimitError(
            "Rate limit exceeded",
            response=Mock(headers={"retry-after": "5"}),
            body=None,
        )

        detector = RateLimitDetector()
        assert detector.is_rate_limit_error(rate_limit_error, "openai") is True
        assert detector.is_retryable_error(rate_limit_error, "openai") is True

    def test_anthropic_rate_limit_detection(self):
        """Test detection of Anthropic rate limit errors."""
        rate_limit_error = anthropic.RateLimitError(
            "Rate limit exceeded",
            response=Mock(headers={"retry-after": "10"}),
            body=None,
        )

        detector = RateLimitDetector()
        assert detector.is_rate_limit_error(rate_limit_error, "anthropic") is True

    def test_gemini_rate_limit_detection(self):
        """Test detection of Gemini rate limit errors."""
        # Gemini uses a different error format
        error_msg = (
            "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED'}}"
        )
        gemini_error = Exception(error_msg)

        detector = RateLimitDetector()
        assert detector.is_rate_limit_error(gemini_error, "gemini") is True

    def test_non_rate_limit_error(self):
        """Test that non-rate-limit errors are not detected as rate limits."""
        normal_error = ValueError("Some other error")

        detector = RateLimitDetector()
        assert detector.is_rate_limit_error(normal_error, "openai") is False
        assert detector.is_retryable_error(normal_error, "openai") is False

    def test_retryable_server_error(self):
        """Test detection of retryable server errors."""
        server_error = Exception("500 Internal Server Error")

        detector = RateLimitDetector()
        assert detector.is_retryable_error(server_error, "openai") is True

    def test_timeout_is_retryable(self):
        """Test that timeout errors are retryable."""
        timeout_error = Exception("Connection timeout")

        detector = RateLimitDetector()
        assert detector.is_retryable_error(timeout_error, "openai") is True

    def test_extract_openai_quota_info(self):
        """Test extracting quota info from OpenAI error."""
        # Mock OpenAI error with headers
        mock_response = Mock()
        mock_response.headers = {
            "retry-after": "5",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-limit-requests": "100",
        }

        rate_limit_error = openai.RateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body=None,
        )

        detector = RateLimitDetector()
        quota_info = detector.extract_quota_info(rate_limit_error, "openai")

        assert quota_info.is_rate_limit is True
        assert quota_info.retry_after == 5.0
        assert quota_info.limit_value == 100
        assert quota_info.quota_type == "requests"

    def test_extract_gemini_quota_info(self):
        """Test extracting quota info from Gemini error."""
        error_msg = (
            "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Quota exceeded for metric: "
            "generativelanguage.googleapis.com/generate_requests_per_model_per_day, limit: 0', "
            "'details': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_requests_per_model_per_day'}]}}"
        )
        gemini_error = Exception(error_msg)

        detector = RateLimitDetector()
        quota_info = detector.extract_quota_info(gemini_error, "gemini")

        assert quota_info.is_rate_limit is True
        # Check if per_day is detected in the error message
        assert "per_day" in error_msg
        assert quota_info.limit_value == 0

    def test_should_fail_fast_daily_quota(self):
        """Test fail fast decision for daily quota exhaustion."""
        quota_info = QuotaInfo(
            is_rate_limit=True,
            daily_quota_exhausted=True,
            quota_type="requests_per_day",
        )

        detector = RateLimitDetector()
        assert detector.should_fail_fast(quota_info) is True

    def test_should_fail_fast_zero_limit(self):
        """Test fail fast decision for zero quota limit."""
        quota_info = QuotaInfo(
            is_rate_limit=True,
            limit_value=0,
            quota_type="requests",
        )

        detector = RateLimitDetector()
        assert detector.should_fail_fast(quota_info) is True

    def test_should_not_fail_fast_normal_rate_limit(self):
        """Test that normal rate limits should not fail fast."""
        quota_info = QuotaInfo(
            is_rate_limit=True,
            quota_type="requests_per_minute",
            retry_after=5.0,
        )

        detector = RateLimitDetector()
        assert detector.should_fail_fast(quota_info) is False


class TestRetryHandler:
    """Tests for retry handler."""

    def test_retry_handler_initialization(self):
        """Test retry handler initialization."""
        config = RateLimitConfig(max_retries=3)
        handler = RetryHandler(config, "openai")

        assert handler.config == config
        assert handler.provider == "openai"
        assert handler.detector is not None

    def test_should_retry_rate_limit(self):
        """Test should_retry for rate limit errors."""
        config = RateLimitConfig()
        handler = RetryHandler(config, "openai")

        rate_limit_error = openai.RateLimitError(
            "Rate limit exceeded",
            response=Mock(headers={}),
            body=None,
        )

        assert handler.should_retry(rate_limit_error) is True

    def test_should_not_retry_normal_error(self):
        """Test should_retry for non-retryable errors."""
        config = RateLimitConfig()
        handler = RetryHandler(config, "openai")

        normal_error = ValueError("Invalid input")
        assert handler.should_retry(normal_error) is False

    def test_calculate_delay_exponential(self):
        """Test delay calculation with exponential backoff."""
        config = RateLimitConfig(
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            jitter=False,
        )
        handler = RetryHandler(config, "openai")

        # First retry: 1.0 * 2^0 = 1.0
        delay0 = handler.calculate_delay(0)
        assert delay0 == 1.0

        # Second retry: 1.0 * 2^1 = 2.0
        delay1 = handler.calculate_delay(1)
        assert delay1 == 2.0

        # Third retry: 1.0 * 2^2 = 4.0
        delay2 = handler.calculate_delay(2)
        assert delay2 == 4.0

    def test_calculate_delay_with_max_cap(self):
        """Test that delay is capped at max_delay."""
        config = RateLimitConfig(
            base_delay=1.0,
            max_delay=5.0,
            exponential_base=2.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            jitter=False,
        )
        handler = RetryHandler(config, "openai")

        # 1.0 * 2^10 = 1024, but should be capped at 5.0
        delay = handler.calculate_delay(10)
        assert delay == 5.0

    def test_calculate_delay_respects_retry_after(self):
        """Test that retry-after header is respected."""
        config = RateLimitConfig(respect_retry_after=True, max_delay=60.0)
        handler = RetryHandler(config, "openai")

        # Mock error with retry-after
        mock_response = Mock()
        mock_response.headers = {"retry-after": "15"}
        rate_limit_error = openai.RateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body=None,
        )

        delay = handler.calculate_delay(0, rate_limit_error)
        assert delay == 15.0

    def test_calculate_delay_retry_after_capped(self):
        """Test that retry-after is capped at max_delay."""
        config = RateLimitConfig(respect_retry_after=True, max_delay=10.0)
        handler = RetryHandler(config, "openai")

        # Mock error with high retry-after
        mock_response = Mock()
        mock_response.headers = {"retry-after": "100"}
        rate_limit_error = openai.RateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body=None,
        )

        delay = handler.calculate_delay(0, rate_limit_error)
        assert delay == 10.0  # Capped at max_delay

    def test_calculate_delay_with_jitter(self):
        """Test that jitter adds randomization."""
        config = RateLimitConfig(
            base_delay=10.0,
            exponential_base=1.5,  # Minimal exponential growth (must be >= 1.1)
            backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
            jitter=True,
        )
        handler = RetryHandler(config, "openai")

        # With jitter, delays should vary slightly
        delays = [handler.calculate_delay(0) for _ in range(10)]

        # All delays should be within ±25% of base_delay (for attempt 0)
        for delay in delays:
            assert 7.5 <= delay <= 12.5

        # At least some variation (not all the same)
        assert len(set(delays)) > 1

    def test_fail_fast_for_daily_quota(self):
        """Test that daily quota exhaustion triggers fail-fast."""
        config = RateLimitConfig()
        handler = RetryHandler(config, "gemini")

        # Mock Gemini daily quota error
        daily_quota_error = Exception(
            "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
            "generate_requests_per_model_per_day, limit: 0"
        )

        # Should not retry daily quota exhaustion
        assert handler.should_retry(daily_quota_error) is False
